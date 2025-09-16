import json
import re
import time
from pathlib import Path
from statistics import mean

from django.core.management import BaseCommand, CommandError
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from copilot.llm.provider import get_chat
from copilot.management.commands.eval_retrieval import read_jsonl
from copilot.management.commands.rag_ask import ANSWER_PROMPT
from copilot.rag.retrieve import get_retriever
from copilot.rag.settings import DATA_DIR, TOP_K, LLM_MODEL


def tokenize(s: str) -> list[str]:
    return re.findall(r"[0-9A-Za-zÀ-ỹ]+", (s or "").strip())

def f1_score(pred: str, ref: str) -> float:
    p = tokenize(pred)
    r = tokenize(ref)
    if not p or not r:
        return 0.0

    p_set = set(p)
    r_set = set(r)
    overlap = len(p_set & r_set)

    if overlap == 0:
        return 0.0

    precision = overlap / len(p_set)
    recall = overlap / len(r_set)
    return 2 * (precision * recall) / (precision + recall)

'''ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Bạn là trợ lý phong thủy. Trả lời ngắn gọn, dựa trên ngữ cảnh cung cấp. "
     "Nếu ngữ cảnh không đủ, nói 'Tôi không chắc từ tài liệu hiện có.'"),
    ("human",
     "Câu hỏi: {question}\n\nNgữ cảnh:\n{context}\n\n"
     "Yêu cầu: trả lời 2–4 câu tiếng Việt, bám sát ngữ cảnh và tránh bịa đặt.")
])'''

JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Bạn là chuyên gia, giám khảo về huyền học. Cho điểm 0..1 về ĐỘ CHÍNH XÁC so với câu tham chiếu. "
     "Chỉ chấm độ đúng (không chấm văn phong). Trả về đúng JSON: "
     '{{"score": <float>, "rationale": "<ngắn gọn>"}}'),
    ("human",
     "Câu hỏi: {question}\nTham chiếu: {ref}\nTrả lời: {pred}\n"
     "Chấm điểm và giải thích ngắn.")
])

class Command(BaseCommand):
    help = "Đánh giá chất lượng câu trả lời: F1 lexical + (tuỳ chọn) LLM judge mini."

    def add_arguments(self, parser):
        parser.add_argument("--file", default=str(DATA_DIR / "eval" / "qa.jsonl"))
        parser.add_argument("--k", type=int, default=TOP_K)
        parser.add_argument("--model", default=LLM_MODEL)
        parser.add_argument("--judge", action="store_true",
                            help="Bật chấm điểm bằng LLM")

    def handle(self, *args, **opts):
        eval_path = Path(opts["file"])
        k = opts["k"]
        model = opts["model"]
        use_judge = opts["judge"]

        data = list(read_jsonl(eval_path))
        if not data:
            raise CommandError(f"Eval set rỗng: {eval_path}")

        retriever = get_retriever(k)
        llm = get_chat(model=model)

        f1s, judge_scores = [], []
        t0 = time.time()

        for i, item in enumerate(data):
            q = item["q"]
            ref = item.get("ref", "").strip()

            docs = retriever.get_relevant_documents(q)

            ctx_lines = []

            for j, d in enumerate(docs):
                snippet = d.page_content.strip().replace("\n", " ")

                if len(snippet) > 500:
                    snippet = snippet[:500] + "..."
                src = d.metadata.get("source", "")
                ctx_lines.append(f"[{j + 1}] {snippet} (SOURCE: {src})")

            context = "\n\n".join(ctx_lines) if ctx_lines else "(Không có ngữ cảnh)"
            print(context)

            pred = (ANSWER_PROMPT | llm).invoke({
                "question": q,
                "context": context
            }).content.strip()

            f1 = f1_score(pred, ref)
            f1s.append(f1)
            line = f"[{i+1}] Q: {q}\n REF: {ref}\n PRED: {pred}\n F1: {f1:.3f}"

            if use_judge:
                judge = (JUDGE_PROMPT | llm).invoke({
                    "question": q,
                    "ref": ref,
                    "pred": pred
                }).content.strip()

                m = re.search(r"\{.*\}", judge, re.DOTALL)
                if m:
                    try:
                        score = float(json.loads(m.group(0))["score"])
                    except Exception:
                        score = 0.0

                judge_scores.append(score)
                line += f" | Judge: {score:.3f} ({judge})"

            self.stdout.write(line + "\n")

        dt = time.time() - t0
        summary = f"\nDone in {dt:.2f}s | k={k}\n Avg F1: {mean(f1s):.3f}"
        if use_judge and judge_scores:
            summary += f" | Avg Judge: {mean(judge_scores):.3f}"
        self.stdout.write(self.style.SUCCESS(summary))
