import time

from django.core.management import BaseCommand
from langchain_core.prompts import ChatPromptTemplate

from copilot.llm.provider import get_chat
from copilot.rag.retriever import get_retriever
from copilot.rag.settings import TOP_K, LLM_MODEL

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Bạn là trợ lý phong thuỷ. Trả lời ngắn gọn, dựa trên ngữ cảnh được cung cấp."
     " Nếu ngữ cảnh không đủ, hãy nói 'Tôi không chắc từ tài liệu hiện có.'"),
    ("human",
     "Câu hỏi: {question}\n\n"
     "Ngữ cảnh (có thể rút gọn):\n{context}\n\n"
     "Yêu cầu:\n- Trả lời 2–4 câu tiếng Việt, bám sát ngữ cảnh.\n"
     "- Liệt kê nguồn (tên file) đã dùng ở cuối câu trả lời.")
])

class Command(BaseCommand):
    help = "Hỏi–đáp với RAG (retriever + LLM), in câu trả lời kèm nguồn."

    def add_arguments(self, parser):
        parser.add_argument("--q", dest="question", required=True,
                            help="Câu hỏi phong thủy (tiếng Việt)")
        parser.add_argument("--k", dest="top_k", type=int, default=TOP_K,
                            help="Số đoạn trích dẫn lấy về (top k)")
        parser.add_argument("--model", default=LLM_MODEL,
                            help="Tên model Ollama")
        parser.add_argument("--temp", type=float, default=0.0)

    def handle(self, *args, **opts):
        q = opts["question"]
        k = opts["top_k"]
        model = opts["model"]
        temp = opts["temp"]

        # Lấy ngữ cảnh
        retriever = get_retriever(k)
        docs = retriever.get_relevant_documents(q)

        # Ghép đoạn trích tài liệu + nguồn để tạo ngữ cảnh
        ctx_lines = []
        used_files = set()

        for i, d in enumerate(docs):
            snippet = d.page_content.strip().replace("\n", " ")

            if len(snippet) > 500:
                snippet = snippet[:500] + "..."

            src = d.metadata.get("source", "")
            used_files.add(src)

            ctx_lines.append(f"[{i+1}] {snippet} (SOURCE: {src})")

        context = "\n\n".join(ctx_lines) if ctx_lines else "(Không có ngữ cảnh)"

        llm = get_chat(model, temperature=temp)
        chain = ANSWER_PROMPT | llm

        t0 = time.time()
        res = chain.invoke({
            "question": q,
            "context": context
        })
        dt = time.time() - t0

        answer = res.content.strip()
        sources = ", ".join(sorted(used_files)) if used_files else "Không có"
        self.stdout.write(f"[{model}] {answer}\n\n"
                          f"Sources: {sources}\n"
                          f"Took {dt:.2f}s")
