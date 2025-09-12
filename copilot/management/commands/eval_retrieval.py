import json
import time
from pathlib import Path
from statistics import mean

from django.core.management import BaseCommand, CommandError

from copilot.rag.retrieve import get_retriever
from copilot.rag.settings import DATA_DIR, TOP_K


def read_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path, 'r', encoding='utf') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

class Command(BaseCommand):
    help = "Đánh giá retrieval ở file-level với 2 phương pháp: Recall@k và MRR@k"

    def add_arguments(self, parser):
        parser.add_argument("--file", default=str(DATA_DIR / "eval" / "qa.jsonl"))
        parser.add_argument("--k", type=int, default=TOP_K)

    def handle(self, *args, **opts):
        eval_path = Path(opts["file"])
        k = opts["k"]

        data = list(read_jsonl(eval_path))
        if not data:
            raise CommandError(f"Eval set rỗng: {eval_path}")

        retriever = get_retriever(k)

        recalls, mrrs = [], []
        t0 = time.time()

        for i, item in enumerate(data):
            q = item["q"]
            gold = item.get("sources", "")
            docs = retriever.get_relevant_documents(q)

            ranked_src = [d.metadata.get("source", "") for d in docs]
            hit = any(r in gold for r in ranked_src)
            recalls.append(hit)

            rr = 0.0
            for rank, src in enumerate(ranked_src):
                if src in gold:
                    rr = 1.0 / (rank + 1)
                    break
            mrrs.append(rr)

            self.stdout.write(
                f"[{i+1}] Q: {q}\n"
                f"gold: {gold}\n"
                f"got: {ranked_src}\n"
                f"Hit: {hit}, Reciprocal Rank (rr): {rr:.3f}\n"
            )

        dt = time.time() - t0
        self.stdout.write(self.style.SUCCESS(
              f"\nDone in {dt:.2f}s | k={k}\n"
              f"Recall@{k}: {mean(recalls):.3f} | MRR@{k}: {mean(mrrs):.3f} "
              f"{len(data)} câu."
        ))

