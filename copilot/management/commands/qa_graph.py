import hashlib

from django.core.management import BaseCommand

from copilot.graph.rag_graph import build_graph
from copilot.rag.settings import TOP_K


class Command(BaseCommand):
    help = "Generate a QA graph from the database."

    def add_arguments(self, parser):
        parser.add_argument("--q", required=True, help="Câu hỏi")
        parser.add_argument("--k", type=int, default=TOP_K, help="số đoạn lấy ở retriever")
        parser.add_argument("--max_iters", type=int, default=2, help="số vòng tối đa")

    def handle(self, *args, **opts):
        q = opts["q"]
        k = opts["k"]
        max_iters = opts["max_iters"]

        app, memory = build_graph(max_iters)

        state = {
            "question": q,
            "k": k,
            "iterations": 0
        }

        tid = "cli-" + hashlib.md5(q.encode("utf-8")).hexdigest()[:8]
        cfg = {
            "configurable":
                {
                    "thread_id": tid,
                }
        }

        final = app.invoke(state, config=cfg)
        print("[INVOKE] DONE")
        print(f"Memory:")
        checkpoints = list(memory.list(cfg))
        for cp in checkpoints:
            print(cp)
        print()

        answer = final.get("answer", "").strip()
        verdict = final.get("verdict", "good")
        self.stdout.write(self.style.SUCCESS(f"[{verdict}]\n{answer}"))
