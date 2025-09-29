from django.core.management import BaseCommand

from copilot.graph.run import run_graph
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

        result = run_graph(q, k, max_iters, make_thread_id_from_question=True)

        answer = result.get("answer", "").strip()
        verdict = result.get("verdict", "good")
        self.stdout.write(self.style.SUCCESS(f"[{verdict}]\n{answer}"))
