import time

from django.core.management.base import BaseCommand

from copilot.rag.ingest import ingest_corpus
from copilot.rag.supa import *


class Command(BaseCommand):
    help = "Ingest tài liệu phong thuỷ vào Chroma (embed->split)"

    def add_arguments(self, parser):
        parser.add_argument(
            "--reset", action="store_true",
            help="Xóa index cũ trước khi ingest"
        )

    def handle(self, *args, **opts):
        if opts["reset"]:
            self.stdout.write("Xoá index cũ...")
            supa_client = get_supabase_client()
            # table = get_supabase_table_name()
            # supa_client.table(table).delete().neq("uid", None).execute()
            # self.stdout.write(self.style.WARNING(f"Đã reset bảng Supabase: {table}"))
            # gọi RPC thay vì delete()
            supa_client.rpc("reset_documents").execute()
            self.stdout.write(self.style.SUCCESS("Đã TRUNCATE + RESTART IDENTITY cho bảng documents."))

        t0 = time.time()
        stats = ingest_corpus()
        dt = time.time() - t0

        self.stdout.write(self.style.SUCCESS(
            f"INGEST DONE in {dt:.2f}s | files={stats['files']} "
            f"chunks={stats['chunks']} added={stats['added']} deleted={stats['deleted']}\n"
            f"corpus={stats['corpus_dir']} "
            # f"| chroma={stats['chroma_dir']} | collection={stats['collection']}"
        ))
