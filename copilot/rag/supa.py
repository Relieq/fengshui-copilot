import os
from typing import Optional

from supabase import client, create_client

_SUPABASE_CLIENT: Optional[client] = None


def get_supabase_client() -> client:
    global _SUPABASE_CLIENT
    if _SUPABASE_CLIENT is not None:
        return _SUPABASE_CLIENT

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("Thiếu SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY trong .env")

    _SUPABASE_CLIENT = create_client(url, key)
    return _SUPABASE_CLIENT


def get_supabase_table_name() -> str:
    return os.getenv("SUPABASE_TABLE", "documents")


def get_supabase_query_name() -> str:
    # tên function tìm kiếm (match_documents) bạn đã tạo trong DB
    return os.getenv("SUPABASE_QUERY_NAME", "match_documents")
