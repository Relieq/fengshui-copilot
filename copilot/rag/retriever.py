import os

from langchain_chroma import Chroma
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_ollama import OllamaEmbeddings
from supabase import create_client

from copilot.llm.embeddings import get_embeddings
from copilot.rag.settings import TOP_K, EMBEDDING_MODEL, CHROMA_DIR, COLLECTION_NAME


def _supabase_client():
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    assert url and key, "Thiếu SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY trong .env"
    return create_client(url, key)

def get_vectorstore():
    client = _supabase_client()
    embeddings = get_embeddings()
    table = os.getenv("SUPABASE_TABLE", "documents")
    query = os.getenv("SUPABASE_QUERY_NAME", "match_documents")
    return SupabaseVectorStore(client=client, embedding=embeddings, table_name=table, query_name=query)

def get_retriever(top_k: int | None = None):
    vs = get_vectorstore()

    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": max(20, 5 * top_k)}
    )