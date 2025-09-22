import os

from langchain_chroma import Chroma
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_ollama import OllamaEmbeddings
from supabase import create_client

from copilot.llm.embeddings import get_embeddings
from copilot.rag.settings import TOP_K, EMBEDDING_MODEL, CHROMA_DIR, COLLECTION_NAME
from copilot.rag.supa import *


def get_vectorstore():
    client = get_supabase_client()
    embeddings = get_embeddings()
    table = get_supabase_table_name()
    query = get_supabase_query_name()
    return SupabaseVectorStore(client=client, embedding=embeddings, table_name=table, query_name=query)


def get_retriever(top_k: int | None = TOP_K):
    vs = get_vectorstore()

    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": max(20, 5 * top_k)}
        # search_kwargs={"k": top_k}
    )
