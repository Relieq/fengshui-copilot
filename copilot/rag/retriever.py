from langchain.retrievers import MultiQueryRetriever
from langchain_chroma import Chroma
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_ollama import OllamaEmbeddings

from copilot.llm.embeddings import get_embeddings
from copilot.llm.provider import get_chat
from copilot.rag.settings import TOP_K, EMBEDDING_MODEL, CHROMA_DIR, COLLECTION_NAME, RETRIEVER_MODE
from copilot.rag.supa import *


def get_vectorstore():
    supa_client = get_supabase_client()
    embeddings = get_embeddings()
    table = get_supabase_table_name()
    query = get_supabase_query_name()
    return SupabaseVectorStore(client=supa_client, embedding=embeddings, table_name=table, query_name=query)


def get_retriever(top_k: int = TOP_K):
    vs = get_vectorstore()
    mode = RETRIEVER_MODE
    print("Retriever mode:", mode, "| top_k:", top_k)

    base = vs.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )
    if mode == "mq":
        llm = get_chat("MQR", temperature=0)
        mqr = MultiQueryRetriever.from_llm(retriever=base, llm=llm, include_original=True)

        class _Adapter:
            @staticmethod # phương thức không cần tham chiếu đến lớp (không truyền self)
            def invoke(q: str):
                docs = mqr.invoke(q)
                return docs[:top_k]
        return _Adapter()

    if mode == "similarity":
        return base

    return vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": max(20, 5 * top_k)}
        # search_kwargs={"k": top_k}
    )
