from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

from copilot.rag.settings import TOP_K, EMBEDDING_MODEL, CHROMA_DIR, COLLECTION_NAME


def get_retriever(top_k: int | None = None):
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    return vs.as_retriever(search_kwargs={"k": top_k or TOP_K})