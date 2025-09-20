from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_ollama import OllamaEmbeddings
from mpmath.libmp import normalize
from openai import max_retries

from copilot.llm import env


class EmbeddingProviderError(RuntimeError):...


def get_embeddings():
    provider = env("EMBED_PROVIDER", "hf_endpoint")
    model = env("EMBEDDING_MODEL", "BAAI/bge-m3")

    if provider == "ollama":
        return OllamaEmbeddings(model=model)

    if provider == "hf_inference":
        api_key = env("HUGGINGFACEHUB_API_TOKEN")
        if not api_key:
            raise EmbeddingProviderError("Thiếu HUGGINGFACEHUB_API_TOKEN trong .env")
        return HuggingFaceInferenceAPIEmbeddings(api_key=api_key, model_name=model)

    if provider == "hf_endpoint":
        api_key = env("HUGGINGFACEHUB_API_TOKEN")
        if not api_key:
            raise EmbeddingProviderError("Thiếu HUGGINGFACEHUB_API_TOKEN trong .env")
        return HuggingFaceEndpointEmbeddings(
            model=model,
            task="feature-extraction",
            huggingfacehub_api_token=api_key
        )

    raise EmbeddingProviderError(f"EMBEDDING_PROVIDER không được hỗ trợ: {provider}")
