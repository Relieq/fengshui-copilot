import os

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


class ProviderError(RuntimeError):
    ...

def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name, default)
    return v.strip() if isinstance(v, str) else v

def get_chat(model: str | None, temperature: float = 0.0):
    provider = _env("LLM_PROVIDER", "ollama").lower()
    model = _env("LLM_MODEL", "llama3.1:8b").lower()
    print(f"[LLM] Provider={provider}, Model={model}, Temp={temperature}")

    if provider == "ollama":
        return ChatOllama(model=model, temperature=temperature)

    if provider == "openrouter":
        api_key = _env("OPENROUTER_API_KEY")
        if not api_key:
            raise ProviderError("Thiếu OPENROUTER_API_KEY trong .env")

        base_url = _env("OPENROUTER_BASE_URL")
        headers = {
            "HTTP-Referer": _env("OPENROUTER_HTTP_REFERER", "http://localhost"),
            "X-Title": _env("OPENROUTER_APP_TITLE", "fengshui-copilot-dev"),
        }

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
            default_headers=headers,
        )

    if provider == "openai":
        api_key = _env("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError("Thiếu OPENAI_API_KEY trong .env")

        return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)

    raise ProviderError(f"LLM_PROVIDER không được hỗ trợ: {provider}")
