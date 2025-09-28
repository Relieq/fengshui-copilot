from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from copilot.llm import env


class ProviderError(RuntimeError):
    ...


def get_chat(role: str | None = None, temperature: float = 0.0):
    provider = env("LLM_PROVIDER", "ollama").lower()  # Trong project này thì chỉ dùng provider chung thôi
    model = env(f"{role}_MODEL".upper(), env("LLM_MODEL", "llama3.1:8b")).lower()
    print(f"[LLM] Provider={provider}, Model={model}, Temp={temperature}")

    if provider == "ollama":
        return ChatOllama(model=model, temperature=temperature)

    if provider == "openrouter":
        api_key = env("OPENROUTER_API_KEY")
        if not api_key:
            raise ProviderError("Thiếu OPENROUTER_API_KEY trong .env")

        base_url = env("OPENROUTER_BASE_URL")
        headers = {
            "HTTP-Referer": env("OPENROUTER_HTTP_REFERER", "http://localhost"),
            "X-Title": env("OPENROUTER_APP_TITLE", "fengshui-copilot-dev"),
        }

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
            base_url=base_url,
            default_headers=headers,
        )

    if provider == "openai":
        api_key = env("OPENAI_API_KEY")
        if not api_key:
            raise ProviderError("Thiếu OPENAI_API_KEY trong .env")

        return ChatOpenAI(model=model, temperature=temperature, api_key=api_key)

    raise ProviderError(f"LLM_PROVIDER không được hỗ trợ: {provider}")
