"""Environment configuration and lazily constructed LLM clients."""

import os

from dotenv import load_dotenv


load_dotenv()


class Config:
    def __init__(self) -> None:
        self.base_url = os.getenv("FP2MP_CHAT_URL") or os.getenv("CHAT_URL")
        self.api_key = os.getenv("FP2MP_API_KEY") or os.getenv("CHAT_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("FP2MP_MODEL") or os.getenv("CHAT_MODEL") or "openai/gpt-4.1"


config = Config()


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def init_llm(model: str | None = None, temperature: float = 0.5):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model or config.model,
        base_url=config.base_url,
        api_key=config.api_key,
        temperature=temperature,
    )


def __getattr__(name: str):
    if name == "llm":
        value = init_llm(temperature=_float_env("CHAT_TEMPERATURE", 0.5))
    elif name == "embedding":
        from langchain_openai import OpenAIEmbeddings

        value = OpenAIEmbeddings(
            model=os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small",
            base_url=os.getenv("EMBEDDING_URL"),
            api_key=os.getenv("EMBEDDING_API_KEY") or config.api_key,
        )
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


__all__ = [
    "config",
    "init_llm",
    "llm",
    "embedding",
]
