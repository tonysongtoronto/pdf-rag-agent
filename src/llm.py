"""
Single entry point for the DeepSeek chat LLM. DeepSeek exposes an
OpenAI-compatible API, so we use `langchain_openai.ChatOpenAI` pointed
at DeepSeek's base URL.
"""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from src import config

_llm_instance: ChatOpenAI | None = None


def get_llm(temperature: float | None = None) -> ChatOpenAI:
    """Return a (lazily-created, cached) DeepSeek chat model instance."""
    global _llm_instance
    if _llm_instance is None:
        if not config.DEEPSEEK_API_KEY:
            raise RuntimeError(
                "DEEPSEEK_API_KEY is not set. Add it to your .env file before "
                "talking to the agent."
            )
        _llm_instance = ChatOpenAI(
            model=config.DEEPSEEK_MODEL,
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL,
            temperature=temperature if temperature is not None else config.LLM_TEMPERATURE,
        )
    return _llm_instance
