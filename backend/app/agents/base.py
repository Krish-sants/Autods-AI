import logging
from functools import lru_cache
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def get_llm() -> Any | None:
    """Returns a configured Gemini chat model, or None if no API key is set.

    Callers MUST handle the None case by falling back to deterministic/templated
    behavior — the whole pipeline is designed to work with zero API keys.
    """
    settings = get_settings()
    if not settings.GOOGLE_API_KEY:
        logger.info("GOOGLE_API_KEY not set — LLM-dependent agents will use template fallback.")
        return None

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.GEMINI_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.3,
        )
    except Exception:
        logger.exception("Failed to initialize Gemini LLM — falling back to templates.")
        return None


def is_llm_available() -> bool:
    return get_llm() is not None


async def safe_llm_call(prompt: str, fallback: str) -> tuple[str, bool]:
    """Returns (text, used_llm). Never raises — any LLM failure returns the fallback text."""
    llm = get_llm()
    if llm is None:
        return fallback, False
    try:
        response = await llm.ainvoke(prompt)
        text = getattr(response, "content", None) or str(response)
        text = text.strip()
        return (text if text else fallback), bool(text)
    except Exception:
        logger.exception("LLM call failed — falling back to template text.")
        return fallback, False
