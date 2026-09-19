"""
Provider-agnostic LLM factory.

Usage:
    from ai.llm.llm_factory import get_llm, get_embedder

Supported providers (set LLM_PROVIDER in .env):
    google      → gemini-1.5-flash via langchain-google-genai  [DEFAULT]
    openai      → gpt-4o-mini via langchain-openai
    anthropic   → claude-3-haiku via langchain-anthropic
    groq        → llama3-8b-8192 via langchain-groq
    ollama      → any local model — India-hosted / open-source swap-in point

Embedding providers (set EMBEDDING_MODEL in .env):
    sentence-transformers → all-MiniLM-L6-v2, local, no API cost  [DEFAULT]
    openai                → text-embedding-3-small
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

from langchain_core.language_models import BaseChatModel

from backend.config.settings import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """
    Return a cached LangChain chat model based on LLM_PROVIDER env var.
    If no valid API key is set, automatically falls back to MockChatModel
    so the system runs fully end-to-end without errors until the user adds their key.
    """
    provider = settings.LLM_PROVIDER.lower()
    model_name = settings.llm_model_name

    logger.info("Initialising LLM: provider=%s model=%s", provider, model_name)

    def _is_placeholder_key(key: str) -> bool:
        if not key or not key.strip():
            return True
        k = key.strip().lower()
        return (
            k.startswith("your_")
            or "actual_key" in k
            or "placeholder" in k
            or len(k) < 15
        )

    if provider in ["mock", "local", "offline"]:
        from ai.llm.mock_llm import MockChatModel
        return MockChatModel()

    elif provider == "google":
        if _is_placeholder_key(settings.GOOGLE_API_KEY):
            logger.warning(
                "GOOGLE_API_KEY is not set or is a placeholder in .env. "
                "Using built-in MockChatModel. Add your real Google Gemini API key to .env anytime to switch automatically."
            )
            from ai.llm.mock_llm import MockChatModel
            return MockChatModel()
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.1,
            convert_system_message_to_human=True,
        )

    elif provider == "openai":
        if _is_placeholder_key(settings.OPENAI_API_KEY):
            logger.warning("OPENAI_API_KEY is not set in .env. Using built-in MockChatModel.")
            from ai.llm.mock_llm import MockChatModel
            return MockChatModel()
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.1,
        )

    elif provider == "anthropic":
        if _is_placeholder_key(settings.ANTHROPIC_API_KEY):
            logger.warning("ANTHROPIC_API_KEY is not set in .env. Using built-in MockChatModel.")
            from ai.llm.mock_llm import MockChatModel
            return MockChatModel()
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model_name,
            api_key=settings.ANTHROPIC_API_KEY,
            temperature=0.1,
        )

    elif provider == "groq":
        if _is_placeholder_key(settings.GROQ_API_KEY):
            logger.warning("GROQ_API_KEY is not set in .env. Using built-in MockChatModel.")
            from ai.llm.mock_llm import MockChatModel
            return MockChatModel()
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model_name,
            api_key=settings.GROQ_API_KEY,
            temperature=0.1,
        )

    elif provider == "ollama":
        # Local / India-hosted open-source model swap-in point.
        # To use: set LLM_PROVIDER=ollama, OLLAMA_MODEL=<model_name>
        # and ensure Ollama is running (ollama serve).
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            model=model_name,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0.1,
        )

    else:
        logger.warning("Unknown provider '%s' — using MockChatModel fallback", provider)
        from ai.llm.mock_llm import MockChatModel
        return MockChatModel()


# ── Embedding model ───────────────────────────────────────────────────────────

_embedder = None


def get_embedder():
    """
    Return a cached embedding model.
    Returns an object with an .encode(texts) method
    (compatible with sentence-transformers and openai wrappers).
    """
    global _embedder
    if _embedder is not None:
        return _embedder

    provider = settings.EMBEDDING_MODEL.lower()
    model_name = settings.EMBEDDING_MODEL_NAME

    logger.info("Initialising embedder: provider=%s model=%s", provider, model_name)

    if provider == "sentence-transformers":
        from sentence_transformers import SentenceTransformer
        _embedder = _STEmbedder(SentenceTransformer(model_name))

    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        _oe = OpenAIEmbeddings(
            model=model_name, api_key=settings.OPENAI_API_KEY
        )
        _embedder = _LangChainEmbedder(_oe)

    else:
        raise ValueError(
            f"Unsupported EMBEDDING_MODEL: '{provider}'. "
            "Supported: sentence-transformers | openai"
        )

    return _embedder


class _STEmbedder:
    """Thin wrapper around SentenceTransformer."""

    def __init__(self, model):
        self._model = model

    def encode(self, texts: List[str], batch_size: int = 64, show_progress: bool = False) -> List[List[float]]:
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    def encode_one(self, text: str) -> List[float]:
        return self.encode([text])[0]


class _LangChainEmbedder:
    """Wrapper around LangChain embedding models."""

    def __init__(self, model):
        self._model = model

    def encode(self, texts: List[str], **kwargs) -> List[List[float]]:
        return self._model.embed_documents(texts)

    def encode_one(self, text: str) -> List[float]:
        return self._model.embed_query(text)
