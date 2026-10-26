"""Embeddings through OpenAI — only when a key is configured.

Without ``OPENAI_API_KEY`` every function here returns ``None`` and the
caller falls back to full-text search. Embeddings are never faked:
a document with no embedding simply has a NULL vector column.
"""

from __future__ import annotations

from . import settings

_EMBEDDING_DIMENSIONS = 1536  # text-embedding-3-small


def _client():
    if not settings.embeddings_configured():
        return None
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.OPENAI_EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
    )


def embed_texts(texts: list[str]) -> list[list[float]] | None:
    """Embed a batch of texts. None when embeddings are not configured."""
    client = _client()
    if client is None:
        return None
    return client.embed_documents(texts)


def embed_query(text: str) -> list[float] | None:
    """Embed one query string. None when embeddings are not configured."""
    client = _client()
    if client is None:
        return None
    return client.embed_query(text)
