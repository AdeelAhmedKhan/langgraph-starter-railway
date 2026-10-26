"""Settings: environment variables only. No secrets in the repo.

Every optional integration reports its state honestly through
``config_status()`` — "configured" or "not_configured". Nothing in this
app invents a capability it does not have: without ``OPENAI_API_KEY``
there is no LLM and no embeddings, and the graph says so in every
response that touches them.
"""

import os

APP_VERSION = "0.1.0"


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default) or default


def _get_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


DATABASE_URL = _get("DATABASE_URL")
# No default on purpose: the local-dev credential must live in .env,
# never as a literal in the repo. Unset means "no database" — the app
# boots anyway and /api/health reports db: "error".
APP_URL = _get("APP_URL", "http://localhost:8000")
PORT = int(_get("PORT", "8000"))

OPENAI_API_KEY = _get("OPENAI_API_KEY")
OPENAI_MODEL = _get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = _get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

LANGSMITH_API_KEY = _get("LANGSMITH_API_KEY")
LANGSMITH_TRACING = _get_bool("LANGSMITH_TRACING", False)
LANGSMITH_PROJECT = _get("LANGSMITH_PROJECT", "langgraph-starter")


def llm_configured() -> bool:
    return bool(OPENAI_API_KEY)


def embeddings_configured() -> bool:
    # Embeddings go through the same OpenAI key as the LLM.
    return bool(OPENAI_API_KEY)


def langsmith_configured() -> bool:
    return bool(LANGSMITH_API_KEY and LANGSMITH_TRACING)


def config_status() -> dict:
    """Honest capability report for /api/health and /api/graph/info."""
    return {
        "llm": "configured" if llm_configured() else "not_configured",
        "embeddings": "configured" if embeddings_configured() else "not_configured",
        "langsmith": "configured" if langsmith_configured() else "not_configured",
    }
