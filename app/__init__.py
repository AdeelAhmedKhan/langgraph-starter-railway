"""LangGraph starter — FastAPI + LangGraph + Postgres 18 + pgvector."""

# Load .env at package import so `cp .env.example .env` works out of the
# box for every entry point (app, scripts, tests). The TanStack starter's
# seed script tripped on exactly this — tsx never loaded .env.
from dotenv import load_dotenv as _load_dotenv

_load_dotenv()
