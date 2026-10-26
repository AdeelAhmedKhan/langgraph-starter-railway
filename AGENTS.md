# AGENTS.md — working in this repo

Read this before changing anything. The user of this repo is a developer
deploying a LangGraph starter on Railway; keep it runnable, minimal, and
honest.

## Stack (pinned — do not bump majors casually)

Python 3.12 + FastAPI 0.162.0 + Uvicorn 0.54.0 + LangGraph 1.2.17 +
LangChain 1.4.7 + langchain-openai 1.7.2 + langgraph-checkpoint-postgres
3.2.0 + LangSmith 0.14.12 + psycopg 3.3.6 (binary, pool) + pgvector
0.5.1 + Pydantic 2.14.1. Versions verified on PyPI 2026-10-26; see
README "Stack & versions".

## The one rule that matters

**Never fake a capability.** No LLM key means the draft node returns a
verbatim extract of the retrieved documents, labelled `llm_used: false`.
No embeddings means retrieval is full-text search, and the method used
is recorded in state. No LangSmith key means tracing is off and health
reports `langsmith: "not_configured"`. Anything not configured reports
clearly instead of crashing — and nothing in a response is invented.

## Layout

- `app/main.py` — FastAPI app, routes, landing page. Loads `.env`
  first (via `app/__init__.py`), so `cp .env.example .env` works out
  of the box. The graph is built lazily: the app boots and
  `/api/health` answers even when Postgres is down.
- `app/settings.py` — env vars only. `config_status()` is the honest
  capability report.
- `app/db.py` — the shared psycopg pool, retrieval (vector when
  embeddings exist, full-text otherwise), notes, `db_ok()`. Full-text
  uses OR over significant terms with prefix matching (`term:*`), not
  `plainto_tsquery` — the AND semantics of the latter made natural
  questions match nothing. The `vector` extension needs a superuser:
  `migrate.py` says so plainly instead of crashing with a raw error.
- `app/embeddings.py` — OpenAI embeddings, env-gated; returns None
  when not configured.
- `app/graph.py` — the demo graph: `retrieve → draft → review →
  publish`. The `review` node calls `interrupt()`; nothing is saved
  until the run is resumed with an approve.
- `scripts/migrate.py` — idempotent DDL + `PostgresSaver.setup()`.
  This is the pre-deploy command in `railway.toml`.
- `scripts/seed.py` — idempotent demo documents; embeddings only
  when `OPENAI_API_KEY` is set.

## Commands

```bash
cp .env.example .env   # then fill in DATABASE_URL
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/migrate.py
python scripts/seed.py
uvicorn app.main:app --reload --port 8000
python -m pytest       # DB-free API tests (needs requirements-dev.txt)
python -m compileall app scripts   # syntax check after changes
```

## Do / Don't

- DO run `python -m compileall app scripts` + `python -m pytest`
  after changes and fix real errors. Do not claim a test passes if it
  did not. Do not claim a flow works from a status code alone — check
  the response body.
- DO keep `requirements.txt` as the single source of truth for pinned
  versions. If you bump one, verify it on PyPI first and update the
  README version table in the same change.
- DO keep secrets out of the repo. `.env.example` holds placeholders
  only. There are no required secrets for this starter — the LLM and
  LangSmith keys are optional. `DATABASE_URL` has no default in code:
  it must come from `.env` locally or the Postgres reference variable
  on Railway; unset means "no database", and the app says so honestly.
- DON'T add a feature that needs a new env var without documenting it
  in `.env.example` and README.
- DON'T invent versions. Check `pypi.org/pypi/<pkg>/json` before
  pinning.
- DON'T generate fake embeddings, fake LLM output, or fake traces.
  A NULL embedding and a `not_configured` status are honest states.
- DON'T push, deploy, or publish from here. This repo is prepared
  locally; pushing and Railway template creation are separate,
  human-approved steps.
