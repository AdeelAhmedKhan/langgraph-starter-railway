# LangGraph Starter — Deploy on Railway

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/deploy/langgraph-starter)

A production-wired LangGraph starter: **FastAPI** + **LangGraph 1.2**
with a **Postgres 18 checkpointer** + **pgvector** retrieval, a
**human-in-the-loop** review step, and optional LangSmith tracing.
One-click deploy on Railway; nothing in here is a demo stub — the graph
runs against a real database, persistence survives restarts, and
anything not configured tells you so instead of crashing.

## Stack & versions (pinned, verified 2026-10-26)

| Piece | Version | Notes |
| --- | --- | --- |
| Python | 3.12 | `requires-python = ">=3.11,<3.14"` |
| FastAPI | 0.162.0 | |
| Uvicorn | 0.54.0 | Production server |
| LangGraph | 1.2.17 | StateGraph + `interrupt()` |
| LangChain / LangChain Core | 1.4.7 / 1.7.0 | |
| langchain-openai | 1.7.2 | Chat + embeddings, env-gated |
| langgraph-checkpoint-postgres | 3.2.0 | `PostgresSaver` persistence |
| LangSmith | 0.14.12 | Tracing, env-gated |
| psycopg (binary, pool) | 3.3.6 | Postgres driver + pool |
| pgvector | 0.5.1 | `vector(1536)` embeddings |
| Pydantic | 2.14.1 | Request validation |
| httpx | 0.28.1 | |
| python-dotenv | 1.2.4 | Loads `.env` at import |

## What you get

- **The demo graph** — `retrieve → draft → review → publish`. A
  research assistant: it retrieves real documents from Postgres, drafts
  an answer, then **stops for a human decision** before anything is
  saved. Resume it with an approve or a reject; the run completes or
  ends cleanly either way.
- **Persistence** — every graph step is checkpointed in Postgres
  (`PostgresSaver`). A paused run survives a server restart; resume it
  later with the same `thread_id`.
- **Retrieval** — pgvector cosine search when embeddings exist,
  PostgreSQL full-text search otherwise. The method used is recorded in
  the run state and returned by the API — never hidden.
- **Honest LLM behaviour** — with `OPENAI_API_KEY` set, drafting uses
  `gpt-4o-mini` over the retrieved documents. Without it, the draft is
  a verbatim extract of the retrieved documents, labelled
  `llm_used: false`. No fake LLM output, ever.
- **LangSmith tracing** — env-gated. Set `LANGSMITH_API_KEY` and
  `LANGSMITH_TRACING=true` to trace runs; without them, tracing is off
  and health reports `langsmith: "not_configured"`.
- **Healthcheck** — `GET /api/health` returns HTTP 200 with real DB
  and integration status; Railway uses it as the deploy healthcheck
  (`railway.toml`).
- **API** — `POST /api/runs` to start a run, `GET /api/runs/{id}`
  to inspect it, `POST /api/runs/{id}/resume` to approve or reject,
  `GET /api/notes` for published answers, `GET /api/graph/info` for
  the graph shape. Swagger docs at `/docs`.

## Run it locally

```bash
cp .env.example .env   # then fill in DATABASE_URL
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/migrate.py   # creates extension, tables, checkpoint tables
python scripts/seed.py      # seeds demo documents (embeds them if OPENAI_API_KEY is set)
uvicorn app.main:app --reload --port 8000
```

You need a Postgres 18 for anything past the landing page. The landing
page and `/api/health` work without one (health reports `db: "error"`).

**Local superuser note:** the `vector` extension can only be created by
a Postgres superuser. If your local database user is not one, the
migrate script stops with a clear message — run once as the superuser
(`psql -d <your-db> -c "CREATE EXTENSION vector"`) and re-run it. On
Railway the database user is a superuser, so the pre-deploy migrate
needs no extra step.

Try it:

```bash
curl -X POST localhost:8000/api/runs \
  -H 'Content-Type: application/json' \
  -d '{"question": "How do human-in-the-loop interrupts work?"}'

# approve the draft (thread_id from the response):
curl -X POST localhost:8000/api/runs/<thread_id>/resume \
  -H 'Content-Type: application/json' \
  -d '{"approved": true}'
```

## Deploy on Railway (exact steps)

1. Deploy this repo from GitHub (or from the Railway marketplace
   template once published).
2. In the new Railway project, add a **Postgres** service.
3. On the app service, add a reference variable:
   `DATABASE_URL = ${{Postgres.DATABASE_URL}}`
4. Set the **pre-deploy command** on the app service to
   `python scripts/migrate.py` (also in `railway.toml`). Migrations run
   before each deploy; a failed migration stops the deploy before the
   app starts.
5. Generate a public domain (Networking → Generate Domain).
6. Deploy. Railway healthchecks `GET /api/health` — it must answer 200.
7. Optional, once: run the seed against the Railway database
   (`DATABASE_URL` from Railway, then `python scripts/seed.py`).
8. Optional LLM: set `OPENAI_API_KEY` to enable LLM drafting and
   vector embeddings (re-run `python scripts/seed.py` to embed the
   seeded documents).
9. Optional tracing: set `LANGSMITH_API_KEY` and
   `LANGSMITH_TRACING=true` (and `LANGSMITH_PROJECT` if you want a
   custom project name).

## What it costs to run

Railway charges for what the services actually use — there is no fixed
price for this template. A realistic small deployment:

| Service | Typical usage | Notes |
| --- | --- | --- |
| App (1 vCPU, ~512 MB) | ~$5–10/mo | The app idles low |
| Postgres | ~$5/mo | Small volume; grows with data |

Roughly **~$10–15/mo** for a small deployment, plus OpenAI usage only
when you actually call the LLM/embeddings (pay per token at OpenAI's
rates; nothing is charged when no key is set, because no calls happen).
Verify against [Railway's pricing page](https://railway.com/pricing)
before promising a number to anyone; Railway's prices change.

## How to add API keys

- **OpenAI** — platform.openai.com → API keys. Set `OPENAI_API_KEY`
  as a Railway service variable (never in the repo). Rotate in the
  dashboard; update the variable; redeploy. Without it, drafting and
  embeddings stay off and the app says so.
- **LangSmith** — smith.langchain.com → API keys. Set
  `LANGSMITH_API_KEY` and `LANGSMITH_TRACING=true`.
- **Any other provider** — add the key as a Railway service variable
  (never in the repo). Read it with `os.environ` in server code only.

## How to back up

- **Postgres** — Railway's Postgres has backups/PITR as an opt-in
  feature; enable it in the database service settings. Additionally,
  take a logical dump on a schedule you control:
  `pg_dump "$DATABASE_URL" > backup-$(date +%F).sql` and store it
  somewhere other than Railway. Test a restore at least once — an
  untested backup is not a backup. The checkpoint tables are inside
  the same database, so the dump covers paused runs too.
- **Code + config** — the repo is the backup for code. Service
  variables are not in the repo by design; keep a copy of the variable
  *names* (not values) in `.env.example` and the values in Railway.

## How to update

- **Dependencies** — check PyPI for current versions, bump one at a
  time, then `pip install -r requirements.txt && python -m pytest &&
  python -m compileall app scripts`. Pay attention to LangGraph and
  langgraph-checkpoint-postgres (they move together) and to the
  checkpointer's table format — read its changelog before upgrading,
  because a checkpoint schema change can strand paused runs.
- **Schema** — edit the DDL in `scripts/migrate.py` (it is
  idempotent: `CREATE ... IF NOT EXISTS`), then re-run it.
- **This template** — if you deployed from the marketplace, you get a
  notification when the template updates; pull the changes into your
  repo yourself — Railway does not auto-update your copy.

## Hardening before real users

- Put the API behind authentication before exposing it publicly —
  this starter has no auth by design, so anyone with the URL can start
  runs. Add an API-key check or an auth proxy first.
- Add rate limiting on `POST /api/runs` if the endpoint is public.
- Set `OPENAI_API_KEY` from a secret store, not a pasted value, and
  rotate it if it ever leaks.
- Review the `notes` table: approved answers are stored in plain text
  — do not let the graph save sensitive user input without a policy.

## Project layout

```
app/
  __init__.py       loads .env at package import
  main.py           FastAPI app + routes + landing page
  settings.py       env vars + honest config_status()
  db.py             psycopg pool, retrieval, notes, db_ok()
  embeddings.py     OpenAI embeddings, env-gated (None when unset)
  graph.py          StateGraph: retrieve → draft → review → publish
scripts/
  migrate.py        idempotent DDL + PostgresSaver.setup()
  seed.py           idempotent demo documents (+ embeddings if key set)
tests/
  test_api.py       DB-free API tests (health honesty, validation)
railway.toml        build/start/healthcheck/pre-deploy config
requirements.txt    pinned runtime dependencies (source of truth)
```

See `AGENTS.md` for the rules AI agents should follow in this repo.

## License

MIT — see `LICENSE`.
