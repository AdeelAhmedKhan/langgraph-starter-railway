"""Create the schema: pgvector extension, app tables, checkpoint tables.

Idempotent — safe to re-run. This is the pre-deploy command in
railway.toml, so a failed migration stops the deploy before the app
starts. Run locally after `cp .env.example .env`:

    python scripts/migrate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Running `python scripts/migrate.py` puts scripts/ on sys.path, not the
# project root — add the root so `import app` works from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import settings
from app.db import get_pool

APP_DDL = """
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    slug TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    embedding vector(1536),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS documents_search_idx
    ON documents USING gin (to_tsvector('english', title || ' ' || body));
CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS notes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    thread_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    approved BOOLEAN NOT NULL DEFAULT TRUE,
    feedback TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS notes_created_idx ON notes (created_at DESC);
"""


def main() -> None:
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.errors import InsufficientPrivilege

    if not settings.DATABASE_URL:
        print(
            "ERROR: DATABASE_URL is not set. Copy .env.example to .env "
            "and fill it in, or set it in the environment. On Railway, "
            "reference the Postgres service's DATABASE_URL."
        )
        raise SystemExit(1)

    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            # The vector extension is not trusted: only a superuser can
            # create it. On Railway the Postgres user is a superuser, so
            # this just works. Locally, a plain role hits
            # InsufficientPrivilege — say so plainly instead of dying
            # with a raw traceback.
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            except InsufficientPrivilege:
                print(
                    "ERROR: cannot CREATE EXTENSION vector as this user — "
                    "it needs a Postgres superuser.\n"
                    "Run once as the superuser, then re-run this script:\n"
                    "  psql -d <your-db> -c \"CREATE EXTENSION vector\"\n"
                    "On Railway the database user is a superuser, so the "
                    "pre-deploy migrate needs no extra step."
                )
                raise SystemExit(1)
            cur.execute(APP_DDL)
    # LangGraph checkpoint tables (checkpoints, checkpoint_blobs,
    # checkpoint_writes, checkpoint_migrations). setup() is idempotent.
    saver = PostgresSaver(pool)
    saver.setup()
    print("migrations applied successfully!")


if __name__ == "__main__":
    main()
