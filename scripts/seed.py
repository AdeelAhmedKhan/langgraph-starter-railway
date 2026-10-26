"""Seed the demo documents. Idempotent — safe to re-run.

Documents are real content about this template, so the retrieval demo
returns something meaningful. Embeddings are generated only when
OPENAI_API_KEY is set; without it, docs are seeded with NULL
embeddings and retrieval uses full-text search instead. Nothing is
faked: a NULL embedding is an honest "not embedded" state.

    python scripts/seed.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Running `python scripts/seed.py` puts scripts/ on sys.path, not the
# project root — add the root so `import app` works from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import settings
from app.db import get_pool

DOCS = [
    (
        "starter-overview",
        "LangGraph starter overview",
        "This template deploys a production-wired LangGraph starter: FastAPI "
        "serves the API, LangGraph runs the demo graph, Postgres 18 stores "
        "both the LangGraph checkpointer state and the application tables, "
        "and pgvector holds document embeddings for retrieval. The demo graph "
        "retrieves documents, drafts an answer, and stops for a human decision "
        "before anything is published.",
    ),
    (
        "human-in-the-loop",
        "Human-in-the-loop interrupts",
        "The review node calls LangGraph interrupt() with the draft and the "
        "retrieved documents, then pauses. Nothing is saved until the API "
        "resumes the run with an approve or reject decision. The thread state "
        "lives in the Postgres checkpointer, so the pause survives a server "
        "restart. Resume with POST /api/runs/{thread_id}/resume and an "
        "approved true/false body.",
    ),
    (
        "postgres-checkpointer",
        "Postgres checkpointer",
        "LangGraph persistence uses PostgresSaver from "
        "langgraph-checkpoint-postgres. Every graph step is checkpointed in "
        "Postgres tables created by the migrate script, so a run can be "
        "inspected, resumed, or replayed after the fact. The same Postgres "
        "service also hosts the documents and notes tables.",
    ),
    (
        "railway-deployment",
        "Railway deployment",
        "Deploy with the Railway template: add a Postgres service and "
        "reference it as DATABASE_URL, set the pre-deploy command to "
        "python scripts/migrate.py so migrations run before each deploy, "
        "and healthcheck GET /api/health. The app binds $PORT. Optional "
        "integrations are env-gated: set OPENAI_API_KEY for LLM drafting "
        "and embeddings, and LANGSMITH_API_KEY with LANGSMITH_TRACING=true "
        "for LangSmith tracing. Without them the app reports not_configured "
        "and keeps working.",
    ),
    (
        "pgvector-retrieval",
        "pgvector retrieval",
        "Documents carry a 1536-dimension embedding column for "
        "text-embedding-3-small. When embeddings exist and an OpenAI key is "
        "set, retrieval is cosine similarity over pgvector; otherwise it is "
        "PostgreSQL full-text search over title and body. The retrieval "
        "method used is recorded in the run state and returned by the API.",
    ),
]


def main() -> None:
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            for slug, title, body in DOCS:
                cur.execute(
                    "INSERT INTO documents (slug, title, body) VALUES (%s, %s, %s) "
                    "ON CONFLICT (slug) DO UPDATE SET title = EXCLUDED.title, body = EXCLUDED.body",
                    (slug, title, body),
                )
    print(f"Seeded {len(DOCS)} documents.")

    if settings.embeddings_configured():
        from app.embeddings import embed_texts

        pool = get_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT slug, title, body FROM documents WHERE embedding IS NULL")
                pending = cur.fetchall()
        if not pending:
            print("All documents already have embeddings.")
            return
        texts = [f"{title}\n{body}" for _, title, body in pending]
        vectors = embed_texts(texts)
        if vectors is None:
            print("Embeddings not configured — skipping.")
            return
        with pool.connection() as conn:
            with conn.cursor() as cur:
                from pgvector.psycopg import register_vector

                register_vector(conn)
                for (slug, _, _), vector in zip(pending, vectors):
                    cur.execute(
                        "UPDATE documents SET embedding = %s WHERE slug = %s",
                        (vector, slug),
                    )
        print(f"Embedded {len(pending)} documents.")
    else:
        print(
            "OPENAI_API_KEY not set — documents seeded without embeddings; "
            "retrieval will use full-text search."
        )


if __name__ == "__main__":
    main()
