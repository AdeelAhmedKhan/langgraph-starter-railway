"""Postgres access: connection pool, schema, retrieval, health.

One pool is shared by the app tables and the LangGraph checkpointer.
The pool is created lazily so the app can boot (and /api/health can
answer honestly) even when the database is down.
"""

from __future__ import annotations

import datetime
import uuid
from typing import Any

from psycopg_pool import ConnectionPool

from . import settings

_pool: ConnectionPool | None = None


def _jsonable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert UUID/datetime values to strings so rows JSON-serialize."""
    out = []
    for row in rows:
        clean = {}
        for key, value in row.items():
            if isinstance(value, uuid.UUID):
                clean[key] = str(value)
            elif isinstance(value, datetime.datetime):
                clean[key] = value.isoformat()
            else:
                clean[key] = value
        out.append(clean)
    return out


def get_pool() -> ConnectionPool:
    """Lazily create the shared pool. Never raises for a down DB at
    construction time — checkout is where a dead DB surfaces."""
    global _pool
    if _pool is None:
        # min_size=0: no eager connections, so a dead DB costs nothing
        # at boot. timeout=3: checkout waits at most 3s, so /api/health
        # answers quickly even when the DB is down (the first pytest run
        # took 121s with the pool defaults — timeout 30s per checkout).
        _pool = ConnectionPool(
            settings.DATABASE_URL,
            min_size=0,
            max_size=10,
            timeout=3,
            open=True,
            kwargs={"autocommit": True, "connect_timeout": 2},
        )
    return _pool


def db_ok() -> bool:
    """True only if a real query succeeds. Used by /api/health."""
    try:
        pool = get_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                row = cur.fetchone()
                return row is not None and row[0] == 1
    except Exception:
        return False


_STOPWORDS = frozenset(
    {
        "how", "what", "why", "when", "where", "which", "who", "whom",
        "does", "do", "did", "is", "are", "was", "were", "can", "could",
        "should", "would", "will", "the", "and", "for", "with", "about",
        "tell", "me", "explain", "describe", "you", "your", "a", "an",
        "of", "to", "in", "on", "it", "its", "this", "that", "there",
        "here", "from", "by", "as", "at", "be", "an", "or", "if", "so",
    }
)


def _search_terms(question: str) -> list[str]:
    """Significant lowercase word tokens for the full-text OR query."""
    import re

    tokens = re.findall(r"[a-z0-9]+", question.lower())
    terms = [t for t in tokens if len(t) > 2 and t not in _STOPWORDS]
    return terms or [t for t in tokens if len(t) > 2] or tokens


def search_docs(question: str, limit: int = 5) -> tuple[list[dict[str, Any]], str]:
    """Retrieve seeded documents for a question.

    Vector search when embeddings exist *and* an OpenAI key is set;
    otherwise PostgreSQL full-text search. Both paths return real rows
    from the seeded ``documents`` table — never invented content.
    Returns (rows, method) where method is "vector" or "fulltext".
    """
    from .embeddings import embed_query

    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            if settings.embeddings_configured():
                cur.execute("SELECT 1 FROM documents WHERE embedding IS NOT NULL LIMIT 1")
                if cur.fetchone() is not None:
                    vector = embed_query(question)
                    if vector is not None:
                        from pgvector.psycopg import register_vector

                        register_vector(conn)
                        cur.execute(
                            "SELECT id, slug, title, body FROM documents "
                            "WHERE embedding IS NOT NULL "
                            "ORDER BY embedding <=> %s::vector LIMIT %s",
                            (vector, limit),
                        )
                        cols = [d.name for d in cur.description]
                        return _jsonable(
                            [dict(zip(cols, row)) for row in cur.fetchall()]
                        ), "vector"
            # Full-text search with OR + prefix matching. plainto_tsquery
            # ANDs every term, so a natural question ("How do interrupts
            # work?") matches nothing the moment it contains a word the
            # doc lacks. OR over the significant terms (prefix-matched, so
            # "interrupt" finds "interrupts") is the lenient behaviour a
            # demo needs — still real full-text search, still ranked.
            terms = _search_terms(question)
            tsq = " | ".join(f"{t}:*" for t in terms)
            cur.execute(
                "SELECT id, slug, title, body, "
                "ts_rank(to_tsvector('english', title || ' ' || body), "
                "to_tsquery('english', %s)) AS rank "
                "FROM documents "
                "WHERE to_tsvector('english', title || ' ' || body) "
                "@@ to_tsquery('english', %s) "
                "ORDER BY rank DESC LIMIT %s",
                (tsq, tsq, limit),
            )
            cols = [d.name for d in cur.description]
            return _jsonable(
                [dict(zip(cols, row)) for row in cur.fetchall()]
            ), "fulltext"


def save_note(
    thread_id: str,
    question: str,
    answer: str,
    approved: bool,
    feedback: str | None,
) -> str:
    """Persist an approved answer. Returns the note id."""
    import uuid

    note_id = str(uuid.uuid4())
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO notes (id, thread_id, question, answer, approved, feedback) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (note_id, thread_id, question, answer, approved, feedback or ""),
            )
    return note_id


def list_notes(limit: int = 20) -> list[dict[str, Any]]:
    """Most recent published notes, newest first."""
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, thread_id, question, answer, approved, feedback, created_at "
                "FROM notes ORDER BY created_at DESC LIMIT %s",
                (limit,),
            )
            cols = [d.name for d in cur.description]
            rows = _jsonable([dict(zip(cols, row)) for row in cur.fetchall()])
            return rows
