"""FastAPI app: landing page, health, and the demo graph over HTTP.

Startup order matters: ``load_dotenv()`` runs before anything reads
environment variables, so ``cp .env.example .env`` works out of the box
(the TanStack starter's seed script tripped on this exact issue).

The graph and its checkpointer are built lazily — the app boots and
/api/health answers even when Postgres is down. Endpoints that need the
database return a clear 503 instead of crashing.
"""

import os
import uuid

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from . import settings
from .db import db_ok, list_notes
from .graph import get_graph, graph_info

if settings.langsmith_configured():
    # LangGraph/LangChain pick these up at run time. Only set when the
    # user actually configured tracing — never pretend to trace.
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_API_KEY", settings.LANGSMITH_API_KEY)
    os.environ.setdefault("LANGSMITH_PROJECT", settings.LANGSMITH_PROJECT)

app = FastAPI(
    title="LangGraph Starter",
    description=(
        "Production-wired LangGraph starter: FastAPI + Postgres 18 "
        "checkpointer + pgvector, human-in-the-loop review, Railway-ready."
    ),
    version=settings.APP_VERSION,
)


class RunRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class ResumeRequest(BaseModel):
    approved: bool
    feedback: str = Field(default="", max_length=2000)


LANDING_HTML = """<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>LangGraph Starter</title>
<style>body{font-family:system-ui,sans-serif;max-width:640px;margin:3rem auto;padding:0 1rem;line-height:1.6}code{background:#eee;padding:.1rem .3rem;border-radius:4px}pre{background:#111;color:#eee;padding:1rem;border-radius:8px;overflow:auto}</style>
</head>
<body>
<h1>LangGraph Starter</h1>
<p>FastAPI + LangGraph + Postgres 18 checkpointer + pgvector. A research-assistant
graph that retrieves real documents, drafts an answer, and <b>stops for a human
decision</b> before anything is published.</p>
<ul>
<li><a href="/docs">API docs</a> (Swagger)</li>
<li><a href="/api/health">GET /api/health</a> — status, DB, and which integrations are configured</li>
<li><a href="/api/graph/info">GET /api/graph/info</a> — graph nodes, edges, checkpointer</li>
<li><a href="/api/notes">GET /api/notes</a> — published (approved) answers</li>
</ul>
<h2>Try it</h2>
<pre>curl -X POST localhost:8000/api/runs \\
  -H 'Content-Type: application/json' \\
  -d '{"question": "How do human-in-the-loop interrupts work?"}'

# approve the draft (thread_id from the response):
curl -X POST localhost:8000/api/runs/&lt;thread_id&gt;/resume \\
  -H 'Content-Type: application/json' \\
  -d '{"approved": true}'</pre>
<p>Without <code>OPENAI_API_KEY</code> the draft is a verbatim extract of the
retrieved documents, labelled as such — never a fake LLM answer. Without
<code>LANGSMITH_API_KEY</code> tracing is off. See <code>.env.example</code>.</p>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def landing() -> str:
    return LANDING_HTML


@app.get("/api/health")
def health() -> dict:
    ok = db_ok()
    return {
        "status": "ok" if ok else "degraded",
        "db": "ok" if ok else "error",
        **settings.config_status(),
        "version": settings.APP_VERSION,
    }


@app.get("/api/graph/info")
def info() -> dict:
    return {**graph_info(), **settings.config_status()}


def _require_db() -> None:
    if not db_ok():
        raise HTTPException(
            status_code=503,
            detail="Database is not available. Set DATABASE_URL and run python scripts/migrate.py.",
        )


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _interrupt_payload(graph, config: dict) -> dict | None:
    snapshot = graph.get_state(config)
    for task in snapshot.tasks or []:
        for intr in getattr(task, "interrupts", []) or []:
            return intr.value
    return None


@app.post("/api/runs", status_code=201)
def create_run(body: RunRequest) -> dict:
    _require_db()
    try:
        graph = get_graph()
    except Exception as exc:  # e.g. checkpointer tables missing
        raise HTTPException(
            status_code=503,
            detail=f"Graph could not start (did you run python scripts/migrate.py?): {exc}",
        ) from exc
    thread_id = uuid.uuid4().hex
    config = _config(thread_id)
    initial = {
        "question": body.question,
        "thread_id": thread_id,
        "docs": [],
        "retrieval": "",
        "draft": "",
        "llm_used": False,
        "approved": None,
        "feedback": None,
        "final_answer": None,
        "status": "started",
    }
    result = graph.invoke(initial, config)
    return {
        "thread_id": thread_id,
        "status": "awaiting_approval",
        "question": result.get("question"),
        "draft": result.get("draft"),
        "docs": result.get("docs"),
        "retrieval": result.get("retrieval"),
        "llm_used": result.get("llm_used"),
        "interrupt": _interrupt_payload(graph, config),
    }


@app.get("/api/runs/{thread_id}")
def get_run(thread_id: str) -> dict:
    _require_db()
    graph = get_graph()
    snapshot = graph.get_state(_config(thread_id))
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="Unknown thread_id.")
    values = dict(snapshot.values)
    return {
        "thread_id": thread_id,
        "status": values.get("status"),
        "next": list(snapshot.next or []),
        "values": values,
        "interrupt": _interrupt_payload(graph, _config(thread_id)),
    }


@app.post("/api/runs/{thread_id}/resume")
def resume_run(thread_id: str, body: ResumeRequest) -> dict:
    _require_db()
    graph = get_graph()
    config = _config(thread_id)
    snapshot = graph.get_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="Unknown thread_id.")
    if not snapshot.next:
        raise HTTPException(status_code=409, detail="Run already finished.")
    result = graph.invoke(
        Command(resume={"approved": body.approved, "feedback": body.feedback}),
        config,
    )
    return {
        "thread_id": thread_id,
        "status": result.get("status"),
        "approved": result.get("approved"),
        "feedback": result.get("feedback"),
        "final_answer": result.get("final_answer"),
    }


@app.get("/api/notes")
def notes() -> dict:
    _require_db()
    return {"notes": list_notes()}
