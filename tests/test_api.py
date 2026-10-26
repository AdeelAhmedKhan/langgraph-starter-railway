"""DB-free API tests: health honesty, validation, and graph info.

These tests deliberately run WITHOUT Postgres — they prove the app boots
and answers honestly when the database is down, and that bad input is
rejected before any graph work happens. The end-to-end graph flows are
covered by the local parity test (see agent_notes), which runs against
a real Postgres 18.
"""

import os

# Point at a dead database BEFORE importing the app, so settings pick
# up an unreachable DATABASE_URL.
os.environ["DATABASE_URL"] = "postgresql://langgraph:dummy-password-not-real@127.0.0.1:59999/langgraph_starter"
os.environ.pop("OPENAI_API_KEY", None)
os.environ["LANGSMITH_TRACING"] = "false"

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app, raise_server_exceptions=False)


def test_landing_page():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "LangGraph Starter" in resp.text


def test_health_degraded_without_db():
    resp = client.get("/api/health")
    assert resp.status_code == 200  # Railway healthcheck needs 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["db"] == "error"
    assert body["llm"] == "not_configured"
    assert body["langsmith"] == "not_configured"


def test_graph_info_without_db():
    resp = client.get("/api/graph/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["nodes"] == ["retrieve", "draft", "review", "publish"]
    assert body["checkpointer"].startswith("PostgresSaver")


def test_create_run_validation():
    resp = client.post("/api/runs", json={"question": ""})
    assert resp.status_code == 422
    resp = client.post("/api/runs", json={})
    assert resp.status_code == 422


def test_create_run_without_db_is_503_not_crash():
    resp = client.post("/api/runs", json={"question": "How do interrupts work?"})
    assert resp.status_code == 503
    assert "Database" in resp.json()["detail"]


def test_resume_unknown_thread_without_db_is_503():
    resp = client.post("/api/runs/doesnotexist/resume", json={"approved": True})
    assert resp.status_code == 503


def test_notes_without_db_is_503():
    resp = client.get("/api/notes")
    assert resp.status_code == 503
