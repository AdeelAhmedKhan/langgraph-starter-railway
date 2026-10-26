"""Demo LangGraph: retrieve -> draft -> human review -> publish.

The graph is a small research assistant. It retrieves real documents
from Postgres, drafts an answer, then *stops* for a human decision
before anything is saved. That stop is the point of the demo:
LangGraph persistence + human-in-the-loop, exercisable over HTTP.

Honesty rules for this graph:
- Without OPENAI_API_KEY there is no LLM. The draft node then returns
  a verbatim extract of the retrieved documents, labelled
  ``llm_used: false`` — never a fake LLM answer.
- Retrieval is vector search only when embeddings exist *and* a key is
  set; otherwise it is PostgreSQL full-text search. The method used is
  recorded in state (``retrieval``) and returned by the API.
- Nothing is published without an explicit approve. A reject ends the
  run with status "rejected" and saves nothing.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from . import settings
from .db import get_pool, save_note, search_docs


class AgentState(TypedDict, total=False):
    question: str
    thread_id: str
    docs: list[dict]
    retrieval: str
    draft: str
    llm_used: bool
    approved: bool | None
    feedback: str | None
    final_answer: str | None
    status: str


def retrieve(state: AgentState) -> dict:
    docs, method = search_docs(state["question"])
    return {"docs": docs, "retrieval": method}


def draft(state: AgentState) -> dict:
    docs = state.get("docs") or []
    if settings.llm_configured():
        from langchain_openai import ChatOpenAI

        context = "\n\n".join(
            f"[{d['slug']}] {d['title']}: {d['body']}" for d in docs
        )
        prompt = (
            "Answer the question using ONLY the context below. "
            "If the context does not contain the answer, say so plainly.\n\n"
            f"Question: {state['question']}\n\n"
            f"Context:\n{context if context else '(no documents retrieved)'}"
        )
        llm = ChatOpenAI(model=settings.OPENAI_MODEL, api_key=settings.OPENAI_API_KEY)
        response = llm.invoke(prompt)
        return {"draft": str(response.content), "llm_used": True}
    if docs:
        extract = "\n\n".join(f"{d['title']}: {d['body']}" for d in docs)
        text = (
            "No OPENAI_API_KEY is set, so this draft is a verbatim extract "
            "of the retrieved documents (no LLM was used):\n\n" + extract
        )
    else:
        text = (
            "No OPENAI_API_KEY is set and no documents matched the question, "
            "so there is no draft to review."
        )
    return {"draft": text, "llm_used": False}


def review(state: AgentState) -> dict:
    """Human-in-the-loop: pause here until the API resumes the run."""
    decision = interrupt(
        {
            "question": state["question"],
            "draft": state["draft"],
            "docs": state["docs"],
            "llm_used": state.get("llm_used", False),
            "retrieval": state.get("retrieval", ""),
        }
    )
    if isinstance(decision, dict):
        approved = bool(decision.get("approved"))
        feedback = str(decision.get("feedback") or "")
    else:
        approved, feedback = False, ""
    return {"approved": approved, "feedback": feedback}


def publish(state: AgentState) -> dict:
    if state.get("approved"):
        save_note(
            thread_id=state["thread_id"],
            question=state["question"],
            answer=state["draft"],
            approved=True,
            feedback=state.get("feedback"),
        )
        return {"final_answer": state["draft"], "status": "published"}
    return {"final_answer": None, "status": "rejected"}


_graph = None


def get_graph():
    """Build (once) the compiled graph with the Postgres checkpointer."""
    global _graph
    if _graph is not None:
        return _graph
    from langgraph.checkpoint.postgres import PostgresSaver

    saver = PostgresSaver(get_pool())
    builder = StateGraph(AgentState)
    builder.add_node("retrieve", retrieve)
    builder.add_node("draft", draft)
    builder.add_node("review", review)
    builder.add_node("publish", publish)
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "draft")
    builder.add_edge("draft", "review")
    builder.add_edge("review", "publish")
    builder.add_edge("publish", END)
    _graph = builder.compile(checkpointer=saver)
    return _graph


def graph_info() -> dict:
    """Static description of the graph for /api/graph/info."""
    return {
        "nodes": ["retrieve", "draft", "review", "publish"],
        "edges": [
            ["retrieve", "draft"],
            ["draft", "review"],
            ["review", "publish"],
            ["publish", "END"],
        ],
        "human_in_the_loop": "review node calls interrupt(); resume via POST /api/runs/{thread_id}/resume",
        "checkpointer": "PostgresSaver (Postgres 18, langgraph-checkpoint-postgres)",
        "persistence": "thread state survives server restarts (checkpoint tables in Postgres)",
    }
