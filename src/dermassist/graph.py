"""LangGraph wiring for the DermAssist pipeline.

Graph shape (HANDOFF.md §5):

    START -> ingest -> preprocess -> classify -> interpret -> literature
          -> build_report -> human_review --(approved)--> finalize -> END
                                          --(rejected/edited)--> interpret

The human_review node pauses via ``interrupt()``; a checkpointer persists state
so the graph survives the pause and resumes on the same ``thread_id``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from dermassist import nodes
from dermassist.config import get_settings
from dermassist.schemas import (
    ABCDEFeatures,
    ClassifierResult,
    LesionReport,
    LesionState,
    LiteratureRef,
)

# Pydantic models we store inside graph state and therefore round-trip through the
# checkpointer. Registering them explicitly silences the "unregistered type"
# deserialization warnings and keeps us forward-compatible with strict msgpack.
_CHECKPOINT_TYPES = (
    LesionState,
    LesionReport,
    ClassifierResult,
    ABCDEFeatures,
    LiteratureRef,
)


def build_graph_builder() -> StateGraph:
    """Construct the (uncompiled) graph topology."""
    builder = StateGraph(LesionState)

    builder.add_node("ingest", nodes.ingest)
    builder.add_node("preprocess", nodes.preprocess)
    builder.add_node("classify", nodes.classify)
    builder.add_node("interpret", nodes.interpret)
    builder.add_node("literature", nodes.literature)
    builder.add_node("build_report", nodes.build_report)
    builder.add_node("human_review", nodes.human_review)
    builder.add_node("finalize", nodes.finalize)

    builder.add_edge(START, "ingest")
    builder.add_edge("ingest", "preprocess")
    builder.add_edge("preprocess", "classify")
    builder.add_edge("classify", "interpret")
    builder.add_edge("interpret", "literature")
    builder.add_edge("literature", "build_report")
    builder.add_edge("build_report", "human_review")

    # Conditional edge: the review gate either finalizes or loops back.
    builder.add_conditional_edges(
        "human_review",
        nodes.route_review,
        {"finalize": "finalize", "interpret": "interpret"},
    )
    builder.add_edge("finalize", END)

    return builder


def make_checkpointer(db_path: Path | None = None) -> SqliteSaver:
    """Create a persistent on-disk SQLite checkpointer.

    We construct ``SqliteSaver`` directly from a long-lived ``sqlite3`` connection
    (rather than the ``from_conn_string`` context manager, which closes the
    connection on block exit) so state persists across separate CLI invocations.
    """
    if db_path is None:
        db_path = get_settings().checkpoint_db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    serde = JsonPlusSerializer(allowed_msgpack_modules=list(_CHECKPOINT_TYPES))
    return SqliteSaver(conn, serde=serde)


def build_graph(db_path: Path | None = None):
    """Compile the graph with a SQLite checkpointer. Returns a compiled graph."""
    builder = build_graph_builder()
    checkpointer = make_checkpointer(db_path)
    return builder.compile(checkpointer=checkpointer)


__all__ = ["build_graph_builder", "make_checkpointer", "build_graph"]
