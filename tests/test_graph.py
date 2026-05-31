"""Phase 1 acceptance test: the mocked graph runs end-to-end, pauses at the
human-review gate via interrupt(), and resumes via the checkpointer to finalize.

No real model is touched.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langgraph.types import Command

from dermassist.compliance import DISCLAIMER
from dermassist.graph import build_graph
from dermassist.schemas import LesionState


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


@pytest.fixture()
def graph(tmp_path: Path):
    return build_graph(db_path=tmp_path / "checkpoints.sqlite")


def test_pauses_at_human_review(graph):
    cfg = _config("t-pause")
    graph.invoke(LesionState(image_path="fake.jpg"), config=cfg)

    snap = graph.get_state(cfg)
    # Paused: there are pending nodes and an interrupt is registered.
    assert snap.next, "graph should be paused, not finished"
    interrupts = [i for t in snap.tasks for i in (t.interrupts or [])]
    assert interrupts, "expected an interrupt at the human-review gate"
    payload = interrupts[0].value
    assert payload["action"] == "human_review_required"
    assert payload["report"] is not None
    assert payload["report"]["disclaimer"] == DISCLAIMER


def test_resume_approved_reaches_finalize(graph):
    cfg = _config("t-approve")
    graph.invoke(LesionState(image_path="fake.jpg"), config=cfg)

    graph.invoke(Command(resume={"status": "approved", "notes": "looks right"}), config=cfg)

    snap = graph.get_state(cfg)
    assert not snap.next, "graph should be finalized (no pending nodes)"
    values = snap.values
    assert values["review_status"] == "approved"
    assert values["report"].disclaimer == DISCLAIMER
    assert values["review_cycles"] == 1


def test_reject_loops_back_then_approve(graph):
    cfg = _config("t-loop")
    graph.invoke(LesionState(image_path="fake.jpg"), config=cfg)

    # Reject -> loops back to interpret -> pauses again at review.
    graph.invoke(Command(resume={"status": "rejected", "notes": "re-examine border"}), config=cfg)
    snap = graph.get_state(cfg)
    assert snap.next, "rejected decision should loop back and pause again"
    interrupts = [i for t in snap.tasks for i in (t.interrupts or [])]
    assert interrupts, "expected a second interrupt after loop-back"

    # Now approve -> finalize.
    graph.invoke(Command(resume={"status": "approved"}), config=cfg)
    snap = graph.get_state(cfg)
    assert not snap.next
    assert snap.values["review_status"] == "approved"
    assert snap.values["review_cycles"] == 2


def test_resume_survives_new_graph_instance(graph, tmp_path: Path):
    """State persists on disk: a fresh compiled graph over the same DB resumes."""
    db = tmp_path / "persist.sqlite"
    g1 = build_graph(db_path=db)
    cfg = _config("t-persist")
    g1.invoke(LesionState(image_path="fake.jpg"), config=cfg)

    # Brand-new graph object, same on-disk checkpointer file.
    g2 = build_graph(db_path=db)
    g2.invoke(Command(resume={"status": "approved"}), config=cfg)
    snap = g2.get_state(cfg)
    assert not snap.next
    assert snap.values["review_status"] == "approved"
