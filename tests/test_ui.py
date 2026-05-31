"""Phase 6 tests for the review UI's pure helpers.

Importing ``dermassist.ui`` does not start Streamlit (the app body is guarded by
``__name__ == "__main__"``), so these run headless.
"""

from __future__ import annotations

import pytest

from dermassist import ui


class _Task:
    def __init__(self, interrupts):
        self.interrupts = interrupts


class _Snapshot:
    def __init__(self, tasks=(), next_=(), values=None):
        self.tasks = tasks
        self.next = next_
        self.values = values or {}


def test_thread_config():
    assert ui.thread_config("abc") == {"configurable": {"thread_id": "abc"}}


def test_is_paused_true_when_interrupts_present():
    snap = _Snapshot(tasks=(_Task(["payload"]),), next_=("human_review",))
    assert ui.is_paused(snap)
    assert not ui.is_finalized(snap)


def test_is_finalized_when_no_next_and_values_present():
    snap = _Snapshot(tasks=(), next_=(), values={"report": object()})
    assert ui.is_finalized(snap)
    assert not ui.is_paused(snap)


def test_running_state_is_neither():
    snap = _Snapshot(tasks=(), next_=("classify",), values={})
    assert not ui.is_paused(snap)
    assert not ui.is_finalized(snap)


def test_probabilities_frame_sorted_descending():
    frame = ui.probabilities_frame({"a": 0.1, "b": 0.7, "c": 0.2})
    assert list(frame.index) == ["b", "c", "a"]
    assert list(frame["probability"]) == [0.7, 0.2, 0.1]


def test_save_upload_strips_path_components(tmp_path):
    out = ui.save_upload("../../evil.png", b"data", upload_dir=tmp_path / "uploads")
    assert out.name == "evil.png"
    assert out.parent == tmp_path / "uploads"
    assert out.read_bytes() == b"data"


def test_app_renders_initial_state():
    """Run the real Streamlit script headless; the render body must execute clean."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("src/dermassist/ui.py", default_timeout=60).run()

    assert not at.exception
    # Title and the persistent compliance disclaimer must be present.
    assert any("DermAssist" in t.value for t in at.title)
    assert any("research and educational" in w.value for w in at.warning)
    # Initial state prompts for an upload.
    assert any("Upload" in i.value for i in at.info)
