"""Phase 6 — Streamlit human-review UI.

Launch with:

    uv run streamlit run src/dermassist/ui.py

Drives the same LangGraph pipeline as the CLI: upload a dermatoscopic image, run
the analysis until the human-review gate, inspect the report side-by-side
(original + preprocessed image, classifier probabilities, Claude's ABCDE
interpretation, cited literature), then Approve / Reject / Edit to resume the graph
through the SQLite checkpointer. The compliance disclaimer is pinned on screen.

The module is import-safe: the Streamlit body runs only under ``streamlit run``
(``__name__ == "__main__"``), so the pure helpers below can be unit-tested.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from dermassist.compliance import DISCLAIMER, DISCLAIMER_LONG
from dermassist.schemas import HAM10000_CLASSES

UPLOAD_DIR = Path("artifacts/uploads")
_ABCDE_FIELDS = ("asymmetry", "border", "color", "diameter", "evolution")


# ------------------------------- pure helpers ------------------------------ #


def thread_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def collect_interrupts(snapshot: Any) -> list:
    return [
        itr
        for task in getattr(snapshot, "tasks", ()) or ()
        for itr in (getattr(task, "interrupts", ()) or ())
    ]


def is_paused(snapshot: Any) -> bool:
    return bool(collect_interrupts(snapshot))


def is_finalized(snapshot: Any) -> bool:
    return not snapshot.next and bool(snapshot.values)


def probabilities_frame(probabilities: dict[str, float]):
    """A descending-sorted DataFrame for the probability bar chart."""
    import pandas as pd

    ordered = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    return pd.DataFrame({"probability": dict(ordered)})


def save_upload(name: str, data: bytes, upload_dir: Path = UPLOAD_DIR) -> Path:
    upload_dir.mkdir(parents=True, exist_ok=True)
    # Basename guards against any path components in the uploaded filename.
    path = upload_dir / Path(name).name
    path.write_bytes(data)
    return path


# ------------------------------ Streamlit app ------------------------------ #


def _render() -> None:
    import streamlit as st
    from langgraph.types import Command

    from dermassist.graph import build_graph
    from dermassist.schemas import LesionState

    @st.cache_resource
    def get_graph():
        return build_graph()

    st.set_page_config(page_title="DermAssist Review", layout="wide")
    graph = get_graph()

    # Persistent compliance banner (HANDOFF.md: every surface).
    st.title("DermAssist — Lesion Review")
    st.warning(DISCLAIMER_LONG)

    with st.sidebar:
        st.header("New analysis")
        uploaded = st.file_uploader("Dermatoscopic image", type=["png", "jpg", "jpeg"])
        if st.button("Run analysis", type="primary", disabled=uploaded is None):
            path = save_upload(uploaded.name, uploaded.getbuffer())
            thread_id = uuid4().hex[:12]
            st.session_state["thread_id"] = thread_id
            with st.spinner("Running pipeline (preprocess → classify → interpret → literature)…"):
                graph.invoke(LesionState(image_path=str(path)), config=thread_config(thread_id))
            st.rerun()
        thread_id = st.session_state.get("thread_id")
        if thread_id:
            st.caption(f"thread_id: `{thread_id}`")

    thread_id = st.session_state.get("thread_id")
    if not thread_id:
        st.info("Upload a dermatoscopic image and click **Run analysis** to begin.")
        return

    snapshot = graph.get_state(thread_config(thread_id))
    values = snapshot.values or {}

    # --- Images side-by-side ---
    col_a, col_b = st.columns(2)
    if values.get("image_path") and Path(values["image_path"]).exists():
        col_a.image(values["image_path"], caption="Original", use_container_width=True)
    pre = values.get("preprocessed_path")
    if pre and Path(pre).exists():
        col_b.image(pre, caption="Preprocessed", use_container_width=True)

    # --- Classifier probabilities ---
    classifier = values.get("classifier_result")
    if classifier is not None:
        st.subheader("Classifier probabilities (owns the diagnostic numbers)")
        st.bar_chart(probabilities_frame(classifier.probabilities))
        st.caption(f"Top label: **{classifier.label}** ({classifier.top_confidence:.3f})")

    # --- ABCDE interpretation ---
    interp = values.get("interpretation")
    if interp is not None:
        st.subheader("ABCDE interpretation (Claude — narrative only)")
        for field in _ABCDE_FIELDS:
            st.markdown(f"**{field.title()}** — {getattr(interp, field)}")
    if values.get("reconciliation"):
        st.markdown(f"**Reconciliation:** {values['reconciliation']}")

    # --- Cited literature ---
    refs = values.get("literature") or []
    if refs:
        st.subheader("Cited literature")
        for ref in refs:
            st.markdown(f"- **PMID {ref.pmid}** — {ref.title}  \n  _{ref.relevance_note}_")

    st.divider()

    # --- Review gate / outcome ---
    cycles = values.get("review_cycles", 0)
    if is_paused(snapshot):
        st.subheader("🔴 Human review required — HARD STOP")
        if cycles:
            st.caption(f"Review cycle {cycles + 1} (previous decision sent it back).")
        notes = st.text_area(
            "Reviewer notes (sent back to Claude on Reject / Edit):", key="notes"
        )
        c1, c2, c3 = st.columns(3)
        decision = None
        if c1.button("✅ Approve", use_container_width=True):
            decision = "approved"
        if c2.button("❌ Reject (re-run)", use_container_width=True):
            decision = "rejected"
        if c3.button("✏️ Edit (re-run)", use_container_width=True):
            decision = "edited"
        if decision:
            with st.spinner(f"Resuming graph ({decision})…"):
                graph.invoke(
                    Command(resume={"status": decision, "notes": notes or None}),
                    config=thread_config(thread_id),
                )
            st.rerun()
    elif is_finalized(snapshot):
        st.subheader("✅ Finalized")
        st.success(f"Review status: **{values.get('review_status')}** "
                   f"after {cycles} review cycle(s).")
        report = values.get("report")
        if report is not None:
            st.markdown(f"**Recommendation:** {report.recommendation}")
            with st.expander("Full report JSON"):
                st.json(report.model_dump())

    # Footer disclaimer — pinned regardless of state.
    st.divider()
    st.caption(f"⚠️ {DISCLAIMER}")


if __name__ == "__main__":
    _render()


__all__ = [
    "thread_config",
    "collect_interrupts",
    "is_paused",
    "is_finalized",
    "probabilities_frame",
    "save_upload",
]
