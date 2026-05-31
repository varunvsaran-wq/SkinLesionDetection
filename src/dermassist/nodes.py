"""Pipeline nodes.

Phase 1: every node is a MOCK returning hardcoded-but-plausible output, so the
full graph (including the resumable human-review gate) runs end-to-end before any
real model is touched. Each node keeps the exact input/output contract its real
counterpart will use, so later phases are drop-in swaps.

Real implementations land per HANDOFF.md build order:
    ingest/preprocess -> Phase 2
    classify          -> Phase 3
    interpret         -> Phase 4
    literature        -> Phase 5
"""

from __future__ import annotations

from pathlib import Path

from langgraph.types import interrupt

from dermassist.compliance import DISCLAIMER
from dermassist.schemas import (
    ABCDEFeatures,
    ClassifierResult,
    LesionReport,
    LesionState,
    LiteratureRef,
)


def ingest(state: LesionState) -> dict:
    """Validate the input image path exists. (Phase 1 mock: existence check only.)"""
    path = Path(state.image_path)
    # In the mock we don't require the file to exist on disk (no dataset yet),
    # but we surface a note so the real Phase 2 loader has an obvious hook.
    exists = path.exists()
    if not exists:
        print(f"[ingest] note: '{path}' not found on disk — OK for mocked run.")
    return {}


def preprocess(state: LesionState) -> dict:
    """Resize + color constancy + hair removal. (Phase 2 will make this real.)"""
    src = Path(state.image_path)
    preprocessed = str(src.with_name(f"{src.stem}__preprocessed{src.suffix or '.png'}"))
    return {"preprocessed_path": preprocessed}


def classify(state: LesionState) -> dict:
    """Dedicated vision classifier. The classifier OWNS calibrated probabilities
    over the 7 HAM10000 classes (never Claude). (Phase 3 swaps in a real model.)
    """
    # Mock distribution: plausible melanoma-leaning case. Sums to 1.0.
    probabilities = {
        "akiec": 0.03,
        "bcc": 0.05,
        "bkl": 0.07,
        "df": 0.02,
        "mel": 0.62,
        "nv": 0.18,
        "vasc": 0.03,
    }
    top_label = max(probabilities, key=probabilities.__getitem__)
    result = ClassifierResult(
        label=top_label,
        probabilities=probabilities,
        top_confidence=probabilities[top_label],
    )
    return {"classifier_result": result}


def interpret(state: LesionState) -> dict:
    """Claude's ABCDE narrative, reconciled with the classifier's probabilities.
    (Phase 4 swaps in a real Claude tool-use call.)

    On a review loop-back, reviewer notes are acknowledged so the re-interpretation
    visibly responds to feedback.
    """
    note_suffix = ""
    if state.reviewer_notes:
        note_suffix = f" (revised per reviewer note: {state.reviewer_notes})"

    features = ABCDEFeatures(
        asymmetry="Asymmetric across one axis." + note_suffix,
        border="Irregular, notched border with focal indistinctness.",
        color="Multiple shades: brown, black, and slate-blue.",
        diameter="Approximately 8 mm.",
        evolution="Reported recent change in size/color (per intake).",
    )
    return {"interpretation": features}


def literature(state: LesionState) -> dict:
    """PubMed retrieval keyed off the top differential. (Phase 5 makes this real.)"""
    refs = [
        LiteratureRef(
            pmid="00000000",
            title="Dermoscopy of melanoma and its mimics: a practical review.",
            relevance_note="ABCDE and dermoscopic criteria supporting the mock differential.",
        ),
        LiteratureRef(
            pmid="11111111",
            title="Deep learning for skin lesion classification on HAM10000.",
            relevance_note="Context for classifier probability calibration.",
        ),
    ]
    return {"literature": refs}


def build_report(state: LesionState) -> dict:
    """Assemble the structured, human-reviewable report from state."""
    assert state.interpretation is not None, "interpret must run before build_report"
    assert state.classifier_result is not None, "classify must run before build_report"

    differential = sorted(
        state.classifier_result.probabilities,
        key=state.classifier_result.probabilities.__getitem__,
        reverse=True,
    )[:3]

    report = LesionReport(
        abcde=state.interpretation,
        classifier=state.classifier_result,
        differential=differential,
        literature=state.literature,
        overall_confidence=round(state.classifier_result.top_confidence * 0.9, 3),
        recommendation=(
            "Mocked recommendation: findings are consistent with the top "
            "differential; route to a qualified human reviewer before any action."
        ),
        disclaimer=DISCLAIMER,
    )
    return {"report": report}


def human_review(state: LesionState) -> dict:
    """HARD-STOP human-review gate.

    Pauses the graph via ``interrupt()`` and waits indefinitely. The graph is
    resumed by re-invoking with ``Command(resume=<decision>)`` where ``<decision>``
    is a dict like ``{"status": "approved"|"rejected"|"edited", "notes": "..."}``.

    This gate is never auto-approved or skippable (HANDOFF.md §6).
    """
    decision = interrupt(
        {
            "action": "human_review_required",
            "prompt": (
                "Review this report. Respond with a decision dict: "
                '{"status": "approved" | "rejected" | "edited", "notes": "..."}'
            ),
            "report": state.report.model_dump() if state.report else None,
            "disclaimer": DISCLAIMER,
        }
    )

    # `decision` is whatever was passed to Command(resume=...).
    if isinstance(decision, dict):
        status = decision.get("status", "pending")
        notes = decision.get("notes")
    else:
        # Tolerate a bare string decision for convenience.
        status = str(decision)
        notes = None

    if status not in ("approved", "rejected", "edited"):
        raise ValueError(
            f"Invalid review status {status!r}; "
            "expected one of: approved, rejected, edited."
        )

    return {
        "review_status": status,
        "reviewer_notes": notes,
        "review_cycles": state.review_cycles + 1,
    }


def finalize(state: LesionState) -> dict:
    """Terminal node. Guarantees the disclaimer is present on the final report."""
    if state.report is not None and state.report.disclaimer != DISCLAIMER:
        updated = state.report.model_copy(update={"disclaimer": DISCLAIMER})
        return {"report": updated}
    return {}


def route_review(state: LesionState) -> str:
    """Conditional edge out of the review gate.

    approved            -> finalize
    rejected / edited   -> interpret (loop back to re-interpret with feedback)
    """
    return "finalize" if state.review_status == "approved" else "interpret"


__all__ = [
    "ingest",
    "preprocess",
    "classify",
    "interpret",
    "literature",
    "build_report",
    "human_review",
    "finalize",
    "route_review",
]
