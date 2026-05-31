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
    """Resize + Shades-of-Gray color constancy + DullRazor hair removal (Phase 2, real).

    If the source image exists on disk, run the real pipeline and write the result
    to ``artifacts/preprocessed/``. If it doesn't (e.g. the mocked, dataset-free
    smoke demo), fall back to deriving a path string so the graph still flows.
    """
    src = Path(state.image_path)
    out_dir = Path("artifacts/preprocessed")
    out_path = out_dir / f"{src.stem}__preprocessed.png"

    if not src.exists():
        print(f"[preprocess] note: '{src}' not found — skipping real preprocessing (mock path).")
        return {"preprocessed_path": str(out_path)}

    # Imported lazily so the core (Phase 1) install doesn't require opencv/numpy.
    from dermassist.preprocessing import preprocess_image

    written = preprocess_image(src, out_path)
    print(f"[preprocess] wrote preprocessed image -> {written}")
    return {"preprocessed_path": str(written)}


def _mock_classifier_result() -> ClassifierResult:
    """Plausible melanoma-leaning distribution (sums to 1.0). Used as a fallback
    when torch/transformers or a real image are unavailable (dataset-free demo)."""
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
    return ClassifierResult(
        label=top_label,
        probabilities=probabilities,
        top_confidence=probabilities[top_label],
    )


def classify(state: LesionState) -> dict:
    """Dedicated vision classifier (Phase 3, real).

    The classifier OWNS the calibrated probabilities over the 7 HAM10000 classes
    (never Claude — HANDOFF.md §6). Runs the real HuggingFace model on the
    preprocessed image; falls back to a mock distribution when torch/transformers
    aren't installed or no real image exists, so the graph still flows.
    """
    img_path = Path(state.preprocessed_path or state.image_path)

    if not img_path.exists():
        print(f"[classify] note: '{img_path}' not found — using mock classifier output.")
        return {"classifier_result": _mock_classifier_result()}

    try:
        from dermassist.classifier import classify_image

        result = classify_image(img_path)
        print(f"[classify] real classifier -> {result.label} ({result.top_confidence:.3f})")
        return {"classifier_result": result}
    except ImportError as exc:
        print(f"[classify] vision deps not installed ({exc}); using mock output. "
              "Install with: uv sync --extra vision")
        return {"classifier_result": _mock_classifier_result()}


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
