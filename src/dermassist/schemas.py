"""Pydantic schemas for pipeline state and the structured lesion report.

Mirrors the schema in HANDOFF.md §4, with the 7 HAM10000 classes pinned so the
classifier interface is stable across the mocked (Phase 1) and real (Phase 3)
implementations.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from dermassist.compliance import DISCLAIMER

# The 7 HAM10000 diagnostic categories. The classifier owns calibrated
# probabilities over exactly these labels.
HAM10000_CLASSES: tuple[str, ...] = (
    "akiec",  # Actinic keratoses / intraepithelial carcinoma
    "bcc",    # Basal cell carcinoma
    "bkl",    # Benign keratosis-like lesions
    "df",     # Dermatofibroma
    "mel",    # Melanoma
    "nv",     # Melanocytic nevi
    "vasc",   # Vascular lesions
)

ReviewStatus = Literal["pending", "approved", "rejected", "edited"]


class ABCDEFeatures(BaseModel):
    """Claude's narrative interpretation along the ABCDE dermoscopy axes."""

    asymmetry: str
    border: str
    color: str
    diameter: str
    evolution: str


class ClassifierResult(BaseModel):
    """Calibrated output of the dedicated vision classifier.

    The classifier — not Claude — owns these probabilities (HANDOFF.md §6).
    """

    label: str
    probabilities: dict[str, float]  # class -> prob over the 7 HAM10000 classes
    top_confidence: float = Field(ge=0, le=1)


class LiteratureRef(BaseModel):
    """A PubMed reference cited in the report."""

    pmid: str
    title: str
    relevance_note: str


class LesionReport(BaseModel):
    """The structured, human-reviewable report."""

    abcde: ABCDEFeatures
    classifier: ClassifierResult
    differential: list[str]
    literature: list[LiteratureRef]
    overall_confidence: float = Field(ge=0, le=1)
    recommendation: str
    disclaimer: str = DISCLAIMER


class LesionState(BaseModel):
    """Graph state carried across nodes and persisted by the checkpointer."""

    image_path: str
    preprocessed_path: Optional[str] = None
    classifier_result: Optional[ClassifierResult] = None
    interpretation: Optional[ABCDEFeatures] = None
    literature: list[LiteratureRef] = Field(default_factory=list)
    report: Optional[LesionReport] = None
    review_status: ReviewStatus = "pending"
    reviewer_notes: Optional[str] = None
    review_cycles: int = 0  # how many times the review gate has been hit


__all__ = [
    "HAM10000_CLASSES",
    "ReviewStatus",
    "ABCDEFeatures",
    "ClassifierResult",
    "LiteratureRef",
    "LesionReport",
    "LesionState",
]
