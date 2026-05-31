"""Phase 3 tests for the classifier's numeric core.

These exercise label normalization and probability aggregation directly, so they
run without downloading torch or model weights. The end-to-end ``classify_image``
path (which needs the real model) is validated separately/manually.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dermassist.classifier import build_classifier_result, normalize_label
from dermassist.schemas import HAM10000_CLASSES


# The default model's verbatim labels -> expected canonical codes.
ANWARKH_LABELS = {
    "benign_keratosis-like_lesions": "bkl",
    "basal_cell_carcinoma": "bcc",
    "actinic_keratoses": "akiec",
    "vascular_lesions": "vasc",
    "melanocytic_Nevi": "nv",
    "melanoma": "mel",
    "dermatofibroma": "df",
}


@pytest.mark.parametrize("label,expected", ANWARKH_LABELS.items())
def test_normalize_default_model_labels(label, expected):
    assert normalize_label(label) == expected


@pytest.mark.parametrize("code", HAM10000_CLASSES)
def test_normalize_passes_through_canonical_codes(code):
    assert normalize_label(code) == code


@pytest.mark.parametrize(
    "label,expected",
    [
        ("MEL", "mel"),
        ("Melanocytic nevus", "nv"),
        ("naevus", "nv"),
        ("Basal Cell Carcinoma", "bcc"),
        ("Actinic Keratosis / Bowen's", "akiec"),
        ("seborrheic keratosis", "bkl"),
        ("haemangioma", "vasc"),
    ],
)
def test_normalize_handles_variants(label, expected):
    assert normalize_label(label) == expected


def test_normalize_rejects_unknown():
    with pytest.raises(ValueError):
        normalize_label("something_unrelated")


def test_actinic_keratoses_not_misrouted_to_bkl():
    # "actinic keratoses" contains "keratoses" but must resolve to akiec, not bkl.
    assert normalize_label("actinic_keratoses") == "akiec"


def test_build_result_sums_to_one_and_covers_all_classes():
    id2label = dict(enumerate(ANWARKH_LABELS.keys()))
    logits = np.array([0.1, 0.2, 0.05, 0.05, 3.0, 0.4, 0.1])  # index 4 = melanocytic_Nevi

    result = build_classifier_result(id2label, logits)

    assert set(result.probabilities) == set(HAM10000_CLASSES)
    assert math.isclose(sum(result.probabilities.values()), 1.0, abs_tol=1e-3)
    assert result.label == "nv"  # highest logit maps to melanocytic_Nevi -> nv
    assert result.probabilities["nv"] == pytest.approx(result.top_confidence)
    assert 0.0 <= result.top_confidence <= 1.0


def test_build_result_string_keyed_id2label():
    # HF configs sometimes key id2label by string indices.
    id2label = {str(i): lbl for i, lbl in enumerate(ANWARKH_LABELS.keys())}
    logits = np.zeros(7)
    logits[5] = 5.0  # melanoma
    result = build_classifier_result(id2label, logits)  # type: ignore[arg-type]
    assert result.label == "mel"


def test_temperature_softens_confidence():
    id2label = dict(enumerate(ANWARKH_LABELS.keys()))
    logits = np.array([0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0])
    sharp = build_classifier_result(id2label, logits, temperature=1.0)
    soft = build_classifier_result(id2label, logits, temperature=5.0)
    # Higher temperature -> less peaked -> lower top confidence.
    assert soft.top_confidence < sharp.top_confidence
