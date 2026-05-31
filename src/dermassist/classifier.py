"""Phase 3 — real vision classifier.

Loads a pretrained HuggingFace ``transformers`` image-classification checkpoint
(a ViT by default) and produces probabilities over the 7 canonical HAM10000
classes. The dedicated classifier — never Claude — owns these probabilities
(HANDOFF.md §6).

The numeric core (label normalization, softmax, aggregation to canonical classes)
is pure numpy so it is unit-testable without downloading torch or model weights.
Only ``classify_image`` pulls in ``torch``/``transformers``, and it does so lazily.

Output contract matches the Phase-1 mock exactly: a ``ClassifierResult`` with a
``probabilities`` dict over all 7 classes — a drop-in swap.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from dermassist.config import get_settings
from dermassist.schemas import HAM10000_CLASSES, ClassifierResult

# Ordered keyword rules mapping arbitrary model label strings to canonical HAM10000
# codes. First match wins, so order resolves ambiguity (e.g. "actinic keratoses"
# must hit `akiec` before the generic `keratosis` -> `bkl` rule; "melanoma" is a
# full word so it never collides with "melanocytic ...").
_LABEL_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("akiec", "actinic", "intraepithelial", "bowen"), "akiec"),
    (("bcc", "basal"), "bcc"),
    (("dermatofibroma",), "df"),
    (("melanoma",), "mel"),
    (("melanocytic", "nevi", "nevus", "naevus"), "nv"),
    (("vascular", "angioma", "haemangioma", "hemangioma"), "vasc"),
    (("keratosis", "keratoses", "seborrheic", "lichen", "bkl"), "bkl"),
)


def normalize_label(label: str) -> str:
    """Map a model's class label to one of the 7 canonical HAM10000 codes.

    Raises ``ValueError`` on an unmappable label rather than guessing — a silent
    mismap would corrupt the diagnostic probabilities.
    """
    s = label.strip().lower().replace("-", " ").replace("_", " ")
    if s in HAM10000_CLASSES:
        return s
    for keywords, canonical in _LABEL_RULES:
        if any(k in s for k in keywords):
            return canonical
    raise ValueError(
        f"Cannot map classifier label {label!r} to a HAM10000 class. "
        "Check the model's id2label or extend _LABEL_RULES."
    )


def _softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64).ravel() / temperature
    z = z - z.max()  # numerical stability
    e = np.exp(z)
    return e / e.sum()


def build_classifier_result(
    id2label: dict[int, str],
    logits: np.ndarray,
    temperature: float = 1.0,
    round_to: int = 4,
) -> ClassifierResult:
    """Turn model logits + its id2label into a canonical ``ClassifierResult``.

    Probabilities for any model labels that map to the same canonical class are
    summed; canonical classes the model never emits get 0.0. Probabilities sum to
    1.0 (up to rounding).
    """
    probs = _softmax(logits, temperature)
    canonical: dict[str, float] = {c: 0.0 for c in HAM10000_CLASSES}
    for idx, p in enumerate(probs):
        # HF configs sometimes key id2label by str; tolerate both.
        label = id2label.get(idx, id2label.get(str(idx)))  # type: ignore[arg-type]
        if label is None:
            raise ValueError(f"id2label has no entry for index {idx}.")
        canonical[normalize_label(label)] += float(p)

    top_label = max(canonical, key=canonical.__getitem__)
    return ClassifierResult(
        label=top_label,
        probabilities={k: round(v, round_to) for k, v in canonical.items()},
        top_confidence=round(canonical[top_label], round_to),
    )


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    """Load (processor, model) once per model name. Lazily imports transformers."""
    import torch  # noqa: F401  (ensures a clear error if torch is missing)
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    model.eval()
    return processor, model


def classify_image(
    image_path: str | Path,
    model_name: str | None = None,
    temperature: float | None = None,
) -> ClassifierResult:
    """Run the real classifier on a (preprocessed) image and return canonical probs."""
    import torch

    from dermassist.preprocessing import load_image

    settings = get_settings()
    model_name = model_name or settings.classifier_model
    temperature = settings.classifier_temperature if temperature is None else temperature

    processor, model = _load_model(model_name)
    image = load_image(image_path)  # RGB uint8 (H, W, 3); processor handles resize/normalize

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    return build_classifier_result(
        id2label, logits.numpy(), temperature=temperature
    )


__all__ = [
    "normalize_label",
    "build_classifier_result",
    "classify_image",
]
