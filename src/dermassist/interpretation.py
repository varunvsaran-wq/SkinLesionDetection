"""Phase 4 — Claude ABCDE interpretation + structured output.

Claude looks at the (preprocessed) dermoscopy image, assesses the ABCDE features
(Asymmetry, Border, Color, Diameter, Evolution), and reconciles its visual
narrative with the dedicated classifier's probabilities. Output is forced through
a Pydantic-validated schema via tool-use (HANDOFF.md §4).

§6 guardrail: Claude owns the *narrative/interpretation* only — it is explicitly
instructed never to emit its own diagnostic probabilities. The classifier owns
those calibrated numbers.

The Anthropic SDK is imported lazily inside ``interpret_image`` so the rest of the
package (and the offline tests for the pure prompt/parse helpers) doesn't require
``anthropic`` to be installed.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from dermassist.compliance import DISCLAIMER
from dermassist.config import get_settings
from dermassist.schemas import ABCDEFeatures, ClassifierResult

TOOL_NAME = "record_abcde_interpretation"

SYSTEM_PROMPT = (
    "You are a dermoscopy image-analysis assistant for a RESEARCH AND EDUCATIONAL "
    "tool. You are NOT a medical device and your output is NOT a diagnosis.\n\n"
    "You will be shown a preprocessed dermatoscopic image of a skin lesion together "
    "with the calibrated class probabilities from a dedicated vision classifier "
    "(over the 7 HAM10000 classes: akiec, bcc, bkl, df, mel, nv, vasc).\n\n"
    "Your job:\n"
    "1. Describe the lesion along the ABCDE axes (Asymmetry, Border, Color, "
    "Diameter/dimension, Evolution) based on what is visible in the image. For "
    "Evolution, note that change-over-time is not observable from a single image; "
    "say so rather than inventing history.\n"
    "2. Reconcile your visual impression with the classifier's probabilities — note "
    "where they agree or disagree and why.\n\n"
    "HARD RULES:\n"
    "- Do NOT output your own diagnostic probabilities, percentages, or a single "
    "diagnosis. The classifier owns the calibrated probabilities; you own the "
    "qualitative narrative only.\n"
    "- Record your assessment by calling the provided tool. Do not answer in prose.\n\n"
    f"{DISCLAIMER}"
)

# Strict tool schema. We hand-write it (rather than generating from Pydantic) so it
# is guaranteed strict-compatible: all fields required, additionalProperties false.
INTERPRETATION_TOOL: dict[str, Any] = {
    "name": TOOL_NAME,
    "description": (
        "Record the ABCDE dermoscopy interpretation and the reconciliation with the "
        "classifier's probabilities. Narrative only — no probabilities or diagnosis."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "asymmetry": {"type": "string", "description": "Asymmetry assessment."},
            "border": {"type": "string", "description": "Border regularity assessment."},
            "color": {"type": "string", "description": "Colour variegation assessment."},
            "diameter": {"type": "string", "description": "Diameter / size assessment."},
            "evolution": {
                "type": "string",
                "description": "Evolution note (not observable from one image — say so).",
            },
            "reconciliation": {
                "type": "string",
                "description": (
                    "How the visual ABCDE impression agrees or disagrees with the "
                    "classifier's probabilities. No probabilities of your own."
                ),
            },
        },
        "required": ["asymmetry", "border", "color", "diameter", "evolution", "reconciliation"],
        "additionalProperties": False,
    },
}

_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}


class _InterpretationToolInput(BaseModel):
    """Validates the tool input Claude returns."""

    asymmetry: str
    border: str
    color: str
    diameter: str
    evolution: str
    reconciliation: str


def media_type_for(path: str | Path) -> str:
    """Map a file suffix to an Anthropic-supported image media type."""
    suffix = Path(path).suffix.lower()
    if suffix not in _MEDIA_TYPES:
        raise ValueError(
            f"Unsupported image type {suffix!r}; expected one of {sorted(_MEDIA_TYPES)}."
        )
    return _MEDIA_TYPES[suffix]


def _format_probabilities(classifier: ClassifierResult) -> str:
    ranked = sorted(
        classifier.probabilities.items(), key=lambda kv: kv[1], reverse=True
    )
    lines = [f"  {label}: {prob:.3f}" for label, prob in ranked]
    return "\n".join(lines)


def build_user_content(
    image_b64: str,
    media_type: str,
    classifier: ClassifierResult,
    reviewer_notes: str | None = None,
) -> list[dict[str, Any]]:
    """Assemble the user turn: the image plus the classifier context and the ask."""
    text = (
        "Dermatoscopic image attached. The dedicated classifier reported these "
        f"calibrated probabilities (top label: {classifier.label}, "
        f"confidence {classifier.top_confidence:.3f}):\n"
        f"{_format_probabilities(classifier)}\n\n"
        "Assess the lesion along the ABCDE axes from the image, reconcile your "
        "impression with the probabilities above, and record it via the tool."
    )
    if reviewer_notes:
        text += (
            "\n\nA human reviewer returned this report for revision with the note: "
            f"\"{reviewer_notes}\". Address it in your reassessment."
        )
    return [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": image_b64},
        },
        {"type": "text", "text": text},
    ]


def _block_field(block: Any, name: str) -> Any:
    """Read a field from a content block that may be a dict or an SDK object."""
    if isinstance(block, dict):
        return block.get(name)
    return getattr(block, name, None)


def parse_interpretation(content: list[Any]) -> tuple[ABCDEFeatures, str]:
    """Extract + validate the tool call from a response's content blocks.

    Returns ``(ABCDEFeatures, reconciliation)``. Raises ``ValueError`` if the
    expected tool call is absent or fails validation.
    """
    for block in content:
        if _block_field(block, "type") == "tool_use" and _block_field(block, "name") == TOOL_NAME:
            validated = _InterpretationToolInput.model_validate(_block_field(block, "input"))
            features = ABCDEFeatures(
                asymmetry=validated.asymmetry,
                border=validated.border,
                color=validated.color,
                diameter=validated.diameter,
                evolution=validated.evolution,
            )
            return features, validated.reconciliation
    raise ValueError(f"No '{TOOL_NAME}' tool call found in Claude's response.")


def interpret_image(
    image_path: str | Path,
    classifier: ClassifierResult,
    reviewer_notes: str | None = None,
) -> tuple[ABCDEFeatures, str]:
    """Call Claude to produce the ABCDE interpretation + reconciliation."""
    import anthropic  # lazy: only needed for the real path

    settings = get_settings()
    client = (
        anthropic.Anthropic(api_key=settings.anthropic_api_key)
        if settings.anthropic_api_key
        else anthropic.Anthropic()
    )

    image_b64 = base64.standard_b64encode(Path(image_path).read_bytes()).decode("utf-8")
    media_type = media_type_for(image_path)

    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=2000,
        system=[
            {"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}
        ],
        messages=[
            {
                "role": "user",
                "content": build_user_content(
                    image_b64, media_type, classifier, reviewer_notes
                ),
            }
        ],
        tools=[INTERPRETATION_TOOL],
        tool_choice={"type": "tool", "name": TOOL_NAME},
    )
    return parse_interpretation(response.content)


__all__ = [
    "TOOL_NAME",
    "SYSTEM_PROMPT",
    "INTERPRETATION_TOOL",
    "media_type_for",
    "build_user_content",
    "parse_interpretation",
    "interpret_image",
]
