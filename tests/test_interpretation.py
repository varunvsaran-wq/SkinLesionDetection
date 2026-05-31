"""Phase 4 tests for the interpretation module's pure helpers.

These cover the tool schema, user-content assembly, media-type detection, and the
tool-response parser — all without calling the Anthropic API. The live
``interpret_image`` path is validated separately/manually.
"""

from __future__ import annotations

import pytest

from dermassist import interpretation as it
from dermassist.schemas import ABCDEFeatures, ClassifierResult


def _classifier() -> ClassifierResult:
    return ClassifierResult(
        label="mel",
        probabilities={
            "akiec": 0.03, "bcc": 0.05, "bkl": 0.07, "df": 0.02,
            "mel": 0.62, "nv": 0.18, "vasc": 0.03,
        },
        top_confidence=0.62,
    )


def _tool_block(**overrides):
    payload = {
        "asymmetry": "Asymmetric across both axes.",
        "border": "Irregular, notched.",
        "color": "Brown, black, blue-grey.",
        "diameter": "~8 mm.",
        "evolution": "Not observable from a single image.",
        "reconciliation": "Visual impression aligns with the classifier's melanoma lean.",
    }
    payload.update(overrides)
    return {"type": "tool_use", "name": it.TOOL_NAME, "input": payload}


def test_tool_schema_is_strict():
    schema = it.INTERPRETATION_TOOL["input_schema"]
    assert it.INTERPRETATION_TOOL["strict"] is True
    assert schema["additionalProperties"] is False
    # Every property is required (strict tool-use requirement).
    assert set(schema["required"]) == set(schema["properties"])
    assert "reconciliation" in schema["properties"]


@pytest.mark.parametrize(
    "name,expected",
    [
        ("x.png", "image/png"),
        ("x.PNG", "image/png"),
        ("x.jpg", "image/jpeg"),
        ("x.jpeg", "image/jpeg"),
        ("x.webp", "image/webp"),
    ],
)
def test_media_type_for(name, expected):
    assert it.media_type_for(name) == expected


def test_media_type_rejects_unknown():
    with pytest.raises(ValueError):
        it.media_type_for("scan.tiff")


def test_build_user_content_includes_image_and_probs():
    content = it.build_user_content("BASE64", "image/png", _classifier())
    assert content[0]["type"] == "image"
    assert content[0]["source"] == {
        "type": "base64", "media_type": "image/png", "data": "BASE64",
    }
    text = content[1]["text"]
    # Probabilities and the top label are surfaced for reconciliation.
    assert "mel: 0.620" in text
    assert "top label: mel" in text


def test_build_user_content_appends_reviewer_notes():
    content = it.build_user_content("B64", "image/png", _classifier(), reviewer_notes="re-check border")
    assert "re-check border" in content[1]["text"]


def test_parse_interpretation_valid():
    features, reconciliation = it.parse_interpretation([_tool_block()])
    assert isinstance(features, ABCDEFeatures)
    assert features.border == "Irregular, notched."
    # Reconciliation is returned separately — not folded into ABCDEFeatures.
    assert "melanoma" in reconciliation


def test_parse_interpretation_missing_tool_raises():
    text_only = [{"type": "text", "text": "no tool call here"}]
    with pytest.raises(ValueError):
        it.parse_interpretation(text_only)


def test_parse_interpretation_invalid_input_raises():
    bad = _tool_block()
    del bad["input"]["color"]  # missing required field
    with pytest.raises(ValueError):
        it.parse_interpretation([bad])


def test_parse_interpretation_supports_sdk_object_blocks():
    # Real SDK returns objects with attributes, not dicts.
    class Block:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    block = Block(type="tool_use", name=it.TOOL_NAME, input=_tool_block()["input"])
    features, _ = it.parse_interpretation([block])
    assert features.diameter == "~8 mm."
