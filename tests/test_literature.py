"""Phase 5 tests for the literature retrieval module.

The ChromaDB ranking is tested with an injected deterministic embedder, so no
~400MB PubMedBERT download is needed. The MCP fetch is covered at the
result-parsing level (the network leg needs a live MCP server).
"""

from __future__ import annotations

import json

import pytest

from dermassist import literature as lit
from dermassist.schemas import LiteratureRef


# ----------------------------- query building ------------------------------ #


def test_build_pubmed_query_expands_top_differential():
    q = lit.build_pubmed_query(["mel", "nv", "bkl"])
    assert "melanoma" in q
    assert "dermoscopy" in q


def test_build_index_query_includes_interpretation():
    q = lit.build_index_query(["vasc"], interpretation_text="uniform violaceous color")
    assert "vascular" in q
    assert "violaceous" in q


# ---------------------------- MCP result parsing --------------------------- #


def test_parse_list_of_dicts():
    raw = [
        {"pmid": "1", "title": "A", "abstract": "abs one"},
        {"pmid": "2", "title": "B", "abstract": "abs two"},
    ]
    arts = lit.parse_mcp_articles(raw)
    assert [a.pmid for a in arts] == ["1", "2"]
    assert arts[0].title == "A"


def test_parse_json_string_with_articles_key():
    raw = json.dumps({"articles": [{"uid": "9", "name": "T", "summary": "S"}]})
    arts = lit.parse_mcp_articles(raw)
    assert len(arts) == 1
    assert arts[0].pmid == "9"      # uid alias
    assert arts[0].title == "T"     # name alias
    assert arts[0].abstract == "S"  # summary alias


def test_parse_call_tool_result_like_object():
    class Block:
        def __init__(self, text):
            self.text = text

    class Result:
        def __init__(self, content):
            self.content = content

    payload = json.dumps([{"pmid": "5", "title": "X", "abstract": "y"}])
    arts = lit.parse_mcp_articles(Result([Block(payload)]))
    assert arts[0].pmid == "5"


def test_parse_garbage_returns_empty():
    assert lit.parse_mcp_articles("not json at all") == []
    assert lit.parse_mcp_articles(12345) == []


# ------------------------------ index ranking ------------------------------ #

_VOCAB = ["melanoma", "nevus", "vascular", "keratosis"]


def _fake_embed(texts):
    """Deterministic bag-of-keywords embedding (cosine-friendly, non-zero)."""
    out = []
    for t in texts:
        tl = t.lower()
        v = [float(tl.count(w)) for w in _VOCAB]
        if sum(v) == 0:
            v = [1.0, 0.0, 0.0, 0.0]
        out.append(v)
    return out


def _index(tmp_path):
    return lit.LiteratureIndex(embed_fn=_fake_embed, path=tmp_path / "chroma")


def test_index_add_and_query_ranks_by_similarity(tmp_path):
    idx = _index(tmp_path)
    idx.add([
        lit.Article("mel1", "Melanoma review", "melanoma melanoma dermoscopy of melanoma"),
        lit.Article("nv1", "Nevus review", "nevus benign nevus dermoscopy"),
        lit.Article("vasc1", "Vascular review", "vascular lesion angioma"),
    ])
    top = idx.query("melanoma diagnosis", k=1)
    assert len(top) == 1
    assert top[0].pmid == "mel1"


def test_index_skips_empty_abstracts(tmp_path):
    idx = _index(tmp_path)
    idx.add([lit.Article("e1", "Empty", "   "), lit.Article("v1", "Vasc", "vascular lesion")])
    top = idx.query("vascular", k=5)
    assert [a.pmid for a in top] == ["v1"]


def test_query_empty_index_returns_empty(tmp_path):
    assert _index(tmp_path).query("anything", k=3) == []


# ---------------------------- notes / refs --------------------------------- #


def test_extractive_note_uses_first_sentence():
    art = lit.Article("1", "T", "First sentence here. Second sentence ignored.")
    note = lit.extractive_note(art, ["mel"])
    assert note == "First sentence here."


def test_extractive_note_handles_empty_abstract():
    note = lit.extractive_note(lit.Article("1", "T", ""), ["mel"])
    assert "mel" in note


def test_to_literature_refs_shape():
    refs = lit.to_literature_refs([lit.Article("42", "Title", "An abstract sentence.")], ["mel"])
    assert isinstance(refs[0], LiteratureRef)
    assert refs[0].pmid == "42"
    assert refs[0].title == "Title"
