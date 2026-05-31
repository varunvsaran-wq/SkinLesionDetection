"""Offline tests for the PubMed MCP server's XML parser.

The network round-trip (esearch/efetch) needs NCBI; here we verify the pure
``parse_efetch_xml`` against a representative efetch document, and that the parsed
output is consumable by the dermassist literature client.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from dermassist.literature import parse_mcp_articles

_SERVER = Path(__file__).resolve().parents[1] / "tools" / "pubmed_mcp_server.py"


def _load_server():
    spec = importlib.util.spec_from_file_location("pubmed_mcp_server", _SERVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SAMPLE_XML = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <ArticleTitle>Dermoscopy of melanoma: a review</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Melanoma shows atypical networks.</AbstractText>
          <AbstractText Label="CONCLUSION">ABCDE aids recognition.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>22222222</PMID>
      <Article>
        <ArticleTitle>Benign nevi on dermoscopy</ArticleTitle>
        <Abstract>
          <AbstractText>Symmetric regular pigment network.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


def test_parse_efetch_xml_extracts_articles():
    server = _load_server()
    arts = server.parse_efetch_xml(SAMPLE_XML)
    assert len(arts) == 2
    first = arts[0]
    assert first["pmid"] == "12345678"
    assert first["title"] == "Dermoscopy of melanoma: a review"
    # Labelled sections are concatenated with their labels.
    assert "BACKGROUND:" in first["abstract"]
    assert "CONCLUSION:" in first["abstract"]
    # Unlabelled abstract has no label prefix.
    assert arts[1]["abstract"] == "Symmetric regular pigment network."


def test_server_output_is_consumable_by_literature_client():
    """The server returns JSON the dermassist parser normalizes 1:1."""
    server = _load_server()
    arts = server.parse_efetch_xml(SAMPLE_XML)
    import json

    normalized = parse_mcp_articles(json.dumps(arts))
    assert [a.pmid for a in normalized] == ["12345678", "22222222"]
    assert normalized[0].title.startswith("Dermoscopy of melanoma")
