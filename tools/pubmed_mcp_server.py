"""A minimal PubMed MCP server for DermAssist (Phase 5).

Wraps NCBI E-utilities (esearch + efetch) and exposes a single Streamable-HTTP MCP
tool, ``search_pubmed(query, retmax)``, returning articles in exactly the shape the
``dermassist.literature`` client expects: ``[{"pmid", "title", "abstract"}, ...]``.

Run it:

    uv run python tools/pubmed_mcp_server.py            # serves http://127.0.0.1:8000/mcp

then point DermAssist at it in ``.env``:

    MCP_ENDPOINT=http://127.0.0.1:8000/mcp
    MCP_PUBMED_TOOL=search_pubmed

Environment:
    PUBMED_MCP_HOST   (default 127.0.0.1)
    PUBMED_MCP_PORT   (default 8000)
    NCBI_API_KEY      (optional — raises the NCBI rate limit from 3 to 10 req/s)

Compliance: this only retrieves public literature metadata. DermAssist remains a
research/educational tool, not a medical device.
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET

import httpx

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_TOOL_TAG = "dermassist"  # NCBI asks callers to identify themselves


def _params(extra: dict) -> dict:
    params = {"db": "pubmed", "tool": _TOOL_TAG, **extra}
    api_key = os.environ.get("NCBI_API_KEY")
    if api_key:
        params["api_key"] = api_key
    email = os.environ.get("NCBI_EMAIL")
    if email:
        params["email"] = email
    return params


def esearch_pmids(client: httpx.Client, query: str, retmax: int) -> list[str]:
    """Return PMIDs for a query via E-utilities esearch (JSON)."""
    resp = client.get(
        f"{EUTILS}/esearch.fcgi",
        params=_params({"term": query, "retmax": retmax, "retmode": "json"}),
    )
    resp.raise_for_status()
    return resp.json().get("esearchresult", {}).get("idlist", [])


def parse_efetch_xml(xml_text: str) -> list[dict]:
    """Parse an efetch PubMed XML document into article dicts.

    Pure function (no network) so it can be unit-tested with a fixture.
    """
    articles: list[dict] = []
    root = ET.fromstring(xml_text)
    for art in root.findall(".//PubmedArticle"):
        pmid_el = art.find(".//MedlineCitation/PMID")
        pmid = (pmid_el.text or "").strip() if pmid_el is not None else ""

        title_el = art.find(".//Article/ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""

        # Abstracts may be split into multiple labelled sections.
        sections = []
        for ab in art.findall(".//Article/Abstract/AbstractText"):
            text = "".join(ab.itertext()).strip()
            if not text:
                continue
            label = ab.get("Label")
            sections.append(f"{label}: {text}" if label else text)
        abstract = " ".join(sections)

        if pmid or abstract:
            articles.append({"pmid": pmid, "title": title, "abstract": abstract})
    return articles


def efetch_articles(client: httpx.Client, pmids: list[str]) -> list[dict]:
    """Fetch titles + abstracts for PMIDs via E-utilities efetch (XML)."""
    if not pmids:
        return []
    resp = client.get(
        f"{EUTILS}/efetch.fcgi",
        params=_params({"id": ",".join(pmids), "retmode": "xml", "rettype": "abstract"}),
    )
    resp.raise_for_status()
    return parse_efetch_xml(resp.text)


def fetch_pubmed(query: str, retmax: int = 20) -> list[dict]:
    """esearch → efetch round-trip. Returns ``[{"pmid","title","abstract"}, ...]``."""
    with httpx.Client(timeout=20.0, headers={"User-Agent": f"{_TOOL_TAG}/0.1"}) as client:
        pmids = esearch_pmids(client, query, retmax)
        return efetch_articles(client, pmids)


def build_server():
    """Construct the FastMCP server (imported lazily so the parser stays testable)."""
    from mcp.server.fastmcp import FastMCP

    host = os.environ.get("PUBMED_MCP_HOST", "127.0.0.1")
    port = int(os.environ.get("PUBMED_MCP_PORT", "8000"))
    mcp = FastMCP("pubmed", host=host, port=port)

    @mcp.tool()
    def search_pubmed(query: str, retmax: int = 20) -> str:
        """Search PubMed and return matching articles (pmid, title, abstract).

        Args:
            query: PubMed search expression (e.g. "melanoma dermoscopy diagnosis").
            retmax: Maximum number of articles to return.
        """
        try:
            articles = fetch_pubmed(query, retmax)
        except Exception as exc:  # surface a structured error to the caller
            return json.dumps({"error": str(exc), "articles": []})
        return json.dumps(articles)

    return mcp


def main() -> None:
    server = build_server()
    host = os.environ.get("PUBMED_MCP_HOST", "127.0.0.1")
    port = os.environ.get("PUBMED_MCP_PORT", "8000")
    print(f"PubMed MCP server -> http://{host}:{port}/mcp  (tool: search_pubmed)")
    server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
