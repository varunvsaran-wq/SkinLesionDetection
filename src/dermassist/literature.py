"""Phase 5 — literature retrieval.

Pipeline (HANDOFF.md §5, Phase 5):

    top differential ─▶ PubMed (via MCP) ─▶ embed abstracts (PubMedBERT)
                     ─▶ store in ChromaDB ─▶ retrieve top-k for grounding
                     ─▶ relevance notes (extractive, or opt-in Claude-authored)

Design notes:
- The PubMed fetch goes through a **configurable MCP server** (`MCP_ENDPOINT`);
  ``parse_mcp_articles`` normalizes the common result shapes. When the endpoint is
  unset/unreachable the ``literature`` node falls back to mock abstracts.
- ``LiteratureIndex`` takes an **injectable** ``embed_fn`` so the ranking logic is
  unit-testable without downloading the ~400MB PubMedBERT weights; the default
  ``embed_fn`` lazily loads sentence-transformers.
- Heavy imports (chromadb, sentence_transformers, mcp, anthropic) are all lazy.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable, Optional

from dermassist.config import get_settings
from dermassist.schemas import LiteratureRef

EmbedFn = Callable[[list[str]], list[list[float]]]

# HAM10000 code → full term, used to expand the differential into a PubMed query.
_CLASS_TERMS = {
    "akiec": "actinic keratosis intraepithelial carcinoma",
    "bcc": "basal cell carcinoma",
    "bkl": "benign keratosis seborrheic keratosis",
    "df": "dermatofibroma",
    "mel": "melanoma",
    "nv": "melanocytic nevus",
    "vasc": "vascular lesion",
}


class Article:
    """A normalized PubMed article."""

    __slots__ = ("pmid", "title", "abstract")

    def __init__(self, pmid: str, title: str, abstract: str):
        self.pmid = str(pmid)
        self.title = title or ""
        self.abstract = abstract or ""

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Article(pmid={self.pmid!r}, title={self.title[:40]!r})"


# --------------------------- query construction ---------------------------- #


def build_pubmed_query(differential: list[str]) -> str:
    """Build a PubMed search query keyed off the top differential(s)."""
    terms = [_CLASS_TERMS.get(code, code) for code in differential[:2]]
    primary = terms[0] if terms else "skin lesion"
    return f"{primary} dermoscopy diagnosis"


def build_index_query(differential: list[str], interpretation_text: str | None = None) -> str:
    """The text used to retrieve the most relevant stored abstracts for this case."""
    terms = " ".join(_CLASS_TERMS.get(code, code) for code in differential[:2])
    query = f"dermoscopy {terms}"
    if interpretation_text:
        query = f"{query}. {interpretation_text}"
    return query


# ------------------------------- MCP client -------------------------------- #


def parse_mcp_articles(raw: Any) -> list[Article]:
    """Normalize a PubMed MCP tool result into ``Article`` objects.

    Tolerates the shapes PubMed MCP servers commonly return:
    - a list of dicts with pmid/title/abstract (or aliases like uid/pmid, name/title)
    - a JSON string encoding such a list (or ``{"articles": [...]}` / ``{"results": [...]}``)
    - the MCP SDK ``CallToolResult`` whose ``.content`` holds text blocks of the above
    """
    # Unwrap an MCP CallToolResult-like object: join its text content blocks.
    content = getattr(raw, "content", None)
    if content is not None and not isinstance(raw, (list, dict, str)):
        texts = []
        for block in content:
            text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if text:
                texts.append(text)
        raw = "\n".join(texts)

    if isinstance(raw, str):
        raw = raw.strip()
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return []

    if isinstance(raw, dict):
        for key in ("articles", "results", "data", "esearchresult"):
            if key in raw and isinstance(raw[key], list):
                raw = raw[key]
                break
        else:
            raw = [raw]

    if not isinstance(raw, list):
        return []

    articles: list[Article] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        pmid = item.get("pmid") or item.get("uid") or item.get("id") or ""
        title = item.get("title") or item.get("name") or ""
        abstract = item.get("abstract") or item.get("summary") or item.get("text") or ""
        if pmid or abstract:
            articles.append(Article(pmid=pmid, title=title, abstract=abstract))
    return articles


async def _mcp_call(endpoint: str, tool: str, arguments: dict) -> Any:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    async with streamablehttp_client(endpoint) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool(tool, arguments)


def pubmed_search(query: str, retmax: int | None = None) -> list[Article]:
    """Search PubMed through the configured MCP server. Raises on misconfiguration."""
    settings = get_settings()
    if not settings.mcp_endpoint:
        raise RuntimeError("MCP_ENDPOINT not configured.")
    retmax = retmax or settings.pubmed_retmax
    raw = asyncio.run(
        _mcp_call(
            settings.mcp_endpoint,
            settings.mcp_pubmed_tool,
            {"query": query, "retmax": retmax},
        )
    )
    return parse_mcp_articles(raw)


# ----------------------------- vector index -------------------------------- #


def _default_embed_fn(model_name: str) -> EmbedFn:
    """Lazily build a sentence-transformers embedding function (downloads weights)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def embed(texts: list[str]) -> list[list[float]]:
        return model.encode(texts, normalize_embeddings=True).tolist()

    return embed


class LiteratureIndex:
    """A local ChromaDB collection of PubMed abstracts with PubMedBERT embeddings."""

    def __init__(
        self,
        embed_fn: Optional[EmbedFn] = None,
        path: Optional[Path] = None,
        collection_name: str = "pubmed_abstracts",
    ):
        import chromadb

        settings = get_settings()
        self._embed_fn = embed_fn or _default_embed_fn(settings.embedding_model)
        if path is None:
            path = settings.chroma_path
        path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(path))
        # We supply embeddings ourselves, so no Chroma-side embedding function.
        self._collection = self._client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, articles: list[Article]) -> None:
        articles = [a for a in articles if a.abstract.strip()]
        if not articles:
            return
        ids = [a.pmid or f"noid-{i}" for i, a in enumerate(articles)]
        documents = [a.abstract for a in articles]
        metadatas = [{"pmid": a.pmid, "title": a.title} for a in articles]
        embeddings = self._embed_fn(documents)
        self._collection.upsert(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
        )

    def query(self, text: str, k: int) -> list[Article]:
        count = self._collection.count()
        if count == 0:
            return []
        embedding = self._embed_fn([text])[0]
        res = self._collection.query(query_embeddings=[embedding], n_results=min(k, count))
        out: list[Article] = []
        for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
            out.append(Article(pmid=meta.get("pmid", ""), title=meta.get("title", ""), abstract=doc))
        return out


# --------------------------- relevance notes ------------------------------- #


def extractive_note(article: Article, differential: list[str]) -> str:
    """A free, deterministic relevance note: the abstract's first sentence."""
    abstract = article.abstract.strip()
    if not abstract:
        return f"Relevant to the {differential[0] if differential else 'lesion'} differential."
    first = abstract.split(". ")[0].strip()
    if len(first) > 280:
        first = first[:277] + "..."
    return first if first.endswith(".") else first + "."


# ------------------------------ orchestrator ------------------------------- #


def to_literature_refs(articles: list[Article], differential: list[str]) -> list[LiteratureRef]:
    return [
        LiteratureRef(
            pmid=a.pmid or "n/a",
            title=a.title or "(untitled)",
            relevance_note=extractive_note(a, differential),
        )
        for a in articles
    ]


def retrieve_literature(
    differential: list[str],
    interpretation_text: str | None = None,
    embed_fn: Optional[EmbedFn] = None,
) -> list[LiteratureRef]:
    """End-to-end: MCP search → embed + store → retrieve top-k → relevance notes."""
    settings = get_settings()
    articles = pubmed_search(build_pubmed_query(differential))
    index = LiteratureIndex(embed_fn=embed_fn)
    index.add(articles)
    top = index.query(
        build_index_query(differential, interpretation_text), k=settings.literature_top_k
    )
    return to_literature_refs(top, differential)


__all__ = [
    "Article",
    "build_pubmed_query",
    "build_index_query",
    "parse_mcp_articles",
    "pubmed_search",
    "LiteratureIndex",
    "extractive_note",
    "to_literature_refs",
    "retrieve_literature",
]
