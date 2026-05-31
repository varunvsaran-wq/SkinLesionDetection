"""Application configuration loaded from environment / .env via pydantic-settings.

All fields are optional for Phase 0/1 so the mocked graph runs with no secrets.
Real credentials (ANTHROPIC_API_KEY, MCP endpoint) become required as later
phases are switched on.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Claude / Anthropic (Phase 4+)
    anthropic_api_key: str | None = Field(default=None)
    # Claude model for ABCDE interpretation + structured output.
    claude_model: str = Field(default="claude-opus-4-8")

    # State persistence. SQLite for dev; a Postgres URL for prod (open decision).
    database_url: str = Field(default="sqlite:///checkpoints/dermassist.sqlite")

    # Dataset (Phase 2+)
    dataset_path: Path = Field(default=Path("data/ham10000"))

    # Vision classifier (Phase 3). A HuggingFace transformers image-classification
    # checkpoint trained on the 7 HAM10000 classes. Open decision (HANDOFF.md §8) —
    # this is a sensible default ViT; override CLASSIFIER_MODEL to swap it.
    classifier_model: str = Field(default="Anwarkh1/Skin_Cancer-Image_Classification")
    # Temperature for softmax. 1.0 = raw model output; fit >1 on a validation set
    # to calibrate over-confident probabilities (temperature scaling).
    classifier_temperature: float = Field(default=1.0, gt=0)

    # Literature retrieval (Phase 5+)
    # PubMed MCP server endpoint (Streamable HTTP). Unset → node uses mock abstracts.
    mcp_endpoint: str | None = Field(default=None)
    # Name of the search tool exposed by your PubMed MCP server.
    mcp_pubmed_tool: str = Field(default="search_pubmed")
    # Local biomedical embedding model (sentence-transformers). Downloads on first use.
    embedding_model: str = Field(default="NeuML/pubmedbert-base-embeddings")
    # Local ChromaDB persistence directory.
    chroma_path: Path = Field(default=Path("artifacts/chroma"))
    # How many PubMed candidates to fetch, and how many to keep after re-ranking.
    pubmed_retmax: int = Field(default=20, gt=0)
    literature_top_k: int = Field(default=3, gt=0)
    # Opt-in: have Claude author the per-reference relevance notes (one extra API
    # call). Off by default — the extractive notes are free and deterministic.
    literature_claude_citations: bool = Field(default=False)

    # Environment label
    app_env: str = Field(default="dev")

    @property
    def checkpoint_db_path(self) -> Path:
        """Filesystem path for the SQLite checkpointer.

        Parses the ``sqlite:///`` form of ``database_url``. For non-sqlite URLs
        (e.g. Postgres in prod) the checkpointer wiring in ``graph.py`` should be
        swapped accordingly; this helper only handles the dev SQLite case.
        """
        url = self.database_url
        prefix = "sqlite:///"
        if url.startswith(prefix):
            return Path(url[len(prefix):])
        # Fallback: treat the whole value as a path.
        return Path(url)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()


__all__ = ["Settings", "get_settings"]
