# DermAssist — Agentic Skin-Lesion Analysis Pipeline

> ⚠️ **For research and educational use only. Not for clinical diagnosis.**
> DermAssist is a research/educational tool, **NOT a medical device**. Its output is
> not a diagnosis and must never drive clinical decisions. Every analysis requires
> review by a qualified human — the human-review gate is a **hard stop**, never a
> rubber stamp.

A [LangGraph](https://langchain-ai.github.io/langgraph/)-orchestrated pipeline that
ingests a dermatoscopic skin-lesion image and produces a structured,
literature-grounded report that a human reviews before finalization.

```
START → ingest → preprocess → classify → interpret → literature → build_report
      → human_review ──(approved)──→ finalize → END
                      ──(rejected/edited)──→ interpret   # loop back
```

The novel part is the **orchestration + resumable human-review gate**, not the ML.

## Status

- **Phase 1 complete** — the full graph runs end-to-end, pausing at the
  human-review gate via LangGraph `interrupt()` and resuming from a SQLite
  checkpointer.
- **Phase 2 complete** — real image preprocessing: PIL/OpenCV loader, resize,
  Shades-of-Gray color constancy, and DullRazor-style hair removal
  ([preprocessing.py](src/dermassist/preprocessing.py)). The `preprocess` node
  runs it on a real image and falls back to a mock path when none is present.
- **Phase 3 complete** — real vision classifier: a pretrained HuggingFace ViT
  (`Anwarkh1/Skin_Cancer-Image_Classification` by default, configurable via
  `CLASSIFIER_MODEL`) producing probabilities over the 7 HAM10000 classes with
  robust label normalization and optional temperature scaling
  ([classifier.py](src/dermassist/classifier.py)). The `classify` node uses it on
  the preprocessed image and falls back to a mock when `torch` isn't installed.
  Enable with `uv sync --extra vision` (pulls in torch; first run downloads the
  checkpoint). The classifier — never Claude — owns the diagnostic probabilities.
- **Phase 4 complete** — Claude ABCDE interpretation + structured output: Claude
  (`claude-opus-4-8` by default) reads the preprocessed image, assesses the ABCDE
  features, and reconciles its narrative with the classifier's probabilities,
  forced through a Pydantic-validated schema via tool-use with prompt caching
  ([interpretation.py](src/dermassist/interpretation.py)). Per §6, Claude is
  instructed to never emit its own probabilities — the classifier owns those. The
  `interpret` node uses it and falls back to a mock when `anthropic` isn't
  installed or no `ANTHROPIC_API_KEY` is set. Enable with `uv sync --extra
  reasoning` and set `ANTHROPIC_API_KEY` in `.env`.

- **Phase 5 complete** — literature retrieval: PubMed is queried via a configurable
  **MCP server** (`MCP_ENDPOINT` + `MCP_PUBMED_TOOL`) keyed off the top differential;
  abstracts are embedded with **PubMedBERT** (local sentence-transformers) and
  stored in a **local ChromaDB** collection, then the top-k most relevant are
  retrieved for grounding ([literature.py](src/dermassist/literature.py)).
  Relevance notes are extractive by default; an opt-in Claude-authored mode is
  gated behind `LITERATURE_CLAUDE_CITATIONS`. The `literature` node falls back to
  mock references when `MCP_ENDPOINT` is unset/unreachable or deps are missing.
  A ready-to-run PubMed MCP server is included
  ([tools/pubmed_mcp_server.py](tools/pubmed_mcp_server.py), see below).

- **Phase 6 complete** — Streamlit human-review UI ([ui.py](src/dermassist/ui.py)):
  upload an image and run the pipeline, then review the report side-by-side
  (original + preprocessed image, classifier probability bar chart, Claude's ABCDE
  interpretation, cited literature) and **Approve / Reject / Edit** to resume the
  graph through the SQLite checkpointer. Reject/Edit send reviewer notes back to
  Claude and loop; the compliance disclaimer is pinned on every screen. Launch with
  `uv sync --extra ui` then `uv run streamlit run src/dermassist/ui.py`.

All six phases are implemented. See [HANDOFF.md](HANDOFF.md) for the original spec.

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev                    # core + dev (pytest) deps
uv sync --extra preprocess --extra dev # + Phase 2 preprocessing (numpy/pillow/opencv)
cp .env.example .env                   # optional now; fill in for later phases
```

The pipeline needs no API keys; preprocessing runs on any local image. Optional
dependency groups: `preprocess` (Phase 2), `vision` (Phase 3, pulls in torch),
`reasoning` (Phase 4), `literature` (Phase 5), `ui` (Phase 6).

## Run the pipeline (CLI)

```bash
# 1. Run until the human-review gate. Prints the report + a thread_id, then pauses.
uv run dermassist run --image data/example_lesion.jpg

# 2. Resume that thread with a reviewer decision.
uv run dermassist resume --thread-id <THREAD_ID> --decision approved
#   --decision rejected|edited loops back to re-interpret, then pauses again.
#   --notes "..." attaches reviewer feedback that the re-interpretation responds to.

# Inspect a thread at any time:
uv run dermassist status --thread-id <THREAD_ID>
```

State persists on disk (SQLite), so `run` and `resume` can be separate process
invocations — the graph genuinely pauses and resumes.

## Literature retrieval (PubMed MCP)

A matching PubMed MCP server is bundled. In one terminal:

```bash
uv sync --extra literature
uv run python tools/pubmed_mcp_server.py        # serves http://127.0.0.1:8000/mcp
```

Then add to `.env`:

```bash
MCP_ENDPOINT=http://127.0.0.1:8000/mcp
MCP_PUBMED_TOOL=search_pubmed
# NCBI_API_KEY=...   # optional; raises NCBI rate limit 3 -> 10 req/s
```

Now `dermassist run` (or the UI) queries real PubMed, embeds abstracts with
PubMedBERT, and retrieves the top-k for the report. Without the server set, the
`literature` node uses mock references so the pipeline still runs.

## Review UI (Streamlit)

```bash
uv sync --extra ui
uv run streamlit run src/dermassist/ui.py
```

Upload a dermatoscopic image, run the pipeline, review the report side-by-side, and
Approve / Reject / Edit — the same graph and checkpointer as the CLI, driven from a
browser. The disclaimer banner is pinned on screen.

## Test

```bash
uv run pytest
```

The acceptance tests cover: pausing at the review gate, resuming to `finalize`,
reject → loop-back → approve, and resuming across a fresh graph instance (on-disk
persistence).

## Project layout

```
src/dermassist/
  compliance.py    # single source of the disclaimer (imported everywhere)
  config.py        # pydantic-settings (.env) — keys, DB URL, dataset/model config
  schemas.py       # LesionState + LesionReport (+ 7 HAM10000 classes)
  nodes.py         # pipeline nodes (real impls + mock fallbacks)
  preprocessing.py # Phase 2: load, resize, Shades-of-Gray, DullRazor hair removal
  classifier.py    # Phase 3: HF ViT, label normalization, softmax → 7-class probs
  interpretation.py# Phase 4: Claude vision + forced tool-use → Pydantic ABCDE
  literature.py    # Phase 5: PubMed-via-MCP + PubMedBERT + ChromaDB retrieval
  ui.py            # Phase 6: Streamlit review UI
  graph.py         # LangGraph wiring + SQLite checkpointer
  cli.py           # run / resume / status
tools/
  pubmed_mcp_server.py # bundled PubMed MCP server (NCBI E-utilities, Streamable HTTP)
tests/
  test_graph.py         # Phase 1 graph acceptance tests
  test_preprocessing.py # Phase 2
  test_classifier.py    # Phase 3
  test_interpretation.py# Phase 4
  test_literature.py    # Phase 5
  test_ui.py            # Phase 6
```

## Tech stack

Python 3.11+ · LangGraph · SQLite checkpointer (→ Postgres for prod) · Pydantic /
pydantic-settings. Later phases add: OpenCV/PIL/torchvision preprocessing, a
pretrained dermoscopy EfficientNet/ViT classifier, Claude (Anthropic API) for
ABCDE interpretation + structured output, PubMed-via-MCP + ChromaDB literature
retrieval, and a Streamlit review UI. These live in optional dependency groups
(`vision`, `reasoning`, `literature`, `ui`) and are not installed for Phase 1.

## Open decisions (see HANDOFF.md §8)

Postgres vs SQLite for prod · which pretrained dermoscopy checkpoint · biomedical
(PubMedBERT) vs Voyage embeddings · ChromaDB local vs managed vector search.

---

*For research and educational use only. Not for clinical diagnosis.*
