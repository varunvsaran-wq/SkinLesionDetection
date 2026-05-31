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

The remaining nodes (interpret, literature) are still mocked. See
[HANDOFF.md](HANDOFF.md) for the build order (Phases 4–6 swap real models into the
same interfaces).

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
  compliance.py   # single source of the disclaimer (imported everywhere)
  config.py       # pydantic-settings (.env) — API keys, DB URL, dataset path
  schemas.py      # LesionState + LesionReport (+ 7 HAM10000 classes)
  nodes.py        # pipeline nodes (preprocess real; classify/interpret/literature mocked)
  preprocessing.py# Phase 2: load, resize, Shades-of-Gray, DullRazor hair removal
  graph.py        # LangGraph wiring + SQLite checkpointer
  cli.py          # run / resume / status
tests/
  test_graph.py        # Phase 1 acceptance tests
  test_preprocessing.py# Phase 2 preprocessing tests
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
