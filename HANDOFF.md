# Project Handoff: DermAssist — Agentic Skin-Lesion Analysis Pipeline

> **Status:** Greenfield. Nothing built yet. This document is the spec for an autonomous coding agent to bootstrap the project.
>
> **⚠️ Compliance framing (non-negotiable, applies everywhere):** This is a **research/educational tool, NOT a medical device**. Every UI surface, report, and README must carry: *"For research and educational use only. Not for clinical diagnosis."* The human-review gate is a **hard stop**, never a rubber stamp.

---

## 1. What we're building

A LangGraph-orchestrated pipeline that ingests a dermatoscopic skin-lesion image and produces a structured, literature-grounded report that a human reviews before finalization.

**Flow:**
```
Ingest image → Preprocess → Vision classifier → Claude interprets (ABCDE features)
→ Cross-reference PubMed → Generate structured JSON report → Human review gate (pause/resume)
→ Finalize
```

The genuinely novel part is the **orchestration + resumable human gate**, not the ML. Build that skeleton first with mocked nodes, then swap in real models.

---

## 2. Tech stack (locked)

| Layer | Choice |
|---|---|
| Language | Python 3.11+ |
| Orchestration | LangGraph |
| State persistence | SQLite checkpointer (dev) → Postgres (prod) |
| Dataset | HAM10000 / ISIC Archive |
| Preprocessing | OpenCV, PIL, torchvision; Shades-of-Gray color constancy; DullRazor-style hair removal |
| Vision (classification) | Fine-tuned EfficientNet or ViT on dermoscopy (HuggingFace pretrained to start) |
| Vision (interpretation) + reasoning | Claude via Anthropic API |
| Literature | PubMed via MCP + ChromaDB + biomedical embeddings (PubMedBERT/BioBERT or Voyage) |
| Structured output | Claude tool-use → Pydantic-validated JSON |
| Review UI | Streamlit (start here) |
| Package mgmt | uv (preferred) or poetry |

---

## 3. Build order (strict — do not jump ahead)

### Phase 0 — Scaffold
- Initialize repo with `uv`/`poetry`, `.env.example`, `.gitignore`, `pyproject.toml`.
- Set up config loading (pydantic-settings) for `ANTHROPIC_API_KEY`, DB URL, dataset path, MCP endpoint.
- Add the compliance disclaimer constant in one place, imported everywhere.

### Phase 1 — Graph skeleton with MOCKED nodes  ← **highest priority**
Get the full pipeline flowing end-to-end with fake data before touching ML:
- Define the `LesionState` TypedDict/Pydantic state (see §4).
- Build all nodes as stubs returning hardcoded plausible output.
- Wire the graph with the conditional edge for the review gate.
- Implement `interrupt()` at the review node + SQLite checkpointer so the graph pauses and resumes.
- **Acceptance:** run end-to-end, pause at review, resume from CLI, reach `finalize`. No real model touched.

### Phase 2 — Preprocessing (real)
- Image loader (PIL/OpenCV), resize, color constancy (Shades-of-Gray), optional hair removal.
- Unit tests on a handful of HAM10000 samples.

### Phase 3 — Vision classifier (real)
- Load a pretrained dermoscopy EfficientNet/ViT from HuggingFace; output calibrated class probabilities over the 7 HAM10000 classes.
- Keep the interface identical to the Phase-1 mock so it's a drop-in swap.

### Phase 4 — Claude interpretation + structured output
- Claude analyzes the image for ABCDE features (asymmetry, border, color, diameter/dimension, evolution).
- Claude reconciles its narrative with the classifier's probabilities.
- Force output through a Pydantic schema via tool-use; validate before passing on.

### Phase 5 — Literature retrieval
- PubMed MCP query keyed off the top differential.
- Embed + store abstracts in ChromaDB; retrieve top-k relevant for grounding.
- Claude cites retrieved literature in the final report.

### Phase 6 — Streamlit review UI
- Side-by-side: original image, preprocessed image, classifier probs, Claude's ABCDE interpretation, cited literature.
- Approve / Reject / Edit controls that resume the graph via the checkpointer.
- Disclaimer banner persistent on screen.

---

## 4. Core state schema (starting point)

```python
from typing import Optional, Literal
from pydantic import BaseModel, Field

class ABCDEFeatures(BaseModel):
    asymmetry: str
    border: str
    color: str
    diameter: str
    evolution: str

class ClassifierResult(BaseModel):
    label: str
    probabilities: dict[str, float]   # class -> prob over 7 HAM10000 classes
    top_confidence: float

class LiteratureRef(BaseModel):
    pmid: str
    title: str
    relevance_note: str

class LesionReport(BaseModel):
    abcde: ABCDEFeatures
    classifier: ClassifierResult
    differential: list[str]
    literature: list[LiteratureRef]
    overall_confidence: float = Field(ge=0, le=1)
    recommendation: str
    disclaimer: str = "For research and educational use only. Not for clinical diagnosis."

class LesionState(BaseModel):
    image_path: str
    preprocessed_path: Optional[str] = None
    classifier_result: Optional[ClassifierResult] = None
    interpretation: Optional[ABCDEFeatures] = None
    report: Optional[LesionReport] = None
    review_status: Literal["pending", "approved", "rejected", "edited"] = "pending"
    reviewer_notes: Optional[str] = None
```

---

## 5. Graph shape (LangGraph)

```
START → ingest → preprocess → classify → interpret → literature → build_report
      → human_review  ──(approved)──→ finalize → END
                       ──(rejected/edited)──→ interpret   # loop back
```

- Use a **checkpointer** (`SqliteSaver` dev) so state survives the pause.
- Use **`interrupt()`** at `human_review`; resume by invoking the graph with the same thread_id.
- ⚠️ **Verify the current LangGraph `interrupt` / checkpointer API before coding** — it has changed across recent versions. Don't trust any single remembered snapshot; check the live docs.

---

## 6. Key things the agent must NOT do
- Do **not** let Claude's vision alone produce the diagnostic probability — the dedicated classifier owns calibrated probs; Claude owns interpretation/narrative.
- Do **not** make the human gate auto-approve or skippable.
- Do **not** ship without the disclaimer on every surface.
- Do **not** commit dataset images or `.env` to git.
- Do **not** build the real vision model before the mocked graph runs end-to-end.

---

## 7. First commit checklist
- [ ] Repo scaffolded, deps installed, `.env.example` present
- [ ] `LesionState` + report schemas defined
- [ ] Graph with all-mock nodes runs START→END
- [ ] Review gate pauses via `interrupt()` and resumes via checkpointer
- [ ] README with setup steps + compliance disclaimer

---

## 8. Open decisions to surface to the human (don't guess)
- Postgres vs SQLite for the target deployment.
- Which specific pretrained dermoscopy checkpoint to use.
- Embedding model: biomedical (PubMedBERT) vs Voyage — affects literature relevance.
- ChromaDB local vs Vertex AI Vector Search (only if committing to GCP).

---

**Reminder for the agent:** Before writing code against LangGraph, the Anthropic structured-output API, or the PubMed MCP, confirm current versions and API signatures from live documentation. These libraries move fast.
