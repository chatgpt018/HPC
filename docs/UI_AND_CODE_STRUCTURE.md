# UI design, implementation boundaries, and usability

**Submitted title:** AI-Powered Market Intelligence & Document Analysis using RAG + HPC on AWS

## Design objective

The Gradio application uses a distinctive **Market Atlas / Research System** identity with dark editorial header, understated teal and copper signals, off-white paper-like sections, and a single consistent component hierarchy. It avoids typical neon gradient chatbot styling and avoids overwhelming the user with too many independent pages.

The full submitted title remains unchanged and visible in the header. *Market Atlas* is only a visual identity, not a rename of the academic project.

## Four destinations

1. **Overview** — actual document, excerpt, saved-run and completed-local-MPI counts; process walkthrough; clear runtime limitations.
2. **Documents & AI** — upload/validate, indexed-source table, source-aware questions and optional explicitly enabled local Ollama generation.
3. **HPC Analytics** — algorithm, execution method, vector representation, worker count, actual results and an expandable run receipt.
4. **Results & Reports** — saved output browser, charts, source-provenance metadata, downloadable computations, a research brief, source-compatible CSV metrics, and *genuine-only* benchmark views.

## Source code responsibilities

| File | Job |
|---|---|
| `marketintel/ui.py` | Gradio layout and event wiring; no numerical algorithm implementations |
| `marketintel/ui_theme.py` | Single source of truth for the visual system, masthead, flow graphics and CSS |
| `marketintel/ui_actions.py` | UI-facing actions, input handling, result views and charts |
| `marketintel/documents.py` | Local PDF/DOCX/CSV source extraction and chunking |
| `marketintel/vectors.py` | Document-derived numerical features and optional local semantic embeddings |
| `marketintel/rag.py` | Local evidence retrieval and optional model answer assembly |
| `marketintel/analysis.py` | Sequential similarity and K-Means reference |
| `marketintel/mpi_tasks.py` | MPI similarity and distributed K-Means implementation |
| `marketintel/jobs.py` | Real local MPI subprocess launcher and job records; fail-closed |
| `marketintel/provenance.py` | Dataset fingerprints, source manifests, computation explanation |
| `marketintel/metrics.py` | Comparable source-reported market time series |

## Design review checklist

- [x] Full official title preserved.
- [x] Exactly four tabs, with coherent labels and visually distinct roles.
- [x] Source data and computed results drive numbers, not fake analytics.
- [x] Clear empty states and source-first explanation.
- [x] Explicit distinction between sequential, real MPI, TF-IDF, neural retrieval, local LLM, and planned AWS.
- [x] Optional advanced controls and technical metadata are collapsed by default.
- [x] Basic responsive CSS for laptop and smaller screens.
- [ ] **Manual visual review on the student's Windows browser** (required: theme appearance, display scaling, file upload, all tabs, mobile width).
- [ ] **Live multi-rank MPI demonstration** after WSL2 MPI setup.
- [ ] **Real local model invocation** only if an Ollama model is installed.
- [ ] **AWS deployment** (deliberately not completed in the local release).

## Presentation rules

The reviewer should be able to demonstrate: **upload a real report → find its supporting passage → run actual numerical similarity → view the saved output → open the provenance receipt → explain the need for MPI and benchmark results**.

Do not call TF-IDF retrieval a fully configured Bedrock RAG system; do not call a sequential run HPC; do not use synthetic examples as real market findings.
