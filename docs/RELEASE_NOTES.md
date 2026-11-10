# Project changelog

## v0.5 — Research integrity and HPC experiment readiness

- New read-only source-quality audit and actual workload sizing panel; explicit synthetic demo import.
- Memory-safe cluster-label equivalence, shared correctness validation and fail-closed MPI job verification against independent sequential reference.
- Real MPI performance fields: throughput, per-rank CPU time, RSS high-water mark where supported.
- Deterministic, bounded scaling-study script; no MPI simulation or artificial benchmark results.
- Review-only Slurm job scripts; no database or AWS resource provisioning.
- Improved extractive (non-LLM) evidence answers with source snippets.
- New final-year blueprint, dataset case-study guide and experiment protocol.

## v0.3.1 — Windows import hotfix
- Added explicit, Python 3.11-compatible NumPy and pandas pins, including pandas as a direct dependency.
- Added Windows pandas repair and fresh-venv instructions.
- No changes to MPI, analytics, RAG or dashboard logic.
- Linux source regression tests pass, but Windows-specific import must be verified on Windows.

# Local release changes — v0.3

**AI-Powered Market Intelligence & Document Analysis using RAG + HPC on AWS**

Based on the independently cleaned local baseline, rather than overwriting the
original Antigravity ZIP.

## Kept

- PDF/DOCX/CSV ingestion with source locations.
- Sequential exact cosine similarity, K-Means and source-traceable market metrics.
- Four-tab Gradio dashboard and separate educational guide.
- Local evidence retrieval, optional semantic model and optional Ollama adapter.
- Real MPI source code and honest benchmark utilities.

## Improved

- MPI rank-0 input validation propagates errors through collective communication.
- Distributed K-Means now scatters disjoint document row blocks instead of
  loading the full input independently on every worker. Uses `Allreduce` for
  centroid sufficient statistics, and gathers actual assignments.
- Real MPI job results now validate the MPI rank-allocation evidence, not only
  the subprocess return code.
- Saved analytical results include deterministic dataset SHA-256 and exact
  source-document manifests, with one-click traceable research briefs.
- Computed results can be explained by an optional locally installed Ollama
  model with real uploaded-source excerpts. Default briefs remain deterministic.
- TF-IDF index uses corpus-only fitting and is cached for repeat questions.
- Scientific benchmark now stores comparable end-to-end output artifacts for
  sequential and MPI runs and reports median-based speedup.
- Added container packaging, environment doctor, temporary-data smoke test,
  broader integration tests, setup guide, architecture, algorithms, viva and
  final acceptance checklist.

## Removed from new package

- Two redundant local planning/viva notes superseded by the complete README,
  `DEMONSTRATION_AND_VIVA.md` and `SUBMISSION_CHECKLIST.md`.
- Build caches, local runtime data, project history and secrets are excluded
  from the distributed ZIP.

## Important verification boundary

The build environment had no MPI launcher/mpi4py, no local Ollama process and
no authorized AWS connection. Those *live execution stages* must still be
verified independently on the student's WSL2/AWS setup. **The package is a
working local prototype, not a deployed cloud application.**

## v0.4.0 — Editorial UI and structure refinement

- Rebuilt the UI presentation as **Market Atlas**, while preserving the exact academic title.
- Redesigned the four navigation sections with a restrained visual system, real overview KPI counters, clear walkthrough, collapsible advanced settings and source-led results explorer.
- Separated layout (`ui.py`), event/data actions (`ui_actions.py`) and style (`ui_theme.py`) so future Antigravity edits can remain isolated.
- Switched PDF parsing import from deprecated `fitz` alias to `pymupdf`.
- Removed Gradio 6 deprecated constructor parameters; CSS and theme are provided at launch.
- Added `docs/UI_AND_CODE_STRUCTURE.md` and `docs/ANTIGRAVITY_HANDOFF.md`.
- Kept every AWS integration uninvoked and every real MPI result dependent on actual MPI execution.
- Added UI contract and overview-metric tests. Browser automation is restricted in the build environment; a manual Windows visual test is still required.
