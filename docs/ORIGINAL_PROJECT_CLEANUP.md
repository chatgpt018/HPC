# Original Antigravity ZIP: cleanup decisions

The original uploaded ZIP is preserved separately. This is a **new clean baseline** and not an in-place modification of the original 572-entry archive.

## What was wrong in the old project (source inspection)

- The UI could call a sequential loop named simulated MPI while showing multi-rank/success claims.
- The original `mpi_clustering.py` delegated to sequential scikit-learn rather than performing MPI communication.
- Results UI contained fixed speedup and similarity values without current measurements.
- RAG answer generation returned a formatted placeholder; local search was word-overlap rather than real model generation.
- The main HPC dashboard consumed unrelated sample random vectors rather than embeddings derived from the just-uploaded documents.
- Job records were volatile and AWS job submission only wrote Slurm scripts.
- Legacy UI tab files coexisted with the four active views.
- Cache folders, historical runtime JSON, `.env`, and copied raw reports cluttered the archive.

## Why a separate clean baseline was chosen

It avoids deleting data and reduces the number of unverified integrations. This package rebuilds a minimal but coherent local workflow with explicit honest boundaries.

## What is reusable from the original

| Original part | Decision | Why |
|---|---|---|
| Parsing ideas and chunk metadata | Reimplemented compactly | Proven concept, less baggage |
| Sequential cosine / K-Means | Retained mathematical role | Necessary reference implementations |
| MPI similarity row partitioning | Retained principle | Valid MPI workload when launched properly |
| Four-tab Gradio UX | Retained navigation concept | Simple and enough for evaluation |
| Source datasets | **Not copied automatically** | Rights/provenance and user privacy need review |
| Existing experiment logs | **Not claimed as evidence here** | Provenance and execution mode unclear |
| Learning guide | Kept separately in original ZIP | Old material may describe features not yet live |
| AWS infrastructure templates | **Not included as active infrastructure** | User requested AWS later and placeholder subnet/IAM risk |
| Mock MPI and hardcoded results | Removed | Cannot be submitted as real HPC evidence |

## Old code cleanup, only if the student later chooses in-place cleanup

**Review to archive:** unused `app/ui/*_tab.py` implementations, custom test compatibility shims, duplicated dashboard logic. Confirm references first.

**Remove from distribution, not the original working directory:** `__pycache__`, `.pytest_cache`, `.env`, previously generated unverified results, private uploaded reports, virtual environments.

**Do not delete:** source originals, documents with verified provenance, research results, working unit tests, source-level academic records, or learning notes without consent.

## Future porting checklist

1. Compare original and clean algorithms on the same small matrix.
2. Preserve any original tests that catch real edge cases missing in the clean suite.
3. Preserve meaningful report metadata and market metric extraction if its provenance is reliable.
4. Do not add AWS integration until local execution is proven.
5. Regenerate architecture diagrams from actual code rather than copying diagrams that call planned services implemented.

## Current baseline review — 2026-11-10

| Classification | Current files | Evidence and decision |
|---|---|---|
| KEEP | `marketintel/analysis.py`, `mpi_tasks.py`, `documents.py`, `vectors.py`, `tests/`, `examples/synthetic_reports/` | Integrated sequential references, real MPI algorithms, ingestion, retrieval and labelled practice data; local tests and WSL2 execution pass. |
| FIX | `scripts/doctor.py`, `marketintel/correctness.py`, `research_quality.py`, `ui.py`, `ui_theme.py`, `ui_actions.py`, `scripts/benchmark.py` | Corrected broken-import detection, duplicate-row validation, synthetic labels, dark-mode contrast, result refresh and benchmark provenance. |
| KEEP | `pyproject.toml`, `requirements.txt`, `requirements-hpc.txt` | Local and optional MPI dependencies remain separate; keep both installation routes documented and consistent. |
| KEEP | `Dockerfile`, `infrastructure/`, `learning-guide/` | Useful handoff/teaching material; Docker image and AWS deployment are not verified by source presence. |
| ARCHIVE LATER | Historical release notes and old audit claims | Preserve chronology, but use `VERIFICATION.md` for the current machine's results. No file was archived or deleted. |
| RECOMMEND FOR DELETION | None | No current project file was proven unused enough to justify removal. |
