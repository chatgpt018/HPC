# Local verification — academic engineering release v0.5

**Official project title:** AI-Powered Market Intelligence & Document Analysis using RAG + HPC on AWS

## Current evidence

The exact commands, package versions, measured MPI results and remaining limits
are recorded in [`../VERIFICATION.md`](../VERIFICATION.md). On this Windows
machine, the Python 3.11 suite passed **47 tests**, `pip check` passed, the
offline PDF/DOCX/CSV smoke test passed, and the four-page Gradio workflow was
inspected in a browser. In existing Ubuntu WSL2, real four-rank similarity and
distributed K-Means matched sequential references on synthetic numerical and
document-derived inputs.

The portability check in `../VERIFICATION.md` additionally covers fresh ZIP
installs through the Windows and Linux setup scripts, both launch scripts,
and a Windows `--no-index` wheelhouse install. These environments were on one
physical laptop; a second laptop was not available for testing.

## How correctness is tested

- Unit tests cover ingestion, source metadata, TF-IDF retrieval, numerical
  edge cases, market metrics, job validation, UI actions and report contracts.
- The smoke test creates temporary synthetic documents and exercises the local
  document-to-evidence and document-to-analysis paths.
- The WSL2 benchmark launches actual MPI processes and compares each complete
  output with the sequential reference before recording a valid run.

These tests do not establish a live LLM, a real public-report case study,
multi-node speedup, AWS deployment or cloud cost. The synthetic scaling study
is preliminary because it used small inputs and two repetitions.
