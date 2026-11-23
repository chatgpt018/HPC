# Local verification — academic engineering release v0.5

**Official project title:** AI-Powered Market Intelligence & Document Analysis using RAG + HPC on AWS

## Current evidence

The exact commands, package versions, measured MPI results and remaining limits
are recorded in [`../VERIFICATION.md`](../VERIFICATION.md). On this Windows
machine, the Python 3.11 suite passed **68 tests** on 2026-11-22, `pip check` passed, the
offline PDF/DOCX/CSV smoke test passed, and the four-page Gradio workflow was
inspected in a browser, including two synthetic PDF uploads, cited retrieval,
sequential analysis, results and downloads on an earlier UI revision. On the
current Windows host, four actual Microsoft MPI ranks launched and similarity
and distributed K-Means matched Sequential. The current dashboard served HTTP
200, but a fresh browser visual inspection stopped before page navigation.

The previous ZIP passed a fresh Windows Python 3.11 setup, 60 tests, offline
smoke test and localhost launch. The current source passed 68 tests; the new
ZIP's bundled benchmark discovery was checked from a fresh extraction with
the existing Python 3.11 environment; it was not a second dependency install.
Earlier revisions passed Windows wheelhouse
and Ubuntu WSL2 setup checks; the current revision was not run on Linux. All
checks used one physical laptop; a second laptop was unavailable.

## How correctness is tested

- Unit tests cover ingestion, source metadata, TF-IDF retrieval, numerical
  edge cases, market metrics, job validation, UI actions and report contracts.
- The smoke test creates temporary synthetic documents and exercises the local
  document-to-evidence and document-to-analysis paths.
- The current Windows benchmarks launched actual MPI processes in 20 scaling
  configurations across two studies with three repetitions each; all outputs
  passed Sequential comparison. The larger study reached 5,000 synthetic
  documents. See `../output/benchmarks/`.

These tests do not establish a live LLM, a real public-report case study,
multi-node speedup, AWS deployment or cloud cost. The synthetic scaling study
is still limited to one host and at most 5,000 synthetic documents; its
measured speedups were below 1.
