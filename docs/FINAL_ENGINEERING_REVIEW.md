# Final engineering review and professor checklist

**Submitted title: AI-Powered Market Intelligence & Document Analysis using RAG + HPC on AWS**  
**Review date:** 22 November 2026  
**Scope:** Local Windows Python 3.11 and one-host Microsoft MPI. No AWS or database was accessed.

## 1. What the project actually does

The four-page Gradio application indexes PDF, DOCX and CSV text with source
locations, retrieves evidence for a question, converts documents into numeric
features, runs exact similarity or Lloyd K-Means, and saves traceable results.
Sequential computation is the baseline. Genuine MPI distributes similarity
query rows or K-Means vector partitions when a working runtime is installed.
An optional local sentence-transformer and Ollama model are separate from the
default offline path. AWS remains a planned deployment stage.

The tested local workflow is: document → extraction → located chunks → TF-IDF
retrieval → document features → Sequential or real MPI analysis → independent
correctness check → saved result and deterministic research brief. A cited
passage is evidence of what a document says, not proof that its market claim is
true. Similarity and clusters are relationships between selected text features,
not forecasts.

## 2. Technical audit and change log

| Finding | Action and evidence |
|---|---|
| Core environment was healthy, but SciPy and threadpoolctl were only indirect dependencies. | Declared both in `pyproject.toml`; doctor now imports threadpoolctl. Doctor and `pip check` pass. |
| The test report said 48 tests and the dashboard said live MPI was unverified. | Updated the dated test report and the UI wording. Current full suite: 68 passed. The latest Windows run launched ranks 0–3. |
| Real-source attribution was not recorded by normal imports. | Added a validated manifest import, full content hash, explicit real/synthetic classification, publisher/date/scope/usage metadata, and provenance propagation. No real reports were supplied. |
| Optional semantic mode encoded every indexed passage for every question. | Added an incremental, versioned, pickle-free local NPZ index keyed by chunk IDs and text hashes. Offline model-contract tests verify reuse; no neural model was run here. |
| An optional local model could return unknown source numbers. | Numbered citations are checked against retrieved passages for question answers and report explanations. This does not verify factual entailment. |
| Exact top-k sorted every full similarity row. | Replaced full sorting with an exact score cutoff and deterministic sorting of candidates/ties. Regression tests compare with full-sort output. |
| Cluster validation checked labels and inertia, but not centroids or iterations. | Added label-invariant centroid and iteration checks. A live four-rank run passed. |
| A failed MPI candidate could be saved where completed Results are read. | MPI now writes to a job candidate folder and publishes only after rank and Sequential checks pass. A failure contract test confirms no completed result appears. |
| Windows peak memory was reported as zero when `resource` was unavailable. | Added Windows process peak RSS measurement with the system API; measured rank memory is in the new study. |
| Content validation accepted a malformed CSV row or a false DOCX extension. | Added signature/structure checks, source-unit/text limits, and path-free errors. Invalid files leave the index unchanged. |
| Benchmark plots and a current experimental record were missing from the shareable source. | Preserved the original 12 raw records and a second eight-configuration larger study, each with summary CSV/JSON and seven offline charts under `output/benchmarks/`. |
| Bundled measurements were invisible in the dashboard after extraction. | Results & Reports now discovers historical bundled studies and new local runs, validates saved JSON/CSV, and plots actual runtime, speedup, efficiency and throughput. |
| Near-tied float32 cosine scores ranked one pair differently across row partitions at 2,000 documents. | Score multiplication now uses float64; the saved failed partial run was discarded and a partition regression test passes. The rerun passed every MPI correctness check. |
| Windows setup assumed the `py` launcher was registered. | Setup now also detects a Python 3.11 executable and supports `-PythonExecutable`; a fresh extracted-package test is recorded in `VERIFICATION.md`. |

No archived MPI evidence, source data, or working tests were deleted. The
current Git commit before this review was retained as the local checkpoint.

### File disposition

| Classification | Files or group | Decision |
|---|---|---|
| KEEP | `README.md`, `QUICK_START.md`, setup/run scripts, tests, architecture and viva guides | They serve distinct installation, implementation, and presentation needs. |
| IMPROVE | `documents.py`, `vectors.py`, `rag.py`, `analysis.py`, `correctness.py`, `jobs.py`, doctor, benchmarks | Focused correctness, source integrity, and performance changes made. |
| REFACTOR | Peak RSS helpers in MPI and benchmark code | Merged into one cross-platform helper in `performance.py`. |
| ARCHIVE | Older extracted release folder and dated WSL2 measurements | Historical evidence; preserve separately and label by date/environment. |
| RECOMMEND DELETION | None in the current source checkout | No safe deletion was needed for this review. Virtual environments, caches, private data, and temporary files are excluded from the release. |

## 3. Verification record

| Check executed on this Windows host | Result |
|---|---|
| `.venv311/Scripts/python.exe scripts/doctor.py` | Core packages importable, Python 3.11.9 supported; `mpi4py.MPI` importable; no Ollama or neural model. |
| `.venv311/Scripts/python.exe -m pytest -q -p no:cacheprovider` | 68 passed. |
| `.venv311/Scripts/python.exe scripts/verify_local.py` | PASS: temporary synthetic PDF/DOCX/CSV, retrieval, sequential analytics, source-aware brief. |
| `python -m compileall -q marketintel scripts tests` and `python -m pip check` | Passed; no broken requirements. |
| `mpiexec -n 4 .venv311/Scripts/python.exe scripts/mpi_check.py` | Four genuine ranks, all on one Windows host. |
| Four-rank 3 × 16 similarity and K-Means | Both passed independent Sequential checks; partition sizes 1, 1, 1, 0. |
| Scaling study, 100/500/1000 documents × 384 features × 2/4 ranks × two algorithms × three repetitions | 12/12 configurations correct; raw records and charts preserved. |
| Larger scaling study, 2000/5000 documents × 384 features × 2/4 ranks × two algorithms × three repetitions | 8/8 configurations correct; raw records and charts preserved. |
| Gradio `run_windows.ps1` on isolated port 7865 | Local HTTP 200. A fresh visual browser walkthrough did not complete because Computer Use stopped before page navigation. |
| Previous ZIP extraction and `setup_windows.ps1 -PythonExecutable <Python 3.11 path>` | All core imports healthy; extracted 60 tests, offline smoke and `pip check` passed; extracted Gradio returned HTTP 200 on port 7866. Current source passed 68 tests. |
| Current source ZIP extracted into a new folder | Existing Python 3.11 dependencies imported the extracted package; both bundled historical studies were discovered and the larger study displayed verified scaling measurements. This was a discovery check, not a new dependency installation. |

These checks ran on one physical laptop. An older source revision was tested
on Ubuntu WSL2, but the current Windows session has no installed WSL distro.
No second physical laptop, native Ubuntu host, multi-node cluster, live Ollama,
neural embedding model, Docker build, AWS deployment, or database was tested.

## 4. HPC result and interpretation

The input is labelled **SYNTHETIC NUMERICAL BENCHMARK INPUT**. The primary
metric is end-to-end elapsed time including MPI launch, data load, and result
writing. Three repetitions per configuration are summarised by medians.
`speedup = Sequential time / MPI time`; efficiency divides speedup by ranks.
Similarity work units are directed non-self comparisons; K-Means units are
document × cluster × actual iteration distance evaluations.

All 20 measured speedups across both studies were below 1. The earlier study
ranged **0.0043×–0.0436×**; the larger study ranged **0.039×–0.350×**. Median
peak resident memory of the largest MPI rank ranged **142,868–228,916 KiB**.
The study records per-rank CPU time, hostname, row allocations, input
fingerprint, and a timing section that includes MPI collectives. Its outside
section includes process startup, loading, and output; it is **not** an isolated
communication measure. CPU utilisation was not measured. The tested workloads
were too small for their compute work to offset launch and communication costs.
This supports correctness and a measured scaling discussion, not a positive
speedup or multi-node claim.

Open the larger study's [offline interactive charts](../output/benchmarks/native_windows_20261122_large/scaling_charts.html),
[summary CSV](../output/benchmarks/native_windows_20261122_large/scaling_study.csv),
or [summary JSON](../output/benchmarks/native_windows_20261122_large/scaling_study.json).
Each study's `raw/` folder contains benchmark records with repetitions and rank
logs. Re-run with `scripts/scaling_study.py`; use `scripts/plot_scaling.py` to
regenerate charts. A measured weak-scaling experiment and actual AWS
multi-node experiment remain necessary for broader HPC scaling claims.

## 5. RAG status and evaluation

The default retrieval method is local TF-IDF, followed by literal source
excerpts. This mode ran in the offline smoke test. The optional neural mode
uses a locally installed sentence-transformer; its index and incremental
update contract passed tests with a deterministic fixture model. No actual
neural model or Ollama service was available, so **generative RAG is implemented
but not live-verified**. There are no invented generated answers or scores.

An explicitly enabled Ollama answer receives retrieved text as untrusted data,
is instructed to reject insufficient evidence, and must cite only numbered
retrieved sources. The returned appendix maps numbers to filenames, locations,
and chunk IDs. Numeric citation validation does not prove claim support,
publisher credibility, or correct historical-versus-forecast interpretation.
The optional research explanation follows the same citation rule.

`scripts/evaluate_retrieval.py` computes Hit@k, Recall@k, and no-evidence
accuracy from manually labelled indexed locations. The bundled template is
empty because no licensed real source collection was provided. Answer
completeness, unsupported-claim rate, and real model output must be reviewed
with actual documents and a running local model before they can be reported.

## 6. UI, portability, and security

The four pages are Overview, Documents & AI, HPC Analytics, and Results &
Reports. The benchmark view now selects bundled historical studies and new
local records without user file paths. The current server responded with HTTP
200 and automated UI action tests passed. Earlier revisions were browser
checked, but automatic approval review rejected activating the personal Edge
window because its active tab was not confirmed to be the local app. No new
screenshots are claimed. The user should manually check contrast, narrow
viewport, keyboard focus, and the complete upload-to-report journey before
submission.

Offline use requires Python and the installed Python packages; no AWS
credential, database, or paid API is required. The source ZIP contains no
virtual environment, indexed personal reports, or model weights. A completely
disconnected target laptop needs a separately prepared wheelhouse for the
same OS, architecture, and Python version. Windows native MPI and Ubuntu/WSL2
MPI require their own working runtimes and are optional for Sequential mode.

Uploaded text is untrusted. Files are limited to 25 MB; supported formats,
binary signatures, CSV shape, and extracted text size are checked. Real-source
metadata needs explicit attribution. The local JSON store is intended for
one user: it is not a concurrent multi-user database and has no authentication.
Do not expose the app publicly without a secure hosting design.

## 7. Professor's eleven requirements

| # | Requirement | Current status | Implemented files and execution evidence | Missing work / student's action |
|---:|---|---|---|---|
| 1 | Real-world problem | PARTIALLY IMPLEMENTED | PDF/DOCX/CSV ingestion, source manifest, citations, and metrics in `documents.py`, `cli.py`, `metrics.py`; smoke test passes. | Supply permitted real reports and a manually checked market case study. |
| 2 | Computationally intensive task | VERIFIED | `analysis.py` and `mpi_tasks.py` compute exact all-pairs top-k and Lloyd K-Means; 100–5000 document runs completed. | Repeat on the intended research hardware. |
| 3 | HPC justification | VERIFIED | Quadratic similarity complexity, partitioning, and overhead documented in `docs/ALGORITHMS_EXPLAINED.md`; negative speedup is reported honestly. | Show crossover or explain why it is absent with larger data. |
| 4 | Sequential baseline | VERIFIED | Sequential similarity and K-Means in `analysis.py`; smoke and benchmark runs. | None for local baseline. |
| 5 | Parallel implementation | VERIFIED | Actual 2/4-rank Microsoft MPI similarity and distributed K-Means in `mpi_tasks.py`; rank logs preserved. | Multi-node execution not yet established. |
| 6 | Correctness validation | VERIFIED | `correctness.py` checks pairs/scores and label-invariant clusters, centroids, inertia, iterations; 20/20 study configurations passed. | Repeat validation on cloud outputs. |
| 7 | Performance benchmarking | VERIFIED | `scripts/benchmark.py`, `scripts/scaling_study.py`, raw records, three repetitions/configuration. | Repeat on larger real workloads and dedicated compute nodes. |
| 8 | Scalability analysis | PARTIALLY IMPLEMENTED | Input-size and 2/4-rank single-host scaling views from 100 to 5,000 documents; offline charts. | Run weak scaling, more ranks, and a multi-node study. |
| 9 | Speedup, efficiency, throughput, resource use | PARTIALLY IMPLEMENTED | Saved wall times, speedup, efficiency, throughput, rank CPU seconds and peak RSS. | CPU utilisation and isolated communication profiling remain unmeasured. |
| 10 | Actual AWS HPC deployment | BLOCKED BY EXTERNAL REQUIREMENTS | Slurm/Docker handoff templates exist under `infrastructure/`; no AWS call was made. | Student provisions secured AWS resources and captures real job/node IDs. |
| 11 | Cloud performance and cost | BLOCKED BY EXTERNAL REQUIREMENTS | `infrastructure/AWS_HANDOFF.md` and experiment protocol define measurement and cost method. | Run cloud jobs and use actual region, instance, storage, transfer, and bill data. |

## 8. Handoff order

1. Follow `QUICK_START.md` on the submission laptop. Run doctor, the full test
   suite, and `scripts/verify_local.py` before the demo.
2. Add a permitted real-source manifest and reports using
   `docs/REAL_DATA_CASE_STUDY.md`; manually inspect original passages.
3. Prepare manually labelled retrieval questions. If a local model is
   available, record its identity and verify actual Ollama answers and sources.
4. Re-run larger MPI experiments on the intended Linux/AWS hardware and retain
   raw logs, rank/host allocations, fingerprints, correctness checks, and costs.
5. Complete the student-owned AWS deployment and update this checklist only
   after capturing real evidence. The Dockerfile and Slurm scripts alone do
   not prove deployment.

The architecture, algorithms, experimental method, AWS handoff, and viva
questions remain in their focused existing documents. This review is the
single status and evidence index for the latest local engineering work.
