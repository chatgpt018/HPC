# Release verification — 2026-11-22

**Official title:** AI-Powered Market Intelligence & Document Analysis using RAG + HPC on AWS

## Latest engineering recheck — 2026-11-22

- Windows Python 3.11.9: `python -m pytest -q -p no:cacheprovider` passed **68 tests**; compilation, `scripts/doctor.py`, `pip check`, and the offline PDF/DOCX/CSV smoke test passed. Core imports include SciPy and threadpoolctl.
- Microsoft MPI 10.1.12498.52 became importable in this session. `mpiexec -n 4 .venv311/Scripts/python.exe scripts/mpi_check.py` returned ranks 0–3 on one Windows host. Four-rank similarity and clustering on a 3 × 16 synthetic input passed independent Sequential checks with partitions 1, 1, 1, 0. This supersedes the earlier Windows import-block observation below; no security setting was changed.
- A fresh **Windows** scaling study used 100, 500, and 1,000 synthetic documents, 384 features, 2 and 4 ranks, both algorithms, and 3 repetitions: **12/12 configurations verified**. End-to-end median speedups ranged **0.0043×–0.0436×**; MPI was slower. Per-rank median peak RSS ranged **142,868–159,424 KiB**. CPU utilization and isolated communication time were not measured. Saved summary, raw benchmark records, and offline charts: `output/benchmarks/native_windows_20261122/`.
- A second native Windows study used 2,000 and 5,000 synthetic documents, 384 features, 2 and 4 ranks, both algorithms, and three repetitions: **8/8 configurations verified**. Median end-to-end speedups ranged **0.039×–0.350×**, still below 1. The raw records, summary CSV/JSON, and charts are in `output/benchmarks/native_windows_20261122_large/`. A first 2,000-document attempt exposed a float32 near-tie ranking mismatch; score calculation was corrected to float64, regression-tested, and the failed partial run was discarded before the verified rerun.
- The Results page now discovers both bundled historical studies and new local runs, validates saved JSON/CSV metrics, and plots recorded values. Targeted benchmark tests pass. The current app returned **HTTP 200** on port 7868. Automatic approval review rejected activating a personal Edge window to inspect it because the active tab was not confirmed to be the local dashboard; current visual navigation and layout checks remain manual.
- The final source ZIP was extracted into a new temporary project folder. With the existing Python 3.11 environment, imports resolved from that extracted folder and benchmark discovery found both bundled studies without moving files; the larger study displayed 8/8 verified rows and actual scaling traces. This checks packaged discovery, not a second fresh dependency installation. The working-tree full suite, offline smoke, doctor, and dependency check are recorded above.
- An isolated local Gradio instance returned **HTTP 200** on port 7865. Browser visual inspection of this revision did not complete: Computer Use stopped because it could not establish the browser URL with enough confidence. UI construction and action tests passed; do not treat this as a fresh four-page visual check.
- The **previous release ZIP** was extracted into a new temporary folder on this Windows host. The documented `setup_windows.ps1 -PythonExecutable <Python 3.11 path>` created a fresh `.venv311` and installed declared development dependencies. Its extracted suite passed **60 tests**, `scripts/verify_local.py` and `pip check` passed, and `run_windows.ps1` returned **HTTP 200** on port 7866. The current revision's full suite passed **68 tests** in the working tree; see the separate final ZIP discovery check below. All checks use one physical laptop, not a second-laptop test.
- A real public-report manifest workflow and a retrieval evaluation command are implemented, but no licensed real reports, manually labelled real questions, local neural embedding model, or Ollama model were supplied. Those modes have offline contract tests, not live model evidence.
- No WSL distribution is installed in the current session. The older Ubuntu WSL2 evidence below remains historical. No second physical laptop, AWS deployment, database, multi-node MPI, or cloud cost was tested.

The sections below preserve prior observations and their dates; use this section for the latest working-tree status.

## Historical local recheck — earlier 2026-11-22 revision

- Windows Python 3.11.9: `scripts/doctor.py` reported a healthy core environment on a sequential recheck; `import pymupdf` succeeded. An earlier concurrent doctor/smoke invocation gave a transient PyMuPDF import error; no package change was needed.
- Full pytest suite: **50 passed**. `scripts/verify_local.py`: **PASS** with three temporary synthetic PDF/DOCX/CSV documents, source retrieval, sequential similarity and K-Means. `pip check`: no broken requirements.
- `run_windows.ps1` served the Gradio dashboard on a temporary localhost port with **HTTP 200** and the unchanged project title. The temporary test instance was stopped afterward.
- The updated portable ZIP was extracted into a new temporary folder. The documented `setup_windows.ps1` command created a fresh Python 3.11.9 environment from the package registry (including Gradio 6.32.0); doctor, **50 tests**, offline smoke test, `pip check`, and `run_windows.ps1` localhost **HTTP 200** all passed. The fresh data folder stayed inside the extraction.
- Optional Windows `mpi4py.MPI` import remains blocked by Windows Application Control. No genuine MPI, Ubuntu/WSL2, AWS, Ollama model, or second physical laptop was tested in this recheck.

## Historical portable release checks — earlier revisions (one physical laptop)

- The current portable ZIP, including the project report PDF, was extracted into a new Windows temporary folder. `setup_windows.ps1 -Wheelhouse <folder>` created a fresh Python 3.11.9 environment using only local wheels. The doctor reported all core imports healthy, optional `mpi4py` absent and the default data folder inside the extraction. **50 tests**, the offline PDF/DOCX/CSV smoke test and `pip check` passed. `run_windows.ps1` served the dashboard at localhost with HTTP 200 and the exact project title.
- A Windows wheelhouse containing 67 wheels was built using the documented command for an earlier source revision. The dependency specifications did not change for this UI edit. The current ZIP used that wheelhouse with `--no-index` in a fresh extraction; **50 tests** and the smoke test passed. The wheelhouse is separate from the source ZIP.
- An earlier source revision was extracted into a new **Ubuntu WSL2** home directory. `setup_linux.sh` created a fresh Python 3.12.3 environment; doctor, **48 tests**, smoke, `pip check` and `run_linux.sh` HTTP 200 passed. `setup_linux.sh --with-mpi` installed the optional MPI dependency. Four actual Open MPI ranks launched, and one similarity and clustering benchmark each passed an independent sequential correctness check. The current UI revision was not rerun on Ubuntu.
- These are **Windows and Ubuntu WSL2 on one physical laptop**. No second laptop, separate Windows 10 and 11 hosts, native Ubuntu machine, live Ollama model, AWS or database was tested. Package installation normally requires internet; the tested Windows wheelhouse route supports a disconnected target after matching wheels are prepared elsewhere.
- The ZIP excludes virtual environments, caches, credential files and indexed `storage/mpi_case` runtime records containing machine-local source paths. It keeps labelled synthetic inputs and measured MPI evidence. Archive validation found no personal paths or common credential patterns in included text.

## Windows application verification

- Python **3.11.9** in project-local `.venv311`; NumPy 1.26.4, pandas 2.2.3, Gradio 6.31.0, SciPy 1.17.1 and scikit-learn 1.9.1 imported successfully. `python scripts/doctor.py` reports the exact executable and dependency status.
- `.venv311\Scripts\python.exe -m pytest -q` — **50 passed** after the latest dashboard changes.
- `.venv311\Scripts\python.exe -m pip check` — **no broken requirements**.
- `.venv311\Scripts\python.exe -m compileall -q marketintel scripts tests` — **passed**.
- `.venv311\Scripts\python.exe scripts\verify_local.py` — **PASS**: temporary synthetic PDF/DOCX/CSV, source retrieval, 384-dimensional document vectors, sequential similarity/K-Means and traceable CSV growth.
- Local Gradio startup returned **HTTP 200**. Browser inspection covered all four pages, two clearly synthetic PDF uploads, evidence-only retrieval with a page citation, sequential similarity with two source documents, a saved result chart, and JSON/Markdown downloads. The latest UI check used an isolated copy of one indexed PDF plus one labelled synthetic CSV: library and evidence rows had light backgrounds with dark text; the document preview showed source locations and suggested topics; one-document analysis was disabled; two-document sequential similarity produced a score table and populated the saved Results view. No MPI was simulated. Table headers and file download names were checked for readable contrast.
- Follow-up PDF checks used temporary local files: a two-page text PDF produced page-specific excerpts and evidence-only retrieval; a duplicate added no new chunks; image-only, encrypted, corrupt and empty PDFs were rejected without changing the indexed library. The corrupt-PDF error was made path-free. No scanned-PDF OCR is claimed.
- Native Windows Microsoft MPI launcher was present, but `import mpi4py.MPI` and a four-rank attempt failed because **Windows Application Control blocked the DLL**. No security setting was changed.

## Genuine MPI verification on this machine

The recorded artifacts named below are included in the local portable ZIP.
The GitHub source checkout ignores `storage/` runtime state; its scripts and
synthetic examples can regenerate measurements on the destination machine.

- Existing Ubuntu WSL2: Python **3.12.3**, Open MPI **4.1.6**, working `mpi4py`; project-local `.venv-wsl` uses the existing MPI runtime. `mpiexec -n 4 python3 scripts/mpi_check.py` returned ranks **0–3** on one local host.
- On a **3 × 16 synthetic** dataset, four-rank similarity and distributed K-Means each passed **two** independent sequential comparisons. Partitions were **1, 1, 1, 0** rows, verifying an empty rank. Median end-to-end speedups were **0.017×** and **0.020×**, respectively; MPI was slower.
- The bounded study at **20 and 100 documents**, **32 features**, **2 and 4 ranks**, **both algorithms**, and **two repetitions** produced **8/8 correct configurations**. Measured speedups ranged from **0.007× to 0.024×**. Structured raw output is in `storage/benchmarks/20261110T072413Z_d86ee9/scaling_study.csv` and `.json`.
- Nine indexed, explicitly synthetic practice reports produced a **9 × 384 document-derived** dataset. Four-rank similarity and K-Means jobs both completed, matched independent sequential references, and saved all nine source IDs with partitions **3, 2, 2, 2**. Results are in `storage/mpi_case/results/`.
- In the final fresh Ubuntu extraction, `mpiexec -n 4 .venv/bin/python scripts/mpi_check.py` returned ranks **0–3**. One four-rank similarity run and one four-rank clustering run on the archived **3 × 16 synthetic input** both recorded `correct: true` against sequential references. Each is a functionality check with one repetition, not performance evidence.
- These short trials ran from the Windows-mounted `/mnt/c` project folder. They establish functionality and correctness on **one WSL2 host**, not useful speedup or multi-node scaling. The experiment protocol calls for at least three repetitions and larger workloads before academic performance conclusions.

## Still unavailable or unverified

- Neural sentence-transformer inference and live Ollama generation: no local model configured. TF-IDF retrieval and extractive evidence mode are verified; LLM request contracts have automated tests.
- Real public market-report case study: bundled practice reports are fictional.
- Docker image build, AWS, Slurm, Bedrock, database, S3, cloud cost and multi-node execution: not attempted.

## Submission boundary

The local application and single-host MPI computation are verified. Final submission still needs a licensed real-source case study, a larger reproducible performance study, and the student's AWS deployment and cost evidence. No credentials, paid APIs, cloud resources or database were used.
