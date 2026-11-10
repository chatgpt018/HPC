# Release verification — 2026-11-10

**Official title:** AI-Powered Market Intelligence & Document Analysis using RAG + HPC on AWS

## Portable release checks (one physical laptop)

- The final portable ZIP, including the project report PDF, was extracted into a new Windows temporary folder. Running `setup_windows.ps1` from **outside** the project created a fresh Python 3.11.9 environment. The doctor reported all core imports healthy, optional `mpi4py` absent and the default data folder inside the extraction. **48 tests**, the offline PDF/DOCX/CSV smoke test and `pip check` passed. `run_windows.ps1` served the dashboard at localhost with HTTP 200 and the exact project title.
- A Windows wheelhouse containing 67 wheels was built using the documented command for the previous source revision. A second fresh extraction used `setup_windows.ps1 -Wheelhouse <folder>` with `--no-index`; **47 tests** and the smoke test passed at that point. The wheelhouse is separate from the source ZIP; this exact final revision was not rechecked through the offline wheelhouse route.
- The final ZIP was also extracted into a new **Ubuntu WSL2** home directory. `setup_linux.sh` created a fresh Python 3.12.3 environment; doctor, **48 tests**, smoke, `pip check` and `run_linux.sh` HTTP 200 passed. `setup_linux.sh --with-mpi` installed the optional MPI dependency. Four actual Open MPI ranks launched, and one fresh-release similarity and clustering benchmark each passed an independent sequential correctness check.
- These are **Windows and Ubuntu WSL2 on one physical laptop**. No second laptop, separate Windows 10 and 11 hosts, native Ubuntu machine, live Ollama model, AWS or database was tested. Package installation normally requires internet; the tested Windows wheelhouse route supports a disconnected target after matching wheels are prepared elsewhere.
- The ZIP excludes virtual environments, caches, credential files and indexed `storage/mpi_case` runtime records containing machine-local source paths. It keeps labelled synthetic inputs and measured MPI evidence. Archive validation found no personal paths or common credential patterns in included text.

## Windows application verification

- Python **3.11.9** in project-local `.venv311`; NumPy 1.26.4, pandas 2.2.3, Gradio 6.31.0, SciPy 1.17.1 and scikit-learn 1.9.1 imported successfully. `python scripts/doctor.py` reports the exact executable and dependency status.
- `.venv311\Scripts\python.exe -m pytest -q` — **48 passed** after the final dashboard changes.
- `.venv311\Scripts\python.exe -m pip check` — **no broken requirements**.
- `.venv311\Scripts\python.exe -m compileall -q marketintel scripts tests` — **passed**.
- `.venv311\Scripts\python.exe scripts\verify_local.py` — **PASS**: temporary synthetic PDF/DOCX/CSV, source retrieval, 384-dimensional document vectors, sequential similarity/K-Means and traceable CSV growth.
- Local Gradio startup returned **HTTP 200**. Browser inspection covered all four pages, two clearly synthetic PDF uploads, evidence-only retrieval with a page citation, sequential similarity with two source documents, a saved result chart, and JSON/Markdown downloads. A one-document similarity request now gives a clear minimum-input message without saving an empty result. Table headers and file download names were checked for readable contrast. The Results tab populated the newest saved run on selection.
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
