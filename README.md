# AI-Powered Market Intelligence & Document Analysis using RAG + HPC on AWS

**Final-year project · Research workspace release v0.5 · AWS deployment pending**

A focused market-document research application. Users upload PDF/DOCX/CSV
reports, retrieve source-backed evidence, calculate numerical document
similarities and K-Means clusters, and inspect actual results and benchmarks.
An optional local Ollama model can generate grounded answers and explanations.
A real `mpi4py` implementation distributes computational workloads across
processes when an MPI runtime is available. **No HPC results are simulated.**

The [end-semester project report PDF](output/pdf/Final_Year_Project_Report.pdf)
summarizes the implemented system, measured tests, and remaining academic
evidence. It distinguishes the verified local build from the pending AWS stage.

> **Professor's technical requirements:** Genuine HPC workload, real MPI or
> OpenMP execution, clear HPC necessity, and eventual deployment on AWS.
> This package completes a tested **local foundation**. Four-rank MPI
> similarity and K-Means were verified on one WSL2 host using synthetic
> inputs. AWS and multi-node verification remain independent acceptance steps.


## v0.5 academic engineering upgrades

- **Research coverage panel** shows real file counts, industries, source-excerpt integrity and clearly flags bundled synthetic practice data.
- **One-click synthetic practice import** is optional and visibly marked as fictional data, never market evidence.
- **HPC necessity preview** shows actual N(N-1) directed comparisons and warns when collections are too small to justify distributed computing.
- **Stronger result verification:** memory-safe cluster-label equivalence and full top-k rank/score comparisons; local real-MPI jobs run an independent sequential correctness check before reporting completion.
- **Research throughput metrics:** actual rank CPU time, peak RSS on supported systems, measured work units/sec, median speedup and efficiency. These are not simulated.
- **Scaling-study command:** `python -m scripts.scaling_study --sizes 200,500,1000 --processes 2,4 --repetitions 3`; requires installed real MPI and creates no cloud resources.
- **Review-only Slurm templates:** `infrastructure/slurm/` for the student's later AWS deployment. No database setup or deployment is included.
- **Research evidence:** locally retrieved source passages are now presented extractively even without an LLM, clearly labelled as non-generative.
- **Academic guides:** `docs/FINAL_YEAR_BLUEPRINT.md`, `docs/EXPERIMENT_PROTOCOL.md`, `docs/REAL_DATA_CASE_STUDY.md`.

The automated tests verify local logic and UI construction. Separate WSL2
experiments establish real single-host MPI execution; they do **not** establish
an installed Ollama model, cloud deployment or live database. See
`VERIFICATION.md` for exact commands and measured results.

## What is implemented vs not yet verified

| Component | Status |
|---|---|
| PDF, DOCX and CSV ingestion with file/source metadata | Implemented; integration tested |
| Local TF-IDF document evidence retrieval | Implemented; **lexical**, not neural semantic retrieval |
| Optional transformer embeddings from an installed local model | Integration implemented; model installation/runtime not verified |
| Optional local Ollama answers and HPC explanations | Integration and contract tested; live model execution not verified |
| Document-derived numerical vectors with stable source IDs | Implemented; integration tested |
| Sequential cosine top-k and Lloyd K-Means | Implemented; unit and smoke tested |
| Genuine MPI similarity and distributed K-Means source code | Four-rank WSL2 execution verified on synthetic inputs, including an empty partition |
| Real MPI launcher, fail-closed job records and runtime status | Four-rank WSL2 jobs verified against sequential references with source provenance |
| Source-aware saved results, charts, deterministic analysis briefs | Implemented; tested |
| Traceable CSV market metric calculations | Implemented; tested |
| Real sequential vs MPI benchmark script | 8 WSL2 scaling configurations measured and checked; all small cases slower than sequential |
| Four-section Gradio dashboard | Windows localhost HTTP 200; four pages, two synthetic PDF uploads, retrieval, analysis and downloads browser-checked |
| Docker packaging for local web app | Provided; Docker image build not independently tested |
| AWS hosting, Bedrock, ParallelCluster, Slurm, multi-node EC2 | **NOT DEPLOYED**; user-owned final stage |

## 1. Quick start on Windows (without HPC initially)

For the GitHub version, run `git clone https://github.com/chatgpt018/HPC.git`
and `cd HPC`. For the ZIP, open the extracted project folder.

Start with [`QUICK_START.md`](QUICK_START.md) on a new laptop. The setup scripts keep the virtual environment in the extracted project and use Python 3.11 on Windows. If upgrading a much older environment, see `WINDOWS_PANDAS_FIX.md`.

Requires Python **3.11** (recommended), a functioning terminal, and internet
for initial Python package installation unless dependencies are already cached.
No cloud/API account is needed for this stage.

```powershell
cd "PATH\TO\EXTRACTED\PROJECT"
.\setup_windows.ps1
.\.venv311\Scripts\python.exe -m pytest -q
.\.venv311\Scripts\python.exe scripts\verify_local.py
.\run_windows.ps1
```

Open <http://127.0.0.1:7860>. The app binds to your computer's loopback
address by default. Do not publish it without authentication and isolation.
If Windows activation scripts are blocked by local policy, use
`.venv311\Scripts\python.exe` directly instead of activating.

**The one-command verification uses temporary SYNTHETIC test documents.** It
does not touch user files, create AWS resources, call paid APIs, or simulate MPI.

## 2. Four-section dashboard

**Market Atlas / Field Notes design:** The interface now has an editorial research identity, compact actual-data KPIs, a visible four-stage workflow, a guided document intake, a focused HPC configuration panel, and an evidence-led Results Studio. The submitted project title appears in full in the header. The style uses local CSS only: no paid UI kit, external design API, or tracking service.

**Code separation:** `marketintel/ui.py` contains Gradio layout only; `marketintel/ui_actions.py` contains reusable backend handlers and charts; `marketintel/ui_theme.py` contains all CSS and branded presentation HTML. Algorithms are still isolated in `analysis.py`, `mpi_tasks.py`, and other engine modules.


| Section | What to demonstrate |
|---|---|
| **Overview** | Honest feature/environment status and processing pipeline |
| **Documents & AI** | Import PDF/DOCX/CSV, inspect page- or row-labeled text excerpts, ask a topic-specific question and check cited evidence |
| **HPC Analytics** | See input requirements before running similarity or K-Means on document vectors; use genuine MPI only where installed |
| **Results & Reports** | Inspect real result tables/charts, download JSON and research brief, examine market metrics and measured benchmarks |

**Behind the Analysis:** You can explain a saved numerical result using a
deterministic source-aware report, or optionally invoke a configured local
Ollama model with the actual computed results and uploaded source excerpts.
Without a model, the application does *not* pretend to generate AI prose.

## 3. First demonstration with safe sample data

```powershell
python scripts/create_demo.py
```

1. Import at least three files in `examples/synthetic_reports` through
   **02 Documents & AI**. These files are **synthetic practice samples**, not
   actual market studies.
2. Select a document under **Read what was extracted**. Use a suggested topic to ask a question, then inspect the source filename and page/row in the evidence table.
3. Open **03 HPC Analytics** and run **Sequential** similarity or K-Means.
4. Open **04 Results & Reports**, inspect actual scores, and use
   **Explain selected findings**.
5. Download the result JSON and Markdown brief; check the input fingerprint,
   feature type and source-document manifest.
6. For factual numerical trends, upload a CSV with columns
   `market,year,value,unit,region,currency` and use the market metric tool.

**Real reports:** Replace sample inputs with public reports you can lawfully
use, retain publisher/date/source attribution, and never treat clustering or
similarity as a market forecast. See `docs/DATASET_GUIDE.md`.

## 4. Real MPI using WSL2 + Ubuntu on a Windows computer

Windows remains your operating system. WSL2 gives you Linux tooling for Open
MPI and `mpi4py`. See `docs/WINDOWS_WSL_MPI.md` before making system changes.
After checking WSL is installed, inside the Ubuntu terminal:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip openmpi-bin libopenmpi-dev
bash setup_linux.sh --with-mpi
mpiexec -n 4 .venv/bin/python scripts/mpi_check.py
```

The commands above must be run inside a project copy in Ubuntu, not as Windows
PowerShell commands. Package installation may require internet and storage.

Create a dataset from the reports previously indexed in that copy:

```bash
python -m marketintel.cli export --output storage/datasets/uploaded_reports.npz
mpiexec -n 4 python -m marketintel.mpi_tasks similarity \
  --input storage/datasets/uploaded_reports.npz \
  --output storage/results/mpi_similarity.json --expected-processes 4 --top-k 3
mpiexec -n 4 python -m marketintel.mpi_tasks clustering \
  --input storage/datasets/uploaded_reports.npz \
  --output storage/results/mpi_clustering.json --expected-processes 4 --k 3
```

**What makes this genuine MPI?** Rank 0 loads the dataset, query rows are
partitioned for exact similarity, and distributed K-Means uses row scattering
and `MPI.Allreduce` for cluster sums/counts. Output records actual rank IDs,
hostnames and partition ranges. A missing runtime raises an error; it does not
silently run a sequential loop. See `docs/ALGORITHMS_EXPLAINED.md`.

## 5. Scientific HPC benchmarking

After genuine MPI works, use this separate **SYNTHETIC scaling dataset** (not
market evidence):

```bash
python scripts/generate_scaling_data.py --documents 1000 --dimensions 384 --seed 42 \
  --output storage/datasets/synthetic_scaling.npz
python -m scripts.benchmark --input storage/datasets/synthetic_scaling.npz \
  --operation similarity --processes 4 --repetitions 3 --count 3
python -m scripts.benchmark --input storage/datasets/synthetic_scaling.npz \
  --operation clustering --processes 4 --repetitions 3 --count 3
```

Run a few different dataset sizes and MPI counts. Report real rank information,
correctness checks, sequential/MPI wall times (including launcher overhead),
median speedup and efficiency. **Positive speedup is not guaranteed.** Without
`mpiexec`, this script fails with an explicit error; it never fabricates data.

## 6. Optional local AI and semantic retrieval

- Evidence-only TF-IDF retrieval works without any model or API key.
- For a real local LLM, install **Ollama** and a model separately, start the
  service, then set `MARKETINTEL_OLLAMA_MODEL` to your already-installed model
  identifier. The program calls `http://127.0.0.1:11434/api/generate` only.
- For neural semantic retrieval, separately install `sentence-transformers`,
  download a compatible model into a local directory, set
  `MARKETINTEL_LOCAL_EMBED_MODEL` to that folder and
  `MARKETINTEL_RETRIEVAL_MODE=semantic`. The HPC feature vector selector can
  use `semantic` as well.
- Document data may be supplied to the locally running Ollama process when
  generation is explicitly requested. Do not use confidential files without
  verifying your local setup and data-handling requirements.

These integrations require optional software/model resources. They are not
implicitly downloaded or called when you install or start the application.

## 7. Project structure

```text
app.py                    Gradio application entry point
marketintel/
  documents.py             File validation, extraction, source chunks
  storage.py               Local JSON persistence
  vectors.py               Text vectors, retrieval and NPZ I/O
  analysis.py              Sequential cosine similarity and K-Means
  mpi_tasks.py             Genuine MPI algorithms
  jobs.py                  MPI launcher + honest job status
  provenance.py            Input fingerprint and source manifest
  rag.py                   Question answering with local evidence
  local_llm.py             One loopback-only optional LLM adapter
  report_generation.py     RAG evidence + HPC explanation integration
  metrics.py               Source-traceable CSV arithmetic
  ui.py                    Four-section web dashboard
scripts/                   Doctor, offline smoke test, benchmarks, examples
examples/                  Clearly synthetic, optional practice inputs
tests/                     Automated local correctness and integration tests
learning-guide/            Separate interactive beginner notebook
docs/                      Architecture, algorithms, demo, datasets, viva, audit
infrastructure/            Local container and future AWS handoff docs
Dockerfile                 Local application container template
storage/                   Runtime documents/results (ignored by Git)
```

## 8. AWS handoff (student-owned final phase)

This repository does not contain AWS credentials and does not create AWS
resources. For final submission, you must still deploy a secure web app,
configure Bedrock + supported vector retrieval, store inputs/results in S3,
and submit real Slurm/MPI jobs on a ParallelCluster/EC2 cluster. See
`infrastructure/AWS_HANDOFF.md`, `docs/SUBMISSION_CHECKLIST.md` and the
project report. A Dockerfile **does not equal AWS deployment**.

## 9. Learn and present the project

- `learning-guide/index.html` — interactive local learning notebook.
- `docs/ARCHITECTURE_AND_WORKFLOW.md` — current and proposed architecture.
- `docs/ALGORITHMS_EXPLAINED.md` — what MPI computes and why it matters.
- `docs/DEMONSTRATION_AND_VIVA.md` — live demo sequence and viva questions.
- `docs/DATASET_GUIDE.md` — data provenance and realistic experiments.
- `docs/TEST_REPORT.md` — independently checked parts and honest limitations.
- `docs/SUBMISSION_CHECKLIST.md` — exact final acceptance gates.
- `docs/ORIGINAL_PROJECT_CLEANUP.md` — original file cleanup decisions.

## Security, scope and limitations

This local release is a single-user educational app, not a public financial
advisory service. It does not authenticate users, support concurrent writers or
implement production-grade background Slurm job submission. File content is
untrusted; source passages are shown for validation. Do not deploy publicly
without authorization, per-user data separation, an authenticated backend,
reliable database/queue and reviewed IAM permissions.

## Presentation-quality dashboard

The application deliberately uses **only four navigation destinations**. Each button is wired to a real handler; unavailable MPI and local LLM configurations report their limitations instead of pretending to succeed. The Overview KPIs are computed from the local store rather than filled with placeholder totals.

For design decisions, maintainability and manual UI validation, see `docs/UI_AND_CODE_STRUCTURE.md`. For the exact status of verification, see `docs/RELEASE_NOTES.md` and `docs/TEST_REPORT.md`.
