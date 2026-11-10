# Quick start — another laptop

**AI-Powered Market Intelligence & Document Analysis using RAG + HPC on AWS**

Extract the ZIP and open a terminal **inside `Market_Intelligence_HPC_Project`**, or clone the GitHub repository and open its root folder. Follow the steps for your operating system. The local application needs no AWS account, database, API key or paid service. Python packages must be installed once; after that, default TF-IDF retrieval and sequential analytics run without internet.

## Windows 10/11 — Python 3.11

Install Python 3.11 from python.org if it is not already available. In PowerShell:

```powershell
py -3.11 --version
.\setup_windows.ps1
.\.venv311\Scripts\python.exe scripts\doctor.py
.\.venv311\Scripts\python.exe -m pytest -q
.\.venv311\Scripts\python.exe scripts\verify_local.py
.\run_windows.ps1
```

Open `http://127.0.0.1:7860`. If PowerShell blocks `.ps1` execution, use these manual commands inside the project folder without changing a managed security policy:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install -e '.[dev]'
.\.venv311\Scripts\python.exe app.py
```

The setup and run scripts work from any folder because they locate the extracted project themselves.

## Ubuntu Linux or WSL2 — Python 3.11–3.13

Install Python, its `venv` package and pip using your distribution's normal method. In a terminal inside the extracted project:

```bash
python3 --version
bash setup_linux.sh
.venv/bin/python scripts/doctor.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/verify_local.py
bash run_linux.sh
```

For **genuine MPI**, first install a compatible MPI runtime and development headers using your distribution's package manager. Then run `bash setup_linux.sh --with-mpi` and `.venv/bin/python scripts/doctor.py`; check that both `mpi4py.MPI` and `mpiexec` are available before `mpiexec -n 4 .venv/bin/python scripts/mpi_check.py`. MPI is optional for the local dashboard and is never simulated. See `docs/WINDOWS_WSL_MPI.md` for actual algorithms and benchmark commands.

## First offline demonstration

In the dashboard, import the clearly labelled synthetic practice reports or upload your own PDF, DOCX or CSV. Ask a question to see cited source passages in **evidence-only** mode. For similarity, index **at least two documents**; then run **Sequential** analysis and inspect the saved result. This path needs no model download or network service. A local Ollama model and sentence-transformer model are optional and are not installed by the setup scripts. The [end-semester project report](output/pdf/Final_Year_Project_Report.pdf) summarizes what has been verified and what remains for the AWS stage.

The default data folder is `storage/` inside this extracted project, regardless of the terminal's starting folder. To use another location, set `MARKETINTEL_DATA_DIR` in your shell before launching. `.env.example` lists safe settings but is not loaded automatically. It contains no credentials. Do not copy an old laptop's `storage/` state if it contains private documents or absolute local paths.

## Installing on an internet-disconnected laptop

On an **internet-connected computer with the same OS, CPU architecture and Python version** as the offline laptop, extract the same ZIP, open its project folder, and build a separate wheelhouse:

```powershell
# Windows PowerShell (Python 3.11)
py -3.11 -m pip wheel --wheel-dir wheelhouse ".[dev]"
```

```bash
# Ubuntu/WSL2 (use its target Python version)
python3 -m pip wheel --wheel-dir wheelhouse '.[dev]'
```

Copy the ZIP **and** `wheelhouse/` to the offline laptop; place `wheelhouse/` inside the extracted project. Install using `.\setup_windows.ps1 -Wheelhouse wheelhouse` on Windows or `bash setup_linux.sh --wheelhouse wheelhouse` on Linux. Those modes use `--no-index` and fail clearly if a required wheel is missing. Keep Windows and Linux wheelhouses separate. Run the doctor, tests and smoke test above before the first dashboard launch.

The wheelhouse is not part of the small source ZIP. Building one requires a normal package download on the connected computer; using one does not require AWS or paid APIs. If an optional MPI wheel is needed, prepare it separately on Linux with an installed MPI runtime.
