# Start here — project handoff

**AI-Powered Market Intelligence & Document Analysis using RAG + HPC on AWS**

This is the polished *local-first* final-year project package. It runs without
AWS services by default and never pretends that missing MPI or LLM tools ran.

## Start on Windows

1. Extract the ZIP to a new, short folder path such as `C:\Projects\MarketIntel`.
2. Open PowerShell in that folder.
3. Follow [`QUICK_START.md`](QUICK_START.md) to install and check the project. The short Windows route is:

```powershell
.\setup_windows.ps1
.\.venv311\Scripts\python.exe -m pytest -q
.\.venv311\Scripts\python.exe scripts\verify_local.py
.\run_windows.ps1
```

4. Open `http://127.0.0.1:7860` and follow the first demo in `README.md`.
5. For HPC, follow `docs/WINDOWS_WSL_MPI.md` in an Ubuntu WSL2 working copy.

If you see a pandas `_pandas_datetime_CAPI` import error on Windows, see **[WINDOWS_PANDAS_FIX.md](WINDOWS_PANDAS_FIX.md)** before attempting to run pytest again.

## What to read, in order

1. `README.md` — overview and run commands.
2. `learning-guide/index.html` — interactive, beginner-friendly project explanation.
3. `docs/ARCHITECTURE_AND_WORKFLOW.md` — diagrams and every data flow.
4. `docs/ALGORITHMS_EXPLAINED.md` — real computational work and MPI logic.
5. `docs/DEMONSTRATION_AND_VIVA.md` — suggested live demonstration and answers.
6. `docs/SUBMISSION_CHECKLIST.md` — exactly what remains for full completion.

## Important limitations

No AWS resources have been deployed. Real four-rank MPI similarity and K-Means
were verified on one WSL2 host using clearly synthetic inputs. Native Windows
MPI remains blocked by Windows Application Control on this machine. Real LLM
generation requires a separately installed and running local Ollama model.
TF-IDF source retrieval works offline but is not itself LLM generation.
See `docs/TEST_REPORT.md` for the actual verification evidence.

## If using Antigravity

Open the **extracted `Market_Intelligence_HPC_Project` folder** as your workspace.
Start with `docs/ANTIGRAVITY_HANDOFF.md`. Instruct the agent to *inspect and run* the existing project before modifying anything. The local web dashboard has four real sections and no AWS calls in default mode.
