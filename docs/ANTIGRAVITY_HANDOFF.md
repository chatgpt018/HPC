# Instructions for Antigravity — v0.5 academic build

**Project title must remain unchanged:** AI-Powered Market Intelligence & Document Analysis using RAG + HPC on AWS

## Before coding

1. Read `README.md`, `docs/FINAL_YEAR_BLUEPRINT.md`, `docs/EXPERIMENT_PROTOCOL.md` and `infrastructure/AWS_HANDOFF.md`.
2. Start in **read-only inspection** mode; understand what already runs. Do not replace the architecture without evidence.
3. Create a Git commit or separate backup before making changes.

## Verify locally (Windows Python 3.11)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e . pytest
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\verify_local.py
.\.venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:7860`. Test all four tabs: Overview, Documents & AI, HPC Analytics, Results & Reports. A missing MPI launcher is expected on basic Windows and must not be hidden.

## Preserve these boundaries

- No AWS, database, vector-store deployment, cloud accounts, IAM modifications, S3 uploads or billable API calls.
- Do not write fake benchmark values, market predictions, MPI rank metadata, citations or LLM answers.
- Do not rewrite the working four-tab editorial UI for generic AI gradients or unnecessarily add React/FastAPI.
- Keep real source reports separate from synthetic benchmark vectors.
- An installed `mpi4py` package is not proof that multi-rank MPI works.

## Your next technical tasks after local validation

1. Install WSL2 + Ubuntu/Open MPI with user's approval. Run `mpiexec -n 4 python -m scripts.mpi_check`.
2. Run real MPI similarity and K-Means. Preserve log evidence, hostnames and rank allocation.
3. Run the `scripts.scaling_study` experiment grid, validate results, create measured graphs.
4. Configure local neural embeddings and an Ollama model only if desired; do not call paid providers without explicit approval.
5. Help the student build a real-source case study and final-year documentation.

No component should be called 'completed' without logs or test evidence.
