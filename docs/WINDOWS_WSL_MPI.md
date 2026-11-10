# Windows + WSL2 + Ubuntu + genuine MPI setup

**AI-Powered Market Intelligence & Document Analysis using RAG + HPC on AWS**

You can keep Windows as your main operating system. WSL2 runs Linux in Windows;
Ubuntu is a Linux distribution inside WSL2. **Neither one is AWS.**

**Verified local run (2026-11-10):** This machine already had Ubuntu, Open MPI
4.1.6 and `mpi4py`. Four ranks executed both algorithms and matched sequential
results, including a three-document test with one empty rank partition. See
`../VERIFICATION.md` for the measured study. Native Windows MPI detected an
MS-MPI launcher but Windows Application Control blocked the `mpi4py.MPI` DLL;
do not weaken that policy to force a native run.

## Stage 1 — inspect only (PowerShell)

```powershell
wsl --status
wsl --list --verbose
```

If Ubuntu is missing and you have permission to install it, review Microsoft's
WSL setup documentation. A typical Administrator PowerShell command is
`wsl --install -d Ubuntu`; do **not** run it until you've checked installation
requirements, backups, disk space and campus/device policies. A reboot can be
required. You do not have to replace Windows.

## Stage 2 — Ubuntu terminal, not PowerShell

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip openmpi-bin libopenmpi-dev
```

Copy the extracted project to your Ubuntu home directory rather than modifying
the original Windows copy. For example, from Ubuntu, copy the project directory
from your Windows drive's WSL mount to `~/market-intelligence-hpc/` using the
actual path shown on your laptop.
Do not execute a command that overwrites the original copy.

```bash
cd ~/market-intelligence-hpc
bash setup_linux.sh --with-mpi
.venv/bin/python scripts/doctor.py
mpiexec -n 4 .venv/bin/python scripts/mpi_check.py
```

Expected proof: rank IDs `0,1,2,3`, `world=4`, participating in collective
communication. If you see a Python loop printing rank numbers rather than
`mpi4py` launcher output, that is **not** proof of MPI.

## Stage 3 — run actual algorithms

Upload/index example documents in this Ubuntu copy and export the matrix:

```bash
python scripts/create_demo.py
python -m marketintel.cli ingest examples/synthetic_reports/synthetic_01_energy.csv \
  examples/synthetic_reports/synthetic_02_energy.csv \
  examples/synthetic_reports/synthetic_04_healthcare.csv
python -m marketintel.cli export --output storage/datasets/demo.npz
mpiexec -n 4 python -m marketintel.mpi_tasks similarity \
  --input storage/datasets/demo.npz --output storage/results/similarity.json \
  --top-k 2 --expected-processes 4
mpiexec -n 4 python -m marketintel.mpi_tasks clustering \
  --input storage/datasets/demo.npz --output storage/results/clustering.json \
  --k 2 --expected-processes 4
```

The clustering example has fewer documents than ranks: empty partitions are
expected and should be handled. Check saved JSON for `mode: mpi-real`,
`mpi_processes: 4`, four rank allocations with the right row counts, and
numerically valid outputs.

## Stage 4 — measure, do not invent

Use `python -m scripts.benchmark` with a documented NPZ dataset. Benchmark
only after rank diagnostics pass. Save result JSON, actual process/host logs,
dataset fingerprint and multiple timed runs. Tiny workloads may be slower.

## What not to do

- Do not switch the UI to "Real MPI" in native Windows if the installed runtime
  cannot actually load `mpi4py` or its MPI library.
- Do not present WSL2 multi-process output as AWS multi-node execution.
- Do not install unknown binaries or alter Windows security settings just to
  bypass organizational device policies.
