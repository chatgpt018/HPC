# Final-year submission acceptance checklist

**Exact title:** AI-Powered Market Intelligence & Document Analysis using RAG + HPC on AWS

## Local project evidence

- [x] A clean Windows Python 3.11 environment can install the project.
- [x] `python -m pytest -q` passes (47 tests at the portability checkpoint; see `../VERIFICATION.md`).
- [x] `python scripts/verify_local.py` prints `PASS` (offline fixtures).
- [ ] Gradio launches and supports actual uploaded PDF/DOCX/CSV files.
- [x] Evidence table shows file names and page/paragraph/CSV row locations in tests and the local workflow.
- [x] One result has genuine document IDs, vector type and dataset SHA-256.
- [x] Sequential similarity and K-Means generate actual output files; similarity chart was browser-checked.
- [ ] Optional local Ollama model, if used, produces a real grounded answer.

## HPC evidence

- [x] WSL2 + Open MPI + `mpi4py` launch at least four ranks on one host.
- [x] Each rank logs actual rank ID and host/partition.
- [x] Similarity and distributed K-Means both execute with multiple MPI ranks.
- [x] Equivalent sequential/MPI outputs pass correctness checks.
- [x] A preliminary range of small workload sizes and rank counts was measured without fake data.
- [x] Median wall times, speedup and efficiency are reported with raw logs.
- [ ] Repeat larger experiments at least three times on the Ubuntu filesystem for defensible scaling conclusions.
- [ ] Results/plots distinguish synthetic scaling runs from real-report analysis.

## AWS — user-owned final stage (NOT completed by this local package)

- [ ] Budget, IAM roles, network subnets and service quotas reviewed.
- [ ] Actual secured web application deployed.
- [ ] Bedrock and vector search configured and real invocation tested.
- [ ] ParallelCluster provisioned with Slurm on EC2.
- [ ] Authenticated private job submission configured.
- [ ] Slurm job IDs, node assignments and S3 outputs captured.
- [ ] End-to-end cloud workflow verified and charges reviewed.
- [ ] Idle resources deleted or stopped to avoid unnecessary billing.

## Viva and deliverables

- [ ] README, architecture, data dictionary and source licenses are available.
- [ ] Application screenshots reflect actual state.
- [ ] Project report describes tested results, not assumed performance.
- [ ] 10-minute demo can run with a documented fallback for unavailable cloud.
- [ ] Each participant can explain at least one core module and its tests.
