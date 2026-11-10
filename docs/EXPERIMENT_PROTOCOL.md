# Real HPC experimental protocol

## Goal
Evaluate how exact cosine top-k and Lloyd K-Means scale with input size and MPI rank count. This is a performance experiment; it is not an excuse to manufacture positive speedup.

## On Windows

1. Keep Windows and the original source as-is. Install WSL2 Ubuntu only with your approval.
2. Inside Ubuntu create a Python virtual environment and install the project plus Open MPI and `mpi4py`.
3. Verify rank identities before running analytical workloads:

```bash
python -m scripts.doctor
mpiexec -n 4 python -m scripts.mpi_check
```

4. For a short real two-rank experiment:

```bash
python -m scripts.generate_scaling_data --documents 300 --dimensions 384 --output storage/datasets/synthetic_300.npz
python -m scripts.benchmark --input storage/datasets/synthetic_300.npz --operation similarity --processes 2 --repetitions 3 --out storage/benchmarks
```

5. For a systematic study (can be CPU-heavy):

```bash
python -m scripts.scaling_study --sizes 200,500,1000 --processes 2,4 --operations similarity,clustering --repetitions 3 --dimensions 384
```

All vector data in these commands is **SYNTHETIC BENCHMARK INPUT**, not evidence about real markets. The script refuses execution if `mpiexec` / `mpirun` is missing; it does not substitute a Python loop for MPI.

## Fair comparisons

- Hold input, feature dimension, algorithm, initialization and output requirements constant.
- Limit BLAS/OpenMP threads (provided scripts use one thread) to reduce nested parallelism.
- Include dataset loading, process launch and output writing in the end-to-end time used for the primary speedup.
- Report internal compute time separately; do not mix it with total time.
- Use at least three repetitions and compare medians; include machine and process information.
- Verify every MPI output against the sequential baseline. If correctness fails, the experiment is **invalid**.
- If the runtime is slower, explain process startup, communication, memory pressure and load balancing.

## Measurements

`speedup = median sequential wall time / median MPI wall time`

`parallel efficiency (%) = speedup / MPI ranks × 100`

`throughput = validated work units / elapsed seconds`

For similarity the work units are directed non-self document comparisons. For clustering they are document × centroid × iteration distance evaluations. These are *algorithm operation proxies*, not file parsing throughput.

MPI rank metadata includes host names, row partitions, process CPU seconds and peak RSS high-water marks where OS support is available. These high-water marks are **not** instantaneous cluster-wide memory traces.

## Multi-node AWS experiment (later)

Start with 1 node and 2/4 ranks, then 2 nodes with 4 ranks, then increase dataset size. Record EC2 types, Slurm ID, allocation, job duration, CloudWatch resource statistics and billing. No resources are provisioned by this repository.
