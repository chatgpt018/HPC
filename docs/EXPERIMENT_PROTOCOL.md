# Real HPC experimental protocol

## Goal
Evaluate how exact cosine top-k and Lloyd K-Means scale with input size and MPI rank count. This is a performance experiment; it is not an excuse to manufacture positive speedup.

## On Windows

Native Microsoft MPI was verified on this Windows host on 2026-11-22. With a
compatible runtime and optional `mpi4py` installed, verify ranks using
`mpiexec -n 4 .\.venv311\Scripts\python.exe scripts\mpi_check.py` in PowerShell.
WSL2 Ubuntu with Open MPI is an alternative when installed. In either
environment, check the runtime with the doctor before benchmarking.

For WSL2 Ubuntu, create a Python virtual environment and install Open MPI and
`mpi4py`, then verify rank identities before analytical workloads:

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

## Measured larger Windows study — 2026-11-22

The current host had 18 logical CPUs, 15.61 GiB total RAM and approximately
3.38 GiB available before launch. We therefore ran 2,000 and 5,000 synthetic
vectors with 384 features, 2 and 4 native Microsoft MPI ranks, and three
repetitions. All eight configurations passed full Sequential comparison. The
same inputs, fixed seed, controlled BLAS threading and saved outputs were used
for both execution modes. All ranks were on **one Windows host**.

| Documents | Algorithm | Ranks | Sequential median s | MPI median s | Speedup | Efficiency |
|---:|---|---:|---:|---:|---:|---:|
| 2,000 | Similarity | 2 | 0.576 | 5.344 | 0.108× | 5.39% |
| 2,000 | Similarity | 4 | 0.675 | 5.401 | 0.125× | 3.12% |
| 2,000 | Clustering | 2 | 0.206 | 5.122 | 0.040× | 2.01% |
| 2,000 | Clustering | 4 | 0.202 | 5.206 | 0.039× | 0.97% |
| 5,000 | Similarity | 2 | 2.378 | 6.885 | 0.345× | 17.27% |
| 5,000 | Similarity | 4 | 2.234 | 6.380 | 0.350× | 8.75% |
| 5,000 | Clustering | 2 | 0.614 | 5.667 | 0.108× | 5.41% |
| 5,000 | Clustering | 4 | 0.597 | 5.585 | 0.107× | 2.67% |

Runtime, speedup, efficiency, throughput, memory and timing-section charts are
in the [offline chart file](../output/benchmarks/native_windows_20261122_large/scaling_charts.html).
The [summary CSV](../output/benchmarks/native_windows_20261122_large/scaling_study.csv),
[summary JSON](../output/benchmarks/native_windows_20261122_large/scaling_study.json)
and eight raw records preserve the actual measurements. The earlier
100/500/1,000-document study remains in `output/benchmarks/native_windows_20261122/`.
For 5,000-document similarity, measured speedup improved relative to the
smaller cases but remained below 1. End-to-end MPI launch, load and output
overhead dominated. The timing section includes computation and collectives;
isolated communication time and actual CPU utilisation were **not** measured.
This is a single-host scaling study, not AWS or multi-node evidence. A 10,000-
document run was deferred because only about 3.38 GiB RAM was available.

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
