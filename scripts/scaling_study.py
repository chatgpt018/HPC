"""Reproducible *real-MPI-only* strong-scaling and workload-size study.

This script requires an MPI runtime. It never substitutes fake process loops.
Output contains observed measurements, correctness flags, rank/host evidence,
throughput and end-to-end speedup for each experimental configuration.

Example (WSL2/Linux):
    python -m scripts.scaling_study --sizes 200,500,1000 --processes 2,4 \
      --operations similarity,clustering --repetitions 3
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from marketintel.storage import write_json
from scripts.benchmark import benchmark


def parse_positive_csv(raw: str, maximum: int = 20000) -> list[int]:
    """Validate size and rank arguments; keep the experiment bounded."""
    values = [int(part.strip()) for part in raw.split(',') if part.strip()]
    if not values or any(v < 2 or v > maximum for v in values):
        raise ValueError(f'Provide comma-separated integers between 2 and {maximum}')
    return sorted(set(values))


def make_synthetic_vectors(n: int, dim: int, seed: int, output: Path) -> None:
    """Save deterministic synthetic vectors solely for scaling experiments."""
    if not 2 <= n <= 20000 or not 2 <= dim <= 2048:
        raise ValueError('Synthetic experiment exceeds permitted dimensions')
    rng = np.random.default_rng(seed)
    # Clustered but still synthetic: avoids pathological completely unstructured
    # K-Means datasets while remaining reproducible and explicitly nonfactual.
    centers = rng.normal(size=(min(4, n), dim)).astype(np.float32)
    vectors = centers[np.arange(n) % len(centers)] + rng.normal(0, .3, (n, dim)).astype(np.float32)
    ids = np.asarray([f'SYNTHETIC-{i:06d}' for i in range(n)], dtype=str)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, doc_ids=ids, embeddings=vectors)


def run_study(sizes: list[int], processes: list[int], operations: list[str],
              repetitions: int, dimensions: int, output: Path, seed: int = 42) -> dict:
    if not (shutil.which('mpiexec') or shutil.which('mpirun')):
        raise RuntimeError('Real MPI launcher missing. Install MPI + mpi4py in WSL2 before conducting a scaling study.')
    if repetitions < 1 or repetitions > 20 or dimensions < 2 or dimensions > 2048:
        raise ValueError('Invalid repetitions or dimensions')
    if any(p > 64 for p in processes):
        raise ValueError('At most 64 MPI processes are supported in this experiment runner')
    if not operations or any(op not in ('similarity', 'clustering') for op in operations):
        raise ValueError('Supported workloads: similarity, clustering')
    study_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:6]
    folder = output / study_id
    rows = []
    for n in sizes:
        dataset = folder / 'inputs' / f'SYNTHETIC_{n}_d{dimensions}.npz'
        make_synthetic_vectors(n, dimensions, seed, dataset)
        for operation in operations:
            for ranks in processes:
                if n < 3 and operation == 'clustering':
                    continue
                entry = benchmark(dataset, operation, ranks, repetitions,
                                  count=3 if operation=='clustering' else 5,
                                  output_folder=folder / f'{n}docs' / f'{operation}_{ranks}p')
                rank_hosts=sorted({r['hostname'] for run in entry['runs']
                                   for r in run['ranks']})
                rows.append({
                    'dataset_kind':'SYNTHETIC BENCHMARK (NOT MARKET EVIDENCE)',
                    'documents':n, 'dimensions':dimensions,'operation':operation,
                    'mpi_processes':ranks, 'hosts_observed':';'.join(rank_hosts),
                    'repetitions':repetitions,
                    'median_sequential_seconds':entry['median_sequential_wall_seconds'],
                    'median_mpi_seconds':entry['median_mpi_wall_seconds'],
                    'speedup':entry['median_speedup'],
                    'efficiency_percent':entry['median_parallel_efficiency_percent'],
                    'sequential_work_units_per_second':entry['median_sequential_work_units_per_second'],
                    'mpi_work_units_per_second':entry['median_mpi_work_units_per_second'],
                    'verified':all(r['correct'] for r in entry['runs']),
                })
    result={'study_id':study_id,'study_kind':'REAL_MPI_WITH_SYNTHETIC_INPUT',
            'methodology':'Equivalent sequential/MPI workloads, end-to-end times, rank verification',
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'rows':rows}
    write_json(folder/'scaling_study.json',result)
    with (folder/'scaling_study.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader(); writer.writerows(rows)
    return result


def main() -> None:
    p=argparse.ArgumentParser(description='Real MPI scaling experiments; never simulated')
    p.add_argument('--sizes',default='200,500,1000')
    p.add_argument('--processes',default='2,4')
    p.add_argument('--operations',default='similarity,clustering')
    p.add_argument('--repetitions',type=int,default=3)
    p.add_argument('--dimensions',type=int,default=384)
    p.add_argument('--out',type=Path,default=Path('storage/benchmarks'))
    a=p.parse_args()
    studies=run_study(parse_positive_csv(a.sizes),parse_positive_csv(a.processes,64),
        [x.strip() for x in a.operations.split(',') if x.strip()],a.repetitions,a.dimensions,a.out)
    print(json.dumps({'study_id':studies['study_id'],'configurations':len(studies['rows']),
                      'output_directory':str(a.out/studies['study_id'])},indent=2))

if __name__=='__main__':
    main()
