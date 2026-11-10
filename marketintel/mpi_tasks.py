"""Real MPI programs for exact top-k similarity and distributed Lloyd K-Means.

A genuine MPI communicator is mandatory; this module deliberately has no
single-process fallback. Rank 0 loads the input; ranks share work using MPI.
"""
from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

import numpy as np

from .performance import work_units, rate
from .analysis import assign, initial_centroids, local_sums, normalize, topk_rows, updated_centroids
from .storage import write_json
from .vectors import load_dataset


def partitions(total: int, world: int) -> list[tuple[int, int]]:
    """Return contiguous, complete, balanced row partitions (including empties)."""
    if total < 0 or world < 1:
        raise ValueError('Nonnegative row count and a positive process count required')
    q, r = divmod(total, world)
    return [(i * q + min(i, r), (i + 1) * q + min(i + 1, r)) for i in range(world)]


def run(operation: str, input_path: Path, output_path: Path, processes_expected: int | None = None,
        k: int = 3, top_k: int = 5, seed: int = 42, max_iter: int = 60) -> dict | None:
    try:
        from mpi4py import MPI
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError('Real MPI requires mpi4py and an MPI runtime such as Open MPI.') from exc

    comm = MPI.COMM_WORLD
    rank, world = comm.Get_rank(), comm.Get_size()
    if processes_expected is not None and world != processes_expected:
        raise RuntimeError(f'Expected {processes_expected} real MPI ranks, got {world}')
    if world < 2:
        raise RuntimeError('Run with at least two actual MPI ranks (mpiexec -n 2 ...)')
    if operation not in ('similarity', 'clustering') or k < 1 or top_k < 1 or max_iter < 1:
        raise ValueError('Invalid algorithm or arguments')

    # Load once on rank 0 and broadcast any error so that no rank hangs in
    # a collective waiting for another rank to reach it.
    ids = raw = None
    error = None
    if rank == 0:
        try:
            ids, raw = load_dataset(input_path)
            if not ids:
                raise ValueError('Dataset is empty')
            if operation == 'clustering' and k > len(ids):
                raise ValueError(f'k={k} exceeds document count {len(ids)}')
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
    error = comm.bcast(error, root=0)
    if error:
        raise ValueError(f'MPI dataset validation failed: {error}')

    n, dimension = comm.bcast((len(ids), raw.shape[1]) if rank == 0 else None, root=0)
    bounds = partitions(n, world)
    start, stop = bounds[rank]
    info = {'rank': rank, 'hostname': socket.gethostname(), 'rows': stop-start,
            'range_start': start, 'range_stop': stop}
    comm.Barrier()
    started = MPI.Wtime()
    cpu_started = time.process_time()

    if operation == 'similarity':
        # Exact cross-document similarity requires the full reference matrix
        # on each rank; only query rows are partitioned. No fake MPI loops.
        matrix = comm.bcast(normalize(raw) if rank == 0 else None, root=0)
        doc_ids = comm.bcast(ids if rank == 0 else None, root=0)
        partial = topk_rows(doc_ids, matrix, start, stop, k=top_k)
        info.update(cpu_seconds=round(time.process_time()-cpu_started, 6),
                    peak_rss_kib=_peak_rss_kib())
        groups = comm.gather(partial, root=0)
        ranks = comm.gather(info, root=0)
        runtime = comm.allreduce(MPI.Wtime()-started, op=MPI.MAX)
        if rank == 0:
            output = {
                'metadata': {
                    'mode': 'mpi-real', 'algorithm': 'exact-cosine-topk',
                    'documents': n, 'dimensions': dimension, 'top_k': top_k,
                    'mpi_processes': world, 'runtime_seconds': float(runtime),
                    'rank_allocation': ranks, 'input_file': input_path.name,
                    'work_units': n*(n-1), 'work_unit_label': 'directed document comparisons',
                    'work_units_per_second': rate(n*(n-1), float(runtime)),
                },
                'matches': sorted((item for group in groups for item in group),
                                  key=lambda item: (item['doc_id'], item['rank'])),
            }
            write_json(output_path, output)
            return output
        return None

    # In distributed K-Means each rank holds ONLY its assigned rows.
    # The shared dataset is not redundantly loaded on worker ranks.
    if rank == 0:
        normalized = normalize(raw).astype(np.float64)
        initial = initial_centroids(normalized, k, seed)
        chunks = [normalized[a:b] for a, b in bounds]
    else:
        initial = None
        chunks = None
    local_vectors = comm.scatter(chunks, root=0)
    centers = comm.bcast(initial, root=0)
    iterations = 0
    for iterations in range(1, max_iter + 1):
        local_labels = assign(local_vectors, centers)
        sums, counts = local_sums(local_vectors, local_labels, k)
        sums_total = np.zeros_like(sums)
        counts_total = np.zeros_like(counts)
        comm.Allreduce(sums, sums_total, op=MPI.SUM)
        comm.Allreduce(counts, counts_total, op=MPI.SUM)
        next_centers = updated_centroids(centers, sums_total, counts_total)
        movement = float(np.linalg.norm(next_centers-centers))
        centers = next_centers
        if movement <= 1e-6:
            break
    final_labels = assign(local_vectors, centers)
    local_inertia = float(np.sum((local_vectors - centers[final_labels]) ** 2))
    inertia = comm.allreduce(local_inertia, op=MPI.SUM)
    info.update(cpu_seconds=round(time.process_time()-cpu_started, 6),
                peak_rss_kib=_peak_rss_kib())
    label_parts = comm.gather(final_labels.tolist(), root=0)
    ranks = comm.gather(info, root=0)
    runtime = comm.allreduce(MPI.Wtime()-started, op=MPI.MAX)
    if rank == 0:
        ordered_labels = [label for group in label_parts for label in group]
        output = {
            'metadata': {
                'mode': 'mpi-real', 'algorithm': 'distributed-lloyd-kmeans',
                'documents': n, 'dimensions': dimension, 'mpi_processes': world,
                'k': k, 'iterations': iterations, 'inertia': float(inertia),
                'runtime_seconds': float(runtime), 'rank_allocation': ranks,
                'input_file': input_path.name,
                'work_units': n*k*iterations, 'work_unit_label': 'centroid-distance evaluations',
                'work_units_per_second': rate(n*k*iterations, float(runtime)),
            },
            'assignments': [{'doc_id': item, 'cluster_id': int(label)}
                            for item, label in zip(ids, ordered_labels)],
            'centroids': centers.tolist(),
        }
        write_json(output_path, output)
        return output
    return None


def _peak_rss_kib() -> int:
    """Per-rank peak RSS as KiB where supported; unavailable returns zero."""
    try:
        import resource
        import sys
        rss=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return rss // 1024 if sys.platform == 'darwin' else rss
    except ImportError:
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(description='Genuine MPI market analytics (no simulation)')
    parser.add_argument('operation', choices=['similarity', 'clustering'])
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-processes', type=int, default=None)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-iter', type=int, default=60)
    args = parser.parse_args()
    result = run(args.operation, args.input, args.output, args.expected_processes,
                 k=args.k, top_k=args.top_k, seed=args.seed, max_iter=args.max_iter)
    if result is not None:
        print(json.dumps({'status': 'COMPLETED', 'mode': 'mpi-real',
                          'processes': result['metadata']['mpi_processes'],
                          'runtime_seconds': result['metadata']['runtime_seconds'],
                          'rank_allocation': result['metadata']['rank_allocation']}))


if __name__ == '__main__':
    main()
