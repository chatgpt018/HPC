"""Metrics for reproducible sequential-vs-MPI benchmark receipts.

This module never fabricates measurements. All rates are computed from actual
recorded duration and the validated input size.
"""
from __future__ import annotations

import math
from statistics import median


def work_units(operation: str, documents: int, *, iterations: int | None = None, clusters: int | None = None) -> tuple[int, str]:
    """Count useful comparable work units; K-Means uses assignments per iteration."""
    if documents < 0:
        raise ValueError('documents must be nonnegative')
    if operation == 'similarity':
        return documents * max(documents - 1, 0), 'directed document comparisons'
    if operation == 'clustering':
        if iterations is None or iterations < 1:
            raise ValueError('K-Means throughput requires actual iteration count')
        if clusters is None or clusters < 1:
            raise ValueError('K-Means throughput requires actual cluster count')
        return documents * clusters * iterations, 'centroid-distance evaluations'
    raise ValueError('Unknown operation')


def rate(units: int, elapsed_seconds: float) -> float:
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0:
        raise ValueError('Measured time must be finite and positive')
    if units < 0:
        raise ValueError('work units cannot be negative')
    return units / elapsed_seconds


def benchmark_summary(repetitions: list[dict], processes: int, documents: int, operation: str) -> dict:
    """Create computed summary from *actual* completed benchmark iterations."""
    if not repetitions or processes < 2:
        raise ValueError('At least one real run and two MPI ranks are required')
    if not all(r.get('correct') is True for r in repetitions):
        raise ValueError('Invalid benchmark: at least one result failed correctness')
    seq = median(float(r['sequential_wall_seconds']) for r in repetitions)
    mpi = median(float(r['mpi_end_to_end_wall_seconds']) for r in repetitions)
    if seq <= 0 or mpi <= 0:
        raise ValueError('Benchmarks require positive measured times')
    speedup = seq / mpi
    return {'median_sequential_seconds': seq, 'median_mpi_seconds': mpi,
            'speedup': speedup, 'efficiency_percent': 100 * speedup / processes,
            'sequential_docs_per_second': documents / seq,
            'mpi_docs_per_second': documents / mpi,
            'comparison': 'MPI faster' if speedup > 1 else ('Equal' if speedup == 1 else 'MPI slower'),
            'runtime_definition': 'end-to-end wall clock including load, process launch, and result writing'}
