"""Deterministic exact document top-k and Lloyd K-Means reference algorithms."""
from __future__ import annotations

import time
from typing import Any

import numpy as np


def normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError('Expected a finite 2D matrix')
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms > 0, norms, 1)


def topk_rows(ids: list[str], normalized: np.ndarray, start: int, stop: int,
              k: int = 5, block_size: int = 128) -> list[dict]:
    if k < 1 or block_size < 1:
        raise ValueError('top_k and block_size must be positive')
    n = len(ids)
    result = []
    # Full stable sorting per row is intentional for deterministic tie-breaking;
    # it is not the fastest possible top-k algorithm for very large N.
    names = np.asarray(ids)
    for left in range(start, stop, block_size):
        right = min(left + block_size, stop)
        scores = normalized[left:right] @ normalized.T
        for offset, row in enumerate(range(left, right)):
            scores[offset, row] = -np.inf
            order = np.lexsort((names, -scores[offset]))[:min(k, n-1)]
            for rank, j in enumerate(order, start=1):
                if np.isfinite(scores[offset, j]):
                    result.append({'doc_id': ids[row], 'similar_doc_id': ids[j],
                                   'score': float(scores[offset, j]), 'rank': rank})
    return result


def sequential_similarity(ids: list[str], vectors: np.ndarray, k: int = 5,
                          block_size: int = 128) -> dict:
    if np.asarray(vectors).ndim != 2 or len(ids) != vectors.shape[0] or len(set(ids)) != len(ids):
        raise ValueError('Expected a 2D matrix with unique document IDs matching the row count')
    if not ids or vectors.shape[1] == 0:
        raise ValueError('Dataset needs at least one document and one feature')
    t0 = time.perf_counter()
    matrix = normalize(vectors)
    matches = topk_rows(ids, matrix, 0, len(ids), k=k, block_size=block_size)
    return {'metadata': {'mode': 'sequential', 'algorithm': 'exact-cosine-topk', 'documents': len(ids),
                         'dimensions': int(vectors.shape[1]), 'top_k': k,
                         'runtime_seconds': time.perf_counter()-t0},
            'matches': matches}


def initial_centroids(vectors: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    if not 1 <= k <= len(vectors):
        raise ValueError('k must be between 1 and number of documents')
    rng = np.random.default_rng(seed)
    return vectors[rng.choice(len(vectors), size=k, replace=False)].astype(np.float64, copy=True)


def assign(vectors: np.ndarray, centers: np.ndarray) -> np.ndarray:
    if not len(vectors):
        return np.empty((0,), dtype=np.int64)
    # Squared Euclidean distances; all centers use the same deterministic tie rule.
    distances = ((vectors[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return distances.argmin(axis=1)


def local_sums(vectors: np.ndarray, labels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((k, vectors.shape[1]), dtype=np.float64)
    counts = np.bincount(labels, minlength=k).astype(np.float64)
    np.add.at(sums, labels, vectors)
    return sums, counts


def updated_centroids(centers: np.ndarray, sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    new = centers.copy()
    present = counts > 0
    new[present] = sums[present] / counts[present, None]
    # For empty clusters retain previous centroid consistently in both versions.
    return new


def sequential_kmeans(ids: list[str], vectors: np.ndarray, k: int = 3, seed: int = 42,
                      max_iter: int = 60, tol: float = 1e-6) -> dict:
    if np.asarray(vectors).ndim != 2 or len(ids) != vectors.shape[0] or not len(ids) or len(set(ids)) != len(ids):
        raise ValueError('K-Means requires a 2D matrix with matching unique document IDs')
    if vectors.shape[1] == 0:
        raise ValueError('K-Means requires at least one feature')
    if max_iter < 1 or tol < 0:
        raise ValueError('max_iter must be positive and tolerance nonnegative')
    t0 = time.perf_counter()
    vectors = normalize(vectors).astype(np.float64)
    centers = initial_centroids(vectors, k, seed)
    iteration = 0
    for iteration in range(1, max_iter+1):
        labels = assign(vectors, centers)
        sums, counts = local_sums(vectors, labels, k)
        updated = updated_centroids(centers, sums, counts)
        movement = float(np.linalg.norm(updated - centers))
        centers = updated
        if movement <= tol:
            break
    labels = assign(vectors, centers)
    inertia = float(np.sum((vectors - centers[labels]) ** 2))
    return {'metadata': {'mode': 'sequential', 'algorithm': 'lloyd-kmeans', 'documents': len(ids), 'k': k,
                         'iterations': iteration, 'inertia': inertia,
                         'runtime_seconds': time.perf_counter()-t0},
            'assignments': [{'doc_id': d, 'cluster_id': int(c)} for d, c in zip(ids, labels)],
            'centroids': centers.tolist()}
