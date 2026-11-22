"""Pure scientific validation shared by local HPC runner and benchmark suite."""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linear_sum_assignment


def equivalent_cluster_labels(left: list[int], right: list[int]) -> bool:
    """O(n) equivalence under a bijective cluster-label permutation.

    Avoids memory-explosive n*n co-membership matrices in scaling experiments.
    """
    if len(left) != len(right):
        return False
    forward: dict[int, int] = {}
    backward: dict[int, int] = {}
    for a, b in zip(left, right):
        if a in forward and forward[a] != b:
            return False
        if b in backward and backward[b] != a:
            return False
        forward[a], backward[b] = b, a
    return True


def equivalent_results(reference: dict, candidate: dict, operation: str,
                       atol: float = 1e-5, rtol: float = 1e-6) -> bool:
    """Compare full outputs, not just first few rows or self-reported flags."""
    if operation == 'similarity':
        def rows(d):
            matches = d['matches']
            keyed = {(r['doc_id'], r['similar_doc_id'], int(r['rank'])): float(r['score'])
                     for r in matches}
            # A dict alone would silently discard repeated rows in a claimed
            # MPI result, allowing malformed output to pass equivalence checks.
            return keyed if len(keyed) == len(matches) else None
        a, b = rows(reference), rows(candidate)
        return (a is not None and b is not None and a.keys() == b.keys() and
                all(math.isclose(a[k], b[k], rel_tol=rtol, abs_tol=atol) for k in a))
    if operation == 'clustering':
        a, b = reference['assignments'], candidate['assignments']
        if [x['doc_id'] for x in a] != [x['doc_id'] for x in b]:
            return False
        ref_labels = [int(x['cluster_id']) for x in a]
        got_labels = [int(x['cluster_id']) for x in b]
        i0 = float(reference['metadata']['inertia'])
        i1 = float(candidate['metadata']['inertia'])
        if not (equivalent_cluster_labels(ref_labels, got_labels)
                and math.isclose(i0, i1, rel_tol=rtol, abs_tol=atol)):
            return False
        ref_iterations = reference['metadata'].get('iterations')
        got_iterations = candidate['metadata'].get('iterations')
        if ref_iterations != got_iterations:
            return False
        if 'centroids' not in reference and 'centroids' not in candidate:
            return True  # Legacy minimal records lack centroid arrays.
        if 'centroids' not in reference or 'centroids' not in candidate:
            return False
        try:
            ref_centers = np.asarray(reference['centroids'], dtype=np.float64)
            got_centers = np.asarray(candidate['centroids'], dtype=np.float64)
        except (TypeError, ValueError):
            return False
        if (ref_centers.ndim != 2 or ref_centers.shape != got_centers.shape or
                not np.all(np.isfinite(ref_centers)) or not np.all(np.isfinite(got_centers))):
            return False
        mapping = dict(zip(got_labels, ref_labels))
        if any(c < 0 or c >= len(got_centers) or r < 0 or r >= len(ref_centers)
               for c, r in mapping.items()):
            return False
        if any(not np.allclose(got_centers[c], ref_centers[r], rtol=rtol, atol=atol)
               for c, r in mapping.items()):
            return False
        unused_got = [i for i in range(len(got_centers)) if i not in mapping]
        unused_ref = [i for i in range(len(ref_centers)) if i not in mapping.values()]
        if len(unused_got) != len(unused_ref):
            return False
        if unused_got:
            left, right = got_centers[unused_got], ref_centers[unused_ref]
            tolerance = atol + rtol * np.abs(right[None, :, :])
            cost = np.max(np.abs(left[:, None, :] - right[None, :, :]) / tolerance, axis=2)
            rows, columns = linear_sum_assignment(cost)
            return bool(np.all(cost[rows, columns] <= 1))
        return True
    raise ValueError(f'Unknown analysis type: {operation}')
