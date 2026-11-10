"""Pure scientific validation shared by local HPC runner and benchmark suite."""
from __future__ import annotations

import math


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
        return (equivalent_cluster_labels(ref_labels, got_labels)
                and math.isclose(i0, i1, rel_tol=rtol, abs_tol=atol))
    raise ValueError(f'Unknown analysis type: {operation}')
