"""Read-only, source-first research quality indicators for the local workspace.

Quality indicators describe evidence coverage; they are not reliability scores
for real-world financial forecasts. No database or external API is required.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from .storage import chunks, records


def quality_report() -> dict:
    """Summarize traceability gaps without claiming sources are trustworthy."""
    documents = records()
    passages = chunks()
    by_type = Counter(d.get('file_type', 'unknown').lower() for d in documents)
    by_industry = Counter((d.get('industry') or 'Unspecified').strip() for d in documents)
    ids = {d.get('doc_id') for d in documents}
    missing_sources = sum(1 for p in passages if not p.get('source') or not p.get('doc_id'))
    mismatched_chunks = sum(1 for p in passages if p.get('doc_id') not in ids)
    synthetic = sum(
        1 for d in documents
        if any(marker in ((d.get('filename') or '') + ' ' + (d.get('industry') or '')).lower()
               for marker in ('synthetic', 'demo'))
    )
    observations = []
    if not documents:
        observations.append('No reports indexed. Upload permitted PDF, DOCX or CSV sources to begin.')
    if by_industry.get('Unspecified', 0):
        observations.append(f'{by_industry["Unspecified"]} reports lack an industry label.')
    if synthetic:
        observations.append(f'{synthetic} documents are marked or named as synthetic/demo examples; exclude them from real-world findings.')
    if missing_sources or mismatched_chunks:
        observations.append(f'{missing_sources + mismatched_chunks} passages have incomplete or mismatched source references.')
    if documents and len(documents) < 10:
        observations.append('The indexed collection is small; meaningful distributed speedup is unlikely.')
    if documents and not any(d.get('file_type') == '.csv' for d in documents):
        observations.append('No CSV data indexed; numerical market trends require comparable structured values.')
    return {
        'documents': len(documents),
        'passages': len(passages),
        'industries': dict(sorted(by_industry.items())),
        'formats': dict(sorted(by_type.items())),
        'synthetic_document_count': synthetic,
        'missing_source_passages': missing_sources,
        'orphan_source_passages': mismatched_chunks,
        'expected_directed_pairs': len(documents) * max(len(documents) - 1, 0),
        'estimated_dense_matrix_mib': round((len(documents) ** 2 * 4) / 1024**2, 2),
        'observations': observations,
        'explanation': 'Diagnostics indicate coverage and scale, not truthfulness or analytical accuracy.',
    }
