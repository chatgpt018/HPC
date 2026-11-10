"""Validate, parse, chunk, and persist real PDF, DOCX, or CSV documents."""
from __future__ import annotations

import csv
import hashlib
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .storage import chunks, home, records, write_json

MAX_BYTES = 25 * 1024 * 1024
ALLOWED = {'.pdf', '.docx', '.csv'}


def parse_file(path: Path) -> list[tuple[str, str]]:
    """Return (source location, text) pairs; location is page/paragraph/row."""
    suffix = path.suffix.lower()
    if suffix == '.pdf':
        import pymupdf
        try:
            pdf = pymupdf.open(path)
        except pymupdf.FileDataError as exc:
            raise ValueError('Cannot read PDF. Check that the file is a valid, uncorrupted PDF.') from exc
        with pdf as doc:
            if doc.is_encrypted:
                raise ValueError('Encrypted PDFs are not supported')
            return [(f'page {i+1}', page.get_text('text')) for i, page in enumerate(doc)]
    if suffix == '.docx':
        from docx import Document
        doc = Document(str(path))
        units = [(f'paragraph {i+1}', p.text) for i, p in enumerate(doc.paragraphs)]
        for t, table in enumerate(doc.tables):
            for r, row in enumerate(table.rows):
                units.append((f'table {t+1}, row {r+1}', ' | '.join(c.text for c in row.cells)))
        return units
    if suffix == '.csv':
        with path.open('r', encoding='utf-8-sig', newline='') as handle:
            return [(f'row {i+2}', '; '.join(f'{key}: {val}' for key, val in record.items()))
                    for i, record in enumerate(csv.DictReader(handle))]
    raise ValueError(f'Unsupported format: {suffix}')


def chunk_units(units: list[tuple[str, str]], window: int = 180, overlap: int = 35) -> list[dict]:
    """Chunk each source unit with overlap, preserving its original page or row."""
    if not 0 <= overlap < window:
        raise ValueError('overlap must be nonnegative and smaller than window')
    result = []
    for source, text in units:
        words = re.sub(r'\s+', ' ', text).strip().split()
        for start in range(0, len(words), window - overlap):
            part = words[start:start + window]
            if part:
                result.append({'source': source, 'text': ' '.join(part), 'start_word': start})
            if start + window >= len(words):
                break
    return result


def ingest_file(path: str | Path, industry: str = 'Unspecified') -> dict:
    path = Path(path)
    if not path.exists() or not path.is_file():
        raise ValueError('The selected file was not found')
    if path.suffix.lower() not in ALLOWED:
        raise ValueError('Supported file types: PDF, DOCX, CSV')
    if path.stat().st_size == 0 or path.stat().st_size > MAX_BYTES:
        raise ValueError('The file is empty or exceeds the 25 MB local limit')
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    doc_id = digest[:16]
    existing = records()
    found = next((x for x in existing if x['doc_id'] == doc_id), None)
    if found:
        return {'document': found, 'status': 'Already indexed', 'new_chunks': 0}
    extracted = chunk_units(parse_file(path))
    if not extracted:
        raise ValueError('No readable text found. Scanned/image-only PDFs need OCR, not implemented here.')
    raw_dir = home() / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Filename is generated from the hash, never from untrusted upload paths.
    copied = raw_dir / f'{doc_id}{path.suffix.lower()}'
    shutil.copyfile(path, copied)
    new_chunks = [{'chunk_id': f'{doc_id}-{i:05d}', 'doc_id': doc_id,
                   'filename': path.name, 'industry': industry or 'Unspecified', **part}
                  for i, part in enumerate(extracted)]
    document = {'doc_id': doc_id, 'filename': path.name, 'industry': industry or 'Unspecified',
                'file_type': path.suffix.lower(), 'size_bytes': len(raw),
                'chunk_count': len(new_chunks), 'uploaded_utc': datetime.now(timezone.utc).isoformat(),
                'local_path': str(copied)}
    # A simple local single-user store; deployable multi-user concurrency needs a DB.
    write_json(home() / 'chunks.json', chunks() + new_chunks)
    write_json(home() / 'documents.json', existing + [document])
    return {'document': document, 'status': 'Indexed', 'new_chunks': len(new_chunks)}
