"""Validate, parse, chunk, and persist real PDF, DOCX, or CSV documents."""
from __future__ import annotations

import csv
import hashlib
import re
import shutil
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from zipfile import BadZipFile

from .storage import chunks, home, records, write_json

MAX_BYTES = 25 * 1024 * 1024
MAX_EXTRACTED_CHARS = 5_000_000
MAX_SOURCE_UNITS = 10_000
ALLOWED = {'.pdf', '.docx', '.csv'}
CHUNK_WORDS = 180
CHUNK_OVERLAP = 35
SOURCE_FIELDS = {'publisher', 'source_url', 'publication_date', 'reporting_period',
                 'geographic_scope', 'source_type', 'usage_notes', 'source_kind'}


def validate_source_metadata(metadata: dict | None) -> dict[str, str]:
    """Keep only explicit, validated attribution supplied by the document owner."""
    if metadata is None:
        return {'source_kind': 'unspecified'}
    if not isinstance(metadata, dict) or set(metadata) - SOURCE_FIELDS:
        raise ValueError('Unrecognized source metadata field')
    if any(not isinstance(value, str) or len(value) > 500 for value in metadata.values()):
        raise ValueError('Source metadata values must be text of at most 500 characters')
    cleaned = {key: value.strip() for key, value in metadata.items() if value.strip()}
    kind = cleaned.get('source_kind', 'unspecified').lower()
    if kind not in {'real', 'synthetic', 'unspecified'}:
        raise ValueError('source_kind must be real, synthetic, or unspecified')
    cleaned['source_kind'] = kind
    if 'publication_date' in cleaned:
        try:
            if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', cleaned['publication_date']):
                raise ValueError('Expected YYYY-MM-DD')
            date.fromisoformat(cleaned['publication_date'])
        except ValueError as exc:
            raise ValueError('publication_date must be YYYY-MM-DD') from exc
    if 'source_url' in cleaned:
        url = urlparse(cleaned['source_url'])
        if url.scheme not in {'http', 'https'} or not url.netloc:
            raise ValueError('source_url must be an HTTP or HTTPS URL')
    if kind == 'real' and not all(cleaned.get(field) for field in
                                  ('publisher', 'publication_date', 'source_type', 'usage_notes')):
        raise ValueError('Real sources require publisher, publication_date, source_type, and usage_notes')
    return cleaned


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
        from docx.opc.exceptions import PackageNotFoundError
        try:
            doc = Document(str(path))
        except (PackageNotFoundError, BadZipFile) as exc:
            raise ValueError('Cannot read Word file. Check that it is a valid DOCX.') from exc
        units = [(f'paragraph {i+1}', p.text) for i, p in enumerate(doc.paragraphs)]
        for t, table in enumerate(doc.tables):
            for r, row in enumerate(table.rows):
                units.append((f'table {t+1}, row {r+1}', ' | '.join(c.text for c in row.cells)))
        return units
    if suffix == '.csv':
        try:
            with path.open('r', encoding='utf-8-sig', newline='') as handle:
                reader = csv.DictReader(handle)
                fields = reader.fieldnames
                if not fields or any(not field or not field.strip() for field in fields) or len(fields) != len(set(fields)):
                    raise ValueError('CSV needs nonempty, unique column names')
                units = []
                for i, record in enumerate(reader, 2):
                    if None in record or any(value is None for value in record.values()):
                        raise ValueError(f'CSV row {i} has a different number of columns than the header')
                    units.append((f'row {i}', '; '.join(f'{key}: {val}' for key, val in record.items())))
                return units
        except (UnicodeError, csv.Error) as exc:
            raise ValueError('Cannot read CSV. Save it as valid UTF-8 with consistent columns.') from exc
    raise ValueError(f'Unsupported format: {suffix}')


def chunk_units(units: list[tuple[str, str]], window: int = CHUNK_WORDS,
                overlap: int = CHUNK_OVERLAP) -> list[dict]:
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


def ingest_file(path: str | Path, industry: str = 'Unspecified',
                source_metadata: dict | None = None) -> dict:
    path = Path(path)
    metadata = validate_source_metadata(source_metadata)
    industry = (industry or 'Unspecified').strip() or 'Unspecified'
    if len(industry) > 100 or any(ord(char) < 32 for char in industry):
        raise ValueError('Industry label must be at most 100 printable characters')
    if not path.exists() or not path.is_file():
        raise ValueError('The selected file was not found')
    if path.suffix.lower() not in ALLOWED:
        raise ValueError('Supported file types: PDF, DOCX, CSV')
    if path.stat().st_size == 0 or path.stat().st_size > MAX_BYTES:
        raise ValueError('The file is empty or exceeds the 25 MB local limit')
    raw = path.read_bytes()
    if path.suffix.lower() == '.pdf' and b'%PDF-' not in raw[:1024]:
        raise ValueError('Cannot read PDF. Check that the file is a valid, uncorrupted PDF.')
    if path.suffix.lower() == '.docx' and not raw.startswith(b'PK\x03\x04'):
        raise ValueError('Cannot read Word file. Check that it is a valid DOCX.')
    if path.suffix.lower() == '.csv' and b'\x00' in raw:
        raise ValueError('Cannot read CSV. Binary content is not a valid text report.')
    digest = hashlib.sha256(raw).hexdigest()
    doc_id = digest[:16]
    existing = records()
    found = next((x for x in existing if x['doc_id'] == doc_id), None)
    if found:
        changes = {key: value for key, value in metadata.items()
                   if value and (key != 'source_kind' or value != 'unspecified')
                   and found.get(key) != value}
        if changes:
            if any(found.get(key) not in (None, '', 'unspecified') for key in changes):
                raise ValueError('Already indexed with different source metadata')
            found.update(changes)
            found['content_sha256'] = digest
            write_json(home() / 'documents.json', existing)
            return {'document': found, 'status': 'Metadata updated', 'new_chunks': 0}
        return {'document': found, 'status': 'Already indexed', 'new_chunks': 0}
    units = parse_file(path)
    if len(units) > MAX_SOURCE_UNITS or sum(len(text) for _, text in units) > MAX_EXTRACTED_CHARS:
        raise ValueError('Extracted text exceeds the local processing limit')
    extracted = chunk_units(units)
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
                'local_path': str(copied), 'content_sha256': digest,
                'processing_status': 'indexed', **metadata}
    # A simple local single-user store; deployable multi-user concurrency needs a DB.
    write_json(home() / 'chunks.json', chunks() + new_chunks)
    write_json(home() / 'documents.json', existing + [document])
    return {'document': document, 'status': 'Indexed', 'new_chunks': len(new_chunks)}
