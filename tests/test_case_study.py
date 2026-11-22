"""Manifest integration uses labelled test fixtures, never bundled market facts."""
from __future__ import annotations

import csv

import numpy as np
import pytest

from marketintel.cli import ingest_manifest
from marketintel.documents import ingest_file, validate_source_metadata
from marketintel.provenance import attach_provenance
from marketintel.storage import records
from marketintel.ui_actions import document_overview


def test_manifest_preserves_source_attribution_and_reimport(monkeypatch, tmp_path):
    monkeypatch.setenv('MARKETINTEL_DATA_DIR', str(tmp_path / 'store'))
    sources = tmp_path / 'reports'
    sources.mkdir()
    (sources / 'fixture.csv').write_text('topic,detail\nSolar,Sample text only\n', encoding='utf-8')
    manifest = tmp_path / 'manifest.csv'
    fields = ['filename', 'industry', 'publisher', 'source_url', 'publication_date',
              'reporting_period', 'geographic_scope', 'source_type', 'usage_notes', 'source_kind']
    with manifest.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({'filename': 'fixture.csv', 'industry': 'Test sector',
                         'publisher': 'Fixture publisher', 'source_url': 'https://example.org/fixture',
                         'publication_date': '2024-01-02', 'reporting_period': '2023',
                         'geographic_scope': 'Test region', 'source_type': 'test fixture',
                         'usage_notes': 'Synthetic test fixture', 'source_kind': 'synthetic'})
    first = ingest_manifest(manifest, sources)
    assert first[0]['new_chunks'] == 1
    document = records()[0]
    assert document['publisher'] == 'Fixture publisher'
    assert len(document['content_sha256']) == 64
    assert 'Fixture publisher' in document_overview(document['doc_id'])
    result = attach_provenance({'metadata': {}}, [document['doc_id']],
                               np.ones((1, 2)), 'test-features')
    assert result['metadata']['source_documents'][0]['source_kind'] == 'synthetic'
    assert ingest_manifest(manifest, sources)[0]['new_chunks'] == 0
    assert len(records()) == 1


def test_manifest_rejects_path_traversal_and_incomplete_real_source(monkeypatch, tmp_path):
    monkeypatch.setenv('MARKETINTEL_DATA_DIR', str(tmp_path / 'store'))
    source_dir = tmp_path / 'reports'
    source_dir.mkdir()
    manifest = tmp_path / 'manifest.csv'
    manifest.write_text('filename,industry,publisher,publication_date,source_type,usage_notes,source_kind\n'
                        '../outside.csv,Energy,Publisher,2024-01-02,filing,permitted,real\n', encoding='utf-8')
    with pytest.raises(ValueError, match='without directories'):
        ingest_manifest(manifest, source_dir)
    with pytest.raises(ValueError, match='require publisher'):
        validate_source_metadata({'source_kind': 'real', 'publication_date': '2024-01-02'})
    with pytest.raises(ValueError, match='YYYY-MM-DD'):
        validate_source_metadata({'publication_date': '20240102'})
    with pytest.raises(ValueError, match='HTTP or HTTPS'):
        validate_source_metadata({'source_url': 'file:///private/report.pdf'})


def test_duplicate_import_can_add_metadata_but_cannot_replace_it(monkeypatch, tmp_path):
    monkeypatch.setenv('MARKETINTEL_DATA_DIR', str(tmp_path / 'store'))
    path = tmp_path / 'fixture.csv'
    path.write_text('topic\nSolar text\n', encoding='utf-8')
    ingest_file(path)
    updated = ingest_file(path, source_metadata={'source_kind': 'synthetic', 'publisher': 'Fixture'})
    assert updated['status'] == 'Metadata updated'
    with pytest.raises(ValueError, match='different source metadata'):
        ingest_file(path, source_metadata={'source_kind': 'real', 'publisher': 'Other',
                    'publication_date': '2024-01-02', 'source_type': 'test fixture',
                    'usage_notes': 'fixture'})
