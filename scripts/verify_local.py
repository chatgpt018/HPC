"""One-command, offline end-to-end smoke test without touching user documents."""
from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marketintel.analysis import sequential_similarity, sequential_kmeans
from marketintel.documents import ingest_file
from marketintel.metrics import growth_table
from marketintel.provenance import attach_provenance, research_brief
from marketintel.rag import answer
from marketintel.storage import records, write_json
from marketintel.vectors import load_dataset, save_dataset


def verify() -> dict:
    import pymupdf
    from docx import Document

    previous = os.environ.get('MARKETINTEL_DATA_DIR')
    try:
        with TemporaryDirectory(prefix='marketintel_check_') as temporary:
            root = Path(temporary)
            os.environ['MARKETINTEL_DATA_DIR'] = str(root / 'state')
            csv = root / 'solar_example.csv'
            csv.write_text('market,year,value,unit,region,currency\nSolar,2023,100,index,Example,NA\n'
                           'Solar,2024,110,index,Example,NA\n', encoding='utf-8')
            report = Document()
            report.add_paragraph('Semiconductor fabrication involves lithography, wafer production and chip manufacturing.')
            docx_path = root / 'chip_example.docx'; report.save(docx_path)
            pdf = pymupdf.open(); pdf.new_page().insert_text((75, 75),
                    'Electric vehicle lithium battery markets depend on charging infrastructure.')
            pdf_path = root / 'ev_example.pdf'; pdf.save(pdf_path); pdf.close()
            inputs = [csv, docx_path, pdf_path]
            for path in inputs:
                loaded = ingest_file(path, industry='SYNTHETIC TEST EXAMPLE')
                assert loaded['new_chunks'] >= 1
            retrieval = answer('What is stated about lithium battery charging?', generate=False)
            assert retrieval['evidence'] and retrieval['generated'] is False
            out = root / 'analysis.npz'
            exported = save_dataset(out)
            ids, vectors = load_dataset(out)
            similarity = attach_provenance(sequential_similarity(ids, vectors, k=2),
                                           ids, vectors, exported['vector_type'])
            clusters = attach_provenance(sequential_kmeans(ids, vectors, k=2),
                                        ids, vectors, exported['vector_type'])
            result_file = root / 'state' / 'results' / 'smoke_test.json'
            write_json(result_file, similarity)
            summary = research_brief(similarity)
            assert len(clusters['assignments']) == len(ids) == 3
            assert 'Source traceability' in summary
            metrics, warnings = growth_table()
            assert len(metrics) == 1 and not warnings
            return {'result': 'PASS', 'dataset': 'SYNTHETIC TEST EXAMPLES',
                    'documents_ingested': len(records()), 'features': list(vectors.shape),
                    'retrieval_has_sources': True, 'llm_invoked': False,
                    'similarity_pairs_computed': len(similarity['matches']),
                    'cluster_assignments': len(clusters['assignments']),
                    'validated_market_metric_rows': len(metrics),
                    'research_brief_generated': True, 'aws_invoked': False,
                    'mpi_executed': False}
    finally:
        if previous is None:
            os.environ.pop('MARKETINTEL_DATA_DIR', None)
        else:
            os.environ['MARKETINTEL_DATA_DIR'] = previous


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
