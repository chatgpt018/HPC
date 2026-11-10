"""Representative real local workflow; no cloud, no model download, no mock HPC claims."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from marketintel.analysis import sequential_kmeans, sequential_similarity
from marketintel.documents import ingest_file
from marketintel.metrics import growth_table
from marketintel.provenance import attach_provenance, research_brief, vector_fingerprint
from marketintel.rag import answer
from marketintel.storage import chunks, records, write_json
from marketintel.vectors import load_dataset, save_dataset
from marketintel.ui import inspect_result, result_chart


def _make_reports(tmp_path: Path):
    from docx import Document
    import fitz

    csv = tmp_path/'energy_market.csv'
    csv.write_text('market,year,value,unit,region,currency\nSolar,2022,100,billion,Global,USD\nSolar,2023,112,billion,Global,USD\nSolar,2024,125,billion,Global,USD\n', encoding='utf-8')
    doc = Document()
    doc.add_heading('Semiconductor market update', 0)
    doc.add_paragraph('Semiconductor manufacturing expansion requires fabrication equipment and specialist talent.')
    word = tmp_path/'semiconductor.docx'; doc.save(word)
    pdf = fitz.open(); page=pdf.new_page()
    page.insert_text((70,70),'Electric vehicles rely on lithium battery supply chains and charging infrastructure.')
    pdf_file=tmp_path/'ev_market.pdf'; pdf.save(pdf_file); pdf.close()
    more=tmp_path/'energy_note.csv'
    more.write_text('sector,comment\nRenewables,Solar photovoltaic adoption and grid development expand.\n',encoding='utf-8')
    return [csv,word,pdf_file,more]


def test_full_document_to_analysis_to_source_brief(monkeypatch,tmp_path):
    monkeypatch.setenv('MARKETINTEL_DATA_DIR',str(tmp_path/'runtime'))
    files=_make_reports(tmp_path)
    for path in files:
        result=ingest_file(path,industry='Demo — synthetic examples')
        assert result['new_chunks']>0
    assert len(records())==4 and len(chunks())>=4
    response=answer('What is mentioned about lithium battery infrastructure?')
    assert response['generated'] is False and response['evidence']
    assert any('ev_market.pdf'==row['filename'] for row in response['evidence'])
    export=save_dataset(tmp_path/'reports.npz', 'hashing')
    ids, vectors=load_dataset(tmp_path/'reports.npz')
    assert export['documents']==4 and vectors.shape==(4,384)
    similarity=attach_provenance(sequential_similarity(ids,vectors,k=2),ids,vectors,export['vector_type'])
    clustering=attach_provenance(sequential_kmeans(ids,vectors,k=2),ids,vectors,export['vector_type'])
    assert len(similarity['matches'])==8 and len(clustering['assignments'])==4
    assert similarity['metadata']['input_sha256']==clustering['metadata']['input_sha256']
    brief=research_brief(similarity)
    assert 'analysis brief' in brief.lower() and 'ev_market.pdf' in brief
    result_file=tmp_path/'runtime'/'results'/'test_result.json'
    write_json(result_file,similarity)
    description, table, metadata=inspect_result(result_file.name)
    assert 'sequential' in description and table and 'source_documents' in json.loads(metadata)
    assert result_chart(result_file.name) is not None
    trends, warnings=growth_table()
    assert not warnings and len(trends)==2
    assert trends[-1]['series_cagr_percent']>0


def test_fingerprint_independent_of_npz_timestamps(tmp_path):
    ids=['a','b']
    matrix=np.asarray([[1,2],[3,4]],dtype=np.float32)
    one=vector_fingerprint(ids,matrix)
    two=vector_fingerprint(ids.copy(),matrix.copy())
    assert one==two
    assert one!=vector_fingerprint(ids,matrix+1)


def test_invalid_analysis_inputs():
    from marketintel.analysis import sequential_similarity, sequential_kmeans
    with pytest.raises(ValueError):
        sequential_similarity(['a','a'],np.ones((2,2)))
    with pytest.raises(ValueError):
        sequential_kmeans(['a','a'],np.ones((2,2)),k=1)
    with pytest.raises(ValueError):
        sequential_kmeans(['a'],np.ones((1,2)),k=1,max_iter=0)
    with pytest.raises(ValueError):
        sequential_similarity([],np.zeros((0,10)))


def test_real_mpi_refuses_missing_runtime(monkeypatch,tmp_path):
    import marketintel.jobs as jobs
    monkeypatch.setattr(jobs.shutil,'which',lambda name: None)
    monkeypatch.setenv('MARKETINTEL_DATA_DIR',str(tmp_path/'state'))
    with pytest.raises(RuntimeError,match='Real MPI runtime not found'):
        jobs.run_local_mpi('similarity', processes=2)
    assert not list((tmp_path/'state').glob('results/*.json'))
