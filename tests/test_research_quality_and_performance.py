"""Regression coverage for research integrity and measured HPC statistics."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from marketintel.correctness import equivalent_cluster_labels, equivalent_results
from marketintel.performance import work_units, rate, benchmark_summary
from marketintel.research_quality import quality_report
from marketintel.storage import write_json
from marketintel.ui_actions import workspace_quality, import_practice_collection, compute_preview
from scripts.scaling_study import parse_positive_csv, make_synthetic_vectors, run_study
from marketintel.vectors import load_dataset


def test_linear_memory_cluster_equivalence():
    assert equivalent_cluster_labels([0, 0, 1, 2], [4, 4, 9, 3])
    assert not equivalent_cluster_labels([0, 0, 1], [1, 2, 3])
    assert not equivalent_cluster_labels([0, 1, 2], [4, 4, 5])
    assert equivalent_cluster_labels([0]*20000 + [1]*20000, [8]*20000 + [7]*20000)


def test_full_cosine_validation_includes_rank_and_id():
    baseline={'matches':[{'doc_id':'a','similar_doc_id':'b','rank':1,'score':0.5}]}
    candidate={'matches':[{'doc_id':'a','similar_doc_id':'b','rank':1,'score':0.5000001}]}
    assert equivalent_results(baseline,candidate,'similarity')
    candidate['matches'][0]['similar_doc_id']='c'
    assert not equivalent_results(baseline,candidate,'similarity')


def test_similarity_validation_rejects_duplicate_output_rows():
    reference={'matches':[{'doc_id':'a','similar_doc_id':'b','rank':1,'score':0.5}]}
    candidate={'matches':reference['matches'] * 2}
    assert not equivalent_results(reference,candidate,'similarity')


def test_cluster_equivalence_checks_inertia():
    a={'metadata':{'inertia':2.0},'assignments':[{'doc_id':'a','cluster_id':0},{'doc_id':'b','cluster_id':1}]}
    b={'metadata':{'inertia':2.0000001},'assignments':[{'doc_id':'a','cluster_id':4},{'doc_id':'b','cluster_id':3}]}
    assert equivalent_results(a,b,'clustering')
    b['metadata']['inertia']=30
    assert not equivalent_results(a,b,'clustering')


def test_performance_calculations_without_fake_timings():
    assert work_units('similarity',100)==(9900,'directed document comparisons')
    assert work_units('clustering',100,iterations=5,clusters=3)[0]==1500
    assert rate(200,2)==100
    with pytest.raises(ValueError): rate(200,0)
    sample=[{'correct':True,'sequential_wall_seconds':4.0,'mpi_end_to_end_wall_seconds':2.0},
            {'correct':True,'sequential_wall_seconds':3.0,'mpi_end_to_end_wall_seconds':1.5}]
    results=benchmark_summary(sample,2,100,'similarity')
    assert results['speedup']==pytest.approx(2)
    assert results['efficiency_percent']==pytest.approx(100)
    assert results['mpi_docs_per_second']==pytest.approx(100/1.75)
    sample[1]['correct']=False
    with pytest.raises(ValueError,match='correctness'):benchmark_summary(sample,2,100,'similarity')


def test_audit_reads_actual_documents_and_identifies_synthetic(monkeypatch,tmp_path):
    monkeypatch.setenv('MARKETINTEL_DATA_DIR',str(tmp_path))
    write_json(tmp_path/'documents.json',[
        {'doc_id':'A','filename':'synthetic_energy.csv','industry':'Unspecified','file_type':'.csv'},
        {'doc_id':'B','filename':'actual.pdf','industry':'Energy','file_type':'.pdf'},
    ])
    write_json(tmp_path/'chunks.json',[
        {'doc_id':'A','source':'row 2','text':'battery'},
        {'doc_id':'B','source':'page 1','text':'solar'},
    ])
    info=quality_report()
    assert info['documents']==2 and info['expected_directed_pairs']==2
    assert info['synthetic_document_count']==1
    assert info['missing_source_passages']==0
    assert info['estimated_dense_matrix_mib']>=0
    markdown, fig=workspace_quality()
    assert '2 indexed files' in markdown
    assert fig is not None
    assert '2 directed non-self similarities' in compute_preview()


def test_practice_collection_is_explicitly_synthetic(monkeypatch,tmp_path):
    monkeypatch.setenv('MARKETINTEL_DATA_DIR',str(tmp_path))
    note, table=import_practice_collection()
    assert 'SYNTHETIC DEMO ONLY' in note
    assert len(table)==9
    assert quality_report()['synthetic_document_count']==9


def test_synthetic_industry_label_is_not_missed(monkeypatch, tmp_path):
    monkeypatch.setenv('MARKETINTEL_DATA_DIR', str(tmp_path))
    write_json(tmp_path/'documents.json', [
        {'doc_id':'A', 'filename':'report.csv', 'industry':'SYNTHETIC / Energy', 'file_type':'.csv'},
    ])
    assert quality_report()['synthetic_document_count'] == 1


def test_scaling_study_parser_and_generation_are_deterministic(tmp_path):
    assert parse_positive_csv('4,2,4') == [2,4]
    with pytest.raises(ValueError): parse_positive_csv('-1,400')
    first=tmp_path/'one.npz';second=tmp_path/'two.npz'
    make_synthetic_vectors(10,24,42,first)
    make_synthetic_vectors(10,24,42,second)
    a,ma=load_dataset(first);b,mb=load_dataset(second)
    assert a==b and np.array_equal(ma,mb)


def test_scaling_refuses_missing_real_mpi(monkeypatch,tmp_path):
    import scripts.scaling_study as scaling
    monkeypatch.setattr(scaling.shutil,'which',lambda name: None)
    with pytest.raises(RuntimeError,match='MPI launcher missing'):
        run_study([100],[2],['similarity'],1,12,tmp_path)
    assert list(tmp_path.iterdir())==[]


def test_fake_mpi_rank_claim_with_wrong_results_is_rejected(monkeypatch,tmp_path):
    """A contract test: a subprocess *claim* must not bypass correctness checks.

    This is deliberately not described as proof of a real MPI runtime.
    """
    import importlib.util
    from marketintel import jobs
    from marketintel.analysis import sequential_similarity
    from marketintel.storage import read_json

    dataset = tmp_path/'dataset.npz'
    ids=np.array(['a','b','c'])
    mat=np.array([[1,0],[0,1],[1,1]],dtype=np.float32)
    np.savez_compressed(dataset,doc_ids=ids,embeddings=mat)
    monkeypatch.setenv('MARKETINTEL_DATA_DIR',str(tmp_path/'state'))
    monkeypatch.setattr(jobs.shutil,'which',lambda tool: '/fake/mpiexec')
    monkeypatch.setattr(importlib.util,'find_spec',lambda name: object())

    def produce_invalid_result(cmd,**kwargs):
        import json
        from marketintel.storage import write_json
        target=Path(cmd[cmd.index('--output')+1])
        output=sequential_similarity(ids.tolist(),mat,k=2)
        output['metadata'].update(mode='mpi-real',mpi_processes=2,
            rank_allocation=[{'rank':0,'hostname':'fake-a'},{'rank':1,'hostname':'fake-b'}])
        output['matches'][0]['score']=999.0
        write_json(target,output)
        class Finished:
            returncode=0;stdout='TEST STUB';stderr=''
        return Finished()

    monkeypatch.setattr(jobs.subprocess,'run',produce_invalid_result)
    with pytest.raises(RuntimeError,match='disagrees with the sequential reference'):
        jobs.run_local_mpi('similarity',2,top_k=2,dataset_path=dataset)
    entries=list((tmp_path/'state'/'jobs').glob('*.json'))
    assert len(entries)==1
    status=read_json(entries[0],{})
    assert status['status']=='FAILED'
    assert status.get('correctness_verified') is not True
