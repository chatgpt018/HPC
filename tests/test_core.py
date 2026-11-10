from pathlib import Path
import numpy as np
import pytest

from marketintel.analysis import normalize, topk_rows, sequential_similarity, sequential_kmeans
from marketintel.mpi_tasks import partitions
from marketintel.documents import ingest_file,chunk_units
from marketintel.vectors import save_dataset,load_dataset,retrieve
from marketintel.rag import answer


def test_partitions():
    for n, p in [(1,4),(7,4),(10,3),(0,4)]:
        parts=partitions(n,p)
        assert len(parts)==p
        assert parts[0][0]==0 and parts[-1][1]==n
        assert all(parts[i][1]==parts[i+1][0] for i in range(p-1))


def test_topk_tie_boundary():
    ids=['a','b','c','d']
    vectors=normalize(np.array([[1,0],[1,0],[1,0],[0,1]],dtype=float))
    found=topk_rows(ids,vectors,0,4,k=1)
    assert [r for r in found if r['doc_id']=='a'][0]['similar_doc_id']=='b'
    assert [r for r in found if r['doc_id']=='b'][0]['similar_doc_id']=='a'


def test_topk_zero_vectors():
    ids=['a','b','c']
    mat=np.asarray([[0,0],[1,0],[0,1]],dtype=float)
    result=sequential_similarity(ids,mat,k=2)
    assert len(result['matches'])==6
    assert np.isfinite([x['score'] for x in result['matches']]).all()


def test_kmeans_deterministic():
    ids=[f'd{i}' for i in range(8)]
    vectors=np.asarray([[1+i*.01,0] for i in range(4)]+[[0,1+i*.01] for i in range(4)])
    a=sequential_kmeans(ids,vectors,k=2)
    b=sequential_kmeans(ids,vectors,k=2)
    assert a['assignments']==b['assignments']
    assert a['metadata']['inertia']==pytest.approx(b['metadata']['inertia'])


def test_chunk_source():
    result=chunk_units([('page 12','one two three four five')],window=3,overlap=1)
    assert result[0]['source']=='page 12'
    assert result[1]['start_word']==2


def test_upload_to_hpc_dataset_and_retrieval(monkeypatch,tmp_path):
    monkeypatch.setenv('MARKETINTEL_DATA_DIR',str(tmp_path/'data'))
    for name,text in [('solar.csv','solar,renewable,photovoltaic'),('battery.csv','battery,lithium,electric vehicle')]:
        path=tmp_path/name
        path.write_text('topic,keywords\n'+text+'\n',encoding='utf-8')
        out=ingest_file(path,industry='Test')
        assert out['new_chunks']>0
    assert retrieve('solar photovoltaic')
    dataset=save_dataset(tmp_path/'vectors.npz')
    ids,v=load_dataset(tmp_path/'vectors.npz')
    assert dataset['documents']==2 and v.shape==(2,384)
    assert all(len(x)==16 for x in ids)
    assert answer('solar photovoltaic')['generated'] is False


def test_safe_npz_no_pickle(tmp_path):
    path=tmp_path/'bad.npz'
    np.savez(path,doc_ids=np.asarray(['a'],dtype=object),embeddings=np.ones((1,3)))
    with pytest.raises(ValueError):
        load_dataset(path)


def test_bad_extension(monkeypatch,tmp_path):
    monkeypatch.setenv('MARKETINTEL_DATA_DIR',str(tmp_path/'data'))
    path=tmp_path/'file.exe'
    path.write_text('bad')
    with pytest.raises(ValueError):
        ingest_file(path)
