"""Offline contract checks for the optional local embedding index."""
from __future__ import annotations

import numpy as np
import pytest

from marketintel.documents import ingest_file
from marketintel.vectors import retrieve


class LocalFixtureModel:
    def __init__(self):
        self.batch_sizes = []

    def encode(self, texts, normalize_embeddings=True):
        self.batch_sizes.append(len(texts))
        vectors = np.asarray([[float('solar' in text.lower()),
                               float('battery' in text.lower())] for text in texts],
                             dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.where(norms > 0, norms, 1)


def test_semantic_index_reuses_and_incrementally_adds_vectors(monkeypatch, tmp_path):
    monkeypatch.setenv('MARKETINTEL_DATA_DIR', str(tmp_path / 'store'))
    monkeypatch.setenv('MARKETINTEL_RETRIEVAL_MODE', 'semantic')
    model_dir = tmp_path / 'local-model'
    model_dir.mkdir()
    monkeypatch.setenv('MARKETINTEL_LOCAL_EMBED_MODEL', str(model_dir))
    model = LocalFixtureModel()
    monkeypatch.setattr('marketintel.vectors._local_sentence_model', lambda: model)
    for name, text, industry in [('solar.csv', 'solar panels', 'Energy'),
                                 ('battery.csv', 'battery cells', 'Transport')]:
        path = tmp_path / name
        path.write_text('topic\n' + text + '\n', encoding='utf-8')
        ingest_file(path, industry)
    hits = retrieve('solar', industry='Energy')
    assert len(hits) == 1 and hits[0]['filename'] == 'solar.csv'
    assert retrieve('solar', document_id=hits[0]['doc_id'])[0]['chunk_id'] == hits[0]['chunk_id']
    assert model.batch_sizes == [2, 1, 1]

    path = tmp_path / 'other.csv'
    path.write_text('topic\nsolar generation\n', encoding='utf-8')
    ingest_file(path, 'Energy')
    assert len(retrieve('solar', industry='Energy')) == 2
    assert model.batch_sizes[-2:] == [1, 1]
    with np.load(tmp_path / 'store' / 'datasets' / 'semantic_index.npz', allow_pickle=False) as index:
        assert int(index['index_version']) == 1
        assert index['vectors'].shape == (3, 2)
        assert len(str(index['dataset_fingerprint'])) == 64


def test_semantic_mode_fails_clearly_without_model(monkeypatch, tmp_path):
    monkeypatch.setenv('MARKETINTEL_DATA_DIR', str(tmp_path / 'store'))
    path = tmp_path / 'report.csv'
    path.write_text('topic\nsolar panels\n', encoding='utf-8')
    ingest_file(path)
    monkeypatch.setenv('MARKETINTEL_RETRIEVAL_MODE', 'semantic')
    monkeypatch.delenv('MARKETINTEL_LOCAL_EMBED_MODEL', raising=False)
    with pytest.raises(RuntimeError, match='sentence-transformers|MARKETINTEL_LOCAL_EMBED_MODEL'):
        retrieve('solar')
    monkeypatch.setenv('MARKETINTEL_RETRIEVAL_MODE', 'typo')
    with pytest.raises(ValueError, match='tfidf or semantic'):
        retrieve('solar')


def test_manual_retrieval_evaluation_uses_indexed_source_labels(monkeypatch, tmp_path):
    from scripts.evaluate_retrieval import evaluate
    monkeypatch.setenv('MARKETINTEL_DATA_DIR', str(tmp_path / 'store'))
    monkeypatch.setenv('MARKETINTEL_RETRIEVAL_MODE', 'tfidf')
    path = tmp_path / 'fixture.csv'
    path.write_text('topic\nphotovoltaic silicon wafers\n', encoding='utf-8')
    document = ingest_file(path)['document']
    questions = tmp_path / 'questions.csv'
    questions.write_text('question,question_type,expected_document_id,expected_source\n'
                         f'Which text mentions photovoltaic wafers?,factual,{document["doc_id"]},row 2\n'
                         'Is Mars mentioned?,insufficient-evidence,,\n', encoding='utf-8')
    result = evaluate(questions, top_k=1)
    assert result['questions'] == 2
    assert result['hit_at_k'] == 1.0
    assert result['no_evidence_accuracy'] == 1.0
    assert len(result['dataset_fingerprint']) == 64
    questions.write_text('question,question_type,expected_document_id,expected_source\n'
                         'Wrong source,factual,not-indexed,page 9\n', encoding='utf-8')
    with pytest.raises(ValueError, match='unindexed source'):
        evaluate(questions)
