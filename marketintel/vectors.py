"""Local retrieval and reproducible real document-derived numerical features.

Default is honest lexical TF-IDF for retrieval and stable hashing features for HPC;
these are NOT transformer semantic embeddings. Optional sentence-transformers
can produce real neural embeddings when a model is installed locally.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .storage import chunks, home, project_path


@lru_cache(maxsize=2)
def _load_sentence_model(model_id: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_id, local_files_only=True)


def _local_sentence_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError('Install sentence-transformers to enable semantic retrieval') from exc
    model_id = os.environ.get('MARKETINTEL_LOCAL_EMBED_MODEL')
    if not model_id or not project_path(model_id).is_dir():
        raise RuntimeError('MARKETINTEL_LOCAL_EMBED_MODEL must be an existing local model directory')
    return _load_sentence_model(str(project_path(model_id)))


@lru_cache(maxsize=2)
def _tfidf_corpus(texts: tuple[str, ...]):
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    return vectorizer, vectorizer.fit_transform(texts)


def retrieve(question: str, top_k: int = 4) -> list[dict]:
    units = chunks()
    if top_k < 1:
        raise ValueError('top_k must be positive')
    if not units or not question.strip():
        return []
    texts = [u['text'] for u in units]
    if os.environ.get('MARKETINTEL_RETRIEVAL_MODE', 'tfidf').lower() == 'semantic':
        model = _local_sentence_model()
        vectors = np.asarray(model.encode(texts, normalize_embeddings=True),dtype=np.float32)
        query = np.asarray(model.encode([question], normalize_embeddings=True),dtype=np.float32)
        scores = (query @ vectors.T).ravel()
    else:
        try:
            vectorizer, corpus_matrix = _tfidf_corpus(tuple(texts))
            query_matrix = vectorizer.transform([question])
            scores = cosine_similarity(query_matrix, corpus_matrix).ravel()
        except ValueError:  # A corpus containing only stop words has no vocabulary.
            return []
    indices = sorted(range(len(units)), key=lambda i: (-scores[i], units[i]['chunk_id']))[:top_k]
    return [{**units[i], 'score': float(scores[i])} for i in indices if scores[i] > 0]


def document_vectors(embedding_mode: str = 'hashing', dimension: int = 384) -> tuple[list[str], np.ndarray, list[dict]]:
    """Aggregate every document's uploaded chunk text into numerical vectors."""
    units = chunks()
    doc_ids = sorted(set(u['doc_id'] for u in units))
    if not doc_ids:
        raise ValueError('Upload and index documents first')
    by_doc = {d: ' '.join(u['text'] for u in units if u['doc_id'] == d) for d in doc_ids}
    texts = [by_doc[d] for d in doc_ids]
    if embedding_mode == 'semantic':
        vectors = np.asarray(_local_sentence_model().encode(texts), dtype=np.float32)
        kind = 'sentence-transformer-local'
    elif embedding_mode == 'hashing':
        vectorizer = HashingVectorizer(n_features=dimension, norm='l2', alternate_sign=False, stop_words='english')
        vectors = vectorizer.transform(texts).toarray().astype(np.float32)
        kind = 'hashed-lexical-features'
    else:
        raise ValueError('Embedding mode must be hashing or semantic')
    metadata = [{'doc_id': doc_id, 'filename': next(u['filename'] for u in units if u['doc_id'] == doc_id),
                 'industry': next(u['industry'] for u in units if u['doc_id'] == doc_id)}
                for doc_id in doc_ids]
    return doc_ids, vectors, [{**m, 'vector_type': kind} for m in metadata]


def save_dataset(destination: str | Path, embedding_mode: str = 'hashing') -> dict:
    path = Path(destination)
    ids, vectors, metadata = document_vectors(embedding_mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, doc_ids=np.asarray(ids, dtype=str), embeddings=vectors)
    path.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return {'path': str(path.resolve()), 'documents': len(ids), 'dimensions': int(vectors.shape[1]),
            'vector_type': metadata[0]['vector_type']}


def load_dataset(path: str | Path) -> tuple[list[str], np.ndarray]:
    """Load only safe numeric/string arrays; no pickle deserialization."""
    with np.load(path, allow_pickle=False) as content:
        ids = [str(d) for d in content['doc_ids'].tolist()]
        matrix = np.asarray(content['embeddings'], dtype=np.float32)
    if matrix.ndim != 2 or len(ids) != matrix.shape[0] or len(set(ids)) != len(ids):
        raise ValueError('Invalid dataset dimensions or duplicate document IDs')
    if matrix.shape[1] == 0 or not np.all(np.isfinite(matrix)):
        raise ValueError('Invalid features: empty dimension or non-finite values')
    return ids, matrix
