"""Local retrieval and reproducible real document-derived numerical features.

Default is honest lexical TF-IDF for retrieval and stable hashing features for HPC;
these are NOT transformer semantic embeddings. Optional sentence-transformers
can produce real neural embeddings when a model is installed locally.
"""
from __future__ import annotations

import json
import os
import hashlib
import logging
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import BadZipFile

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .storage import chunks, home, project_path
from .documents import CHUNK_WORDS, CHUNK_OVERLAP

SEMANTIC_INDEX_VERSION = 1


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


def _semantic_corpus(units: list[dict], model, model_id: str) -> np.ndarray:
    """Reuse unchanged chunk vectors; persist a versioned, pickle-free local index."""
    index_path = home() / 'datasets' / 'semantic_index.npz'
    ids = [unit['chunk_id'] for unit in units]
    hashes = [hashlib.sha256(unit['text'].encode('utf-8')).hexdigest() for unit in units]
    fingerprint = hashlib.sha256(json.dumps(list(zip(ids, hashes)),
                                    separators=(',', ':')).encode('utf-8')).hexdigest()
    cached: dict[tuple[str, str], np.ndarray] = {}
    if index_path.is_file():
        try:
            with np.load(index_path, allow_pickle=False) as saved:
                if (int(saved['index_version']) == SEMANTIC_INDEX_VERSION and
                        str(saved['model_id']) == model_id and
                        int(saved['chunk_words']) == CHUNK_WORDS and
                        int(saved['chunk_overlap']) == CHUNK_OVERLAP):
                    vectors = np.asarray(saved['vectors'], dtype=np.float32)
                    old_ids = saved['chunk_ids'].tolist()
                    old_hashes = saved['text_hashes'].tolist()
                    if (vectors.ndim == 2 and vectors.shape[0] == len(old_ids) == len(old_hashes)
                            and vectors.shape[1] > 0 and np.all(np.isfinite(vectors))):
                        if str(saved['dataset_fingerprint']) == fingerprint and old_ids == ids:
                            return vectors
                        cached = dict(zip(zip(old_ids, old_hashes), vectors))
        except (OSError, ValueError, KeyError, BadZipFile) as exc:
            logging.warning('Rebuilding unreadable local semantic index: %s', type(exc).__name__)
    missing = [i for i, pair in enumerate(zip(ids, hashes)) if pair not in cached]
    if missing:
        fresh = np.asarray(model.encode([units[i]['text'] for i in missing],
                                        normalize_embeddings=True), dtype=np.float32)
        if fresh.ndim != 2 or fresh.shape[0] != len(missing) or not np.all(np.isfinite(fresh)):
            raise ValueError('Local embedding model returned invalid document vectors')
        cached.update({(ids[i], hashes[i]): vector for i, vector in zip(missing, fresh)})
    vectors = np.asarray([cached[pair] for pair in zip(ids, hashes)], dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] == 0:
        raise ValueError('Local embedding model returned empty document vectors')
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=index_path.parent, suffix='.npz', delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, index_version=SEMANTIC_INDEX_VERSION,
                            model_id=model_id, chunk_words=CHUNK_WORDS,
                            chunk_overlap=CHUNK_OVERLAP, dataset_fingerprint=fingerprint,
                            chunk_ids=np.asarray(ids, dtype=str),
                            text_hashes=np.asarray(hashes, dtype=str), vectors=vectors)
        temporary.replace(index_path)
    finally:
        temporary.unlink(missing_ok=True)
    return vectors


def retrieve(question: str, top_k: int = 4, document_id: str | None = None,
             industry: str | None = None) -> list[dict]:
    units = chunks()
    if not 1 <= top_k <= 20:
        raise ValueError('top_k must be between 1 and 20')
    if not units or not question.strip():
        return []
    texts = [u['text'] for u in units]
    mode = os.environ.get('MARKETINTEL_RETRIEVAL_MODE', 'tfidf').lower()
    if mode == 'semantic':
        model = _local_sentence_model()
        model_id = str(project_path(os.environ['MARKETINTEL_LOCAL_EMBED_MODEL']))
        vectors = _semantic_corpus(units, model, model_id)
        query = np.asarray(model.encode([question], normalize_embeddings=True),dtype=np.float32)
        if query.shape != (1, vectors.shape[1]) or not np.all(np.isfinite(query)):
            raise ValueError('Local embedding model returned an incompatible question vector')
        scores = (query @ vectors.T).ravel()
    elif mode == 'tfidf':
        try:
            vectorizer, corpus_matrix = _tfidf_corpus(tuple(texts))
            query_matrix = vectorizer.transform([question])
            scores = cosine_similarity(query_matrix, corpus_matrix).ravel()
        except ValueError:  # A corpus containing only stop words has no vocabulary.
            return []
    else:
        raise ValueError('MARKETINTEL_RETRIEVAL_MODE must be tfidf or semantic')
    eligible = (i for i, unit in enumerate(units)
                if (document_id is None or unit['doc_id'] == document_id)
                and (industry is None or unit['industry'] == industry))
    indices = sorted(eligible, key=lambda i: (-scores[i], units[i]['chunk_id']))[:top_k]
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
