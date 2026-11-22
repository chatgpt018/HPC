"""Evaluate retrieval against manually checked source locations; no LLM scores."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from marketintel.storage import chunks, home, write_json
from marketintel.vectors import retrieve


def evaluate(manifest: Path, top_k: int = 4) -> dict:
    if not 1 <= top_k <= 20:
        raise ValueError('top_k must be between 1 and 20')
    passages = chunks()
    known = {(p['doc_id'], p['source']) for p in passages}
    if not known:
        raise ValueError('Index the evaluation documents first')
    grouped: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    with manifest.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {'question', 'question_type', 'expected_document_id', 'expected_source'}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError('Evaluation CSV needs question, question_type, expected_document_id, expected_source')
        for row_number, row in enumerate(reader, 2):
            question = (row['question'] or '').strip()
            kind = (row['question_type'] or '').strip()
            doc_id = (row['expected_document_id'] or '').strip()
            source = (row['expected_source'] or '').strip()
            if not question or not kind or bool(doc_id) != bool(source):
                raise ValueError(f'Invalid evaluation row {row_number}')
            if (doc_id, source) not in known and doc_id:
                raise ValueError(f'Evaluation row {row_number} references an unindexed source')
            grouped[(question, kind)].update({(doc_id, source)} if doc_id else set())
    if not grouped:
        raise ValueError('Add manually verified evaluation questions before scoring')
    details = []
    positive_hits = []
    positive_recalls = []
    negative_correct = []
    for (question, kind), expected in grouped.items():
        found = retrieve(question, top_k=top_k)
        observed = {(p['doc_id'], p['source']) for p in found}
        if expected:
            hits = len(expected & observed)
            positive_hits.append(int(hits > 0))
            positive_recalls.append(hits / len(expected))
        else:
            negative_correct.append(int(not found))
        details.append({'question': question, 'question_type': kind,
                        'expected_sources': sorted(expected),
                        'retrieved_sources': [
                            {'doc_id': p['doc_id'], 'source': p['source'], 'score': p['score']}
                            for p in found], 'hit': bool(expected & observed) if expected else not found})
    fingerprint = hashlib.sha256(json.dumps(
        [(p['chunk_id'], p['text']) for p in passages], ensure_ascii=False,
        separators=(',', ':')).encode('utf-8')).hexdigest()
    return {'evaluation_type': 'manually-labelled-retrieval-only',
            'retrieval_mode': os.getenv('MARKETINTEL_RETRIEVAL_MODE', 'tfidf').lower(),
            'dataset_fingerprint': fingerprint, 'top_k': top_k,
            'questions': len(details), 'positive_questions': len(positive_hits),
            'no_evidence_questions': len(negative_correct),
            'hit_at_k': sum(positive_hits) / len(positive_hits) if positive_hits else None,
            'recall_at_k': sum(positive_recalls) / len(positive_recalls) if positive_recalls else None,
            'no_evidence_accuracy': sum(negative_correct) / len(negative_correct) if negative_correct else None,
            'details': details,
            'limitation': 'This evaluates retrieval against supplied labels, not LLM factuality or source truth.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--top-k', type=int, default=4)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    result = evaluate(args.questions, args.top_k)
    destination = args.output or home() / 'evaluations' / 'retrieval.json'
    write_json(destination, result)
    print(json.dumps({key: result[key] for key in ('retrieval_mode', 'questions', 'hit_at_k',
        'recall_at_k', 'no_evidence_accuracy', 'dataset_fingerprint')}, indent=2))


if __name__ == '__main__':
    main()
