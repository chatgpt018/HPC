"""CLI for local end-to-end demonstrations and testing."""
from __future__ import annotations

import argparse
import csv
import json
import uuid
from pathlib import Path

from .analysis import sequential_kmeans, sequential_similarity
from .documents import ALLOWED, ingest_file, validate_source_metadata
from .rag import answer
from .storage import home,write_json
from .vectors import load_dataset, save_dataset
from .provenance import attach_provenance, research_brief


def ingest_manifest(manifest: Path, source_dir: Path) -> list[dict]:
    """Import a checked attribution manifest and its local, permitted reports."""
    required = {'filename', 'industry', 'publisher', 'publication_date',
                'source_type', 'usage_notes', 'source_kind'}
    with manifest.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f'Manifest needs columns: {", ".join(sorted(required))}')
        prepared = []
        for row_number, row in enumerate(reader, 2):
            filename = (row.get('filename') or '').strip()
            if not filename or Path(filename).name != filename or Path(filename).suffix.lower() not in ALLOWED:
                raise ValueError(f'Manifest row {row_number}: use a PDF/DOCX/CSV filename without directories')
            path = source_dir / filename
            if not path.is_file() or not path.resolve().is_relative_to(source_dir.resolve()):
                raise ValueError(f'Manifest row {row_number}: source file missing: {filename}')
            metadata = {key: row[key] or '' for key in row if key not in {'filename', 'industry'}
                        and key in {'publisher', 'source_url', 'publication_date',
                                    'reporting_period', 'geographic_scope', 'source_type',
                                    'usage_notes', 'source_kind'}}
            try:
                metadata = validate_source_metadata(metadata)
            except ValueError as exc:
                raise ValueError(f'Manifest row {row_number}: {exc}') from exc
            if metadata['source_kind'] == 'unspecified':
                raise ValueError(f'Manifest row {row_number}: source_kind must be real or synthetic')
            prepared.append((path, (row.get('industry') or '').strip() or 'Unspecified', metadata))
    return [ingest_file(path, industry, metadata) for path, industry, metadata in prepared]


def main():
    parser=argparse.ArgumentParser(description='Local market-intelligence CLI (no AWS calls)')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('ingest');p.add_argument('files',nargs='+',type=Path);p.add_argument('--industry',default='Unspecified')
    p=sub.add_parser('ingest-manifest');p.add_argument('manifest',type=Path)
    p.add_argument('--source-dir',type=Path,default=Path('data/raw/reports'))
    p=sub.add_parser('ask');p.add_argument('question');p.add_argument('--ollama',action='store_true')
    p=sub.add_parser('export');p.add_argument('--output',type=Path,default=Path('storage/datasets/uploaded_reports.npz'))
    p.add_argument('--mode',choices=['hashing','semantic'],default='hashing')
    p=sub.add_parser('analyze');p.add_argument('operation',choices=['similarity','clustering']);p.add_argument('--count',type=int,default=3)
    p.add_argument('--mode',choices=['hashing','semantic'],default='hashing')
    args=parser.parse_args()
    if args.command=='ingest':
        out=[ingest_file(p,args.industry) for p in args.files]
    elif args.command=='ingest-manifest':
        out=ingest_manifest(args.manifest,args.source_dir)
    elif args.command=='ask':
        out=answer(args.question,generate=args.ollama)
    elif args.command=='export':
        out=save_dataset(args.output,args.mode)
    elif args.command=='analyze':
        dest=home()/'datasets'/'uploaded_reports.npz'
        dataset = save_dataset(dest,args.mode)
        ids,matrix=load_dataset(dest)
        out=(sequential_similarity(ids,matrix,k=args.count)
             if args.operation=='similarity' else sequential_kmeans(ids,matrix,k=args.count))
        out = attach_provenance(out, ids, matrix, dataset['vector_type'])
        target=home()/'results'/f'{args.operation}_{uuid.uuid4().hex[:8]}.json'
        write_json(target,out)
        target.with_suffix('.md').write_text(research_brief(out), encoding='utf-8')
        print(f'Saved actual sequential result: {target}')
    else:
        raise RuntimeError('Unknown command')
    print(json.dumps(out,indent=2,default=str))

if __name__=='__main__':
    main()
