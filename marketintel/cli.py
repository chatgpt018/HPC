"""CLI for local end-to-end demonstrations and testing."""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from .analysis import sequential_kmeans, sequential_similarity
from .documents import ingest_file
from .rag import answer
from .storage import home,write_json
from .vectors import load_dataset, save_dataset
from .provenance import attach_provenance, research_brief


def main():
    parser=argparse.ArgumentParser(description='Local market-intelligence CLI (no AWS calls)')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('ingest');p.add_argument('files',nargs='+',type=Path);p.add_argument('--industry',default='Unspecified')
    p=sub.add_parser('ask');p.add_argument('question');p.add_argument('--ollama',action='store_true')
    p=sub.add_parser('export');p.add_argument('--output',type=Path,default=Path('storage/datasets/uploaded_reports.npz'))
    p.add_argument('--mode',choices=['hashing','semantic'],default='hashing')
    p=sub.add_parser('analyze');p.add_argument('operation',choices=['similarity','clustering']);p.add_argument('--count',type=int,default=3)
    p.add_argument('--mode',choices=['hashing','semantic'],default='hashing')
    args=parser.parse_args()
    if args.command=='ingest':
        out=[ingest_file(p,args.industry) for p in args.files]
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
