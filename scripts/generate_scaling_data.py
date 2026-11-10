"""Reproducible SYNTHETIC vectors solely for algorithm scaling, not market evidence."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description='Synthetic vectors for MPI benchmark only')
    p.add_argument('--documents',type=int,default=1000)
    p.add_argument('--dimensions',type=int,default=384)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',type=Path,default=Path('storage/datasets/synthetic_scaling.npz'))
    a=p.parse_args()
    if not 2<=a.documents<=20000 or not 2<=a.dimensions<=2048:
        raise ValueError('Out-of-range size; choose 2-20,000 docs and 2-2,048 dimensions')
    rng=np.random.default_rng(a.seed)
    vectors=rng.normal(0,1,size=(a.documents,a.dimensions)).astype(np.float32)
    ids=np.asarray([f'SYNTHETIC-{i:06d}' for i in range(a.documents)],dtype=str)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output,doc_ids=ids,embeddings=vectors)
    print(f'Created {a.documents}x{a.dimensions} SYNTHETIC benchmark-only matrix at {a.output} (seed={a.seed})')

if __name__=='__main__':
    main()
