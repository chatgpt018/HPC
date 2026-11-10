"""Run real sequential vs launched MPI benchmarks, saving honest measurements.

Use after installing Open MPI and mpi4py in WSL2/Linux.
No MPI installation means no fake MPI results; exits with a clear error.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from marketintel.analysis import sequential_similarity, sequential_kmeans
from marketintel.storage import write_json
from marketintel.provenance import vector_fingerprint
from marketintel.vectors import load_dataset


from marketintel.correctness import equivalent_results
from marketintel.performance import work_units, rate, benchmark_summary


def validate(baseline: dict, parallel: dict, operation: str) -> bool:
    """Memory-safe, full-output validation against sequential reference."""
    return equivalent_results(baseline, parallel, operation)


def _peak_rss_kib() -> int:
    """Return an advisory process RSS high-water mark (KiB); 0 if unsupported."""
    try:
        import resource
        import sys
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value // 1024 if sys.platform == 'darwin' else value
    except ImportError:
        return 0


def benchmark(input_path: Path, operation: str, world: int, repetitions: int, count: int,
              output_folder: Path) -> dict:
    if not 2 <= world <= 64 or repetitions < 1 or count < 1:
        raise ValueError('Require 2-64 MPI processes, positive repetitions, and positive analysis count')
    launcher=shutil.which('mpiexec') or shutil.which('mpirun')
    if not launcher:
        raise RuntimeError('Real MPI launcher not found. Install Open MPI in WSL2/Ubuntu first.')
    ids,vectors=load_dataset(input_path)
    if operation=='clustering' and count>len(ids):
        raise ValueError('Cluster count exceeds number of documents')
    output_folder.mkdir(parents=True,exist_ok=True)
    runs=[]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    for i in range(repetitions):
        t0=time.perf_counter()
        ids,vectors=load_dataset(input_path)  # Include load/I/O as MPI does.
        t_compute=time.perf_counter()
        with threadpool_limits(limits=1):
            baseline=(sequential_similarity(ids,vectors,k=count) if operation=='similarity'
                      else sequential_kmeans(ids,vectors,k=count))
        sequential_compute=time.perf_counter()-t_compute
        # Fair end-to-end comparison: the MPI command writes a result JSON,
        # so the sequential baseline must persist an equivalent result too.
        write_json(output_folder/f'{operation}_sequential_run{i}.json', baseline)
        sequential_total=time.perf_counter()-t0
        target=output_folder/f'{operation}_mpi_{world}p_run{i}.json'
        cmd=[launcher,'-n',str(world),sys.executable,'-m','marketintel.mpi_tasks',operation,
             '--input',str(input_path.resolve()),'--output',str(target.resolve()),
             '--expected-processes',str(world),'--top-k',str(count),'--k',str(count)]
        t1=time.perf_counter()
        completed=subprocess.run(cmd,capture_output=True,text=True,env=env,timeout=600,
                                 cwd=Path(__file__).resolve().parents[1])
        mpi_launch_total=time.perf_counter()-t1
        if completed.returncode!=0 or not target.exists():
            raise RuntimeError(f'MPI execution failed: {completed.stderr[-1200:]}')
        mpi=json.loads(target.read_text(encoding='utf-8'))
        ranks=mpi.get('metadata',{}).get('rank_allocation',[])
        if (mpi.get('metadata',{}).get('mode')!='mpi-real' or
                mpi['metadata'].get('mpi_processes')!=world or
                sorted(x.get('rank') for x in ranks)!=list(range(world))):
            raise AssertionError('Invalid actual MPI rank-allocation evidence')
        correct=validate(baseline,mpi,operation)
        if not correct:
            raise AssertionError('Sequential and genuine MPI outputs do not match; benchmark invalid')
        runs.append({'run':i+1,'sequential_wall_seconds':sequential_total,
                     'mpi_end_to_end_wall_seconds':mpi_launch_total,
                     'sequential_compute_seconds':sequential_compute,
                     'mpi_reported_compute_seconds':mpi['metadata']['runtime_seconds'],
                     'mpi_rank_cpu_seconds':sum(float(x.get('cpu_seconds', 0)) for x in ranks),
                     'mpi_rank_peak_rss_kib':max((int(x.get('peak_rss_kib', 0)) for x in ranks),default=0),
                     'sequential_peak_rss_kib':_peak_rss_kib(),
                     'speedup_end_to_end':sequential_total/mpi_launch_total,
                     'parallel_efficiency_percent':100*sequential_total/(mpi_launch_total*world),
                     'correct':correct,'mpi_stdout':completed.stdout[-1500:],
                     'ranks':mpi['metadata'].get('rank_allocation',[])})
    seq_median=statistics.median(r['sequential_wall_seconds'] for r in runs)
    mpi_median=statistics.median(r['mpi_end_to_end_wall_seconds'] for r in runs)
    extra = benchmark_summary(runs, world, len(ids), operation)
    units, unit_label = work_units(operation, len(ids),
        iterations=mpi['metadata'].get('iterations'), clusters=count)
    record={'operation':operation,'documents':len(ids),'dimensions':int(vectors.shape[1]),
            'dataset_kind':('SYNTHETIC NUMERICAL BENCHMARK INPUT'
                            if ids and all(d.startswith('SYNTHETIC-') for d in ids)
                            else 'DOCUMENT-DERIVED OR EXTERNAL INPUT — VERIFY SOURCE'),
            'input_sha256':vector_fingerprint(ids,vectors),
            'created_utc':datetime.now(timezone.utc).isoformat(),
            'environment':{'platform':platform.platform(),'python':platform.python_version(),
                           'cpu_count':os.cpu_count(),'blas_threads':1},
            'median_sequential_wall_seconds':seq_median,
            'median_mpi_wall_seconds':mpi_median,
            'median_speedup':seq_median/mpi_median,
            'median_parallel_efficiency_percent':100*seq_median/(mpi_median*world),
            'mpi_processes':world,'repetitions':repetitions,'parameter':count,
            'measurement':'end-to-end wall time including process startup',
            'note':'MPI speedup is not guaranteed; no simulated values',
            'work_units':units,'work_unit_label':unit_label,
            'median_mpi_work_units_per_second':rate(units,mpi_median),
            'median_sequential_work_units_per_second':rate(units,seq_median),
            'summary':extra, 'runs':runs}
    write_json(output_folder/f'{operation}_benchmark_{world}p.json',record)
    return record


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--operation',choices=['similarity','clustering'],default='similarity')
    p.add_argument('--processes',type=int,default=2)
    p.add_argument('--repetitions',type=int,default=3)
    p.add_argument('--count',type=int,default=3,help='Top-k or number of clusters')
    p.add_argument('--out',type=Path,default=Path('storage/benchmarks'))
    args=p.parse_args()
    print(json.dumps(benchmark(args.input,args.operation,args.processes,args.repetitions,args.count,args.out),indent=2))

if __name__=='__main__':
    main()
