"""Safe subprocess-based local MPI runner, with durable job records."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .storage import home, read_json, write_json
from .vectors import save_dataset, load_dataset
from .provenance import attach_provenance
from .correctness import equivalent_results
from .analysis import sequential_similarity, sequential_kmeans


def mpi_available() -> bool:
    return bool(shutil.which('mpiexec') or shutil.which('mpirun'))


def run_local_mpi(operation: str, processes: int, top_k: int = 5, k: int = 3,
                  timeout_seconds: int = 300, embedding_mode: str = 'hashing',
                  dataset_path: Path | None = None) -> dict:
    if operation not in {'similarity','clustering'}:
        raise ValueError('Unsupported analysis operation')
    if processes < 2 or processes > 8:
        raise ValueError('Choose between 2 and 8 MPI processes')
    launcher = shutil.which('mpiexec') or shutil.which('mpirun')
    if not launcher:
        raise RuntimeError('Real MPI runtime not found. Use WSL2/Ubuntu with Open MPI, or run the sequential mode.')
    import importlib.util
    if importlib.util.find_spec('mpi4py') is None:
        raise RuntimeError('mpi4py is not installed in the current Python environment')
    data_path = Path(dataset_path) if dataset_path is not None else home() / 'datasets' / 'uploaded_reports.npz'
    if dataset_path is None:
        save_dataset(data_path, embedding_mode=embedding_mode)
    ids, vectors = load_dataset(data_path)
    if len(ids) < 2:
        raise ValueError('Upload at least two documents for MPI analysis')
    run_id = uuid.uuid4().hex[:12]
    output = home() / 'results' / f'{run_id}.json'
    entry = {'run_id':run_id,'status':'RUNNING','operation':operation,'mode':'mpi-real',
             'processes_requested':processes,'started_utc':datetime.now(timezone.utc).isoformat()}
    job_path = home() / 'jobs' / f'{run_id}.json'
    write_json(job_path,entry)
    cmd = [launcher, '-n', str(processes), sys.executable, '-m', 'marketintel.mpi_tasks',
           operation,'--input',str(data_path),'--output',str(output),
           '--expected-processes',str(processes),'--top-k',str(top_k),'--k',str(k)]
    env = dict(os.environ)
    env['OPENBLAS_NUM_THREADS']='1'
    env['OMP_NUM_THREADS']='1'
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds, env=env,
                                   cwd=Path(__file__).resolve().parent.parent, check=False)
        entry['return_code'] = completed.returncode
        entry['stdout'] = completed.stdout[-4000:]
        entry['stderr'] = completed.stderr[-4000:]
        entry['status'] = 'COMPLETED' if completed.returncode == 0 and output.is_file() else 'FAILED'
        if entry['status'] == 'COMPLETED':
            result = read_json(output, {})
            meta = result.get('metadata', {})
            ranks = meta.get('rank_allocation', [])
            actual = meta.get('mpi_processes')
            if (meta.get('mode') != 'mpi-real' or actual != processes or
                    len(ranks) != processes or
                    sorted(x.get('rank') for x in ranks) != list(range(processes))):
                entry['status'] = 'FAILED'
                entry['stderr'] = 'MPI output did not prove the expected rank allocation'
            else:
                # Independently recalculate the same workload; never trust only
                # the MPI job's self-reported success or process count.
                from threadpoolctl import threadpool_limits
                with threadpool_limits(limits=1):
                    reference = (sequential_similarity(ids, vectors, k=top_k)
                        if operation == 'similarity' else sequential_kmeans(ids, vectors, k=k))
                if not equivalent_results(reference, result, operation):
                    entry['status'] = 'FAILED'
                    entry['stderr'] = 'MPI computation disagrees with the sequential reference.'
                else:
                    entry['correctness_verified'] = True
                    result['metadata']['correctness_verified_against_sequential'] = True
                    attach_provenance(result, ids, vectors,
                                  'sentence-transformer-local' if embedding_mode == 'semantic'
                                      else 'hashed-lexical-features')
                    write_json(output, result)
                    entry['actual_ranks'] = ranks
                    entry['result_path'] = str(output)
    except subprocess.TimeoutExpired:
        entry['status']='FAILED'
        entry['stderr']='MPI execution timed out.'
    except OSError as exc:
        entry['status']='FAILED'
        entry['stderr']=f'MPI launcher error: {exc}'
    finally:
        entry['finished_utc']=datetime.now(timezone.utc).isoformat()
        write_json(job_path,entry)
    if entry['status'] != 'COMPLETED':
        raise RuntimeError(f"Real MPI job failed ({run_id}): {entry.get('stderr','No result produced')}")
    return entry


def saved_jobs() -> list[dict]:
    jobs = [read_json(p,{}) for p in (home()/'jobs').glob('*.json')]
    return sorted((j for j in jobs if j),key=lambda x:x.get('started_utc',''),reverse=True)
