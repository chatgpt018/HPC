#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"
with_mpi=0
wheelhouse=''
while (($#)); do
  case "$1" in
    --with-mpi) with_mpi=1; shift ;;
    --wheelhouse) [[ $# -ge 2 ]] || { echo 'Missing folder after --wheelhouse' >&2; exit 2; }; wheelhouse="$2"; shift 2 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

python_cmd="${PYTHON:-python3}"
command -v "$python_cmd" >/dev/null || { echo 'Python 3.11–3.13 is required.' >&2; exit 1; }
"$python_cmd" -c 'import sys; assert (3, 11) <= sys.version_info[:2] < (3, 14), "Python 3.11–3.13 is required"'
if ((with_mpi)) && ! command -v mpiexec >/dev/null && ! command -v mpirun >/dev/null; then
  echo 'Real MPI needs Open MPI (or compatible runtime) and its development headers. Install those separately, then retry.' >&2
  exit 1
fi

"$python_cmd" -m venv .venv
venv_python="$project_root/.venv/bin/python"
extra='dev'
if ((with_mpi)); then extra='dev,hpc'; fi
if [[ -n "$wheelhouse" ]]; then
  [[ -d "$wheelhouse" ]] || { echo "Wheelhouse folder not found: $wheelhouse" >&2; exit 1; }
  "$venv_python" -m pip install --no-index --find-links "$wheelhouse" "market-intelligence-hpc-local[$extra]"
else
  "$venv_python" -m pip install -e ".[${extra}]"
fi
"$venv_python" scripts/doctor.py
echo 'Setup complete. Run bash run_linux.sh from this project folder.'
