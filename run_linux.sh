#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="$project_root/.venv/bin/python"
[[ -x "$python" ]] || { echo 'Project environment not found. Run ./setup_linux.sh first.' >&2; exit 1; }
cd "$project_root"
exec "$python" app.py
