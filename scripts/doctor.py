"""Read-only environment check; never invokes AWS or downloads models."""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import platform
import shutil
import sys
from pathlib import Path


def diagnostics() -> dict:
    required = ['numpy', 'pandas', 'gradio', 'scipy', 'sklearn', 'pymupdf', 'docx', 'plotly']
    optional = ['mpi4py', 'sentence_transformers', 'pytest']
    packages = {name: _probe_import(name) for name in required + optional if name != 'mpi4py'}
    # Import the extension itself: `import mpi4py` can succeed while MPI.pyd
    # cannot load an MPI runtime or is blocked by Windows Application Control.
    packages['mpi4py'] = _probe_import('mpi4py.MPI')
    config = _configuration()
    python_supported = (3, 11) <= sys.version_info[:2] < (3, 14)
    data_setting = Path(os.getenv('MARKETINTEL_DATA_DIR', 'storage')).expanduser()
    data_directory = (data_setting if data_setting.is_absolute() else
                      Path(__file__).resolve().parents[1] / data_setting).resolve()
    status = {
        'python': sys.version.split()[0],
        'python_executable': sys.executable,
        'python_supported': python_supported,
        'platform': platform.platform(),
        'required_packages': {name: packages[name] for name in required},
        'optional_packages': {name: packages[name] for name in optional},
        'environment_healthy': (python_supported and config['valid'] and
                                all(packages[name]['status'] == 'importable' for name in required)),
        'mpi_launcher': shutil.which('mpiexec') or shutil.which('mpirun'),
        'ollama_executable': shutil.which('ollama'),
        'local_llm_model_configured': bool(os.getenv('MARKETINTEL_OLLAMA_MODEL')),
        'local_semantic_model_directory': os.getenv('MARKETINTEL_LOCAL_EMBED_MODEL'),
        'application_config': config,
        'data_directory': str(data_directory),
        'aws_deployed': 'NOT CHECKED — no AWS access made',
    }
    return status


def _probe_import(name: str) -> dict:
    """Check runtime importability, including broken installed binary wheels."""
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError:
        # A dotted optional import can fail while locating its missing parent.
        return {'status': 'not_installed'}
    except Exception as exc:
        return {'status': 'installed_broken', 'error': f'{type(exc).__name__}: {exc}'[:240]}
    if spec is None:
        return {'status': 'not_installed'}
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return {'status': 'installed_broken', 'error': f'{type(exc).__name__}: {exc}'[:240]}
    return {'status': 'importable', 'version': getattr(module, '__version__', 'unknown')}


def _configuration() -> dict:
    """Validate local settings without creating files or contacting services."""
    issues = []
    retrieval = os.getenv('MARKETINTEL_RETRIEVAL_MODE', 'tfidf').lower()
    if retrieval not in {'tfidf', 'semantic'}:
        issues.append('MARKETINTEL_RETRIEVAL_MODE must be tfidf or semantic')
    model = os.getenv('MARKETINTEL_LOCAL_EMBED_MODEL')
    model_path = Path(model).expanduser() if model else None
    if model_path and not model_path.is_absolute():
        model_path = Path(__file__).resolve().parents[1] / model_path
    if (retrieval == 'semantic' or model) and (not model_path or not model_path.is_dir()):
        issues.append('Semantic retrieval needs an existing MARKETINTEL_LOCAL_EMBED_MODEL directory')
    if not os.getenv('MARKETINTEL_BIND', '127.0.0.1').strip():
        issues.append('MARKETINTEL_BIND cannot be empty')
    try:
        port = int(os.getenv('MARKETINTEL_PORT', '7860'))
        if not 1 <= port <= 65535:
            raise ValueError
    except ValueError:
        issues.append('MARKETINTEL_PORT must be an integer from 1 to 65535')
    return {'valid': not issues, 'issues': issues}


def main() -> None:
    print(json.dumps(diagnostics(), indent=2))
    print('\nThis is a read-only environment check, not proof that MPI or an LLM runs successfully.')


if __name__ == '__main__':
    main()
