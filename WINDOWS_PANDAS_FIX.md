# Windows Python 3.11 pandas import repair (v0.3.1)

If pytest fails during import with `_pandas_datetime_CAPI` or `partially initialized module pandas`:

1. Stop the running Gradio server in its PowerShell window using **Ctrl+C**.
2. Open PowerShell **inside the project folder** (the one containing `pyproject.toml`).
3. Repair the existing project environment:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y pandas numpy
.\.venv\Scripts\python.exe -m pip install --no-cache-dir --only-binary=:all: "numpy==1.26.4" "pandas==2.2.3"
.\.venv\Scripts\python.exe -c "import numpy, pandas, gradio; print('numpy',numpy.__version__,'pandas',pandas.__version__,'gradio',gradio.__version__)"
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\verify_local.py
```

If the pandas import still fails, make a new environment rather than repeatedly modifying your current one:

```powershell
Rename-Item .venv .venv-old
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-cache-dir -e . pytest
.\.venv\Scripts\python.exe -c "import numpy, pandas, gradio; print(numpy.__version__,pandas.__version__,gradio.__version__)"
.\.venv\Scripts\python.exe -m pytest -q
```

*Don't delete `.venv-old` until the new environment has been verified.*

This hotfix pins pandas 2.2.3 and NumPy 1.26.4 on Python 3.11. Both have published Windows x86-64 wheels. Python 3.13 and other environments keep separate dependencies. The package cannot be verified on the user's Windows machine from our Linux build environment.

The notification about a newer pip version is not the cause of the failure. The PyMuPDF `fitz` deprecation notice also does not cause this pandas import error.
