@echo off
cd /d "%~dp0"
set "PROJECT_PYTHON=.venv311\Scripts\python.exe"
if not exist "%PROJECT_PYTHON%" set "PROJECT_PYTHON=.venv\Scripts\python.exe"
if not exist "%PROJECT_PYTHON%" (
  echo Python virtual environment not found.
  echo Read README.md and run: py -3.11 -m venv .venv311
  echo Then install: .venv311\Scripts\python.exe -m pip install -e . pytest
  pause
  exit /b 1
)
"%PROJECT_PYTHON%" app.py
if errorlevel 1 pause
