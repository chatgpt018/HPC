$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSCommandPath
$python = Join-Path $projectRoot '.venv311\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'Project environment not found. Run .\setup_windows.ps1 first.'
}
Push-Location -LiteralPath $projectRoot
try {
    & $python app.py
    if ($LASTEXITCODE -ne 0) { throw 'The application exited with an error. Run scripts\doctor.py for diagnostics.' }
} finally {
    Pop-Location
}
