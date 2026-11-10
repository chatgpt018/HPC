param([string]$Wheelhouse = '')

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSCommandPath
Push-Location -LiteralPath $projectRoot
try {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        throw 'Python launcher not found. Install Python 3.11 for Windows, then retry.'
    }
    & py -3.11 -c 'import sys; print(sys.version); assert sys.version_info[:2] == (3, 11)'
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 is required. Check with: py -0p' }

    & py -3.11 -m venv .venv311
    if ($LASTEXITCODE -ne 0) { throw 'Could not create .venv311.' }
    $python = Join-Path $projectRoot '.venv311\Scripts\python.exe'

    if ($Wheelhouse) {
        $wheels = if ([IO.Path]::IsPathRooted($Wheelhouse)) { $Wheelhouse } else { Join-Path $projectRoot $Wheelhouse }
        if (-not (Test-Path -LiteralPath $wheels -PathType Container)) { throw "Wheelhouse folder not found: $wheels" }
        & $python -m pip install --no-index --find-links $wheels 'market-intelligence-hpc-local[dev]'
    } else {
        & $python -m pip install -e '.[dev]'
    }
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check Python version and package source.' }
    & $python scripts\doctor.py
    if ($LASTEXITCODE -ne 0) { throw 'Environment diagnostic failed.' }
    Write-Host 'Setup complete. Run .\run_windows.ps1 from this project folder.'
} finally {
    Pop-Location
}
