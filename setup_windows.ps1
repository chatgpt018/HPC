param([string]$Wheelhouse = '', [string]$PythonExecutable = '')

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSCommandPath
Push-Location -LiteralPath $projectRoot
try {
    $pythonExe = ''
    $pythonArgs = @()
    if ($PythonExecutable) {
        $pythonExe = $PythonExecutable
    } else {
        if (Get-Command py -ErrorAction SilentlyContinue) {
            & py -3.11 -c 'import sys; assert sys.version_info[:2] == (3, 11)' 2>$null
            if ($LASTEXITCODE -eq 0) {
                $pythonExe = 'py'
                $pythonArgs = @('-3.11')
            }
        }
        if (-not $pythonExe) {
            foreach ($candidate in @('python3.11', 'python')) {
                $command = Get-Command $candidate -ErrorAction SilentlyContinue
                if (-not $command) { continue }
                & $command.Source -c 'import sys; assert sys.version_info[:2] == (3, 11)' 2>$null
                if ($LASTEXITCODE -eq 0) {
                    $pythonExe = $command.Source
                    break
                }
            }
        }
    }
    if (-not $pythonExe) {
        throw 'Python 3.11 was not found. Install it or pass -PythonExecutable <path-to-python.exe>.'
    }
    & $pythonExe @pythonArgs -c 'import sys; print(sys.version); assert sys.version_info[:2] == (3, 11)'
    if ($LASTEXITCODE -ne 0) { throw 'The selected Python executable must be Python 3.11.' }

    & $pythonExe @pythonArgs -m venv .venv311
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
