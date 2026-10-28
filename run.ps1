<#
.SYNOPSIS
    CodeLens AI Windows Launcher.
.DESCRIPTION
    Convenience wrapper to run CodeLens CLI inside the local virtual environment.
.EXAMPLE
    .\run.ps1 check
    .\run.ps1 tools
    .\run.ps1 run "Fix the median calculation in examples/off_by_one.py"
    .\run.ps1 fix examples/off_by_one.py
    .\run.ps1 learn examples/off_by_one.py
    .\run.ps1 demo
    .\run.ps1 web
#>

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "[ERROR] Virtual environment not found. Please run .\setup.ps1 first." -ForegroundColor Red
    Exit 1
}

& $venvPython -m codelens.cli @args
