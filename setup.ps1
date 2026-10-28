<#
.SYNOPSIS
    CodeLens AI Windows Environment Setup Script.
.DESCRIPTION
    Initializes virtual environment, installs Python dependencies,
    and verifies Ollama service and gemma4:e2b model readiness.
#>

$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "         CodeLens AI - Windows Setup Wizard              " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Check Python
Write-Host "`n[1/4] Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "  Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "  [ERROR] Python is not found in PATH. Please install Python 3.10+ from python.org" -ForegroundColor Red
    Exit 1
}

# 2. Setup Virtual Environment
Write-Host "`n[2/4] Setting up local virtual environment (.venv)..." -ForegroundColor Yellow
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$venvPip = Join-Path $PSScriptRoot ".venv\Scripts\pip.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "  Creating new virtual environment at .venv..." -ForegroundColor Gray
    python -m venv .venv
    Write-Host "  Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "  Existing .venv detected." -ForegroundColor Green
}

# 3. Install Dependencies
Write-Host "`n[3/4] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
$reqFile = Join-Path $PSScriptRoot "requirements.txt"
if (Test-Path $reqFile) {
    & $venvPip install -r $reqFile
    Write-Host "  Dependencies installed successfully." -ForegroundColor Green
} else {
    Write-Host "  requirements.txt not found!" -ForegroundColor Red
    Exit 1
}

# 4. Verify Ollama & Model
Write-Host "`n[4/4] Checking local Ollama service and gemma4:e2b model..." -ForegroundColor Yellow
try {
    & $venvPython -m codelens.cli check
} catch {
    Write-Host "  [WARNING] Ollama check failed. Ensure Ollama is running ('ollama serve') and model 'gemma4:e2b' is installed." -ForegroundColor Yellow
}

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "  CodeLens AI setup completed successfully!               " -ForegroundColor Green
Write-Host "  You can run commands using: .\run.ps1 <command>        " -ForegroundColor Green
Write-Host "  Example: .\run.ps1 check                                " -ForegroundColor Gray
Write-Host "  Example: .\run.ps1 tools                                " -ForegroundColor Gray
Write-Host "  Example: .\run.ps1 demo                                 " -ForegroundColor Gray
Write-Host "==========================================================" -ForegroundColor Green
