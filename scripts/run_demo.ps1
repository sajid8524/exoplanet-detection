$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$env:PYTHONPATH = (Resolve-Path "src")

$python = "python"
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    $python = "C:\Users\sajid\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
}

& $python -m tess_exoplanet.cli run-demo --workdir runs/demo --n 160 --epochs 80

