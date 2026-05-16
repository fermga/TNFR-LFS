# Build the standalone "LFS Race Engineer" Windows application.
#
# Usage (from repo root, with the project venv activated):
#
#     .\scripts\build_app.ps1
#
# Output:
#     dist\lfs-race-engineer\lfs-race-engineer.exe
#     dist\lfs-race-engineer\_internal\...
#     dist\lfs-race-engineer\config\, racing_lines\, tracks\
#
# To produce the single-file installer afterwards, install Inno Setup
# (winget install JRSoftware.InnoSetup) and run:
#
#     iscc installer\lfs-race-engineer.iss
#
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "[build] FATAL: virtualenv not found at $python" -ForegroundColor Red
    exit 1
}

Write-Host "[build] verifying build dependencies..." -ForegroundColor Cyan
& $python -m pip install --quiet --upgrade pip
& $python -m pip install --quiet -e ".[studio,build]"

Write-Host "[build] cleaning previous build..." -ForegroundColor Cyan
Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue

Write-Host "[build] running PyInstaller..." -ForegroundColor Cyan
& $python -m PyInstaller lfs-race-engineer.spec --noconfirm --clean

$exe = "dist\lfs-race-engineer\lfs-race-engineer.exe"
if (Test-Path $exe) {
    $sizeMB = [math]::Round((Get-ChildItem -Recurse "dist\lfs-race-engineer" |
        Measure-Object -Property Length -Sum).Sum / 1MB, 1)
    Write-Host ""
    Write-Host "[build] SUCCESS" -ForegroundColor Green
    Write-Host "[build]   exe : $exe"
    Write-Host "[build]   size: $sizeMB MB (full folder)"
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "  1) Test it:   .\$exe"
    Write-Host "  2) Package:   iscc installer\lfs-race-engineer.iss"
    Write-Host "                 (output: installer\Output\lfs-race-engineer-setup-<ver>.exe)"
} else {
    Write-Host "[build] FAILED: $exe not found" -ForegroundColor Red
    exit 1
}
