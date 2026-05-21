# Minimal PyInstaller wrapper for "LFS Race Engineer" (single-file exe).
#
# Use this when you just need a quick rebuild of the bundled .exe for
# manual smoke-testing. Skips:
#   - Pre-flight validation (Python version, venv, .spec sanity).
#   - Output verification and bundle-size diff.
#   - Installer generation.
#
# When to use this script:
#   .\scripts\build_app_simple.ps1            # fast iteration
#   .\scripts\build_app_simple.ps1 -Full      # also clean build/ first
#
# When to use the full pipeline instead: see scripts/build_app.ps1.
# When you only need the installer from an existing build: see
# scripts/build_installer.ps1.

param([switch]$Full)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$ConfirmPreference = "None"

$root = Split-Path -Parent $PSScriptRoot
$python = "$root\.venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "ERROR: venv not found" -ForegroundColor Red
    exit 1
}

Set-Location $root

# Read version from pyproject.toml (single source of truth)
$pyproject = Get-Content "$root\pyproject.toml" -Raw
if ($pyproject -match 'version\s*=\s*"([^"]+)"') {
    $version = $Matches[1]
} else {
    Write-Host "ERROR: could not parse version from pyproject.toml" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Building LFS Race Engineer v$version" -ForegroundColor Cyan

# Clean
Write-Host "Cleaning..." -ForegroundColor Yellow
Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue

# Dependencies
Write-Host "Installing dependencies..." -ForegroundColor Cyan
& $python -m pip install --quiet --upgrade pip setuptools wheel
& $python -m pip install --quiet -e ".[studio,build]"

# Build
Write-Host "Building exe (may take 1-2 minutes)..." -ForegroundColor Cyan
& $python -m PyInstaller lfs-race-engineer.spec --noconfirm --clean

$exe = "$root\dist\lfs-race-engineer\lfs-race-engineer.exe"
if (-not (Test-Path $exe)) {
    Write-Host "ERROR: exe not created" -ForegroundColor Red
    exit 1
}

$size = [math]::Round((Get-ChildItem -Recurse "$root\dist\lfs-race-engineer" | Measure-Object -Sum -Property Length).Sum / 1MB, 1)
Write-Host "✓ Built: $exe ($size MB)" -ForegroundColor Green

if ($Full) {
    Write-Host "Building installer..." -ForegroundColor Cyan
    $iscc = Get-Command "iscc.exe" -ErrorAction SilentlyContinue
    if ($iscc) {
        & iscc.exe "/DMyAppVersion=$version" "$root\installer\lfs-race-engineer.iss"
        $installer = "$root\installer\Output\lfs-race-engineer-setup-$version.exe"
        if (Test-Path $installer) {
            $insSize = [math]::Round((Get-Item $installer).Length / 1MB, 1)
            Write-Host "✓ Installer: $installer ($insSize MB)" -ForegroundColor Green
        }
    } else {
        Write-Host "⚠ Inno Setup not found (skipping installer)" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Done!" -ForegroundColor Green
