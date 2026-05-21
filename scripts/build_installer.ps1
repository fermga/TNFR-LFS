# Build the "LFS Race Engineer" Windows installer (.exe) via Inno Setup.
#
# Assumes the PyInstaller bundle already exists under build/lfs-race-engineer/.
# Locates `iscc.exe` across the standard Inno Setup 5/6 install paths and
# compiles installer/lfs-race-engineer.iss. Output lands in
# installer/Output/.
#
# When to use this script:
#   .\scripts\build_installer.ps1          # build installer from current bundle
#   .\scripts\build_installer.ps1 -Force   # rebuild even if output exists
#
# Prerequisite: run scripts/build_app.ps1 (or build_app_simple.ps1) first
# so the PyInstaller bundle is present. To do both in one step, use
# `scripts/build_app.ps1 -Full`.

param([switch]$Force)

$repoRoot = Split-Path -Parent $PSScriptRoot

# Try to find iscc.exe
$isccPaths = @(
    "C:\Program Files\Inno Setup 6\iscc.exe",
    "C:\Program Files (x86)\Inno Setup 6\iscc.exe",
    "$env:LocalAppData\Programs\Inno Setup 6\iscc.exe",
    "C:\Program Files\Inno Setup 5\iscc.exe",
    "C:\Program Files (x86)\Inno Setup 5\iscc.exe"
)

$iscc = $null
foreach ($path in $isccPaths) {
    if (Test-Path $path) {
        $iscc = $path
        break
    }
}

if (-not $iscc) {
    Write-Host ""
    Write-Host "Inno Setup not found in standard locations." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To install Inno Setup:" -ForegroundColor Cyan
    Write-Host "  1. winget install JRSoftware.InnoSetup" -ForegroundColor Gray
    Write-Host "  2. Or download from: https://jrsoftware.org/isdl.php" -ForegroundColor Gray
    Write-Host ""
    exit 1
}

Write-Host ""
Write-Host "Building installer with Inno Setup..." -ForegroundColor Cyan

$issPath = Join-Path $repoRoot "installer" "lfs-race-engineer.iss"
$outDir = Join-Path $repoRoot "installer" "Output"

if (-not (Test-Path $outDir)) {
    New-Item -ItemType Directory $outDir -Force | Out-Null
}

& $iscc $issPath

if ($LASTEXITCODE -eq 0) {
    $setupExe = Join-Path $outDir "lfs-race-engineer-setup-0.3.9.exe"
    if (Test-Path $setupExe) {
        $size = [math]::Round((Get-Item $setupExe).Length / 1MB, 1)
        Write-Host ""
        Write-Host "✓ Installer created: $setupExe ($size MB)" -ForegroundColor Green
    }
} else {
    Write-Host ""
    Write-Host "ERROR: Inno Setup compilation failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
