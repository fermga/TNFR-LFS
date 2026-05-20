# Build the standalone "LFS Race Engineer" Windows application (v0.3.7+).
#
# Modern build pipeline with validation, cleanup, and installer generation.
#
# Usage (from repo root, with the project venv activated):
#
#     .\scripts\build_app.ps1                 # Build exe only
#     .\scripts\build_app.ps1 -Full           # Build exe + installer
#     .\scripts\build_app.ps1 -Full -Sign     # Build exe + installer (signed, if cert available)
#
# Output:
#     dist\lfs-race-engineer\lfs-race-engineer.exe
#     dist\lfs-race-engineer\_internal\...
#     dist\lfs-race-engineer\config\, racing_lines\, tracks\
#     installer\Output\lfs-race-engineer-setup-<ver>.exe (if -Full)
#
# Requirements:
#   - Python venv at .venv/ (auto-verified)
#   - PySide6, pyqtgraph, pyinstaller from [studio,build] extras
#   - Inno Setup 6.3+ for -Full mode (winget install JRSoftware.InnoSetup)

param(
    [switch]$Full = $false,
    [switch]$Sign = $false,
    [switch]$SkipTests = $false
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$ConfirmPreference = "None"

$repoRoot = Split-Path -Parent $PSScriptRoot
$distDir = Join-Path $repoRoot "dist"
$buildDir = Join-Path $repoRoot "build"
$appDir = Join-Path $distDir "lfs-race-engineer"

# ===== Utility functions =====

function Write-Banner { param([string]$msg, [string]$color = "Cyan")
    Write-Host ""
    Write-Host "══" ($msg) "══" -ForegroundColor $color
}

function Invoke-Cmd { param([string]$exe, [string[]]$cmdArgs, [string]$desc)
    Write-Host "→ $desc" -ForegroundColor Gray
    & $exe @cmdArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED (exit $LASTEXITCODE): $desc" -ForegroundColor Red
        exit 1
    }
}

function Get-ProjectVersion {
    $pyprojectPath = Join-Path $repoRoot "pyproject.toml"
    if (Test-Path $pyprojectPath) {
        $content = Get-Content $pyprojectPath -Raw
        if ($content -match 'version\s*=\s*"([^"]+)"') {
            return $matches[1]
        }
    }
    return "0.3.9"
}

# ===== Setup & Validation =====

Write-Banner "LFS Race Engineer Builder (modern)" "Cyan"

$version = Get-ProjectVersion
Write-Host "Version: $version" -ForegroundColor Green

Set-Location $repoRoot

$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Host "FATAL: virtualenv not found at $python" -ForegroundColor Red
    Write-Host "Create with: python -m venv .venv" -ForegroundColor Yellow
    exit 1
}

# ===== Run tests (optional) =====

if (-not $SkipTests) {
    Write-Banner "Running tests" "Magenta"
    Invoke-Cmd $python @("-m", "pytest", "-q") "pytest validation"
}

# ===== Dependency check & upgrade =====

Write-Banner "Verifying build environment" "Cyan"
Invoke-Cmd $python @("-m", "pip", "install", "--quiet", "--upgrade", "pip", "setuptools", "wheel") `
    "pip upgrade"
Invoke-Cmd $python @("-m", "pip", "install", "--quiet", "-e", ".[studio,build]") `
    "dependencies (studio + build)"

# ===== Cleanup =====

Write-Banner "Cleaning previous builds" "Yellow"
Remove-Item -Recurse -Force $buildDir -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force $distDir -ErrorAction SilentlyContinue
Write-Host "✓ Cleaned build/ and dist/" -ForegroundColor Green

# ===== PyInstaller build =====

Write-Banner "Running PyInstaller (may take 1–2 min)" "Cyan"
$specPath = Join-Path $repoRoot "lfs-race-engineer.spec"
Invoke-Cmd $python @("-m", "PyInstaller", $specPath, "--noconfirm", "--clean") `
    "PyInstaller compilation"

# ===== Validation =====

if (-not (Test-Path $appDir)) {
    Write-Host "FAILED: $appDir not created" -ForegroundColor Red
    exit 1
}

$exe = Join-Path $appDir "lfs-race-engineer.exe"
if (-not (Test-Path $exe)) {
    Write-Host "FAILED: $exe not found" -ForegroundColor Red
    exit 1
}

# Size computation
$totalSize = 0
Get-ChildItem -Recurse $appDir | Where-Object { $_.PSIsContainer -eq $false } | ForEach-Object {
    $totalSize += $_.Length
}
$sizeMB = [math]::Round($totalSize / 1MB, 1)

Write-Banner "Build successful" "Green"
Write-Host "Executable : $(Resolve-Path $exe -Relative)" -ForegroundColor Green
Write-Host "App folder : $(Resolve-Path $appDir -Relative)" -ForegroundColor Green
Write-Host "Total size : $sizeMB MB" -ForegroundColor Green

# ===== Code signing (optional) =====

if ($Sign) {
    Write-Banner "Code signing" "Yellow"
    # Optional: integrate with your cert store
    Write-Host "Signing not yet configured. Integrate with your certificate." -ForegroundColor DarkYellow
}

# ===== Installer generation (optional) =====

if ($Full) {
    Write-Banner "Generating installer" "Cyan"

    # Locate iscc: PATH first, then well-known install dirs (mirror of
    # build_installer.ps1). Inno Setup installs to %LocalAppData% when
    # the user picks per-user install via winget.
    $isccCmd = Get-Command "iscc.exe" -ErrorAction SilentlyContinue
    $iscc = if ($isccCmd) { $isccCmd.Source } else { $null }
    if (-not $iscc) {
        $candidates = @(
            "C:\Program Files\Inno Setup 6\iscc.exe",
            "C:\Program Files (x86)\Inno Setup 6\iscc.exe",
            (Join-Path $env:LocalAppData "Programs\Inno Setup 6\iscc.exe"),
            "C:\Program Files\Inno Setup 5\iscc.exe",
            "C:\Program Files (x86)\Inno Setup 5\iscc.exe"
        )
        foreach ($p in $candidates) {
            if (Test-Path $p) { $iscc = $p; break }
        }
    }
    if (-not $iscc) {
        Write-Host "Inno Setup not found. Install with: winget install JRSoftware.InnoSetup" -ForegroundColor Yellow
        Write-Host "Skipping installer generation." -ForegroundColor Yellow
    } else {
        Write-Host "Using iscc: $iscc" -ForegroundColor Gray
        $issPath = Join-Path $repoRoot "installer" "lfs-race-engineer.iss"
        $installerOutput = Join-Path $repoRoot "installer" "Output"

        # Ensure output dir exists
        New-Item -ItemType Directory $installerOutput -Force | Out-Null

        Invoke-Cmd $iscc @("/DMyAppVersion=$version", $issPath) `
            "Inno Setup compilation (lfs-race-engineer.iss)"

        $setupExe = Join-Path $installerOutput "lfs-race-engineer-setup-$version.exe"
        if (Test-Path $setupExe) {
            $setupSize = [math]::Round((Get-Item $setupExe).Length / 1MB, 1)
            Write-Host "Installer : $(Resolve-Path $setupExe -Relative)" -ForegroundColor Green
            Write-Host "Setup size: $setupSize MB" -ForegroundColor Green
        } else {
            Write-Host "Installer generation completed but output not found." -ForegroundColor Yellow
        }
    }
}

# ===== Summary =====

Write-Banner "Build complete" "Green"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1) Test:    $exe" -ForegroundColor Gray
Write-Host "  2) Package: .\scripts\build_app.ps1 -Full" -ForegroundColor Gray
Write-Host "  3) Release: Push installer\Output\lfs-race-engineer-setup-$version.exe" -ForegroundColor Gray
Write-Host ""
