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
    $setupExe = Join-Path $outDir "lfs-race-engineer-setup-0.3.7.exe"
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
