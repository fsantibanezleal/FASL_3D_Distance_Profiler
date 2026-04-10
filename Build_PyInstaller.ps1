# =============================================================================
# Build_PyInstaller.ps1
# SurfaceScope -- PyInstaller Build Script
# =============================================================================
#
# Usage:
#   .\Build_PyInstaller.ps1
#
# Prerequisites:
#   - Python 3.10+ with pip
#   - Virtual environment at .venv/
#   - All dependencies installed
#
# Output:
#   dist\SurfaceScope\SurfaceScope.exe
# =============================================================================

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Write-Host "============================================" -ForegroundColor Cyan
Write-Host " SurfaceScope -- Build" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment
$VenvActivate = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    Write-Host "[1/4] Activating virtual environment..." -ForegroundColor Yellow
    & $VenvActivate
} else {
    Write-Host "[1/4] No .venv found. Using system Python." -ForegroundColor Yellow
}

# Install PyInstaller if missing
Write-Host "[2/4] Ensuring PyInstaller is installed..." -ForegroundColor Yellow
pip install pyinstaller --quiet

# Clean previous build
Write-Host "[3/4] Cleaning previous build artifacts..." -ForegroundColor Yellow
$DistDir = Join-Path $ProjectRoot "dist"
$BuildDir = Join-Path $ProjectRoot "build"
if (Test-Path $DistDir) { Remove-Item -Recurse -Force $DistDir }
if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir }

# Build
Write-Host "[4/4] Running PyInstaller..." -ForegroundColor Yellow
$SpecFile = Join-Path $ProjectRoot "build.spec"
pyinstaller $SpecFile --noconfirm

# Report
$ExePath = Join-Path $DistDir "SurfaceScope\SurfaceScope.exe"
if (Test-Path $ExePath) {
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Green
    Write-Host " BUILD SUCCESSFUL" -ForegroundColor Green
    Write-Host " Output: $ExePath" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Red
    Write-Host " BUILD FAILED" -ForegroundColor Red
    Write-Host "============================================" -ForegroundColor Red
    exit 1
}
