# ========================================
# Скрипт сборки RadiobiologyAnalysis.exe
# PowerShell версия
# ========================================

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Radiobiology Analysis - Build Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Проверка Poetry
try {
    $poetryVersion = poetry --version 2>&1
    Write-Host "[✓] Poetry found: $poetryVersion" -ForegroundColor Green
} catch {
    Write-Host "[✗] Poetry not found!" -ForegroundColor Red
    Write-Host "Please install Poetry: https://python-poetry.org/docs/#installation" -ForegroundColor Yellow
    pause
    exit 1
}

# Информация об окружении
Write-Host ""
Write-Host "[1/4] Checking Poetry environment..." -ForegroundColor Yellow
poetry env info

# Очистка старой сборки
Write-Host ""
Write-Host "[2/4] Cleaning previous build..." -ForegroundColor Yellow
if (Test-Path "dist\RadiobiologyAnalysis.exe") {
    Remove-Item "dist\RadiobiologyAnalysis.exe" -Force
    Write-Host "Previous build deleted." -ForegroundColor Gray
} else {
    Write-Host "No previous build found." -ForegroundColor Gray
}

# Сборка
Write-Host ""
Write-Host "[3/4] Building application with PyInstaller..." -ForegroundColor Yellow
Write-Host "This may take several minutes..." -ForegroundColor Gray
Write-Host ""

$buildStart = Get-Date
poetry run pyinstaller radiobiology_app.spec --clean
$buildEnd = Get-Date
$buildTime = $buildEnd - $buildStart

# Проверка результата
Write-Host ""
Write-Host "[4/4] Checking build result..." -ForegroundColor Yellow

if (Test-Path "dist\RadiobiologyAnalysis.exe") {
    $fileInfo = Get-Item "dist\RadiobiologyAnalysis.exe"
    $fileSizeMB = [math]::Round($fileInfo.Length / 1MB, 2)

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  BUILD SUCCESSFUL!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "  File: dist\RadiobiologyAnalysis.exe" -ForegroundColor White
    Write-Host "  Size: $fileSizeMB MB" -ForegroundColor White
    Write-Host "  Build time: $($buildTime.TotalSeconds) seconds" -ForegroundColor White
    Write-Host ""

    # Запрос на запуск
    $response = Read-Host "Do you want to run the application now? (Y/N)"
    if ($response -eq 'Y' -or $response -eq 'y') {
        Write-Host ""
        Write-Host "Launching application..." -ForegroundColor Cyan
        Start-Process "dist\RadiobiologyAnalysis.exe"
    }
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "  BUILD FAILED!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "Check the errors above." -ForegroundColor Yellow
    Write-Host ""
    pause
    exit 1
}

Write-Host ""
Write-Host "Build script completed." -ForegroundColor Cyan
Write-Host ""
pause
