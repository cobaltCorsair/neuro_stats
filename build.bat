@echo off
REM ========================================
REM Скрипт сборки RadiobiologyAnalysis.exe
REM ========================================

echo.
echo ========================================
echo   Radiobiology Analysis - Build Script
echo ========================================
echo.

REM Проверка наличия Poetry
where poetry >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Poetry not found! Please install Poetry first.
    echo https://python-poetry.org/docs/#installation
    pause
    exit /b 1
)

echo [1/4] Checking Poetry environment...
poetry env info

echo.
echo [2/4] Cleaning previous build...
if exist dist\RadiobiologyAnalysis.exe (
    del /F /Q dist\RadiobiologyAnalysis.exe
    echo Previous build deleted.
) else (
    echo No previous build found.
)

echo.
echo [3/4] Building application with PyInstaller...
echo This may take several minutes...
echo.
poetry run pyinstaller radiobiology_app.spec --clean

echo.
echo [4/4] Checking build result...
if exist dist\RadiobiologyAnalysis.exe (
    echo.
    echo ========================================
    echo   BUILD SUCCESSFUL!
    echo ========================================
    echo.
    for %%A in (dist\RadiobiologyAnalysis.exe) do echo   File: dist\RadiobiologyAnalysis.exe
    for %%A in (dist\RadiobiologyAnalysis.exe) do echo   Size: %%~zA bytes (approx. %%~zA / 1048576 MB)
    echo.

    REM Запрашиваем, запустить ли приложение
    choice /C YN /M "Do you want to run the application now"
    if errorlevel 2 goto end
    if errorlevel 1 goto run

    :run
    echo.
    echo Launching application...
    start dist\RadiobiologyAnalysis.exe
    goto end
) else (
    echo.
    echo ========================================
    echo   BUILD FAILED!
    echo ========================================
    echo Check the errors above.
    echo.
    pause
    exit /b 1
)

:end
echo.
echo Build script completed.
echo.
pause
