@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%.."
set "MODULE_NAME=survival.pipeline_geant4_to_prediction"
set "PREFERRED_PYTHON="

if "%~1"=="" (
    echo Usage:
    echo   run_geant4_pipeline.bat dose.pb geometry.ivz [pipeline options]
    echo.
    echo Example:
    echo   run_geant4_pipeline.bat dose.pb geometry.ivz --alpha 0.1 --beta 0.02 --schedule-days 0,1,2
    echo.
    echo For full help:
    echo   run_geant4_pipeline.bat --help
    exit /b 1
)

if defined PYTHON_EXE (
    set "PREFERRED_PYTHON=%PYTHON_EXE%"
)

if not defined PREFERRED_PYTHON (
    if exist "%PROJECT_DIR%\.venv\Scripts\python.exe" (
        set "PREFERRED_PYTHON=%PROJECT_DIR%\.venv\Scripts\python.exe"
    )
)

if not defined PREFERRED_PYTHON (
    for /d %%D in ("%LOCALAPPDATA%\pypoetry\Cache\virtualenvs\neuro-stats-*") do (
        if not defined PREFERRED_PYTHON (
            if exist "%%~fD\Scripts\python.exe" (
                set "PREFERRED_PYTHON=%%~fD\Scripts\python.exe"
            )
        )
    )
)

if defined PREFERRED_PYTHON (
    pushd "%PROJECT_DIR%" >nul
    "%PREFERRED_PYTHON%" -m %MODULE_NAME% %*
    set "EXIT_CODE=%errorlevel%"
    popd >nul
    exit /b %EXIT_CODE%
)

where py >nul 2>nul
if not errorlevel 1 (
    py -3.10 -c "import numpy" >nul 2>nul
    if not errorlevel 1 (
        pushd "%PROJECT_DIR%" >nul
        py -3.10 -m %MODULE_NAME% %*
        set "EXIT_CODE=%errorlevel%"
        popd >nul
        exit /b %EXIT_CODE%
    )
)

where python >nul 2>nul
if not errorlevel 1 (
    python -c "import numpy" >nul 2>nul
    if not errorlevel 1 (
        pushd "%PROJECT_DIR%" >nul
        python -m %MODULE_NAME% %*
        set "EXIT_CODE=%errorlevel%"
        popd >nul
        exit /b %EXIT_CODE%
    )
)

echo No suitable Python interpreter was found for the GEANT4 pipeline.
echo Set PYTHON_EXE to a project environment, or install numpy/protobuf into the interpreter on PATH.
exit /b 9009
