@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PROTOC_EXE="

for /f "delims=" %%I in ('where protoc 2^>NUL') do (
    set "PROTOC_EXE=%%I"
    goto :protoc_found
)

if not defined PROTOC_EXE (
    set "PROTOC_EXE=C:\dev\dissertation\protobuf\bin\protoc.exe"
)

:protoc_found
if not exist "%PROTOC_EXE%" (
    echo protoc executable was not found.
    exit /b 1
)

pushd "%SCRIPT_DIR%"
"%PROTOC_EXE%" --python_out=. .\NPInputVoxelData.proto .\NPPhaseSpaceData.proto .\NPWiseVoxelData.proto
set "EXIT_CODE=%ERRORLEVEL%"
popd

exit /b %EXIT_CODE%
