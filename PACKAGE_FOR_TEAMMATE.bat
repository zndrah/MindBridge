@echo off
setlocal EnableExtensions enabledelayedexpansion

:: ==============================================================================
:: MindBridge 2.0 -- Portable Release Packaging Launcher
:: Location: PACKAGE_FOR_TEAMMATE.bat
:: Requirements: Windows 10/11, Python 3.11+ (WSL NOT REQUIRED)
:: Purpose: Packages the complete project into release\MindBridge_Portable.zip,
::          excluding secrets, caches, and virtualenvs while including all required
::          models, FAISS indexes, knowledge base assets, and manifests.
:: ==============================================================================

title MindBridge 2.0 -- Package For Teammate

cd /d "%~dp0"
for %%I in ("%~dp0.") do set "PROJECT_ROOT=%%~fI"

echo ==============================================================================
echo   MindBridge 2.0 -- Teammate Packaging Engine (Native Windows)
echo ==============================================================================
echo   Project Root     : %PROJECT_ROOT%
echo   Output Directory : %PROJECT_ROOT%\release\
echo ==============================================================================
echo.

:: Detect Python executable
set "PYTHON_EXE="
if exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%PROJECT_ROOT%\.venv\Scripts\python.exe"
) else (
    where python.exe >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        set "PYTHON_EXE=python.exe"
    ) else (
        where py.exe >nul 2>&1
        if !ERRORLEVEL! equ 0 (
            set "PYTHON_EXE=py.exe"
        )
    )
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] Python was not found in PATH or at .venv\Scripts\python.exe.
    echo Please install Python 3.11 or run FIRST_TIME_SETUP.bat first.
    pause
    exit /b 1
)

echo Executing release packager via %PYTHON_EXE%...
echo.

"%PYTHON_EXE%" "%PROJECT_ROOT%\scripts\package_release.py"

if !ERRORLEVEL! neq 0 (
    echo.
    echo [ERROR] Packaging failed. Please inspect the output above.
    pause
    exit /b 1
)

echo.
echo ==============================================================================
echo   MindBridge portable package is ready in release\MindBridge_Portable.zip!
echo   Share this ZIP file with your teammate.
echo ==============================================================================
echo.
pause
exit /b 0
