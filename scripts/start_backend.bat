@echo off
setlocal EnableExtensions enabledelayedexpansion

:: ==============================================================================
:: MindBridge 2.0 -- Native Windows Backend Launcher
:: Location: scripts\start_backend.bat
:: Requirements: Windows 10/11 (WSL / Docker NOT REQUIRED)
:: ==============================================================================

title MindBridge Backend (FastAPI)

pushd "%~dp0.."
set "PROJECT_ROOT=%CD%"
popd
cd /d "%PROJECT_ROOT%"

set "BACKEND_PORT=8000"
if not "%~1"=="" set "BACKEND_PORT=%~1"

set "PYTHONPATH=!PROJECT_ROOT!"

if not exist "!PROJECT_ROOT!\logs" mkdir "!PROJECT_ROOT!\logs"
if not exist "!PROJECT_ROOT!\.run" mkdir "!PROJECT_ROOT!\.run"

set "PYTHON_EXE=!PROJECT_ROOT!\.venv\Scripts\python.exe"
set "LOG_FILE=!PROJECT_ROOT!\logs\backend.log"

if not exist "!PYTHON_EXE!" (
    echo.
    echo ==============================================================================
    echo [ERROR] Python virtual environment was not found at:
    echo         !PYTHON_EXE!
    echo Please run FIRST_TIME_SETUP.bat to configure the environment.
    echo ==============================================================================
    echo.
    pause
    exit /b 1
)

echo ==============================================================================
echo   MindBridge 2.0 -- FastAPI Backend Service
echo ==============================================================================
echo   Project Root : !PROJECT_ROOT!
echo   Host / Port  : 0.0.0.0:!BACKEND_PORT!
echo   Health Probe : http://127.0.0.1:!BACKEND_PORT!/health
echo   Ready Probe  : http://127.0.0.1:!BACKEND_PORT!/ready
echo   API Docs     : http://127.0.0.1:!BACKEND_PORT!/docs
echo   Log File     : !LOG_FILE!
echo ==============================================================================
echo.
echo Pre-warming RoBERTa crisis classifier and BGE dense vector index...
echo.

if exist "!PROJECT_ROOT!\scripts\tee_runner.py" (
    "!PYTHON_EXE!" "!PROJECT_ROOT!\scripts\tee_runner.py" "!LOG_FILE!" "!PYTHON_EXE!" -m uvicorn backend.api.app:app --host 0.0.0.0 --port !BACKEND_PORT!
) else (
    "!PYTHON_EXE!" -m uvicorn backend.api.app:app --host 0.0.0.0 --port !BACKEND_PORT!
)

set "EXIT_CODE=%ERRORLEVEL%"
if %EXIT_CODE% neq 0 (
    echo.
    echo ==============================================================================
    echo [ERROR] Backend service exited with failure code %EXIT_CODE%.
    echo Check %LOG_FILE% for details.
    echo ==============================================================================
    echo.
    pause
)
exit /b %EXIT_CODE%

