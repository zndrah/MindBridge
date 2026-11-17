@echo off
setlocal EnableExtensions enabledelayedexpansion

:: ==============================================================================
:: MindBridge 2.0 -- Native Windows Frontend Launcher
:: Location: scripts\start_frontend.bat
:: Requirements: Windows 10/11 (WSL / Docker NOT REQUIRED)
:: ==============================================================================

title MindBridge Frontend (Vite)

pushd "%~dp0.."
set "PROJECT_ROOT=%CD%"
popd
cd /d "!PROJECT_ROOT!\frontend"

set "FRONTEND_PORT=5173"
if not "%~1"=="" set "FRONTEND_PORT=%~1"

if not exist "!PROJECT_ROOT!\logs" mkdir "!PROJECT_ROOT!\logs"
if not exist "!PROJECT_ROOT!\.run" mkdir "!PROJECT_ROOT!\.run"

echo ==============================================================================
echo   MindBridge 2.0 -- React + Vite Frontend
echo ==============================================================================
echo   Project Root : !PROJECT_ROOT!
echo   Working Dir  : !PROJECT_ROOT!\frontend
echo   Browser URL  : http://localhost:!FRONTEND_PORT!
echo ==============================================================================
echo.

npm.cmd run dev -- --host 0.0.0.0 --port !FRONTEND_PORT!

set "EXIT_CODE=%ERRORLEVEL%"
if %EXIT_CODE% neq 0 (
    echo.
    echo ==============================================================================
    echo [ERROR] Frontend service exited with failure code %EXIT_CODE%.
    echo ==============================================================================
    echo.
    pause
)
exit /b %EXIT_CODE%

