@echo off
setlocal EnableExtensions enabledelayedexpansion

:: ==============================================================================
:: MindBridge 2.0 -- Native Windows Application Launcher
:: Location: run.bat
:: Requirements: Windows 10/11 (WSL / Docker NOT REQUIRED)
:: ==============================================================================

title MindBridge 2.0 Launcher

set "NON_INTERACTIVE=0"
if /i "%~1"=="--no-pause" set "NON_INTERACTIVE=1"
if /i "%~1"=="-y" set "NON_INTERACTIVE=1"
if /i "%~1"=="--non-interactive" set "NON_INTERACTIVE=1"
if defined CI set "NON_INTERACTIVE=1"

:: Authoritative absolute canonical project root resolution
pushd "%~dp0"
set "PROJECT_ROOT=%CD%"
popd
cd /d "%PROJECT_ROOT%"

set "BACKEND_PORT=8000"
set "FRONTEND_PORT=5173"
if not defined MAX_WAIT_SECONDS set "MAX_WAIT_SECONDS=90"
if defined MINDBRIDGE_READINESS_TIMEOUT set "MAX_WAIT_SECONDS=%MINDBRIDGE_READINESS_TIMEOUT%"

echo ==============================================================================
echo   MindBridge 2.0 -- Grounded First-Response Companion
echo ==============================================================================
echo   Project Root : %PROJECT_ROOT%
echo   Backend Port : %BACKEND_PORT% [FastAPI / PyTorch / BGE]
echo   Frontend Port: %FRONTEND_PORT% [React 18 / Vite]
echo   Readiness Max: %MAX_WAIT_SECONDS%s
echo   Platform     : Native Microsoft Windows
echo ==============================================================================
echo.

:Step1
:PreFlight
:: -----------------------------------------------------------------------------
:: Step 1: Pre-Flight Environment Validation
:: -----------------------------------------------------------------------------
echo [1/5] Verifying environment and installation prerequisites...

set "SETUP_MARKER=%PROJECT_ROOT%\.setup_completed"
if not exist "!SETUP_MARKER!" (
    if exist "%~dp0\.setup_completed" set "SETUP_MARKER=%~dp0\.setup_completed"
)

if not exist "!SETUP_MARKER!" (
    echo [WARNING] MindBridge setup completion marker [.setup_completed] was not found at:
    echo           !PROJECT_ROOT!\.setup_completed
    echo Please run FIRST_TIME_SETUP.bat to set up the environment.
    echo.
    goto PromptSetup
)

if not exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
    echo [WARNING] Python virtual environment was not found at .venv.
    echo Please run FIRST_TIME_SETUP.bat to set up the environment.
    echo.
    goto PromptSetup
)

if not exist "%PROJECT_ROOT%\frontend\node_modules" (
    echo [WARNING] Frontend node_modules directory was not found.
    echo Please run FIRST_TIME_SETUP.bat to install required packages.
    echo.
    goto PromptSetup
)

"%PROJECT_ROOT%\.venv\Scripts\python.exe" -c "import fastapi, uvicorn, torch, transformers, faiss" >nul 2>&1
if !ERRORLEVEL! neq 0 (
    echo [WARNING] Virtual environment exists but required packages are missing or broken.
    echo Please run FIRST_TIME_SETUP.bat to install required packages.
    echo.
    goto PromptSetup
)

if not exist "%PROJECT_ROOT%\models\crisis_detector\roberta\model.safetensors" (
    echo [ERROR] Crisis classifier model weights are missing.
    echo Path: models\crisis_detector\roberta\model.safetensors
    goto ErrorMissingAssets
)

if not exist "%PROJECT_ROOT%\models\embeddings\bge-small-en-v1.5\model.safetensors" (
    echo [ERROR] BGE embedding model weights are missing.
    echo Path: models\embeddings\bge-small-en-v1.5\model.safetensors
    goto ErrorMissingAssets
)

echo [OK] Prerequisites and model assets verified.
echo.

:: -----------------------------------------------------------------------------
:: Step 2: Check For Port Conflicts / Duplicate Instances
:: -----------------------------------------------------------------------------
echo [2/5] Checking port availability (:8000 and :5173)...

netstat -ano | findstr /R /C:":%BACKEND_PORT% .*LISTENING" >nul 2>&1
if !ERRORLEVEL! equ 0 (
    echo.
    echo [ERROR] Port %BACKEND_PORT% is already in use by another process!
    echo A MindBridge instance may already be running, or another service is bound to :%BACKEND_PORT%.
    echo Run stop.bat to stop existing MindBridge processes.
    goto AbortConflict
)

netstat -ano | findstr /R /C:":%FRONTEND_PORT% .*LISTENING" >nul 2>&1
if !ERRORLEVEL! equ 0 (
    echo.
    echo [ERROR] Port %FRONTEND_PORT% is already in use by another process!
    echo Run stop.bat to stop existing MindBridge processes.
    goto AbortConflict
)

echo [OK] Ports %BACKEND_PORT% and %FRONTEND_PORT% are available.
echo.

:: -----------------------------------------------------------------------------
:: Step 3: Launch Native Windows Backend Service
:: -----------------------------------------------------------------------------
echo [3/5] Starting FastAPI Backend on port %BACKEND_PORT%...

if not exist "!PROJECT_ROOT!\logs" mkdir "!PROJECT_ROOT!\logs"
if not exist "!PROJECT_ROOT!\.run" mkdir "!PROJECT_ROOT!\.run"

set "PYTHONPATH=!PROJECT_ROOT!"

:: Launch Backend terminal process
powershell -NoProfile -ExecutionPolicy Bypass -File "!PROJECT_ROOT!\scripts\launch_backend.ps1" -ProjectRoot "!PROJECT_ROOT!" -Port "!BACKEND_PORT!"

echo       Backend process initiated in dedicated terminal.
echo.

:: -----------------------------------------------------------------------------
:: Step 4: Launch Native Windows Frontend Service
:: -----------------------------------------------------------------------------
echo [4/5] Starting React/Vite Frontend on port !FRONTEND_PORT!...

powershell -NoProfile -ExecutionPolicy Bypass -File "!PROJECT_ROOT!\scripts\launch_frontend.ps1" -ProjectRoot "!PROJECT_ROOT!" -Port "!FRONTEND_PORT!"

echo       Frontend process initiated in dedicated terminal.
echo.

:: -----------------------------------------------------------------------------
:: Step 5: Wait for Backend /ready Probe and Launch Browser
:: -----------------------------------------------------------------------------
echo [5/5] Waiting for services to initialize...
echo       Pre-warming RoBERTa crisis classifier and BGE dense vector index...

set /a "WAIT_COUNT=0"

:PollLoop
:: Fast-fail if backend process terminated prematurely
if exist "!PROJECT_ROOT!\.run\backend.pid" (
    set /p "CHECK_PID="<"!PROJECT_ROOT!\.run\backend.pid"
    if defined CHECK_PID (
        tasklist /FI "PID eq !CHECK_PID!" 2>nul | findstr /i "!CHECK_PID!" >nul 2>&1
        if !ERRORLEVEL! neq 0 (
            echo.
            echo [ERROR] Backend process [PID: !CHECK_PID!] exited unexpectedly during initialization.
            echo Check the backend console window and logs for error details.
            goto TimeoutError
        )
    )
)

:: Poll ready probe (curl preferred for speed, PowerShell fallback)
where curl.exe >nul 2>&1
if !ERRORLEVEL! equ 0 (
    curl.exe -s -f -m 1 http://127.0.0.1:%BACKEND_PORT%/ready >nul 2>&1
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:%BACKEND_PORT%/ready' -UseBasicParsing -TimeoutSec 1; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
)

if !ERRORLEVEL! equ 0 goto BackendReady

set /a "WAIT_COUNT+=2"
if !WAIT_COUNT! geq %MAX_WAIT_SECONDS% goto TimeoutError

echo       Initializing models and services (!WAIT_COUNT!s / %MAX_WAIT_SECONDS%s)...
ping 127.0.0.1 -n 3 >nul 2>&1
goto PollLoop

:BackendReady
echo.
echo [OK] Backend /ready probe confirmed healthy!
echo      RoBERTa model and BGE FAISS indexes loaded successfully.

:: Wait for Vite frontend port
set /a "FE_WAIT_COUNT=0"
:FrontendWaitLoop
where curl.exe >nul 2>&1
if !ERRORLEVEL! equ 0 (
    curl.exe -s -f -m 1 http://127.0.0.1:%FRONTEND_PORT%/ >nul 2>&1
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "try { $r = Invoke-WebRequest -Uri 'http://127.0.0.1:%FRONTEND_PORT%/' -UseBasicParsing -TimeoutSec 1; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }" >nul 2>&1
)
if !ERRORLEVEL! equ 0 goto FrontendReady
set /a "FE_WAIT_COUNT+=1"
if !FE_WAIT_COUNT! geq 15 goto FrontendReady
ping 127.0.0.1 -n 2 >nul 2>&1
goto FrontendWaitLoop

:FrontendReady
echo [OK] Frontend responding on http://localhost:%FRONTEND_PORT%
echo.
echo ==============================================================================
echo   MindBridge 2.0 is LIVE!
echo   Frontend  : http://localhost:%FRONTEND_PORT%
echo   Backend   : http://localhost:%BACKEND_PORT%
echo   API Docs  : http://localhost:%BACKEND_PORT%/docs
echo ==============================================================================
echo.
if "!NON_INTERACTIVE!"=="0" (
    echo Opening MindBridge in your default browser...
    start http://localhost:%FRONTEND_PORT%
)
echo.
echo To safely stop all services, run 'stop.bat'.
echo.
if "!NON_INTERACTIVE!"=="0" pause
exit /b 0

:: -----------------------------------------------------------------------------
:: Error Handling Labels
:: -----------------------------------------------------------------------------

:PromptSetup
echo.
echo Would you like to run FIRST_TIME_SETUP.bat now? (Y/N)
if "!NON_INTERACTIVE!"=="1" (
    set "RUN_SETUP_CHOICE=Y"
) else (
    set /p "RUN_SETUP_CHOICE=Select: "
)
if /i "!RUN_SETUP_CHOICE!"=="Y" (
    if "!NON_INTERACTIVE!"=="1" (
        call "%PROJECT_ROOT%\FIRST_TIME_SETUP.bat" --no-pause
    ) else (
        call "%PROJECT_ROOT%\FIRST_TIME_SETUP.bat"
    )
    if !ERRORLEVEL! neq 0 (
        echo.
        echo [ERROR] FIRST_TIME_SETUP.bat encountered errors and exited with code !ERRORLEVEL!.
        echo Setup did not complete successfully.
        if "!NON_INTERACTIVE!"=="0" pause
        exit /b 1
    )
    if not exist "%PROJECT_ROOT%\.setup_completed" (
        echo.
        echo [ERROR] MindBridge setup completion marker [.setup_completed] was not found.
        echo First-time setup did not complete successfully.
        if "!NON_INTERACTIVE!"=="0" pause
        exit /b 1
    )
    goto Step1
)
exit /b 1

:ErrorMissingAssets
echo.
echo [ERROR] Required model assets or knowledge base files are missing.
echo Please ensure you extracted the full MindBridge release ZIP.
echo.
if "!NON_INTERACTIVE!"=="0" pause
exit /b 1

:AbortConflict
echo.
echo Startup aborted to avoid conflicting with existing running processes.
echo.
if "!NON_INTERACTIVE!"=="0" pause
exit /b 1

:TimeoutError
echo.
echo ==============================================================================
echo [ERROR] Startup timed out waiting for backend to become ready (%MAX_WAIT_SECONDS%s).
echo Check the 'MindBridge Backend (FastAPI)' console window for detailed error logs.
echo ==============================================================================
echo.
if "!NON_INTERACTIVE!"=="0" pause
exit /b 1
