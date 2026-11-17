@echo off
setlocal EnableExtensions enabledelayedexpansion

:: ==============================================================================
:: MindBridge 2.0 -- Native Windows First-Time Setup
:: Location: FIRST_TIME_SETUP.bat
:: Requirements: Windows 10/11, Python 3.11+, Node.js 18+ (WSL NOT REQUIRED)
:: ==============================================================================

title MindBridge 2.0 -- Native Windows First-Time Setup

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

echo ==============================================================================
echo   MindBridge 2.0 -- Native Windows First-Time Environment Setup
echo ==============================================================================
echo   Project Root : %PROJECT_ROOT%
echo   OS Platform  : Microsoft Windows Native (No WSL / No Docker Required)
echo ==============================================================================
echo.

:: -----------------------------------------------------------------------------
:: Step 1: Detect Windows OS Version
:: -----------------------------------------------------------------------------
echo [1/8] Checking Windows Operating System...
ver >nul 2>&1
for /f "tokens=*" %%V in ('ver') do echo       Host OS: %%V
echo [OK] Windows host confirmed.
echo.

:: -----------------------------------------------------------------------------
:: Step 2: Check Python Availability and Version
:: -----------------------------------------------------------------------------
echo [2/8] Checking Python 3.11 installation...
set "PYTHON_CMD="

:: Check standard python
where python.exe >nul 2>&1
if !ERRORLEVEL! equ 0 (
    set "PYTHON_CMD=python.exe"
) else (
    where py.exe >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        set "PYTHON_CMD=py.exe -3.11"
    )
)

if "%PYTHON_CMD%"=="" goto ErrorNoPython

:: Verify version
for /f "tokens=2 delims= " %%A in ('%PYTHON_CMD% --version 2^>^&1') do set "DETECTED_PY_VER=%%A"
echo       Detected Python: %DETECTED_PY_VER% (%PYTHON_CMD%)

%PYTHON_CMD% -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) and sys.version_info < (3, 13) else 1)" >nul 2>&1
if !ERRORLEVEL! neq 0 (
    echo [WARNING] Python version %DETECTED_PY_VER% detected. MindBridge 2.0 is calibrated
    echo           for Python 3.11. Setup will proceed, but Python 3.11 is strongly recommended.
) else (
    echo [OK] Python version %DETECTED_PY_VER% is fully compatible.
)
echo.

:: -----------------------------------------------------------------------------
:: Step 3: Check Node.js and npm Availability
:: -----------------------------------------------------------------------------
echo [3/8] Checking Node.js and npm installation...
where node.exe >nul 2>&1
if !ERRORLEVEL! neq 0 goto ErrorNoNode

where npm.cmd >nul 2>&1
if !ERRORLEVEL! neq 0 (
    where npm >nul 2>&1
    if !ERRORLEVEL! neq 0 goto ErrorNoNpm
)

for /f "tokens=*" %%V in ('node.exe --version 2^>^&1') do set "NODE_VER=%%V"
for /f "tokens=*" %%V in ('call npm.cmd --version 2^>^&1') do set "NPM_VER=%%V"
echo       Detected Node.js : %NODE_VER%
echo       Detected npm     : %NPM_VER%
echo [OK] Node.js and npm prerequisites verified.
echo.

:: -----------------------------------------------------------------------------
:: Step 4: Check Available Disk Space
:: -----------------------------------------------------------------------------
echo [4/8] Checking available storage space...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$drive = (Get-Item '%PROJECT_ROOT%').PSDrive; if ($drive) { $freeGB = [math]::Round($drive.Free / 1GB, 2); Write-Host ('       Drive ' + $drive.Name + ': ' + $freeGB + ' GB free'); if ($freeGB -lt 2.0) { exit 1 } else { exit 0 } } else { exit 0 }"

if !ERRORLEVEL! neq 0 (
    echo [WARNING] Low disk space detected ^(less than 2 GB free^).
    echo           Installation of PyTorch and dependencies may fail if space runs out.
) else (
    echo [OK] Sufficient disk space available.
)
echo.

:: -----------------------------------------------------------------------------
:: Step 5: Prepare Python Virtual Environment
:: -----------------------------------------------------------------------------
echo [5/8] Preparing native Python virtual environment (.venv)...
if not exist "!PROJECT_ROOT!\.venv\Scripts\python.exe" (
    echo       Creating clean virtual environment at .venv...
    %PYTHON_CMD% -m venv "!PROJECT_ROOT!\.venv"
    if !ERRORLEVEL! neq 0 goto ErrorVenvCreate
    echo [OK] Virtual environment created.
) else (
    echo [OK] Existing virtual environment found at .venv.
)

set "VENV_PYTHON=!PROJECT_ROOT!\.venv\Scripts\python.exe"
set "VENV_PIP=!PROJECT_ROOT!\.venv\Scripts\pip.exe"

:: Ensure pip is installed inside virtualenv
if not exist "!VENV_PIP!" (
    echo       Bootstrapping pip inside virtual environment...
    "!VENV_PYTHON!" -m ensurepip --upgrade >nul 2>&1
)

echo       Upgrading pip, setuptools, and wheel...
"!VENV_PYTHON!" -m pip install --upgrade pip setuptools wheel >nul 2>&1
echo [OK] Python virtual environment ready.
echo.

:: -----------------------------------------------------------------------------
:: Step 6: Install Backend Python Dependencies
:: -----------------------------------------------------------------------------
echo [6/8] Installing backend dependencies (PyTorch CPU, FAISS, Transformers, FastAPI)...
echo       This step may take 2-4 minutes depending on your internet connection...

where uv.exe >nul 2>&1
if !ERRORLEVEL! equ 0 (
    echo       Using uv package manager for accelerated installation...
    uv pip install -r "%PROJECT_ROOT%\requirements-windows.txt" --python "%VENV_PYTHON%" --index-strategy unsafe-best-match
    if !ERRORLEVEL! neq 0 goto FallbackPip
) else (
    goto FallbackPip
)
goto PipDone

:FallbackPip
echo       Using standard pip installation with PyTorch CPU wheel index...
"%VENV_PYTHON%" -m pip install -r "%PROJECT_ROOT%\requirements-windows.txt" --extra-index-url https://download.pytorch.org/whl/cpu
if !ERRORLEVEL! neq 0 goto ErrorPipInstall

:PipDone
echo [OK] Backend dependencies installed successfully.
echo.

:: -----------------------------------------------------------------------------
:: Step 7: Install Frontend Dependencies
:: -----------------------------------------------------------------------------
echo [7/8] Installing React/Vite frontend packages...
cd /d "%PROJECT_ROOT%\frontend"

if exist "package-lock.json" (
    echo       Running npm ci ^(clean reproducible install^)...
    call npm.cmd ci
    if !ERRORLEVEL! neq 0 (
        echo       npm ci encountered an issue, falling back to npm install...
        call npm.cmd install
    )
) else (
    echo       Running npm install...
    call npm.cmd install
)

if !ERRORLEVEL! neq 0 goto ErrorNpmInstall
cd /d "%PROJECT_ROOT%"
echo [OK] Frontend packages installed successfully.
echo.

:: -----------------------------------------------------------------------------
:: Step 8: Verify ML Models, FAISS Indexes, and Runtime Configuration
:: -----------------------------------------------------------------------------
echo [8/8] Validating models, FAISS vector indexes, and configuration...

:: Verify model checkpoints
if not exist "%PROJECT_ROOT%\models\crisis_detector\roberta\model.safetensors" goto ErrorMissingModel
if not exist "%PROJECT_ROOT%\models\embeddings\bge-small-en-v1.5\model.safetensors" goto ErrorMissingEmbeddings
if not exist "%PROJECT_ROOT%\backend\safety\hotlines.json" goto ErrorMissingHotlines

:: Test Python imports
"%VENV_PYTHON%" -c "import torch; import transformers; import faiss; import fastapi; import uvicorn; import pydantic; import httpx; print('       [OK] Verified imports: PyTorch', torch.__version__, '| Transformers', transformers.__version__, '| FAISS CPU')"
if !ERRORLEVEL! neq 0 goto ErrorImports

:: Setup .env configuration
if not exist "!PROJECT_ROOT!\.env" (
    if exist "!PROJECT_ROOT!\.env.example" (
        copy "!PROJECT_ROOT!\.env.example" "!PROJECT_ROOT!\.env" >nul
        echo       Created initial .env configuration from .env.example.
    )
) else (
    echo       Preserving existing .env configuration.
)

:: Create logs and runtime directories
if not exist "!PROJECT_ROOT!\logs" mkdir "!PROJECT_ROOT!\logs"
if not exist "!PROJECT_ROOT!\.run" mkdir "!PROJECT_ROOT!\.run"

:: Write setup completion stamp using authoritative PROJECT_ROOT
(
    echo setup_timestamp=!DATE! !TIME!
    echo platform=windows
    echo python_executable=!VENV_PYTHON!
) > "!PROJECT_ROOT!\.setup_completed"

echo.
echo ==============================================================================
echo   M I N D B R I D G E   2 . 0  --  S E T U P   C O M P L E T E D !
echo ==============================================================================
echo.
echo   Everything is ready to run natively on your Windows computer!
echo.
echo   [Next Steps]
echo   1. (Optional) Open the '.env' file in Notepad if you want to add your own
echo      Gemini (GEMINI_API_KEY) or Groq (GROQ_API_KEY) keys.
echo      Note: If omitted, MindBridge runs seamlessly in local deterministic mode.
echo.
echo   2. Double-click 'run.bat' to start both Backend and Frontend.
echo      The application will automatically open in your default browser.
echo.
echo   3. Double-click 'stop.bat' whenever you want to shut down the application.
echo ==============================================================================
echo.
if "!NON_INTERACTIVE!"=="0" pause
exit /b 0

:: -----------------------------------------------------------------------------
:: Error Handling Labels
:: -----------------------------------------------------------------------------

:ErrorNoPython
echo.
echo [ERROR] Python was not found on your system!
echo.
echo To install Python:
echo   1. Visit: https://www.python.org/downloads/
echo   2. Download Python 3.11 (e.g., Python 3.11.9 Windows 64-bit installer).
echo   3. IMPORTANT: During installation, CHECK the box:
echo      "Add python.exe to PATH"
echo   4. Re-run FIRST_TIME_SETUP.bat after installation.
echo.
if "!NON_INTERACTIVE!"=="0" pause
exit /b 1

:ErrorNoNode
echo.
echo [ERROR] Node.js was not found on your system!
echo.
echo To install Node.js:
echo   1. Visit: https://nodejs.org/
echo   2. Download the LTS version (Long Term Support).
echo   3. Run the installer with default options.
echo   4. Re-run FIRST_TIME_SETUP.bat after installation.
echo.
if "!NON_INTERACTIVE!"=="0" pause
exit /b 1

:ErrorNoNpm
echo.
echo [ERROR] npm (Node Package Manager) was not found in your PATH!
echo Please reinstall Node.js LTS from https://nodejs.org/ to restore npm.
echo.
if "!NON_INTERACTIVE!"=="0" pause
exit /b 1

:ErrorVenvCreate
echo.
echo [ERROR] Failed to create virtual environment at '%PROJECT_ROOT%\.venv'.
echo Please check your file permissions or antivirus software and try again.
echo.
if "!NON_INTERACTIVE!"=="0" pause
exit /b 1

:ErrorPipInstall
echo.
echo [ERROR] Python dependency installation failed!
echo Please check your internet connection and try running setup again.
echo.
if "!NON_INTERACTIVE!"=="0" pause
exit /b 1

:ErrorNpmInstall
echo.
echo [ERROR] Frontend npm package installation failed!
echo Please check your internet connection and try running setup again.
echo.
cd /d "%PROJECT_ROOT%"
if "!NON_INTERACTIVE!"=="0" pause
exit /b 1

:ErrorMissingModel
echo.
echo [ERROR] Crisis classifier model weights not found at:
echo   models\crisis_detector\roberta\model.safetensors
echo Please make sure the complete MindBridge package was extracted.
echo.
if "!NON_INTERACTIVE!"=="0" pause
exit /b 1

:ErrorMissingEmbeddings
echo.
echo [ERROR] Embedding model weights not found at:
echo   models\embeddings\bge-small-en-v1.5\model.safetensors
echo Please make sure the complete MindBridge package was extracted.
echo.
if "!NON_INTERACTIVE!"=="0" pause
exit /b 1

:ErrorMissingHotlines
echo.
echo [ERROR] Emergency hotlines directory not found at:
echo   backend\safety\hotlines.json
echo.
if "!NON_INTERACTIVE!"=="0" pause
exit /b 1

:ErrorImports
echo.
echo [ERROR] Package import verification failed!
echo Some Python dependencies could not be imported.
echo Please review the output above or re-run setup.
echo.
if "!NON_INTERACTIVE!"=="0" pause
exit /b 1
