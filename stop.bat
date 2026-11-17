@echo off
setlocal EnableExtensions enabledelayedexpansion

:: ==============================================================================
:: MindBridge 2.0 -- Safe Native Windows Process Stopper
:: Location: stop.bat
:: Requirements: Windows 10/11 (WSL / Docker NOT REQUIRED)
:: Invariant: NEVER executes taskkill /IM python.exe or taskkill /IM node.exe
:: ==============================================================================

title MindBridge 2.0 Stopper

pushd "%~dp0"
set "PROJECT_ROOT=%CD%"
popd
cd /d "%PROJECT_ROOT%"

set "BACKEND_PORT=8000"
set "FRONTEND_PORT=5173"

echo ==============================================================================
echo   MindBridge 2.0 -- Safe Service Terminator
echo ==============================================================================
echo   Project Root : %PROJECT_ROOT%
echo ==============================================================================
echo.

:: -----------------------------------------------------------------------------
:: Step 1: Terminate Backend by Tracked PID
:: -----------------------------------------------------------------------------
if exist "!PROJECT_ROOT!\.run\backend.pid" (
    set /p "BACKEND_PID="<"!PROJECT_ROOT!\.run\backend.pid"
    if defined BACKEND_PID (
        echo [INFO] Found Backend process tracking PID: !BACKEND_PID!
        tasklist /FI "PID eq !BACKEND_PID!" 2>nul | findstr /i "!BACKEND_PID!" >nul 2>&1
        if !ERRORLEVEL! equ 0 (
            echo        Stopping Backend process tree ^(PID: !BACKEND_PID!^)...
            taskkill /PID !BACKEND_PID! /T /F >nul 2>&1
            echo [OK] Backend process terminated.
        ) else (
            echo        Process !BACKEND_PID! is not currently running.
        )
    )
    del /f /q "!PROJECT_ROOT!\.run\backend.pid" >nul 2>&1
)

:: -----------------------------------------------------------------------------
:: Step 2: Terminate Frontend by Tracked PID
:: -----------------------------------------------------------------------------
if exist "!PROJECT_ROOT!\.run\frontend.pid" (
    set /p "FRONTEND_PID="<"!PROJECT_ROOT!\.run\frontend.pid"
    if defined FRONTEND_PID (
        echo [INFO] Found Frontend process tracking PID: !FRONTEND_PID!
        tasklist /FI "PID eq !FRONTEND_PID!" 2>nul | findstr /i "!FRONTEND_PID!" >nul 2>&1
        if !ERRORLEVEL! equ 0 (
            echo        Stopping Frontend process tree ^(PID: !FRONTEND_PID!^)...
            taskkill /PID !FRONTEND_PID! /T /F >nul 2>&1
            echo [OK] Frontend process terminated.
        ) else (
            echo        Process !FRONTEND_PID! is not currently running.
        )
    )
    del /f /q "!PROJECT_ROOT!\.run\frontend.pid" >nul 2>&1
)

:: -----------------------------------------------------------------------------
:: Step 3: Check and Free Lingering Sockets on Ports 8000 and 5173
:: -----------------------------------------------------------------------------
echo.
echo Checking port status...

:: Check Port 8000
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%BACKEND_PORT% .*LISTENING"') do (
    set "LINGERING_PID=%%P"
    if defined LINGERING_PID (
        echo [INFO] Releasing port %BACKEND_PORT% from lingering process PID !LINGERING_PID!...
        taskkill /PID !LINGERING_PID! /T /F >nul 2>&1
    )
)

:: Check Port 5173
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":%FRONTEND_PORT% .*LISTENING"') do (
    set "LINGERING_PID=%%P"
    if defined LINGERING_PID (
        echo [INFO] Releasing port %FRONTEND_PORT% from lingering process PID !LINGERING_PID!...
        taskkill /PID !LINGERING_PID! /T /F >nul 2>&1
    )
)

:: Verify ports are now free
netstat -ano | findstr /R /C:":%BACKEND_PORT% .*LISTENING" >nul 2>&1
if !ERRORLEVEL! equ 0 (
    echo [WARNING] Port %BACKEND_PORT% is still held by an active process.
) else (
    echo [OK] Port %BACKEND_PORT% is free.
)

netstat -ano | findstr /R /C:":%FRONTEND_PORT% .*LISTENING" >nul 2>&1
if !ERRORLEVEL! equ 0 (
    echo [WARNING] Port %FRONTEND_PORT% is still held by an active process.
) else (
    echo [OK] Port %FRONTEND_PORT% is free.
)

:: Step 4: Release Virtual Drive Mapping (if any)
subst X: /d >nul 2>&1
subst Y: /d >nul 2>&1

echo.
echo ==============================================================================
echo   MindBridge shutdown complete.
echo ==============================================================================
echo.
ping 127.0.0.1 -n 4 >nul 2>&1
exit /b 0
