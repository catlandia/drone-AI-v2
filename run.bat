@echo off
REM ========================================================================
REM  Drone AI — one-click launcher (Windows)
REM  First run: installs the package + viz deps (pygame-ce).
REM  Subsequent runs: just launches the app.
REM ========================================================================

setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found on PATH. Install Python 3.9+ from python.org.
    pause
    exit /b 1
)

python -c "import drone_ai, pygame" 2>nul
if errorlevel 1 (
    echo [setup] Installing Drone AI and dependencies... this happens once.
    python -m pip install --upgrade pip
    python -m pip install -e ".[viz]"
    if errorlevel 1 (
        echo [ERROR] Installation failed. See messages above.
        pause
        exit /b 1
    )
)

python -m drone_ai.viz.launcher %*
if errorlevel 1 pause
endlocal
