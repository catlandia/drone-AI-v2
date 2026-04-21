@echo off
REM ========================================================================
REM  Drone AI — one-click launcher (Windows)
REM  First run: installs the package + viz deps (pygame-ce).
REM  Subsequent runs: just launches the app.
REM ========================================================================

setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python launcher "py" not found. Install Python 3.9+ from python.org.
    pause
    exit /b 1
)

py -c "import drone_ai, pygame" 2>nul
if errorlevel 1 (
    echo [setup] Installing Drone AI and dependencies... this happens once.
    py -m pip install --upgrade pip
    py -m pip install -e ".[viz]"
    if errorlevel 1 (
        echo [ERROR] Installation failed. See messages above.
        pause
        exit /b 1
    )
)

py -m drone_ai.viz.launcher %*
if errorlevel 1 pause
endlocal
