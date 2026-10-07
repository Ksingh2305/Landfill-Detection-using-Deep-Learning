@echo off
REM ====================================================================
REM  Landfill Detection Project - one-time setup for Windows
REM  Creates a virtual environment and installs every dependency.
REM ====================================================================
setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo   Landfill Detection Project - setup
echo ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found on your PATH.
    echo         Install Python 3.10 - 3.12 from https://www.python.org/downloads/
    echo         and tick "Add python.exe to PATH" during installation.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo Found Python %PYVER%
echo.

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment in .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Could not create the virtual environment.
        pause
        exit /b 1
    )
) else (
    echo Virtual environment already exists - reusing it.
)

call ".venv\Scripts\activate.bat"

echo.
echo Upgrading pip ...
python -m pip install --upgrade pip --quiet

echo.
echo Installing dependencies ^(this pulls in TensorFlow, ~500 MB, be patient^) ...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Dependency installation failed.
    echo         If TensorFlow is the problem, check that your Python is 64-bit
    echo         and version 3.10, 3.11 or 3.12.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Setup complete.
echo.
echo   Next:  run_pipeline.bat    ^(downloads data and trains^)
echo   Then:  run_app.bat         ^(opens the web console^)
echo ============================================================
echo.
pause
