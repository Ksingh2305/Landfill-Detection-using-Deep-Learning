@echo off
REM ====================================================================
REM  Download data, train the model and build the evaluation report.
REM  Pass --quick for a fast run:   run_pipeline.bat --quick
REM  Pass --synthetic to work fully offline.
REM ====================================================================
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else (
    echo [WARN] No .venv found - using the system Python.
    echo        Run setup.bat first if imports fail.
)

echo.
echo Running the full pipeline ...
echo.
python scripts\run_pipeline.py %*

if errorlevel 1 (
    echo.
    echo [ERROR] The pipeline stopped early. Scroll up for the reason.
    pause
    exit /b 1
)

echo.
echo Done. Start the web console with:  run_app.bat
pause
