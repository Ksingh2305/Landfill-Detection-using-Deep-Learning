@echo off
REM ====================================================================
REM  Launch the Streamlit web console at http://localhost:8501
REM ====================================================================
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" (
    call ".venv\Scripts\activate.bat"
) else (
    echo [WARN] No .venv found - using the system Python.
)

echo.
echo Starting the landfill detection console ...
echo Open http://localhost:8501 if your browser does not do it for you.
echo Press Ctrl+C in this window to stop the server.
echo.

python -m streamlit run app\Home.py

pause
