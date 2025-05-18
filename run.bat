@echo off
echo VisualSync Professional Launcher
echo ==============================

:: Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.8 or higher from https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Check Python version
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"
if %ERRORLEVEL% neq 0 (
    echo Error: Python 3.8 or higher is required.
    echo Your current Python version is:
    python --version
    pause
    exit /b 1
)

:: Run the application
python run.py

pause
