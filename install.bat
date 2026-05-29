@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo   MultiModal-QC One-Click Installer
echo ========================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python was not found in PATH.
    echo Install Python 3.10+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create .venv
        pause
        exit /b 1
    )
)

echo Upgrading pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip

echo Installing Python dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Dependency installation failed.
    echo If you are in China, try:
    echo ".venv\Scripts\python.exe" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    pause
    exit /b 1
)

if not exist "storage" mkdir storage
if not exist "outputs\test_detection_data" mkdir outputs\test_detection_data

echo.
echo ========================================
echo   Install complete.
echo   Next:
echo   1. Download Qwen2-VL-2B-Instruct into models\Qwen2-VL-2B-Instruct
echo   2. Download release model/data assets if needed
echo   3. Run start.bat
echo ========================================
pause
