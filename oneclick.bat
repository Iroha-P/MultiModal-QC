@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo   MultiModal-QC One-Click Launcher
echo ========================================
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\oneclick_setup.ps1"
if errorlevel 1 (
    echo.
    echo One-click setup failed. Check the messages above.
    pause
    exit /b 1
)

pause
