@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo   MultiModal-QC Starting...
echo ========================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv not found
    echo Run install.bat first.
    pause
    exit /b 1
)

if not exist "storage" mkdir storage
del /Q storage\api.log storage\gradio.log 2>nul

echo Starting API server (port 8000)...
start /B "" cmd /c ""%CD%\.venv\Scripts\python.exe" -m uvicorn serve.api:app --port 8000 --host 0.0.0.0 > storage\api.log 2>&1"

echo Starting Gradio frontend (port 7860)...
timeout /t 3 /nobreak >nul
start /B "" cmd /c ""%CD%\.venv\Scripts\python.exe" serve\app.py > storage\gradio.log 2>&1"

echo.
echo ========================================
echo   Services starting in background.
echo   Tailing logs below (Ctrl+C to stop tail, services keep running)
echo   Browser will auto-open after 20 sec.
echo ========================================
echo.

REM Auto open browser after delay
start /B "" cmd /c "timeout /t 20 /nobreak >nul && start http://localhost:7860"

REM Live tail both logs
powershell -NoProfile -Command "Get-Content -Path 'storage\api.log','storage\gradio.log' -Wait -Tail 0"
