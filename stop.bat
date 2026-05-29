@echo off
echo ========================================
echo   MultiModal-QC Stopping...
echo ========================================

set found=0

for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr "LISTENING" ^| findstr ":8000 "') do (
    if not "%%a"=="0" (
        echo Killing port 8000 process PID=%%a
        taskkill /F /PID %%a >nul 2>&1
        set found=1
    )
)

for /f "tokens=5" %%a in ('netstat -ano 2^>nul ^| findstr "LISTENING" ^| findstr ":7860 "') do (
    if not "%%a"=="0" (
        echo Killing port 7860 process PID=%%a
        taskkill /F /PID %%a >nul 2>&1
        set found=1
    )
)

taskkill /FI "WINDOWTITLE eq QC-API*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq QC-Gradio*" /F >nul 2>&1

if %found%==0 (
    echo No running services found.
) else (
    echo ========================================
    echo   All services stopped.
    echo ========================================
)
timeout /t 3
