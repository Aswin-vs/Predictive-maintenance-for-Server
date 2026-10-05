@echo off
title Installing Patient Dependencies
color 0B
cd /d "%~dp0"

echo.
echo ================================================================
echo   INSTALLING PATIENT TELEMETRY DEPENDENCIES
echo ================================================================
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found! 
    echo         Download from: https://www.python.org/downloads/
    echo         Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

echo [1/2] Installing core dependencies...
pip install psutil paho-mqtt
echo.

echo [2/2] Checking for NVIDIA GPU...
python -c "import pynvml; pynvml.nvmlInit(); print(f'  GPU found: {pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))}')" 2>nul
if %errorlevel% neq 0 (
    echo   No NVIDIA GPU detected or pynvml not installed.
    echo   GPU metrics will be skipped — this is fine.
    echo   To add GPU support later: pip install pynvml
) else (
    echo   NVIDIA GPU detected! GPU telemetry will be included.
)

echo.
echo ================================================================
echo   DONE! Now double-click START_PATIENT.bat to begin monitoring.
echo ================================================================
echo.
pause
