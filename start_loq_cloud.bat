@echo off
title LOQ Cloud Backend (Hospital Node)
color 0B
cd /d "%~dp0"

set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

REM ---- Load Network Config ----
for /f "usebackq eol=# tokens=1,2 delims==" %%A in ("%~dp0configs\network.env") do (
    set "%%A=%%B"
)

echo ============================================================
echo   Starting LOQ Cloud Backend (Hospital Node)
echo   Inference API: http://localhost:8000
echo   Chat API:      http://localhost:8001
echo   TimescaleDB:   localhost:5432
echo   Grafana:       http://localhost:3000
echo ============================================================
echo.

echo [0/3] Starting Docker containers (TimescaleDB + Grafana)...
docker compose up -d 2>nul
if %errorlevel% neq 0 (
    echo [WARN] Docker not available or containers failed — TSDB features will be disabled.
) else (
    echo [OK] Docker containers started.
)

echo [1/3] Checking Deep Inference Server on port 8000...
netstat -ano | findstr ":8000" | findstr "LISTENING" >nul
if %errorlevel% equ 0 (
    echo [ALREADY RUNNING] Port 8000 is active. Skipping duplicate launch.
) else (
    echo Launching Deep Inference Server on port 8000...
    start "LOQ Inference Server (8000)" cmd /k "venv_loq_py311\Scripts\python.exe -m uvicorn server.deep_analysis.inference_server:app --host 0.0.0.0 --port 8000"
    ping 127.0.0.1 -n 5 >nul
)

echo [2/3] Checking Ollama Chat API on port 8001...
netstat -ano | findstr ":8001" | findstr "LISTENING" >nul
if %errorlevel% equ 0 (
    echo [ALREADY RUNNING] Port 8001 is active. Skipping duplicate launch.
) else (
    echo Launching Ollama Chat API on port 8001...
    start "LOQ Chat API (8001)" cmd /k "venv_loq_py311\Scripts\python.exe -m uvicorn server.deep_analysis.chat_api:app --host 0.0.0.0 --port 8001"
    ping 127.0.0.1 -n 3 >nul
)

echo [3/3] Launching MQTT to TimescaleDB Ingestion Service...
start "MQTT-TSDB Writer" cmd /k "venv_loq_py311\Scripts\python.exe server\deep_analysis\tsdb_writer.py --broker %EDGE_BROKER_HOST% --db-host localhost"

echo.
echo [ONLINE] LOQ Cloud Backend servers running in separate windows.
echo   - Inference Server: http://localhost:8000/api/health
echo   - Chat API:         http://localhost:8001/api/health
echo   - Grafana:          http://localhost:3000 (admin / pm_grafana_2026)
echo   - TimescaleDB:      localhost:5432 (pm_user / pm_edge_2026)
echo.
pause
