@echo off
title PREDICTIVE MAINTENANCE SYSTEM - MASTER LAUNCHER
color 0F
cd /d "%~dp0"

set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

REM ---- Load Network Config ----
for /f "usebackq eol=# tokens=1,2 delims==" %%A in ("%~dp0configs\network.env") do (
    set "%%A=%%B"
)

echo.
echo ================================================================
echo     EDGE-AI PREDICTIVE MAINTENANCE SYSTEM
echo     MASTER ONE-CLICK LAUNCHER v2.0
echo ================================================================
echo   Tier 1 (Patient):  LOQ Hardware Telemetry v2 - MQTT
echo   Tier 2 (Doctor):   RPi4 Edge AI + Dashboard (:8080)
echo   Tier 3 (Hospital): Cloud Inference + Chat + TimescaleDB
echo ================================================================
echo.

REM Save project directory
set "PROJECT_DIR=%~dp0"

REM ---- STEP 0: Check prerequisites ----
echo [0/6] Checking prerequisites...
if not exist "%PROJECT_DIR%venv_loq_py311\Scripts\python.exe" (
    echo [FATAL] venv_loq_py311 not found!
    pause
    exit /b 1
)
echo       OK - Python venv found
echo.

REM ---- STEP 1: Start Docker (TimescaleDB + Grafana) ----
echo [1/6] Starting Docker containers (TimescaleDB + Grafana)...
docker compose up -d 2>nul
if %errorlevel% neq 0 (
    echo       [WARN] Docker not available - TimescaleDB features disabled.
) else (
    echo       OK - TimescaleDB [5432] + Grafana [3000] started
)
echo.

REM ---- STEP 2: Start Cloud Inference Server (Port 8000) ----
echo [2/6] Checking Cloud Inference Server on port 8000...
netstat -ano | findstr ":8000" | findstr "LISTENING" >nul
if %errorlevel% equ 0 (
    echo       [ALREADY RUNNING] Port 8000 is active and healthy. Skipping duplicate launch.
) else (
    echo       Starting Cloud Inference Server on port 8000...
    echo       Models: 1D-CNN, LSTM-AE, MAD-GAN, STGCN, Cloud Fusion, SHAP
    start "Cloud_Inference_8000" cmd /k "venv_loq_py311\Scripts\python.exe -m uvicorn server.deep_analysis.inference_server:app --host 0.0.0.0 --port 8000"
    ping 127.0.0.1 -n 5 >nul
    echo       OK - Inference Server launched
)
echo.

REM ---- STEP 3: Start Chat API (Port 8001) ----
echo [3/6] Checking Ollama Chat API on port 8001...
netstat -ano | findstr ":8001" | findstr "LISTENING" >nul
if %errorlevel% equ 0 (
    echo       [ALREADY RUNNING] Port 8001 is active and healthy. Skipping duplicate launch.
) else (
    echo       Starting Ollama Chat API on port 8001...
    start "Chat_API_8001" cmd /k "venv_loq_py311\Scripts\python.exe -m uvicorn server.deep_analysis.chat_api:app --host 0.0.0.0 --port 8001"
    ping 127.0.0.1 -n 4 >nul
    echo       OK - Chat API launched
)
echo.

REM ---- STEP 4: Start MQTT to TimescaleDB Writer ----
echo [4/6] Starting MQTT to TimescaleDB Ingestion...
start "TSDB_Writer" cmd /k "venv_loq_py311\Scripts\python.exe server\deep_analysis\tsdb_writer.py --broker %RPI4_IP% --db-host localhost"
ping 127.0.0.1 -n 3 >nul
echo       OK - TSDB Writer launched [subscribes to RPi4 on port %MQTT_PORT%]
echo.

REM ---- STEP 5: Start RPi4 Edge Node (via SSH) ----
echo [5/6] Starting RPi4 Edge Node (Doctor) via SSH...
start "RPi4_Edge" cmd /k "venv_loq_py311\Scripts\python.exe scripts\start_rpi4_service.py"
ping 127.0.0.1 -n 4 >nul
echo       OK - RPi4 restart command sent
echo.

REM ---- STEP 6: Start Patient Telemetry v2 (LOQ as Patient) ----
echo [6/6] Starting Patient Telemetry v2.0 (LOQ hardware monitoring)...
echo       Auto-Discovery + Plugin-Based Collection + Hierarchical Schema
start "LOQ_Patient_v2" cmd /k "venv_loq_py311\Scripts\python.exe server\telemetry_agent\mqtt_publisher_v2.py --broker %RPI4_IP% --port %MQTT_PORT% --server-name LOQ-Patient --interval 5"
echo       OK - Patient telemetry launched
echo.

REM ---- DONE ----
echo.
echo ================================================================
echo                   ALL SERVICES LAUNCHED!
echo ================================================================
echo   Cloud Inference:   http://localhost:8000/api/health
echo   Chat API:          http://localhost:8001/api/health
echo   Grafana:           http://localhost:3000  (admin/pm_grafana_2026)
echo   TimescaleDB:       localhost:5432  (pm_user/pm_edge_2026)
echo   RPi4 Dashboard:    http://%RPI4_IP%:8080
echo   RPi4 MQTT Broker:  %RPI4_IP%:%MQTT_PORT%
echo ================================================================
echo   Each service runs in its own window.
echo   Close individual windows to stop specific services.
echo ================================================================
echo.
pause
