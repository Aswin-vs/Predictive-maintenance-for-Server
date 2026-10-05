@echo off
title PREDICTIVE MAINTENANCE - STOP ALL SERVICES
color 0C
cd /d "%~dp0"

REM ---- Load Network Config ----
for /f "usebackq eol=# tokens=1,2 delims==" %%A in ("%~dp0configs\network.env") do (
    set "%%A=%%B"
)

echo.
echo ================================================================
echo              STOPPING ALL SERVICES
echo ================================================================
echo.

echo [1/4] Stopping Python processes...
taskkill /FI "WINDOWTITLE eq Cloud_Inference*" /F 2>nul
taskkill /FI "WINDOWTITLE eq Chat_API*" /F 2>nul
taskkill /FI "WINDOWTITLE eq TSDB_Writer*" /F 2>nul
taskkill /FI "WINDOWTITLE eq LOQ_Patient*" /F 2>nul
taskkill /FI "WINDOWTITLE eq RPi4_Edge*" /F 2>nul
taskkill /FI "WINDOWTITLE eq Patient Telemetry*" /F 2>nul
echo       Done.

echo [2/4] Stopping uvicorn servers on ports 8000 and 8001...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8000" ^| find "LISTENING"') do (
    taskkill /PID %%a /F 2>nul
)
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8001" ^| find "LISTENING"') do (
    taskkill /PID %%a /F 2>nul
)
echo       Done.

echo [3/4] Stopping RPi4 edge-ai service via SSH (if reachable)...
venv_loq_py311\Scripts\python.exe -c "import paramiko; ssh=paramiko.SSHClient(); ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy()); ssh.connect('%RPI4_IP%',username='%RPI4_USER%',password='%RPI4_PASS%',timeout=3); ssh.exec_command('echo %RPI4_PASS% | sudo -S systemctl stop edge-ai'); print('  RPi4 edge-ai stopped.'); ssh.close()" 2>nul
if %errorlevel% neq 0 echo       [SKIP] RPi4 not reachable
echo       Done.

echo [4/4] Stopping Docker containers...
docker compose stop 2>nul
if %errorlevel% neq 0 (
    echo       [SKIP] Docker not running
) else (
    echo       TimescaleDB + Grafana stopped.
)

echo.
echo ================================================================
echo              ALL SERVICES STOPPED
echo ================================================================
echo.
pause
