@echo off
title PREDICTIVE MAINTENANCE SYSTEM - TELEMETRY CACHE PURGE
color 0C
cd /d "%~dp0"

REM ---- Load Network Config ----
if exist "%~dp0configs\network.env" (
    for /f "usebackq eol=# tokens=1,2 delims==" %%A in ("%~dp0configs\network.env") do (
        set "%%A=%%B"
    )
)
if "%EDGE_API_URL%"=="" set "EDGE_API_URL=http://rpi4-edge.tail1ab139.ts.net:8080"

echo.
echo ================================================================
echo     EDGE-AI PREDICTIVE MAINTENANCE SYSTEM
echo     1-CLICK TELEMETRY DATA ^& CACHE PURGE UTILITY
echo ================================================================
echo   Target Edge Node: %EDGE_API_URL%
echo ================================================================
echo.
echo Select cache clear mode:
echo   [1] Clear ALL historical telemetry ^& reset rings across ALL servers
echo   [2] Clear cache specifically for LOQ-Patient (4c007433-db55-4309-8b41-2de057e9449b)
echo   [3] Clear cache specifically for IdeaPad-Patient (76a82f6f-4498-4914-b960-3295224ec28b)
echo   [4] Specify a custom Server ID
echo   [5] Cancel / Exit
echo.

set /p CHOICE="Enter choice [1-5]: "

if "%CHOICE%"=="1" (
    set "TARGET_SID="
    goto :EXECUTE_PURGE
)
if "%CHOICE%"=="2" (
    set "TARGET_SID=4c007433-db55-4309-8b41-2de057e9449b"
    goto :EXECUTE_PURGE
)
if "%CHOICE%"=="3" (
    set "TARGET_SID=76a82f6f-4498-4914-b960-3295224ec28b"
    goto :EXECUTE_PURGE
)
if "%CHOICE%"=="4" (
    set /p TARGET_SID="Enter Server ID: "
    goto :EXECUTE_PURGE
)
if "%CHOICE%"=="5" (
    echo Operation cancelled.
    exit /b 0
)

echo Invalid choice. Exiting.
pause
exit /b 1

:EXECUTE_PURGE
echo.
if "%TARGET_SID%"=="" (
    echo [INFO] Purging ALL telemetry records across all connected servers...
    set "REQ_URL=%EDGE_API_URL%/api/clear_cache"
) else (
    echo [INFO] Purging telemetry records for server: %TARGET_SID%...
    set "REQ_URL=%EDGE_API_URL%/api/clear_cache?server_id=%TARGET_SID%"
)

powershell -NoProfile -Command "try { $res = Invoke-RestMethod -Uri '%REQ_URL%' -Method Post -TimeoutSec 10; Write-Host ('[SUCCESS] ' + $res.message); Write-Host ('Deleted Rows: ' + $res.deleted_rows); Write-Host ('Server Target: ' + $res.server_id); } catch { Write-Host ('[ERROR] Failed to purge cache: ' + $_.Exception.Message) -ForegroundColor Red }"

echo.
echo ================================================================
echo Cache purge complete. Telemetry pipeline will ingest fresh live frames.
echo ================================================================
echo.
pause
