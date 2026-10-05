# ==============================================================================
# Edge-AI Predictive Maintenance - Zero-Touch Patient Installer (Windows)
# Phase 3: Tailscale Mesh Networking & Automated Client Scheduled Task / Service
# ==============================================================================
# Usage (run in elevated PowerShell):
#   .\scripts\Install-Patient-Windows.ps1
# Or with pre-set environment variable:
#   $env:TS_AUTHKEY = "tskey-auth-..."
#   .\scripts\Install-Patient-Windows.ps1
# ==============================================================================

[CmdletBinding()]
param(
    [string]$EdgeBrokerHost = "",
    [int]$EdgeBrokerPort = 1883,
    [string]$TailscaleAuthKey = ""
)

$ErrorActionPreference = "Stop"

# 1. Administrator Privilege Check
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "  Edge-AI Predictive Maintenance - Patient Telemetry Installer (Windows)" -ForegroundColor Cyan
Write-Host "==============================================================================" -ForegroundColor Cyan

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Error "[-] ERROR: This script must be executed with Administrator privileges. Please reopen PowerShell as Administrator."
    exit 1
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$InstallDir = "C:\ProgramData\PredictiveMaintenance"

# 2. Dynamic Network Configuration Resolution
$NetworkEnvPath = Join-Path $RepoRoot "configs\network.env"
if (Test-Path $NetworkEnvPath) {
    Get-Content $NetworkEnvPath | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            $parts = $line.Split("=", 2)
            $k = $parts[0].Trim()
            $v = $parts[1].Trim().Trim('"').Trim("'")
            if ($k -eq "EDGE_BROKER_HOST" -and [string]::IsNullOrEmpty($EdgeBrokerHost)) {
                $EdgeBrokerHost = $v
            }
            if ($k -eq "EDGE_BROKER_PORT" -and $EdgeBrokerPort -eq 1883) {
                [int]::TryParse($v, [ref]$EdgeBrokerPort) | Out-Null
            }
        }
    }
}

if ([string]::IsNullOrEmpty($EdgeBrokerHost)) {
    $EdgeBrokerHost = if ($env:EDGE_BROKER_HOST) { $env:EDGE_BROKER_HOST } else { "rpi4-edge.tail1ab139.ts.net" }
}

Write-Host "[+] Target Edge Broker: ${EdgeBrokerHost}:${EdgeBrokerPort}" -ForegroundColor Green

# 3. Tailscale Installation & Authentication
$TailscaleCmd = Get-Command "tailscale.exe" -ErrorAction SilentlyContinue
if (-not $TailscaleCmd) {
    $tsDefaultPath = "C:\Program Files\Tailscale\tailscale.exe"
    if (Test-Path $tsDefaultPath) {
        $TailscaleCmd = $tsDefaultPath
    }
}

if (-not $TailscaleCmd) {
    Write-Host "[+] Tailscale not detected. Downloading official Tailscale installer..." -ForegroundColor Yellow
    $msiUrl = "https://pkgs.tailscale.com/stable/tailscale-setup-latest.exe"
    $installerPath = Join-Path $env:TEMP "tailscale-setup.exe"
    
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $msiUrl -OutFile $installerPath -UseBasicParsing
    
    Write-Host "[+] Installing Tailscale silently..." -ForegroundColor Green
    Start-Process -FilePath $installerPath -ArgumentList "/quiet /norestart" -Wait
    
    Start-Sleep -Seconds 5
    $TailscaleCmd = "C:\Program Files\Tailscale\tailscale.exe"
    if (-not (Test-Path $TailscaleCmd)) {
        Write-Warning "[-] Tailscale installer finished, but tailscale.exe was not found at standard path."
    }
} else {
    Write-Host "[+] Tailscale is already installed." -ForegroundColor Green
    if ($TailscaleCmd -is [System.Management.Automation.CommandInfo]) {
        $TailscaleCmd = $TailscaleCmd.Source
    }
}

# Check Tailscale status
$tsConnected = $false
try {
    $tsJson = & $TailscaleCmd status --json 2>$null | ConvertFrom-Json
    if ($tsJson -and $tsJson.BackendState -eq "Running") {
        $tsConnected = $true
        Write-Host "[+] Tailscale is active and connected to tailnet." -ForegroundColor Green
    }
} catch {
    $tsConnected = $false
}

if (-not $tsConnected -and (Test-Path $TailscaleCmd)) {
    if ([string]::IsNullOrEmpty($TailscaleAuthKey)) {
        $TailscaleAuthKey = $env:TS_AUTHKEY
    }
    
    if ([string]::IsNullOrEmpty($TailscaleAuthKey)) {
        Write-Host ""
        Write-Host "--------------------------------------------------------------------------" -ForegroundColor Yellow
        Write-Host " Tailscale Authentication Required" -ForegroundColor Yellow
        Write-Host " If you have an Auth Key, enter it below. Otherwise press ENTER for" -ForegroundColor Yellow
        Write-Host " interactive web login." -ForegroundColor Yellow
        Write-Host "--------------------------------------------------------------------------" -ForegroundColor Yellow
        $TailscaleAuthKey = Read-Host "Enter TS_AUTHKEY (optional)"
        Write-Host ""
    }

    $computerName = $env:COMPUTERNAME.ToLower()
    $patientHostname = "patient-$computerName"
    
    if (-not [string]::IsNullOrEmpty($TailscaleAuthKey)) {
        Write-Host "[+] Connecting to Tailscale mesh using Auth Key..." -ForegroundColor Green
        & $TailscaleCmd up --authkey="$TailscaleAuthKey" --advertise-tags=tag:patient --hostname="$patientHostname" --unattended --accept-routes
    } else {
        Write-Host "[+] Initiating Tailscale login..." -ForegroundColor Green
        & $TailscaleCmd up --advertise-tags=tag:patient --hostname="$patientHostname" --unattended --accept-routes
    }
}

# 4. Check Python & Required Packages
Write-Host "[+] Checking Python environment..." -ForegroundColor Green
$pythonCmd = Get-Command "python.exe" -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Error "[-] Python was not found in PATH! Please install Python 3.8+ (ensure 'Add to PATH' is checked) and re-run."
    exit 1
}

$pyPath = $pythonCmd.Source
Write-Host "[+] Found Python at: $pyPath" -ForegroundColor Green

Write-Host "[+] Verifying Python packages (psutil, paho-mqtt)..." -ForegroundColor Green
& $pyPath -m pip install --quiet --upgrade pip
& $pyPath -m pip install --quiet psutil paho-mqtt

# 5. Application Directory Setup
Write-Host "[+] Setting up application directory: $InstallDir..." -ForegroundColor Green
if (-not (Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
}
$pluginsDir = Join-Path $InstallDir "plugins"
if (-not (Test-Path $pluginsDir)) {
    New-Item -ItemType Directory -Force -Path $pluginsDir | Out-Null
}
$trainingDir = Join-Path $InstallDir "server\training"
if (-not (Test-Path $trainingDir)) {
    New-Item -ItemType Directory -Force -Path $trainingDir | Out-Null
}

# Copy files from repository
$telemetryAgentSrc = Join-Path $RepoRoot "server\telemetry_agent"
if (Test-Path $telemetryAgentSrc) {
    Write-Host "[+] Deploying telemetry agent files..." -ForegroundColor Green
    Copy-Item -Path "$telemetryAgentSrc\*" -Destination $InstallDir -Recurse -Force
}

$schemaSrc = Join-Path $RepoRoot "server\training"
if (Test-Path $schemaSrc) {
    Copy-Item -Path "$schemaSrc\feature_schema.py" -Destination $trainingDir -Force
    Copy-Item -Path "$schemaSrc\feature_schema_v2.py" -Destination $trainingDir -Force
    New-Item -ItemType File -Force -Path (Join-Path $InstallDir "server\__init__.py") | Out-Null
    New-Item -ItemType File -Force -Path (Join-Path $trainingDir "__init__.py") | Out-Null
}

# Ensure configs directory
$cfgDir = Join-Path $InstallDir "configs"
if (-not (Test-Path $cfgDir)) {
    New-Item -ItemType Directory -Force -Path $cfgDir | Out-Null
}
if (Test-Path $NetworkEnvPath) {
    Copy-Item -Path $NetworkEnvPath -Destination $cfgDir -Force
}

# 6. Windows Scheduled Task / Auto-Restarting Service Registration
$TaskName = "PredictiveMaintenancePatient"
Write-Host "[+] Registering Windows Scheduled Task '$TaskName' to start on boot with auto-restart..." -ForegroundColor Green

$publisherScript = Join-Path $InstallDir "mqtt_publisher_v2.py"
$taskArg = "`"$publisherScript`" --broker `"$EdgeBrokerHost`" --port $EdgeBrokerPort --mode production --interval 5"

# Unregister existing task if present
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

$Action = New-ScheduledTaskAction -Execute $pyPath -Argument $taskArg -WorkingDirectory $InstallDir
$Trigger = New-ScheduledTaskTrigger -AtStartup
$Principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest

# Set task settings: restart on failure up to 10 times, interval 1 minute
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -RestartCount 10 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Days 0) `
    -StartWhenAvailable

Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings -Description "Edge-AI Predictive Maintenance Patient Server Telemetry Daemon" -Force | Out-Null

# Start the task immediately
Start-ScheduledTask -TaskName $TaskName
Start-Sleep -Seconds 3

$taskStatus = Get-ScheduledTask -TaskName $TaskName
Write-Host ""
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "  [SUCCESS] Patient Telemetry Agent installed and running as Windows Task!" -ForegroundColor Green
Write-Host "  Task Name:        $TaskName" -ForegroundColor Green
Write-Host "  State:            $($taskStatus.State)" -ForegroundColor Green
Write-Host "  Install Path:     $InstallDir" -ForegroundColor Green
Write-Host "  Edge Broker:      ${EdgeBrokerHost}:${EdgeBrokerPort}" -ForegroundColor Green
Write-Host "==============================================================================" -ForegroundColor Cyan
