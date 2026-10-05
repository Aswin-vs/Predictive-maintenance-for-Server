# ⚡ Master Execution Prompt & Handover Guide for Gemini 3.8 Flash (v3.0)

> **Project:** Edge-AI Predictive Maintenance for Server Hardware  
> **Execution Agent:** Gemini 3.8 Flash (Configured with **Thinking Budget: High**) inside Google Antigravity  
> **Companion Architectural Spec:** `master_production_readiness_blueprint_v4.md`  
> **Execution Mode in Antigravity:** Use **Planning Mode** (🧭) for all phases.  
> **Visual Verification Tool:** Antigravity Browser Sub-Agent (Chromium GUI inspection)  

---

## 📌 Executive Architecture & Operating Rules

### Critical Ground Truth Discovered via Live Codebase & RPi4 Inspection:
1. **Gemini 3.8 Flash Context & Output Realities:**
   * Gemini 3.8 Flash has a **1,048,576-token input context window** and a **65,536-token max output per turn**.
   * Truncation does *not* occur from running out of input context; it occurs when a single response attempts to output more than 65K tokens, or when the model takes "lazy coding" shortcuts (e.g., `# ... existing code unchanged ...`).
   * **Rule:** Execute **EXACTLY ONE PHASE PER PROMPT**. Require 100% complete, drop-in replacement files/functions with zero placeholders.
2. **Correction of Claude Sonnet's Phase 2 Mistake:**
   * Claude assumed Phase 2 ("Physical Sensors") was already implemented in the repository. **It is NOT.**
   * In the local repository, `edge/sensor_reader.py` still contains uninstalled Adafruit imports (`adafruit_adxl34x`, `adafruit_mcp3xxx`) and falls back to simulated data.
   * Both sensors (**INMP441** on ALSA Card 3 and **ADXL345** on I2C Bus 1 address `0x53`) were wired and proven working in `/home/pi/mic_test/`, but the production code in `edge/sensor_reader.py` MUST be updated to use `smbus2` and ALSA `plughw:3,0`. Phase 2 is an **ACTIVE CODE IMPLEMENTATION** phase!
3. **Mandatory Documentation Synchronization (Global Rule):**
   * At the end of **EVERY SINGLE PHASE**, Gemini 3.8 Flash MUST execute the documentation sync step:
     * Check off completed tasks `[x]` in `@MEMORIES.md`.
     * Log any discovered technical debt or bugs in `@MEMORIES.md`.
     * Append a plain-English summary of changes to `@PROJECT_SUMMARY.md`.

---

## 🛡️ Global System Prompt for Gemini 3.8 Flash
*(Always include this system header with every phase prompt)*

```text
You are an expert Principal Systems & Machine Learning Engineer implementing the enterprise production hardening for the "Edge-AI Predictive Maintenance for Server Hardware" repository.

CRITICAL OPERATIONAL RULES:
1. NO PLACEHOLDERS: Never output "# ... existing code unchanged ..." or "// TODO". Always write complete, production-ready, drop-in replacement code.
2. VERIFY LIVE LINE NUMBERS: Before editing any file, view its CURRENT content and line numbers using view_file or grep_search. Do not trust static line numbers from planning docs.
3. STRICT SINGLE-PHASE FOCUS: Focus strictly on the CURRENT assigned phase. Do not jump ahead to future phases.
4. MANDATORY GRACEFUL DEGRADATION: Every plugin or data-source read that depends on hardware which may not be present (BMC, ECC RAM, NVMe drive, physical sensor) MUST catch that absence, log it once, report the field as null/"unsupported", and continue running without crashing.
5. EXHAUSTIVE TESTING GATES: After making code changes, ALWAYS execute the provided test commands using run_command. If a test fails, diagnose and fix it immediately before declaring the phase complete.
6. NO HARDCODED SECRETS: Never write Tailscale auth keys, passwords, or tokens directly into committed scripts. Read them from environment variables or prompt interactively.
7. MANDATORY MEMORY SYNCHRONIZATION: At the end of this phase, you MUST update @MEMORIES.md (check off completed sprint tasks, log bugs/lessons learned) and @PROJECT_SUMMARY.md (append a high-level plain-English progress summary).
```

---

## 🚀 Phase 1: Critical Bug Remediation & Backend Hardening 🧭

### Copy-Paste Prompt for Gemini 3.8 Flash:
```text
Execute PHASE 1 of the Production Hardening Plan: Critical Bug Remediation.

Before editing, view each file below to confirm current line numbers and surrounding context.

1. BUG-01: Silent TTL Eviction Crash in `edge/api_server.py`
   - Problem: `now - s["last_seen"] > 1800` raises `TypeError: unsupported operand type(s) for -: 'float' and 'str'` because `last_seen` is stored as an ISO-8601 string while `now` is a float.
   - Fix: In `_cleanup_loop()`, parse `last_seen` safely:
     ```python
     if isinstance(ls, str):
         try:
             ls_ts = datetime.fromisoformat(ls.replace('Z', '+00:00')).timestamp()
         except Exception:
             ls_ts = None
     elif isinstance(ls, (int, float)):
         ls_ts = float(ls)
     else:
         ls_ts = None
     ```
     Evict the server if `ls_ts and (now - ls_ts > 1800)`. Wrap with try/except logging.

2. BUG-02: ST-GCN Data Leakage in `server/training/train_stgcn.py`
   - Problem: `scaler.fit_transform(X_raw)` is executed on the entire dataset BEFORE the 80/20 train/test split. Test set statistics leak into the scaler.
   - Fix: Split raw data into `X_train_raw` and `X_test_raw` first. Call `scaler.fit_transform(X_train_raw)`, and call `scaler.transform(X_test_raw)` on the test set. Compute the graph adjacency matrix `adj` strictly from training correlations.

3. BUG-03: Auto-Retraining Data Leakage in `server/training/auto_retrain.py`
   - Problem: Same data leakage occurs during auto-retraining.
   - Fix: Chronologically split data first (first 90% train, last 10% validation). Fit `StandardScaler` strictly on the train partition.

4. BUG-04: Unbounded Memory Leak in `server/deep_analysis/inference_server.py`
   - Problem: `_rate_tracker = defaultdict(list)` and `_server_windows = defaultdict(deque)` grow infinitely with new client connections.
   - Fix: Replace with bounded TTL caches using `cachetools.TTLCache(maxsize=1000, ttl=3600)` or implement an explicit periodic eviction task removing keys inactive for > 1 hour.

5. BUG-05: Non-Functional Table Sorting & Mojibake in `dashboard/`
   - Problem: Column headers have `sort-col` class but no event listeners. Special characters display Mojibake (`â€”` instead of `—`, `â€¢` instead of `•`).
   - Fix:
     a) In `dashboard/index.html`, ensure `<meta charset="UTF-8">` is the first element in `<head>`.
     b) In `dashboard/assets/app.js`, attach click listeners to all `.sort-col` headers. Implement bidirectional sort (asc/desc) across numeric and string columns and re-render `cache.historyRecords`.
     c) In `edge/api_server.py`, ensure HTML and static responses explicitly set `media_type="text/html; charset=utf-8"`.

VERIFICATION & TESTING GATE:
1. Run automated unit tests validating:
   - ISO timestamp parsing in api_server.py
   - StandardScaler isolation in train_stgcn.py (assert scaler does not see test data)
   - Cache bounds in inference_server.py
2. Visual Browser Verification:
   - Use the Antigravity Browser Sub-Agent to navigate to `http://localhost:8080` (or the active dashboard port).
   - Click a table column header twice. Capture screenshots confirming the sort arrow toggles and rows re-order correctly.
   - Visually confirm that Mojibake (`â€”`) is completely eradicated and characters render cleanly as `—` and `•`.

MANDATORY DOCUMENTATION SYNC:
- Update @MEMORIES.md: Mark BUG-01 through BUG-05 as resolved `[x]`.
- Update @PROJECT_SUMMARY.md: Append summary of Phase 1 bug fixes and verification results.
```

---

## 🚀 Phase 2: Physical Hardware Sensor Driver Integration 🧭

### Copy-Paste Prompt for Gemini 3.8 Flash:
```text
Execute PHASE 2 of the Production Hardening Plan: Physical Hardware Sensor Driver Integration.

Background & Hardware Truth:
- The INMP441 I2S MEMS microphone is active on RPi4 as ALSA Card 3 (`snd_rpi_googlevoicehat_soundcard`, 32-bit S32_LE at 44.1 kHz). Kernel overlay is permanently persisted in `/boot/firmware/config.txt`.
- The ADXL345 3-Axis accelerometer is active on RPi4 on I2C Bus 1 at address `0x53` (DEVID `0xE5`).
- The Python library `smbus2` is installed on RPi4, but `adafruit_adxl34x` and `adafruit_mcp3xxx` are NOT installed.
- Current repository code in `edge/sensor_reader.py` still has uninstalled Adafruit imports and falls back to simulated data. You must replace it with the verified real drivers.

Tasks:
1. Update `edge/sensor_reader.py`:
   - Remove dependency on `adafruit_adxl34x` and `adafruit_mcp3xxx`.
   - Implement ADXL345 driver using `smbus2.SMBus(1)`:
     * Read registers `0x32`–`0x37` in full resolution (+/-16g, scale factor 0.0039 g/LSB).
     * Calculate dynamic vibration RMS: `vibration_rms = abs(sqrt(gx^2 + gy^2 + gz^2) - 1.0)` (removing 1G gravity).
     * Calculate peak acceleration and 32-sample kurtosis (bearing impulse shock detector).
   - Implement INMP441 driver using ALSA capture (`arecord` subprocess or ALSA stream from `plughw:3,0`):
     * Sample 0.2s burst at 44.1 kHz, 32-bit S32_LE.
     * Calculate normalized acoustic RMS: `rms = sqrt(mean(left_channel^2)) / 2^31`.
     * Calculate dominant frequency using FFT (`scipy.fft` or `numpy.fft`): `peak_freq = freqs[argmax(abs(fft_vals))]`.
   - Maintain graceful degradation: if an IOError occurs (e.g. wire loose), log once, fall back to last known good value, increment failure counter, and keep running without crashing.
   - Update `read_all()` and `get_status()` return schemas to report `"source": "inmp441"` and `"source": "adxl345"`.

2. Update `edge/main.py`:
   - Map physical `vibration_rms` and `acoustic_rms` into the edge feature vector.
   - Publish real sensor metrics to MQTT topic `edge/sensors`.

VERIFICATION & TESTING GATE:
1. Deploy updated `edge/sensor_reader.py` to RPi4 (`192.168.137.100`) via SSH/SFTP.
2. Run `python3 -m edge.sensor_reader` on RPi4.
3. Assert that:
   - `adxl345` status is `"OK"`, reading genuine ~1.03g total gravity.
   - `inmp441` status is `"OK"`, capturing dynamic acoustic RMS > 0.001.
   - Zero simulation fallback warnings appear in the log.
4. Restart `edge-ai.service` on RPi4 and confirm via `systemctl status edge-ai` that live sensor metrics stream into the dashboard.

MANDATORY DOCUMENTATION SYNC:
- Update @MEMORIES.md: Mark Phase 2 physical sensor driver integration as complete `[x]`.
- Update @PROJECT_SUMMARY.md: Append summary of real sensor driver deployment and live RPi4 verification.
```

---

## 🚀 Phase 3: Tailscale Mesh Networking & Zero-Touch Client Installer 🧭

### Copy-Paste Prompt for Gemini 3.8 Flash:
```text
Execute PHASE 3 of the Production Hardening Plan: Virtual Local Mesh Networking.

Read master_production_readiness_blueprint_v4.md §2 first for architectural context.

Tasks:
1. Environment & Dynamic Hostname Configuration:
   - Create `configs/network.env`:
     ```env
     TS_TAILNET_SUFFIX="your-tailnet.ts.net"
     EDGE_BROKER_HOST="rpi4-edge.${TS_TAILNET_SUFFIX}"
     EDGE_BROKER_PORT=1883
     EDGE_API_URL="http://rpi4-edge.${TS_TAILNET_SUFFIX}:8080"
     CLOUD_INFERENCE_URL="http://loq-cloud.${TS_TAILNET_SUFFIX}:8000"
     ```
   - Update `edge/mqtt_handler.py`, `server/telemetry_agent/mqtt_publisher_v2.py`, and `dashboard/assets/app.js` to read from environment variables (or fall back to `localhost` / mDNS), removing all hardcoded `192.168.137.x` literals.

2. Create Cross-Platform Zero-Touch Client Installers:
   a) `scripts/install_patient_linux.sh`:
      - Checks for root/sudo privileges.
      - Reads Tailscale auth key from environment variable `TS_AUTHKEY` or prompts interactively (never hardcoded in script text).
      - Installs Tailscale via official repository.
      - Executes: `tailscale up --authkey="$TS_AUTHKEY" --advertise-tags=tag:patient --hostname="patient-$(hostname)"`.
      - Creates `/opt/patient-telemetry/` with Python venv.
      - Copies `server/telemetry_agent/` files and installs requirements.
      - Configures, enables, and starts `systemd` service `patient-telemetry.service`.
      - Confirms first successful MQTT publish to `EDGE_BROKER_HOST`.

   b) `scripts/Install-Patient-Windows.ps1`:
      - Checks for Administrator privileges.
      - Reads `TS_AUTHKEY` from environment or prompt.
      - Silently downloads and installs Tailscale MSI.
      - Joins tailnet with `--advertise-tags=tag:patient`.
      - Copies telemetry agent to `C:\ProgramData\PredictiveMaintenance\`.
      - Registers a Windows Scheduled Task or NSSM service to run `collector_v2.py` at boot with auto-restart on failure.
      - Starts agent and verifies connectivity.

3. Document Tailscale Network Segmentation Policy:
   - Create `docs/tailscale_acl_policy.json` containing the least-privilege ACL rules:
     * `group:edge` (RPi4) reachable by all on port 1883 & 8080.
     * `group:cloud` (LOQ) reachable only by `group:edge` on port 8000.
     * `group:patients` (tag:patient) can only reach `group:edge:1883`. Patients cannot communicate with each other.

VERIFICATION & TESTING GATE:
1. Run `grep_search` across the entire repository for `"192.168.137."` and confirm ZERO hardcoded IP literals remain in code or scripts.
2. Run a secrets scan ensuring no `tskey-auth-...` tokens are committed in any file.
3. Test `mqtt_publisher_v2.py` in dry-run mode ensuring hostname resolution works cleanly via environment configuration.

MANDATORY DOCUMENTATION SYNC:
- Update @MEMORIES.md: Mark Phase 3 mesh networking and installer creation as complete `[x]`.
- Update @PROJECT_SUMMARY.md: Append summary of Tailscale network migration and installer capabilities.
```

---

## 🚀 Phase 4: Tiered Hardware Degradation Telemetry Plugins 🧭

### Copy-Paste Prompt for Gemini 3.8 Flash:
```text
Execute PHASE 4 of the Production Hardening Plan: Tiered Hardware Degradation Telemetry.

Read master_production_readiness_blueprint_v4.md §4 first for degradation physics.
MANDATORY RULE: Every plugin must implement graceful degradation. If hardware or libraries are missing, log once, set field to null/"unsupported", and continue running without throwing unhandled exceptions.

Refactor and expand plugins in `server/telemetry_agent/plugins/`:

1. `plugins/cpu_plugin.py` (Tier 1 Cross-Platform):
   - Linux: Read temps via `lm-sensors` (`sensors -j`), package power via RAPL / `turbostat`, and parse kernel MCE events (`mcelog` / `journalctl -k | grep -i mce`).
   - Windows: Poll LibreHardwareMonitor Remote Web Server JSON (`http://localhost:8085/data.json`) for CPU package temp, power, core load, and clocks. Fall back to WMI/psutil if web server is inactive.
   - Degradation Indicator: Calculate Thermal Resistance slope: $\Delta T / \Delta P = (T_{cpu} - T_{ambient}) / P_{watts}$ over a 60s sliding window. An upward trend flags thermal paste dry-out/pump-out. Track PROCHOT throttle event counts.

2. `plugins/gpu_plugin.py` (Tier 1 Cross-Platform):
   - Primary: Use `pynvml` for NVIDIA GPU temperature, power, utilization, VRAM usage, and `nvmlDeviceGetMemoryErrorCounter` for ECC single-bit / double-bit error counts.
   - Fault Tracking:
     * Linux: Parse `journalctl -k` for `NVRM: Xid` events (classify Xid 79 fatal bus drop, Xid 63 ECC row-remap exhausted, Xid 48 double-bit ECC, Xid 45 row-remap rate).
     * Windows: Query System Event Log for Source=`nvlddmkm` using `win32evtlog` or PowerShell `Get-WinEvent`.
   - Secondary: Fall back to `rocm-smi` for AMD GPUs.
   - Physical Indicator: Calculate Hotspot Delta: $T_{junction} - T_{edge}$. Delta $> 30^\circ\text{C}$ indicates thermal pad deterioration.
   - Note: Document that per-VRM-phase current telemetry is unavailable without proprietary vendor SDKs; board-level power draw is used as proxy.

3. `plugins/motherboard_plugin.py` (Tier 1):
   - Linux: Read PCH/chipset and VRM temperatures via `lm-sensors`.
   - Windows: Poll LibreHardwareMonitor motherboard sensor group.
   - Report "unsupported" if board lacks Super I/O sensors.

4. `plugins/memory_plugin.py` (Tier 1 ECC-Conditional):
   - Linux: Check for EDAC sysfs (`/sys/devices/system/edac/mc/mc*/ce_count`). If present, track single-bit correctable error burst rate. If absent (non-ECC system), report `"ecc_supported": false` and skip ECC metrics.
   - Windows: Best-effort via LibreHardwareMonitor / WHEA event logs.

5. `plugins/storage_plugin.py` (Tier 1 NVMe & SATA):
   - NVMe SSDs: Query `nvme smart-log /dev/nvmeX` (Linux) or `smartctl -a` (Windows/Linux) to extract Log Page 0x02 attributes: Available Spare %, Percentage Used, Media and Data Integrity Errors, Critical Warnings, and Thermal Throttle duration.
   - SATA Drives: Query `smartctl -a` for SMART 5 (Reallocated Sectors), 187 (Reported Uncorrectable), 188 (Command Timeout), 197 (Current Pending), 198 (Offline Uncorrectable).

6. `plugins/psu_fan_plugin.py` (Tier 1 Baseline + Tier 2 BMC Auto-Probe):
   - Tier 1: Linux via `lm-sensors` `inN`/`fanN_input`; Windows via LibreHardwareMonitor voltage/fan groups. Flag +12V DC rail sag below 11.4V.
   - Tier 2 Probe: Attempt `ipmitool mc info` with 1-second timeout. If successful, query `ipmitool sdr list` or Redfish `/redfish/v1/Chassis/{id}/Power` for true dual-redundant PSU rail voltages, current, and fan tachometers. If probe fails, log once ("No BMC detected, using Tier 1 sensors") and proceed with Tier 1.
   - Degradation Indicator: Track Fan RPM deviation from PWM target (motor winding wear).

VERIFICATION & TESTING GATE:
1. Run `python -m server.telemetry_agent.collector_v2 --dry-run` on the current host.
2. Confirm:
   - All plugins execute and return a valid JSON payload.
   - Unsupported hardware fields (e.g. non-ECC RAM, missing BMC) are reported as null/"unsupported" with ZERO unhandled exceptions.
   - Precursor features (thermal resistance slope, hotspot delta, spare capacity) are computed correctly.

MANDATORY DOCUMENTATION SYNC:
- Update @MEMORIES.md: Mark Phase 4 hardware degradation plugins as complete `[x]`.
- Update @PROJECT_SUMMARY.md: Append summary of telemetry plugin enhancements and dry-run results.
```

---

## 🚀 Phase 5: Enterprise ML Training & Survival Analysis TTF Engine 🧭

### Copy-Paste Prompt for Gemini 3.8 Flash:
```text
Execute PHASE 5 of the Production Hardening Plan: Enterprise ML Training & Survival Analysis.

Read master_production_readiness_blueprint_v4.md §5 & §7 first.

Tasks:
1. Enterprise Dataset Ingestion Loaders (`server/training/datasets/`):
   - `alibaba_dram_loader.py`: Ingestion loader for the Alibaba Cloud PAKDD 2021 DRAM Failure Prediction competition dataset (Tianchi / arXiv:2104.15052). Parses kernel logs, address records, and mcelog failure events.
   - `alibaba_gpu_trace_loader.py`: Ingestion loader for Alibaba `cluster-trace-gpu-v2020` (github.com/alibaba/clusterdata, USENIX NSDI '22).
   - `backblaze_loader.py`: Ingestion loader for Backblaze Drive Stats quarterly CSVs (backblaze.com/cloud-storage/resources/hard-drive-test-data). Validates schema at load time and handles SMART column variations.
   - `schema_harmonization.py`: Maps disparate dataset schemas into the unified 36-feature enterprise schema. Explicitly documents policy per field for missing attributes (impute, mask, or drop — do NOT silently zero-fill).

2. Implement Asymmetric Survival TTF Loss (`server/training/loss.py`):
   - Implement the asymmetric penalty loss function:
     $$\mathcal{L}_{asym}(y, \hat{y}) = \begin{cases} \alpha (\hat{y} - y)^2, & \text{if } \hat{y} > y \quad (\alpha = 5.0) \\ (\hat{y} - y)^2, & \text{if } \hat{y} \le y \end{cases}$$
   - Penalize dangerous overestimation of remaining life 5x more heavily than conservative early inspection.
   - Train a Weibull Accelerated Failure Time (AFT) regression head on the ST-GCN model:
     $$S(t) = \exp\left(-\left(\frac{t}{\lambda(x)}\right)^\beta\right), \quad \beta > 1$$

3. Automated Retraining Daemon (`server/deep_analysis/retrain_daemon.py`):
   - Background service computing Population Stability Index (PSI) or Wasserstein drift on incoming telemetry stored in TimescaleDB.
   - Triggers retraining when PSI > 0.25 or 50,000 new samples accumulate.
   - Evaluates candidate model on held-out validation set; auto-quantizes to INT8 and deploys to RPi4 only if F1-score improves by >= 2%.

VERIFICATION & TESTING GATE:
1. Run `python server/training/train_stgcn.py --epochs 2 --dry-run` to verify:
   - Schema harmonization pipeline loads without errors.
   - Asymmetric survival loss backward pass executes and gradients propagate cleanly.
   - Quantization export produces a valid `.tflite` model.

MANDATORY DOCUMENTATION SYNC:
- Update @MEMORIES.md: Mark Phase 5 enterprise training and survival TTF as complete `[x]`.
- Update @PROJECT_SUMMARY.md: Append summary of dataset loaders, asymmetric loss, and retraining daemon.
```

---

## 🚀 Phase 6: Codebase Cleanup, Multi-Machine Manifest & Demo Runbook 🧭

### Copy-Paste Prompt for Gemini 3.8 Flash:
```text
Execute PHASE 6 of the Production Hardening Plan: Codebase Cleanup & Final Production Audit.

Read master_production_readiness_blueprint_v4.md §8 first.

Tasks:
1. Resolve Legacy InfluxDB & Anomaly Threshold Inconsistencies:
   - InfluxDB Cleanup: In Phase G2, TimescaleDB replaced InfluxDB. Clean up dead InfluxDB imports in `server/training/retrain_manager.py` and `server/training/train_adapter.py`. Route all time-series queries to TimescaleDB (`server/deep_analysis/timescaledb_writer.py`).
   - Standardize Anomaly Thresholds: Reconcile threshold definitions across `train_lstm_ae.py`, `train_madgan.py`, and `auto_retrain.py`. Enforce a standardized 97th percentile adaptive quantile threshold across all documentation and configuration files.

2. Generate Multi-Machine File Manifests (`audit/`):
   - Write a python script `audit/generate_manifest.py` that scans the project root and outputs `audit/manifest_<hostname>.json` containing (relative path, file size, last modified timestamp, SHA-256 hash) for every code and non-code file.
   - Run the script locally on LOQ, and remotely on RPi4 via SSH.
   - Diff the manifests to identify:
     * Unreferenced or stale files.
     * Version skew (same filename with differing hashes across machines).

3. Pinned Dependencies Audit (`audit/dependency_versions.md`):
   - Record exact pinned versions for: Python, TFLite runtime, Mosquitto, TimescaleDB, PyTorch, Scikit-Learn, and FastAPI across Edge and Cloud. Flag any train/serve version mismatches.

4. Secrets & Security Sanitization:
   - Grep all `.py`, `.sh`, `.bat`, `.ps1`, `.json`, `.env` files for plaintext passwords or keys (e.g. `edge12345`, `tskey-...`). Ensure all secrets are externalized to `.env` files excluded by `.gitignore`.

5. Demo Day Runbook (`audit/demo_runbook.md`):
   - Create a one-page, numbered step-by-step presentation script:
     1. Power-on and connection sequence (RPi4 gateway -> Cloud backend -> Patient agents).
     2. Dashboard URL to open (`http://rpi4-edge.your-tailnet.ts.net:8080`).
     3. Fault injection demonstration procedure (trigger fan bearing vibration or acoustic cavitation).
     4. One-line rollback & troubleshooting commands (e.g. `tailscale status`, `systemctl status edge-ai`).

VERIFICATION & TESTING GATE:
1. Confirm all files exist in `audit/`:
   - `audit/manifest_loq.json`
   - `audit/manifest_rpi4.json`
   - `audit/dependency_versions.md`
   - `audit/demo_runbook.md`
2. Confirm zero hardcoded secrets remain in the repository.

MANDATORY DOCUMENTATION SYNC:
- Update @MEMORIES.md: Mark Phase 6 audit and cleanup as complete `[x]`.
- Update @PROJECT_SUMMARY.md: Append summary of codebase cleanup, manifest diffs, and demo runbook.
```

---

## 🚀 Phase 7: Comprehensive Multi-Device End-to-End Smoke Test 🧭

### Copy-Paste Prompt for Gemini 3.8 Flash:
```text
Execute PHASE 7 of the Production Hardening Plan: End-to-End Smoke Test & System Certification.

This is the final verification phase that validates the entire multi-device pipeline in real-time.

Tasks:
1. End-to-End Multi-Device Telemetry Flow Test:
   - Start the RPi4 edge service (`edge-ai.service`).
   - Start the Cloud backend services on LOQ (FastAPI `:8000`, Ollama `:8001`).
   - Run the patient telemetry agent on host/IdeaPad streaming to `rpi4-edge`.
   - Measure and record latency at each stage:
     * Physical sensor read latency (INMP441 audio + ADXL345 vibration).
     * Edge TFLite inference latency on RPi4 (target < 15ms).
     * MQTT transit latency over Tailscale mesh.
     * Cloud TimescaleDB ingestion latency.
     * Dashboard WebSocket update latency.
   - Save the latency profile to `audit/e2e_smoke_test.md`.

2. Interactive Visual Audit via Antigravity Browser Sub-Agent:
   - Launch the Browser Sub-Agent to inspect `http://rpi4-edge.your-tailnet.ts.net:8080` (or `http://localhost:8080`).
   - Verify every dashboard view:
     * Live Telemetry Gauge & Charts (vibration RMS, acoustic RMS, CPU temp, GPU load).
     * Active Component Fault Prediction Cards (all 7 subsystems).
     * Table sorting and pagination.
     * Ollama AI Assistant drawer interaction.
   - Capture final visual verification screenshots to `audit/screenshots/`.

3. Anomaly Injection & Automated Offload Verification:
   - Inject a synthetic thermal/vibration anomaly using `tests/inject_anomaly.py`.
   - Confirm that:
     * Edge node flags anomaly with confidence < 0.80.
     * Edge offloads payload to Cloud Hospital (`loq-cloud:8000/predict_deep`).
     * Cloud ST-GCN / Survival TTF regression runs and returns updated root cause explainability.
     * Alert displays on the live dashboard.

VERIFICATION & TESTING GATE:
1. `audit/e2e_smoke_test.md` contains verified timestamps and latency numbers for all hops.
2. The Browser Sub-Agent confirms all dashboard UI elements function cleanly without errors.
3. System is certified 100% PRODUCTION READY.

MANDATORY DOCUMENTATION SYNC:
- Update @MEMORIES.md: Mark Phase 7 end-to-end verification as complete `[x]`. Set project status to "Production Certified".
- Update @PROJECT_SUMMARY.md: Append comprehensive final handover report.
```
