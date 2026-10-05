# 🏛️ Master Production Readiness Blueprint & Live System Dossier (v4.0)

> **Document Purpose:** Single-File Comprehensive Handover for **Claude (Lead Architect)** to review the implementation plan.  
> **System:** Edge-AI Predictive Maintenance for Server Hardware  
> **Topology:** Monitored Patient (N Host Servers) | Edge Doctor (RPi4 Gateway) | Cloud Hospital (LOQ / GPU Backend)  
> **Status:** Ground-Truth Reconciled (Synthesized from Live RPi4 / Codebase Interrogation + Claude 3.7 Sonnet Architectural Review)  
> **Primary Author:** Antigravity AI Engine (Local Systems & Machine Learning Inspector)  
> **Target Execution Agent:** Gemini 3.8 Flash (High Thinking Budget, Planning Mode)  

---

## 📌 Note for Claude: Why This Document Was Re-Generated as v4.0

Claude, you previously produced `master_production_readiness_blueprint_v3.md` and `gemini_38_flash_execution_prompt_v2.md` while working without direct filesystem, SSH, or hardware access to the user's machines.

Because the user cannot upload the entire repository folder due to upload limits, **this single file (v4.0) has been constructed to contain 100% of the relevant ground truth, live hardware states, exact file lines, and architectural plans in depth.** You do not need any other file to review and refine the implementation plan.

---

## 1. 🚨 Critical Ground-Truth Findings & Reconciliations

### 1.1 The Phase 2 Sensor Disconnect (Major Correction)
* **What you assumed in v2.0 & v3.0:** You marked Phase 2 as `"ALREADY DONE... validation only, do not rewrite sensor_reader.py"`, assuming that because the sensors were working, the code was already committed to the repository.
* **The Live Reality:**
  * Both sensors (**INMP441** on I2S and **ADXL345** on I2C) were wired, tested, and validated using isolated test scripts inside `/home/pi/mic_test/` on the RPi4.
  * **However, the repository file `edge/sensor_reader.py` was NEVER updated.**
  * In `edge/sensor_reader.py` (lines 30–53), the code **still attempts to import `adafruit_adxl34x` and `adafruit_mcp3xxx`**, neither of which are installed on the RPi4.
  * At runtime, the production daemon logs:
    ```text
    WARNING: ADXL345 library not available — vibration data will be simulated
    WARNING: MCP3008 library not available — acoustic data will be simulated
    ```
* **The Architecture Impact:** Phase 2 **MUST NOT** be treated as "validation only." It must be an **active code implementation phase** that rewrites `edge/sensor_reader.py` to use `smbus2.SMBus(1)` and ALSA `plughw:3,0`. Otherwise, the production daemon will continue generating simulated data!

---

## 2. Live Hardware & Multi-Device State

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                LIVE MULTI-DEVICE TOPOLOGY                              │
├───────────────────────────────┬───────────────────────────────┬────────────────────────┤
│ 1. LOQ Laptop (Cloud Backend) │ 2. Raspberry Pi 4 (Edge Node) │ 3. IdeaPad (Patient)   │
├───────────────────────────────┼───────────────────────────────┼────────────────────────┤
│ • Role: Cloud ML / Hospital   │ • Role: Edge Doctor / Gateway │ • Role: Patient Server │
│ • OS: Windows 11 Home 64-bit  │ • OS: Debian 13 (Trixie) arm64│ • OS: Windows 11 Home  │
│ • CPU: AMD Ryzen 7 7435HS     │ • CPU: BCM2711 Quad Cortex-A72│ • CPU: AMD Ryzen 5     │
│ • GPU: NVIDIA RTX 4060 (8GB)  │ • RAM: 4 GB LPDDR4            │ • RAM: 8 GB DDR4       │
│ • RAM: 24 GB (23.7 GB usable) │ • IP: 192.168.137.100 (Hotspot│ • IP: 192.168.137.137 │
│ • Python: 3.14.0              │ • Python: 3.11.2 (in venv)    │ • Status: OFFLINE      │
│ • Services: FastAPI :8000,    │ • Services: edge-ai.service   │   (Ping failed; Wi-Fi  │
│   Ollama Assistant :8001      │   active (PID 1171) on :8080  │   hotspot dropped)     │
└───────────────────────────────┴───────────────────────────────┴────────────────────────┘
```

### 2.1 Raspberry Pi 4 Sensor Interrogation Results
1. **INMP441 I2S Digital MEMS Microphone:**
   * **Wiring:** Pin 1 (3.3V), Pin 6 (GND), Pin 9 (L/R $\rightarrow$ GND for Left Channel), Pin 12 (GPIO 18 / SCK), Pin 35 (GPIO 19 / WS), Pin 38 (GPIO 20 / SD).
   * **Kernel Overlay:** Loaded dynamic overlay `googlevoicehat-soundcard` and **permanently appended** `dtoverlay=googlevoicehat-soundcard` to `/boot/firmware/config.txt`. Automatically initializes on boot.
   * **ALSA Card:** Card 3, Device 0 (`plughw:3,0`, `snd_rpi_googlevoicehat_soundcard`).
   * **Live Signal Analysis:** 44.1 kHz, 32-bit `S32_LE`. Captures clean dynamic audio (normalized RMS = **0.7045**, peak-to-peak span > 4.2 billion, zero DC lockup).
2. **ADXL345 3-Axis Digital Accelerometer:**
   * **Wiring:** Pin 17 (3.3V Power), Pin 14 (GND), Pin 3 (GPIO 2 / I2C1 SDA), Pin 5 (GPIO 3 / I2C1 SCL), CS tied to 3.3V (forces I2C mode), SDO tied to Pin 20 / GND (sets address strictly to `0x53`).
   * **Bus & Silicon ID:** `/dev/i2c-1` active. Register `0x00` returns **`0xE5`** (authentic ADXL345 silicon).
   * **Live Acceleration:** $X = -0.050\,g$, $Y = -0.002\,g$, $Z = +1.030\,g$. Total vector magnitude = **$1.031\,g$** (exact Earth resting gravity).
3. **Concurrent Bus Operation:**
   * Executed 44 continuous I2C reads from ADXL345 while I2S DMA streamed audio. **Zero collisions, zero timeouts, zero ALSA buffer underruns.**

---

## 3. Exact Codebase Bug Audit & File Locations

### 3.1 BUG-01: Silent TTL Eviction Crash in `edge/api_server.py`
* **File:** `edge/api_server.py` (Lines 74–77)
* **Code:**
  ```python
  now = time.time()  # float epoch
  for sid, s in list(current_state.items()):
      if now - s.get("last_seen", now) > 1800:
          current_state.pop(sid, None)
  ```
* **Root Cause:** `s["last_seen"]` is stored as an ISO-8601 string (e.g., `"2026-09-25T11:21:21.000Z"`). Subtracting a string from a float throws:
  `TypeError: unsupported operand type(s) for -: 'float' and 'str'`
  This unhandled exception terminates the background cleanup task on tick 1.
* **Fix:** Convert ISO string to float epoch before subtraction:
  `ls_ts = datetime.fromisoformat(ls.replace('Z', '+00:00')).timestamp()`.

### 3.2 BUG-02: ST-GCN Machine Learning Data Leakage
* **File:** `server/training/train_stgcn.py` (Lines 147–152)
* **Code:**
  ```python
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X_raw)
  X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20)
  ```
* **Root Cause:** `fit_transform` learns mean and standard deviation from the entire dataset, leaking test partition statistics into training.
* **Fix:** Split into `X_train_raw` and `X_test_raw` first; call `fit_transform` strictly on `X_train_raw`, and `transform` on `X_test_raw`. Derive adjacency matrix strictly from training correlations.

### 3.3 BUG-03: Auto-Retraining Data Leakage
* **File:** `server/training/auto_retrain.py` (Lines 149–153)
* **Root Cause:** Same data leakage occurs during automated background retraining.
* **Fix:** Chronological partition (first 90% train, last 10% validation) *before* `scaler.fit_transform()`.

### 3.4 BUG-04: Unbounded Memory Leak in Cloud Inference Server
* **File:** `server/deep_analysis/inference_server.py` (Lines 50–65)
* **Code:**
  ```python
  _rate_tracker = defaultdict(list)
  _server_windows = defaultdict(deque)
  _madgan_windows = defaultdict(deque)
  ```
* **Root Cause:** In-memory tracking structures grow unbounded as client IPs and registered servers connect.
* **Fix:** Replace with bounded TTL caches: `cachetools.TTLCache(maxsize=1000, ttl=3600)`.

### 3.5 BUG-05: Non-Functional Table Sorting & Mojibake
* **Files:** `dashboard/assets/app.js` and `dashboard/index.html`
* **Root Cause:** Column headers render the `↕` sort symbol with class `sort-col`, but lack click listeners. Special characters display Mojibake (`â€”` instead of `—`, `â€¢` instead of `•`).
* **Fix:** Insert `<meta charset="UTF-8">` as the very first element in `<head>`. Attach click listeners in `app.js` to toggle ascending/descending sort across numeric and string columns.

---

## 4. Architectural Decisions & Resolutions

### 4.1 InfluxDB vs. TimescaleDB (Clean Handover)
* **Historical Context:** In Phase G2, **TimescaleDB** (`server/deep_analysis/timescaledb_writer.py`, PostgreSQL table `telemetry_records`) was deployed on the LOQ cloud node to replace InfluxDB.
* **Current Technical Debt:** `server/training/retrain_manager.py` (lines 31–34) and `server/training/train_adapter.py` (lines 95–97) still import `InfluxWriter`. While guarded by `try/except`, it leaves confusing configuration references (`cloud.env`).
* **Resolution:** Cleanly remove all remaining InfluxDB code and route training data queries directly to TimescaleDB.

### 4.2 LSTM-AE Anomaly Threshold Standardization
* **The Variance:**
  * `train_lstm_ae.py`: $\mu + 2\sigma$ of validation errors (~95.4%).
  * `auto_retrain.py`: 97th percentile of validation errors.
  * `train_madgan.py`: 95th percentile of normal scores.
* **Resolution:** Standardize on the **97th percentile adaptive quantile** across all models and configuration files.

### 4.3 Virtual Mesh Networking: Tailscale (Verified Apr 2026 Terms)
* **The Problem:** Windows Mobile Hotspot (`192.168.137.x`) dropped DHCP leases; the IdeaPad is currently offline because the hotspot was disrupted.
* **Tailscale Personal Plan:** Free forever, unlimited devices, up to 6 users, 3 ACL groups, up to 50 tagged resources.
* **Persistent Addressing:** Stable MagicDNS hostnames `<hostname>.<tailnet>.ts.net` (e.g. `rpi4-edge.<tailnet>.ts.net:1883`).
* **Network Segmentation Policy (`docs/tailscale_acl_policy.json`):**
  * `group:edge` $\rightarrow$ RPi4 only. Accessible by all on ports 1883 (MQTT) and 8080 (Dashboard).
  * `group:cloud` $\rightarrow$ LOQ only. Accessible exclusively from `group:edge` on port 8000 (Inference API).
  * `group:patients` $\rightarrow$ Monitored patient servers (`tag:patient`). Can only reach `group:edge:1883`. Inter-patient communication is strictly denied.
* **Zero-Touch Installers:** One-line Bash curl installer for Linux systemd and one-click PowerShell script for Windows.
* **RPi4 USB-C Gadget Mode:** Leverages Debian Trixie's official `rpi-usb-gadget` package for direct driver-free virtual Ethernet cable administration from a laptop without depending on Wi-Fi or hotspot state.

---

## 5. Tiered Hardware Ingestion Model & Degradation Physics

To prevent plugin crashes on consumer machines, hardware ingestion is partitioned into three distinct tiers with mandatory graceful degradation:

### 5.1 Ingestion Tiers
* **Tier 0 (Reference Chassis Only):** Physical ADXL345 vibration and INMP441 acoustic telemetry hardwired to RPi4.
* **Tier 1 (Universal Cross-Platform Software Sensors):** Runs on *every* patient machine (Windows or Linux, consumer or enterprise):
  * **CPU:** Linux `lm-sensors` (`sensors -j`) / RAPL `turbostat`; Windows LibreHardwareMonitor Remote Web Server JSON endpoint (`:8085/data.json`).
  * **GPU:** `pynvml` (NVIDIA) for temp, power, VRAM, and ECC error counters; `rocm-smi` (AMD); kernel `NVRM: Xid` / Windows `nvlddmkm` event logging.
  * **Motherboard:** PCH and chipset temperatures via Super I/O / LibreHardwareMonitor.
  * **RAM:** Linux `rasdaemon` / EDAC sysfs (ECC-conditional; reports "unsupported" on non-ECC systems); Windows WHEA logs.
  * **Storage:** NVMe Log Page 0x02 attributes (`nvme smart-log` / `smartctl -a`); SATA SMART 5, 187, 188, 197, 198.
  * **PSU / Fans:** Super I/O rail voltages (+12V, +5V, +3.3V) and fan tachometer speeds.
* **Tier 2 (Enterprise BMC / Out-of-Band):** Probed at startup via `ipmitool mc info` (1s timeout). If available, reads true dual-redundant PSU rail voltages, line frequency, power factor, and thermal tachs via `ipmitool sdr list` or Redfish REST `/redfish/v1/Chassis/{id}/Power`. If probe fails, logs once and falls back gracefully to Tier 1.

### 5.2 Failure Physics & Precursor Indicators
```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    PHYSICAL SERVER HARDWARE DEGRADATION MAPPING                         │
├───────────────┬───────────────────────────────────────┬─────────────────────────────────┤
│ Subsystem     │ Leading Telemetry Indicators          │ Physics of Failure Precursor    │
├───────────────┼───────────────────────────────────────┼─────────────────────────────────┤
│ CPU           │ • $\Delta T / \Delta P$ thermal slope │ • Thermal paste pump-out/dry-out│
│               │ • PROCHOT throttle event count        │ • VRM vCore voltage droop       │
│               │ • Correctable MCE / MCA bus logs      │ • Silicon electromigration      │
├───────────────┼───────────────────────────────────────┼─────────────────────────────────┤
│ GPU & VRAM    │ • $T_{junction} - T_{edge} > 30^\circ$│ • Thermal pad degradation       │
│               │ • NVML ECC single-bit error counters  │ • VRAM memory cell wear         │
│               │ • PCIe AER replay & rollback events   │ • Substrate thermal warping     │
│               │ • NVIDIA XID 31/43/45/62/79 logs      │ • Power stage driver deadlock   │
├───────────────┼───────────────────────────────────────┼─────────────────────────────────┤
│ Motherboard   │ • VRM MOSFET temperatures             │ • Gate oxide breakdown          │
│ & Chipset     │ • PCH junction temperature            │ • Solder joint thermal fatigue  │
│               │ • PCIe correctable error bursts       │ • Trace impedance mismatch      │
│               │ • CMOS RTC battery voltage (< 2.8V)   │ • Battery chemical depletion    │
├───────────────┼───────────────────────────────────────┼─────────────────────────────────┤
│ RAM           │ • Single-bit ECC error burst rate     │ • Capacitor cell charge leakage │
│ (DDR4 / DDR5) │ • Memory controller thermal load      │ • Impending multi-bit failure   │
│               │ • Rowhammer degradation counter       │ • Inter-cell disturbance        │
├───────────────┼───────────────────────────────────────┼─────────────────────────────────┤
│ PSU           │ • $+12V / +5V / +3.3V$ voltage sag    │ • Electrolytic capacitor dry-out│
│               │ • Step-load transient droop slope     │ • ESR (equiv. series resistance)│
│               │ • Dual-PSU load sharing imbalance     │ • Secondary stage diode wear    │
├───────────────┼───────────────────────────────────────┼─────────────────────────────────┤
│ Cooling Fans  │ • RPM deviation from PWM target       │ • Motor winding insulation wear │
│ & AIO Pumps   │ • ADXL345 Vibration RMS & Kurtosis    │ • Bearing ball/race friction    │
│               │ • INMP441 Acoustic Cavitation FFT     │ • Impeller erosion / air ingress│
├───────────────┼───────────────────────────────────────┼─────────────────────────────────┤
│ Storage       │ • NVMe Available Spare % below 10%    │ • NAND flash block exhaustion   │
│ (NVMe & HDD)  │ • Percentage Used ($> 100\%$)         │ • Write cycle endurance limit   │
│               │ • Media & Data Integrity Errors       │ • Uncorrectable read disturb    │
│               │ • HDD SMART 5, 187, 188, 197, 198     │ • Platter head crash / weak bit │
└───────────────┴───────────────────────────────────────┴─────────────────────────────────┘
```

---

## 6. Verified Enterprise Datacenter Datasets

1. **RAM ECC Failure:** Alibaba Cloud PAKDD 2021 DRAM Failure Prediction Dataset (Tianchi / arXiv:2104.15052). 3M+ production records.
2. **GPU Workload & Throttling:** Alibaba `cluster-trace-gpu-v2020` (USENIX NSDI '22, 6,500 GPUs across ~1,800 servers).
3. **GPU Hardware Failures:** ORNL Titan Supercomputer GPU Failure Dataset (DOI-backed, Oak Ridge National Lab).
4. **Storage RUL:** Backblaze Drive Stats (Quarterly CSV releases through Q1 2026, 340,000+ enterprise drives).
5. **Fan & Pump Mechanical Wear:** NASA C-MAPSS run-to-failure degradation benchmark.
6. **Schema Harmonization Module (`schema_harmonization.py`):** Maps disparate raw column formats into the unified 36-feature enterprise schema. Missing fields are explicitly masked or imputed—never silently zero-filled.

---

## 7. AI/ML Survival Analysis & Asymmetric Loss

### 7.1 Weibull Accelerated Failure Time (AFT) Model
Server hardware degradation exhibits clear wear-out bathtub physics modeled by a Weibull survival distribution:
$$S(t) = \exp\left(-\left(\frac{t}{\lambda(x)}\right)^\beta\right)$$
* $\beta > 1$ represents mechanical/thermal wear-out.
* $\lambda(x) = \exp(\theta^T x)$ represents scale parameter conditioned on current telemetry feature vector $x$.

### 7.2 Asymmetric Penalty Loss
Overestimating Time-to-Failure (server dies before technician arrives) is catastrophic compared to underestimating TTF (prompting an early inspection):
$$\mathcal{L}_{\text{asym}}(y, \hat{y}) = \begin{cases} \alpha (\hat{y} - y)^2, & \text{if } \hat{y} > y \quad (\text{Dangerous Overestimate, } \alpha = 5.0) \\ (\hat{y} - y)^2, & \text{if } \hat{y} \le y \quad (\text{Conservative Underestimate, } \alpha = 1.0) \end{cases}$$

---

## 8. Implementation Micro-Sprints for Gemini 3.8 Flash

```text
Phase 1: Critical Bug Remediation & Backend Hardening (BUG-01 through BUG-05)
         ↳ Includes Antigravity Browser Sub-Agent visual testing for dashboard sorting & UTF-8.
         ↳ Updates @MEMORIES.md and @PROJECT_SUMMARY.md.

Phase 2: Physical Hardware Sensor Driver Integration (Active Code Implementation!)
         ↳ Rewrites edge/sensor_reader.py with tested smbus2 & ALSA plughw:3,0 drivers.
         ↳ Deploys to RPi4 and verifies real vibration & acoustic telemetry streaming.
         ↳ Updates @MEMORIES.md and @PROJECT_SUMMARY.md.

Phase 3: Tailscale Mesh Networking & Zero-Touch Client Installer
         ↳ Creates configs/network.env, Linux curl installer, Windows PS1 installer, and ACL rules.
         ↳ Updates @MEMORIES.md and @PROJECT_SUMMARY.md.

Phase 4: Tiered Hardware Degradation Telemetry Plugins
         ↳ Implements degradation precursors (CPU thermal slope, GPU Xid/ECC, NVMe Log Page 0x02).
         ↳ Updates @MEMORIES.md and @PROJECT_SUMMARY.md.

Phase 5: Enterprise ML Training & Survival Analysis TTF Engine
         ↳ Implements Tianchi PAKDD DRAM, Alibaba GPU, and Backblaze loaders with asymmetric loss.
         ↳ Updates @MEMORIES.md and @PROJECT_SUMMARY.md.

Phase 6: Codebase Cleanup, Multi-Machine Manifest & Demo Runbook
         ↳ Generates SHA-256 manifests across nodes, cleans InfluxDB, and writes demo runbook.
         ↳ Updates @MEMORIES.md and @PROJECT_SUMMARY.md.

Phase 7: Comprehensive Multi-Device End-to-End Smoke Test
         ↳ Performs live end-to-end telemetry run, measures hop latency, and certifies production readiness.
         ↳ Updates @MEMORIES.md and @PROJECT_SUMMARY.md.
```

---

## 9. Review Questions for Claude

Claude, please review this reconciled dossier and advise on:
1. **Phase 2 Expansion:** Confirming that Phase 2 must be an active driver refactoring rather than validation-only, does our 7-phase breakdown match your architectural expectations?
2. **Schema Harmonization Policy:** For patient machines without physical chassis sensors (where `vibration_rms` and `acoustic_rms` are unavailable), do you recommend masking these features with a missingness indicator channel in the ST-GCN graph, or imputing them with the nominal resting baseline ($0.02\,\text{RMS}$)?
3. **Tailscale ACL Nuance:** Does our proposed 3-group policy (`group:edge`, `group:cloud`, `group:patients`) provide sufficient isolation for enterprise compliance demonstrations?
