# 🧠 Project Master Memory — Edge-AI Predictive Maintenance for Server Hardware

> **⚠️ CRITICAL FOR ALL AGENTS:** Read this ENTIRE file before doing ANY work. This is the single source of truth for project state across all devices and Antigravity sessions.

---

## 🖥️ Device Context

| Device | Role (Analogy) | Specs | Status |
|--------|---------------|-------|--------|
| **Separate Server/Device** | 🏥 **THE PATIENT** — the machine being monitored for hardware failures | **TBD — hardware not yet confirmed.** Sensors + telemetry agent will be placed on/near this device. | NOT yet decided |
| **Raspberry Pi 4 (4GB RAM)** | 🩺 **THE DOCTOR (Edge)** — runs lightweight AI models, makes fast predictions, professional operator dashboard | ARM Cortex-A72, 4GB RAM, OS: fresh install needed (Raspberry Pi OS Lite 64-bit) | NOT yet configured |
| **Secondary Laptop (Lenovo LOQ, RTX 4060)** | 🏨 **THE HOSPITAL (Cloud/Server)** — heavy AI computation, deep analysis, model training, detailed Grafana dashboard | Ryzen 7, 24GB RAM, NVIDIA RTX 4060 8GB VRAM, NVMe SSD + ext HDD, Windows 11 | NOT currently in use |
| **Primary Laptop (Lenovo IdeaPad S145)** | 💻 **DEV MACHINE** — write all code, test logic, prepare datasets (not part of final system) | Ryzen 5 + Vega 8 (NO CUDA), 8GB RAM | ACTIVE — current workspace |

> **⚠️ CRITICAL DISTINCTION:**
> - The **RTX 4060 laptop is NOT the server being monitored.** It is ONLY the cloud backend that:
>   - (1) Trains models on GPU
>   - (2) Receives offloaded uncertain predictions from RPi4 for deep analysis
>   - (3) Runs the detailed Grafana engineer dashboard
> - The **SEPARATE SERVER DEVICE is the "patient"** — its hardware (CPU, GPU, HDD, PSU, RAM, Fans) is what we predict failures for.
> - The **RPi4 sits near/on the server** — collects external sensor data + receives software telemetry from the server → runs edge AI → sends uncertain cases to RTX 4060 laptop for deep analysis.
>
> **Dev Workflow:** Code is written on IdeaPad S145 → pushed to GitHub → pulled on LOQ RTX 4060 + RPi4. ~65% of the project can be done on the IdeaPad S145 WITHOUT the other devices.
>
> **Cross-Device Handoff:** When token limit is hit on IdeaPad S145, switch to LOQ RTX 4060. All context is in this MEMORIES.md + docs/ folder + GitHub repo.

---

## 📋 Project Metadata

| Field | Value |
|-------|-------|
| **Project Title** | Edge-AI Predictive Maintenance for Server Hardware |
| **Course** | B.Tech CSE with AI Specialization — Final Year Project |
| **Domain** | Edge AI |
| **Paper 1 (Implementation Ref)** | "Edge-AI Enabled Predictive Maintenance" — IJME Dec 2025 — uses RPi4, 1D-CNN, TFLite, vibration/temp/current sensors on motors |
| **Paper 2 (Base Paper)** | "Design of a Lightweight Edge-AI System for Predictive" — Applied Sciences 2026 — uses ESP32-S3, confidence-aware hybrid edge-cloud, asymmetric TTF loss, SHAP explainability |
| **Our Twist** | Instead of motors/bearings → **server hardware** (CPU, GPU, HDD, PSU, RAM, Fans). Instead of ESP32 → **RPi4** (more capable). Both built-in telemetry + external sensors. |
| **Key Prof Requirement** | Must include substantial AI component (specializing in AI) |
| **Key Design Choice** | TTF (Time-to-Failure) regression model, NOT just binary classification |

---

## 🔧 Technical Stack (Decided)

### Software
- **Language:** Python 3.11+ (3.14 if available)
- **Edge ML Runtime:** TensorFlow Lite (INT8 quantized models on RPi4)
- **Server ML Frameworks:** PyTorch or TensorFlow (full precision training on LOQ RTX 4060)
- **Data Collection (on the monitored server):** psutil, GPUtil, LibreHardwareMonitor (via pythonnet), pySMART, nvidia-smi (if GPU present), WMI
- **Data Collection (RPi4 GPIO sensors):** Adafruit libraries for DHT22, ADXL345; spidev for MCP3008 ADC (when sensors are purchased)
- **API Framework:** FastAPI + Uvicorn (on RPi4 edge + LOQ cloud backend)
- **Communication:** MQTT (Mosquitto broker on RPi4) — all devices connect to RPi4
- **Time Series Storage:** InfluxDB on LOQ (for Grafana) + SQLite on RPi4 (lightweight local store)
- **Dashboard (confirmed):** Custom HTML/CSS/JS on RPi4 (operator) + Grafana on LOQ (engineer)
- **AI Chatbot:** Ollama + Llama 3.1 8B (Q4 quantized, ~5GB VRAM) on LOQ — RAG over prediction data + SHAP explanations. Rule-based fallback on RPi4 when LOQ unavailable.
- **Model Optimization:** TFLite quantization (FP32 → INT8), weight pruning, ONNX as alternative export

### Hardware
- **Monitored Server (Patient):** TBD — hardware not yet confirmed (❌ BLOCKING Q21-Q25)
- **Edge (Doctor):** Raspberry Pi 4 Model B, **4GB RAM** (confirmed), Raspberry Pi OS Lite 64-bit
- **Cloud (Hospital):** Lenovo LOQ, Ryzen 7, 24GB RAM, NVIDIA RTX 4060 8GB VRAM, NVMe SSD + ext HDD, Windows 11
- **External Sensors (not yet purchased — software-only first):**
  - DHT22 — ambient temperature + humidity
  - ADXL345 — 3-axis accelerometer (vibration detection)
  - MAX4466 — electret microphone (acoustic monitoring)
  - MCP3008 — SPI ADC for analog sensors (RPi4 has no analog input)

### Datasets
| Dataset | Purpose | Source |
|---------|---------|--------|
| Backblaze Drive Stats Q1 2023 | HDD failure prediction + TTF regression | backblaze.com |
| NASA C-MAPSS (FD001-FD004) | RUL/TTF pretraining benchmark | NASA data repository |
| AI4I 2020 Predictive Maintenance | Multi-failure classification (thermal, power, overstrain) | UCI ML Repository |
| Custom Live Telemetry | Real-world validation from the separate server device | Self-collected |

---

## 🏗️ Architecture Summary (Full Details in `docs/implementation_plan.md`)

### Core Architecture: 3-Device Confidence-Aware Hybrid (from Paper 2)

```
SERVER DEVICE (Patient)        RPi4 (Doctor/Edge)           RTX 4060 LAPTOP (Hospital/Cloud)
┌─────────────────────┐       ┌──────────────────────┐      ┌──────────────────────────┐
│ THE MACHINE BEING   │       │ EDGE AI ENGINE       │      │ CLOUD/SERVER BACKEND     │
│ MONITORED           │       │                      │      │                          │
│                     │ telem │ ├ TFLite INT8 Model  │      │ ├ Full FP32 Models       │
│ Telemetry Agent     │──────▶│ ├ TTF Estimator      │      │ ├ LSTM-AE Anomaly        │
│ ├ psutil, sensors   │ MQTT  │ ├ Isolation Forest   │      │ ├ SHAP Explainer         │
│ ├ SMART data        │       │ └ Confidence Router  │      │ ├ Model Retraining       │
│ └ nvidia-smi (if    │       │     │          │     │      │ ├ Grafana Dashboard      │
│   GPU present)      │       │  conf>0.85   conf<0.85     │ │ (detailed eng. view)   │
│                     │       │     ▼          ▼     │      │ └ AI CHATBOT (Ollama)    │
│ External Sensors    │       │  LOCAL     OFFLOAD───┼─────▶│   Llama 3.1 8B + RAG    │
│ (placed on/near     │       │  DECISION   to cloud │      │   over predictions/SHAP  │
│  this device)       │       │                      │◀─────│                          │
│ ├ DHT22 (temp)      │       │ Operator Dashboard   │      └──────────────────────────┘
│ ├ ADXL345 (vib.)    │       │ ├ Health + TTF cards │
│ └ MAX4466 (acoustic)│       │ ├ Alert log          │
│                     │       │ └ Simple chat        │
│                     │       │   (rule-based        │
└─────────────────────┘       │    fallback)         │
                              └──────────────────────┘
```

### Models (4 ML models + 1 LLM = 5 AI components)
1. **1D-CNN Multi-Task** → Fault classification (7 classes) + TTF regression head (with asymmetric loss)
2. **LSTM-Autoencoder** → Anomaly detection (trained on healthy data only, unsupervised)
3. **XGBoost** → HDD specialist (Backblaze SMART → TTF days)
4. **Isolation Forest** → PSU/Fan anomaly detection (unsupervised, lightweight for edge)
5. **Llama 3.1 8B (via Ollama)** → AI chatbot with RAG — answers maintenance questions using prediction data + SHAP explanations. Rule-based fallback on RPi4 when LOQ is unavailable.

### 7 Fault Classes
| Class | Label | Primary Sensors |
|-------|-------|----------------|
| 0 | HEALTHY | All |
| 1 | HDD_DEGRADATION | SMART attributes |
| 2 | THERMAL_CRITICAL | CPU/GPU temperatures |
| 3 | PSU_ANOMALY | Voltage rails |
| 4 | FAN_FAILURE | RPM + acoustic |
| 5 | MEMORY_FAULT | ECC errors, page faults |
| 6 | GPU_DEGRADATION | ECC, clock drops, thermal |

### 28 Engineered Features
Key features include: thermal ramp rates (°C/sec), Δ-features (rate of change), cross-sensor correlations (cpu_temp - ambient_temp), health indices, voltage deviation %, acoustic kurtosis, vibration RMS/peak frequency. Full list in `docs/implementation_plan.md` §7.

---

## 📊 Active Sprint Tasks

> **Current Phase:** ✅ Planning COMPLETE — Phase A APPROVED for coding. Start from A1.

### Phase A: Work on IdeaPad S145 (current laptop, NO GPU)
- [x] Phase 0: System architecture design & planning ✅ (completed 2026-08-01 by @claude-architect)
- [x] Phase 0.5: Answer open questions + finalize design decisions ✅ (completed 2026-08-02)
- [ ] A1: GitHub repo setup
- [ ] A2: Download & prepare datasets (Backblaze, NASA C-MAPSS, AI4I 2020)
- [ ] A3: Build TTF labels for Backblaze data
- [ ] A4: Feature engineering pipeline (28 features)
- [ ] A5: EDA notebooks
- [ ] A6: Write telemetry agent code (test with dummy data)
- [ ] A7: MQTT pipeline (publisher + subscriber, test locally)
- [ ] A8: Train Isolation Forest (CPU, scikit-learn)
- [ ] A9: Edge inference code + confidence router
- [ ] A10: FastAPI servers (edge + server APIs)
- [ ] A11: Dashboard frontend (HTML/CSS/JS — RPi4 operator view)
- [ ] A12: Chat interface (HTML/JS widget for RPi4 dashboard + LOQ Grafana)
- [ ] A13: Rule-based chat fallback (template responses for RPi4 when LOQ unavailable)
- [ ] A14: RPi4 setup guide
- [ ] A15: All documentation structure

### Phase B: Work on LOQ RTX 4060 (GPU training + cloud backend + chatbot)
- [ ] B1: Train 1D-CNN multi-task model (GPU)
- [ ] B2: Train LSTM-Autoencoder (GPU)
- [ ] B3: Train XGBoost on full Backblaze dataset (GPU-accelerated)
- [ ] B4: Model quantization (FP32 → INT8 TFLite)
- [ ] B5: Deploy edge models to RPi4
- [ ] B6: Set up Grafana engineer dashboard on LOQ
- [ ] B7: SHAP explainability module
- [ ] B8: Deep analysis API (FastAPI on LOQ — receives offloaded predictions from RPi4)
- [ ] B9: Install Ollama + download Llama 3.1 8B Q4 model on LOQ
- [ ] B10: RAG context builder (feeds current predictions + SHAP data to LLM)
- [ ] B11: Chat API endpoint (FastAPI /chat on LOQ)

### Phase C: Integration & Demo (needs all 3 devices)
- [ ] C1: Install telemetry agent on the SEPARATE SERVER DEVICE
- [ ] C2: End-to-end integration test (Server → RPi4 → LOQ)
- [ ] C3: RPi4 WiFi hotspot setup
- [ ] C4: Collect real telemetry from server (2+ weeks for training data)
- [ ] C5: Fine-tune models on real server data
- [ ] C6: Demo preparation
- [ ] C7: Final documentation & report

---

## ❓ Open Questions — Status as of 2026-08-02

| # | Question | Answer | Status |
|---|----------|--------|--------|
| Q1 | RPi4 RAM — 4GB or 8GB? | **4GB RAM** | ✅ |
| Q2 | RPi4 OS — Raspbian? What version? | **Fresh install needed — use Raspberry Pi OS Lite 64-bit** | ✅ |
| Q3 | External sensors — do you have DHT22, ADXL345, MAX4466, MCP3008? | **None currently — will buy later. Software-only first.** | ✅ |
| Q4 | Dashboard preference — custom HTML/JS or Grafana? | **BOTH — RPi4: custom operator view. LOQ: Grafana engineer view.** | ✅ |
| Q5 | Timeline — weeks until submission? | **Has time but wants to move fast** | ✅ |
| Q6 | ~~Server laptop specs~~ → **LOQ (cloud backend) specs** | **Lenovo LOQ: Ryzen 7, 24GB RAM, RTX 4060 8GB, NVMe SSD + ext HDD, Win11** | ✅ (re-labeled) |
| Q7 | ~~Server laptop OS~~ → **LOQ (cloud backend) OS** | **Windows 11** | ✅ (re-labeled) |
| Q8 | Network setup — **3 devices** (server + RPi4 + LOQ) | **RPi4 WiFi Hotspot for demo access. Server↔RPi4 via Ethernet. RPi4↔LOQ via WiFi/Ethernet. Exact topology TBD when server hardware is known.** | ⚠️ Needs revisit |
| Q9 | Team size | **Team exists but Aswin does everything** | ✅ |
| Q10 | MQTT broker location | **On RPi4 (Mosquitto) — central hub** | ✅ |
| Q11 | Primary dev device | **IdeaPad S145 → GitHub → LOQ + RPi4** | ✅ |
| Q12 | Git/GitHub | **GitHub — repo not created yet** | ✅ |
| Q13 | Previous v0.1 prototype | **Removed. Start from scratch.** | ✅ |
| Q14 | Professor deliverables | **Everything: code + working demo + report + presentation** | ✅ |
| Q15 | Software-only fallback | **YES — Sensor Abstraction Layer with auto-detect** | ✅ |
| Q16 | Dashboard location | **RPi4: operator. LOQ: Grafana engineer.** | ✅ |
| Q17 | Network for demo | **RPi4 WiFi hotspot** | ✅ |
| Q18 | RPi4 ready state | **Fresh install needed — include setup guide** | ✅ |
| Q19 | LOQ brand | **Lenovo LOQ** | ✅ |
| Q20 | Dev laptop specs | **IdeaPad S145, Ryzen 5, Vega 8, 8GB RAM** | ✅ |
| **Q21** | **Separate server device — what hardware?** | **TBD — user needs to look into it** | ❌ BLOCKING |
| **Q22** | **Separate server — what OS?** | **TBD** | ❌ BLOCKING |
| **Q23** | **Separate server — does it have a GPU?** | **TBD** | ❌ BLOCKING |
| **Q24** | **Separate server — can Python be installed on it?** | **TBD** | ❌ BLOCKING |
| **Q25** | **Separate server — is it always on or intermittent?** | **TBD** | ❌ BLOCKING |

---

## 📁 Project File Structure (Current)

```
Predictive maintenance for Server/
├── MEMORIES.md                 ← THIS FILE (cross-device context — READ FIRST)
├── PROJECT_SUMMARY.md          ← High-level overview for quick catch-up
├── docs/
│   ├── AGENT_HANDOFF.md        ← Quick-start guide for new agents
│   ├── implementation_plan.md  ← Full system design (746 lines, 44KB)
│   ├── answers_and_decisions.md ← Detailed answers to design questions
│   └── device_roles.md         ← Which device does what + execution order
├── Design of a Lightweight Edge-AI System for Predictive.pdf   ← Paper 2 (base paper)
└── EDGE+-+IJME+-+Current+Issue+Dec+2025+(1)[1] (2).pdf        ← Paper 1 (implementation ref)
```

---

## 🔑 Key Design Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| **3 separate devices** (patient + doctor + hospital) | Separate server = realistic monitoring target. RPi4 = independent edge AI. LOQ = cloud backend only. Cleaner architecture than 2-device where laptop monitors itself. |
| RPi4 instead of ESP32 (Paper 2 used ESP32) | RPi4 is more capable — can run TFLite + full Python stack; still demonstrates edge AI concepts |
| TTF regression instead of binary classification | User's explicit requirement; more useful for maintenance planning; aligns with Paper 2's approach |
| Asymmetric loss function for TTF | Overestimating remaining life is DANGEROUS (miss a failure); underestimating is safe (early replacement). From Paper 2. |
| Confidence-aware offloading | ~85-90% predictions handled locally (fast, no network dependency); only uncertain cases go to LOQ GPU. Key differentiator. |
| Both built-in telemetry + external sensors | Built-in (SMART, nvidia-smi) captures software-visible metrics; external sensors (vibration, acoustic) catch physical degradation that software can't see |
| Multi-model architecture (not one monolithic model) | Different components degrade differently — HDD needs SMART-specific model (XGBoost), GPU/CPU need temporal patterns (LSTM), PSU/Fan need anomaly detection (Isolation Forest) |
| 4 public datasets + custom collection | Public datasets for training + benchmarking; custom data for real-world validation and fine-tuning |
| **Dual dashboards** | RPi4: custom HTML/JS operator view (lightweight, professional, essential info). LOQ: Grafana engineer view (every detail, raw telemetry, SHAP). Demonstrates edge-cloud split visually. |
| **Software-only first** | Start without external sensors. Sensor Abstraction Layer auto-detects and gracefully degrades. Add sensors when purchased. |
| **Local LLM chatbot (Ollama)** | Llama 3.1 8B Q4 on LOQ RTX 4060 (~5GB VRAM). Uses RAG over prediction data + SHAP. Adds NLP as 5th AI component. Rule-based fallback on RPi4 ensures chat works even without LOQ. Runs 100% offline — perfect for demo. |

---

## 📝 Lessons Learned & Technical Debt

### From v0.1 Prototype (HDD only)
- XGBoost on GPU works well for HDD SMART classification (F1=0.99) but doesn't provide TTF estimation
- Backblaze data has extreme class imbalance (~0.01% failure rate) — requires aggressive upsampling
- FastAPI + Uvicorn is solid for inference serving
- Windows console encoding issues — always use `sys.stdout.reconfigure(encoding="utf-8")`

### From Paper 1 (EDGE-IJME 2025)
- 1D-CNN with TFLite INT8 quantization achieved 94.7% accuracy on RPi4 at 30ms latency
- Edge-based processing reduced bandwidth by 70%+ vs cloud-only
- Multi-sensor fusion (vibration + temperature + current) significantly improves accuracy over single-sensor
- 28 features (time-domain + frequency-domain) extracted from raw sensor signals

### From Paper 2 (Applied Sciences 2026)
- Asymmetric loss function penalizes TTF overestimation — critical for safety-first maintenance
- Confidence-aware offloading maintains 0.924 critical fault recall even at 50% cloud outage
- INT8 quantization: model size 5.4MB → 1.3MB, inference ~42ms on edge
- SHAP explainability makes predictions interpretable for non-AI operators
- Custom sample weighting: 3× weight for imminent failures (TTF < 20% of max)

### Technical Notes
- Windows telemetry requires admin privileges for LibreHardwareMonitor DLL (temperature/fan data)
- RPi4 has no analog GPIO — need MCP3008 SPI ADC for analog sensors like MAX4466
- SMART data alone can't predict sudden electronic failures — multi-sensor fusion essential
- MQTT is preferred over pure REST for real-time streaming (lower overhead, pub/sub pattern)

---

## 🏷️ Open Bugs & UI Issues

- (None — project in planning phase, no code yet)

---

## 📌 Agent Role Assignments

| Role | Responsibility | When |
|------|---------------|------|
| **@claude-architect** | System design, architecture, planning, sprint management | ✅ Phase 0 done |
| **@gemini-coder** | Code implementation, feature development | Phase 1 onwards (after questions answered) |
| **@gemini-qa** | Code review, GUI inspection, testing | Phase 5+ |
| **@fix-and-summarize** | Bug fixing, documentation updates | As needed |
