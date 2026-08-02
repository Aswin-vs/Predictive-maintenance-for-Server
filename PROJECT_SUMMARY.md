# 🖥️ Edge-AI Predictive Maintenance for Server Hardware

**Project Goal:** Design and implement a comprehensive Edge-AI predictive maintenance system for server hardware (CPU, GPU, HDD/SSD, PSU, RAM, Fans) using a Raspberry Pi 4 as the edge device, a separate server/machine as the monitored target, and a laptop (RTX 4060) as the cloud backend for heavy computation.

**Status:** ✅ Design Complete — Awaiting server hardware confirmation before implementation

**Course:** B.Tech CSE with AI Specialization — Final Year Project

**Research Papers:**
- Paper 1 (Implementation Reference): "Edge-AI Enabled Predictive Maintenance" — IJME Dec 2025
- Paper 2 (Base Paper): "Design of a Lightweight Edge-AI System for Predictive Maintenance" — Applied Sciences 2026

**Devices:**
- 🏥 Patient (monitored server): **Separate device — hardware TBD**
- 🩺 Doctor (edge): **Raspberry Pi 4** (4GB RAM — lightweight AI models + operator dashboard)
- 🏨 Hospital (cloud): **Lenovo LOQ RTX 4060** (GPU training + deep analysis + Grafana dashboard)
- 💻 Dev machine: **Lenovo IdeaPad S145** (code writing only — not part of final system)

## Recent Progress

- **2026-08-01:** @claude-architect initialized workspace. Extracted and analyzed both research papers (Paper 1: EDGE-IJME 2025, Paper 2: Applied Sciences 2026). Conducted comprehensive research on server hardware telemetry, datasets, model architectures, and RPi4 edge deployment strategies.
- **2026-08-01:** Complete system architecture designed. Covers: 7-class fault classification, TTF regression with asymmetric loss, confidence-aware hybrid edge-cloud inference, 28 engineered features, 4 datasets (Backblaze + NASA C-MAPSS + AI4I 2020 + custom telemetry), 6 server components (CPU/GPU/HDD/PSU/RAM/Fans), external sensor integration (DHT22/ADXL345/MAX4466), TFLite INT8 deployment on RPi4, MQTT communication pipeline, and 4-level alert system.
- **2026-08-02:** All 20 design questions answered by user. Created `docs/` folder with implementation plan, answers, device roles guide, and agent handoff doc. MEMORIES.md rewritten for cross-device continuity.
- **2026-08-02 (CORRECTION):** User clarified that RTX 4060 laptop is **NOT** the server being monitored — it is ONLY the cloud backend. A **separate device** will be the "patient" (hardware TBD). Architecture corrected across all documents.
- **2026-08-02:** Added AI chatbot feature — Ollama + Llama 3.1 8B on LOQ with RAG over predictions/SHAP. Rule-based fallback on RPi4. Now 5 AI components total (1D-CNN, LSTM-AE, XGBoost, Isolation Forest, LLM). All documents finalized. **Planning phase COMPLETE. Ready for @gemini-coder to begin Phase A.**
