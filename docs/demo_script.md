# 🎬 Demo Script — Edge-AI Predictive Maintenance

> **Last Updated: 2026-08-13 — Phase E4 security fixes deployed. All endpoints authenticated.**

## Pre-Demo Setup (5 minutes before)

### 1. LOQ Cloud Backend
```powershell
cd "C:\Users\LOQ\Documents\Antigravity\Predictive maintenance for Server"

# Option A: One-click launch (Master Orchestrator)
.\START_ALL.bat

# Option B: Manual launch
.\venv_loq_py311\Scripts\python.exe server/deep_analysis/inference_server.py  # Terminal 1
.\venv_loq_py311\Scripts\python.exe server/deep_analysis/chat_api.py          # Terminal 2
```

### 2. Pre-load Ollama (avoids first-response delay)
```powershell
ollama run qwen3.5:4b "Hello" --verbose
```

### 3. Verify Everything
```powershell
# Cloud health check (no auth needed for GET /api/health)
curl http://localhost:8000/api/health
curl http://localhost:8001/api/health

# Edge health check (via Tailscale MagicDNS or Mesh IP)
curl http://100.90.160.78:8080/api/health
curl http://rpi4-edge.tail1ab139.ts.net:8080/api/health

# Auth test (should return 401 without key)
curl -X POST http://100.90.160.78:8080/api/update_state -H "Content-Type: application/json" -d "{\"server_id\":\"test\"}"
```

---

## Demo Flow (7-10 minutes)

### Scene 1: Architecture Overview (1 min)
> "This is a 3-device predictive maintenance system for server hardware."

Open the architecture diagram and explain:
- **Patient Server** (IdeaPad S145 / LOQ) -- the machine being monitored
- **RPi4 Edge Node** -- runs lightweight INT8 AI models, makes fast predictions
- **LOQ Cloud Backend** -- heavy GPU inference, LSTM-AE anomaly detection, SHAP explainability, AI chatbot

Key stat: **Edge processes predictions in <50ms** using INT8 quantized models.

### Scene 2: Healthy State (1 min)
1. Open dashboard: `http://rpi4-edge.tail1ab139.ts.net:8080/` (or `http://100.90.160.78:8080/`)
2. Start dummy simulator with all-healthy data:
   ```powershell
   .\start_dummy_simulator.bat
   ```
3. Show: **Health Index 100%**, all components green, no anomalies

### Scene 3: Fault Injection — THERMAL_CRITICAL (2 min)
1. Stop the healthy simulator (Ctrl+C)
2. Modify `run_dummy_simulator.py` to inject `mock_class=2`
3. Watch the dashboard:
   - Health index drops from 100% → ~60% → WARNING
   - CPU component turns yellow/red
   - Anomaly log fills up
   - TTF (Time-to-Failure) countdown begins

> "The system detected a thermal anomaly. The 1D-CNN classifier identified it as THERMAL_CRITICAL with 99% confidence."

### Scene 4: Cloud Offload & SHAP (2 min)
1. Show that when edge confidence is low (<0.85), the prediction is **offloaded to the LOQ GPU**
2. Point to the SHAP explanation in the anomaly details:
   - "SHAP tells us WHY the model made this prediction"
   - Feature `cpu_temp` has the highest positive SHAP value → driving the thermal prediction

### Scene 5: AI Chatbot (2 min)
1. Open the chat widget on the dashboard
2. Ask: **"What's wrong with the server?"**
3. Qwen 3.5 4B responds with:
   - Current health status
   - SHAP-informed diagnosis
   - Maintenance recommendations

> "The chatbot uses RAG — it has access to real-time SHAP explanations and health metrics, generating answers in under 8 seconds."

### Scene 6: Recovery (1 min)
1. Stop the fault injection
2. Restart healthy simulator
3. Watch health index recover: 60% → 80% → 100%

> "The system self-heals once the anomaly is resolved."

---

## Talking Points & Q&A Prep

### "Why 4 ML models?"
| Model | Purpose | Where it runs |
|-------|---------|---------------|
| 1D-CNN Multi-Task | Fault classification (7 classes) + TTF regression | Edge (INT8) + Cloud (FP32) |
| LSTM-Autoencoder | Anomaly detection (unsupervised) | Cloud only |
| Isolation Forest | PSU/Fan anomaly (lightweight) | Edge only |
| XGBoost | HDD TTF specialist (SMART data) | Cloud only |

### "What's the accuracy?"
- **97.57% validation accuracy** on fault classification (7 classes)
- **0.0611 MAE** on TTF regression (normalized 0-1 scale)
- Trained on AI4I 2020 dataset (~10,000 samples)
- **Honest evaluation**: scaler fitted on training data only (no data leakage)

### "What about real data?"
- System supports **auto-retraining**: after 30 days of real telemetry, the transfer learning adapter fine-tunes the model on server-specific data
- `retrain_manager.py` monitors data accumulation and triggers retraining
- Model versioning via SQLite registry

### "How does the confidence router work?"
- Edge runs TFLite INT8 inference (<50ms)
- If confidence > 0.85 → **local decision** (no network latency)
- If confidence < 0.85 → **offload to cloud GPU** for FP32 inference + SHAP explanation
- If cloud is unreachable → **fallback to local** with reduced confidence

### "What about security?"
- **All POST endpoints require API key** (`X-API-Key` header) — prevents LAN injection attacks
- **Rate limiting** on cloud APIs (100 req/min per IP)
- **`POST /api/update_state` was hardened in Phase E4** — previously anyone on the LAN could inject fake anomalies
- Edge node reads `API_KEY` from environment; cloud reads from `configs/api_keys.json`
- Health/state GET endpoints remain open for dashboard polling

---

## Backup Plans

| Risk | Mitigation |
|------|-----------|
| RPi4 loses WiFi | Dashboard can be cached in browser. Edge inference works fully offline. |
| Ollama not loaded | Pre-load before demo. Chat fallback uses rule-based responses. |
| TFLite fails | `mock_class` shortcut bypasses TFLite entirely. |
| InfluxDB not running | All components gracefully degrade. No crashes. |
| Network timeout | Edge has 2-second timeout on cloud offload. Falls back to local immediately. |

---

## Quick Commands Reference

```powershell
# Start everything (Master Orchestrator)
.\START_ALL.bat

# Stop everything
.\STOP_ALL.bat

# Health checks
curl http://localhost:8000/api/health        # Cloud inference
curl http://localhost:8001/api/health        # Cloud chat
curl http://100.90.160.78:8080/api/health   # Edge Doctor

# Manual cloud inference test (note: X-API-Key required!)
curl -X POST http://localhost:8000/api/offload -H "Content-Type: application/json" -H "X-API-Key: pm-edge-ac0008caeb077588ba481e57e1bde8af" -d "{\"cpu_temp\": 92, \"mock_class\": 2}"

# Auth test: should return 401 (no key)
curl -X POST http://100.90.160.78:8080/api/update_state -H "Content-Type: application/json" -d "{\"server_id\":\"test\"}"

# Auth test: should return 200 (valid key)
curl -X POST http://100.90.160.78:8080/api/update_state -H "Content-Type: application/json" -H "X-API-Key: pm-edge-ac0008caeb077588ba481e57e1bde8af" -d "{\"server_id\":\"test\",\"health_index\":50}"

# SSH into RPi4
ssh pi@100.90.160.78  # password: edge12345 (or pi@rpi4-edge.tail1ab139.ts.net)
```
