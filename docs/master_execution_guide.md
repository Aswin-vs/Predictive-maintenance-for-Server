# 🚀 Master Execution Guide — 3-Tier Edge-AI Predictive Maintenance System

This guide outlines the exact, step-by-step execution protocol for all 3 nodes in the system architecture:
1. 🏥 **Patient Server (Monitored Machine):** Collects system telemetry & publishes to MQTT (`server/telemetry/{uuid}`).
2. 🩺 **Doctor Node (Raspberry Pi 4 Edge):** Runs LiteRT INT8 inference (< 0.6ms), Mosquitto broker, physical sensor drivers, and Operator Dashboard on Port `8080`.
3. 🏨 **Hospital Node (LOQ RTX 4060 Cloud):** Handles deep GPU offloading (Port `8000`), Chat API (Port `8001`), and Ollama Qwen 3.5 4B LLM.

> **⚠️ Security & Authentication:** All POST API endpoints require authentication via `X-API-Key` header. Key: `pm-edge-ac0008caeb077588ba481e57e1bde8af` or `PM_EDGE_2026_KEY`. Health/state GET endpoints are open.
>
> **Production Certified (2026-10-02):** 10-Model AI Portfolio, Tailscale WireGuard Mesh, 36/36 Automated Tests Passed.

---

## 📌 System Architecture & Network Topology

### Tailscale Virtual Local Mesh (Primary Production Network):
- **Tailscale Domain:** `tail1ab139.ts.net`
- **Raspberry Pi 4 Edge Doctor:** `100.90.160.78` (`rpi4-edge.tail1ab139.ts.net`) — SSH User: `pi`, Pass: `edge12345`
- **LOQ Cloud Hospital Node:** `100.111.172.121` (`loq.tail1ab139.ts.net`)
- **IdeaPad S145 Patient Node:** `100.97.69.16` (`ideapads145.tail1ab139.ts.net`) — SSH User: `Aswin`, Pass: `479548`

### Local Subnet (Fallback):
- **Subnet:** `192.168.137.x` (LOQ: `.1`, RPi4: `.100`, IdeaPad: `.137`)

---

## ⚡ Master Startup Sequence (One-Click or Step-by-Step)

### Option A: One-Click Master Launcher (Recommended)
On your **LOQ Laptop**:
```powershell
START_ALL.bat
```
This automatically verifies prerequisites, launches TimescaleDB + Grafana in Docker, starts Cloud Inference (:8000), starts Chat API (:8001), starts TSDB Ingestion, sends an SSH reboot signal to RPi4 `edge-ai.service`, and launches local Patient Telemetry.

To stop everything cleanly:
```powershell
STOP_ALL.bat
```

---

### Option B: Step-by-Step Execution

#### STEP 1: Start the LOQ Cloud Backend (Hospital Node)
Run on your **LOQ Laptop (RTX 4060 GPU)**:

```powershell
start_loq_cloud.bat
```

Or execute manually in separate terminals:
1. **Docker Containers (TimescaleDB + Grafana):**
   ```powershell
   docker compose up -d
   # TimescaleDB: localhost:5432 | Grafana: http://localhost:3000
   ```
2. **Ollama LLM Service:**
   Make sure Ollama is running in Windows System Tray (or `ollama serve`). Verify active model:
   ```powershell
   ollama ps
   # Shows qwen3.5:4b (3.1 GB, 100% GPU)
   ```
3. **Deep Analysis Inference Server (Port 8000):**
   ```powershell
   .\venv_loq_py311\Scripts\python.exe -m uvicorn server.deep_analysis.inference_server:app --host 0.0.0.0 --port 8000
   ```
4. **Cloud Chat API (Port 8001):**
   ```powershell
   .\venv_loq_py311\Scripts\python.exe -m uvicorn server.deep_analysis.chat_api:app --host 0.0.0.0 --port 8001
   ```
5. **MQTT → TimescaleDB Ingestion Service:**
   ```powershell
   .\venv_loq_py311\Scripts\python.exe server/deep_analysis/tsdb_writer.py --broker 100.90.160.78 --db-host localhost
   ```

---

#### STEP 2: Start the Raspberry Pi 4 Edge Node (Doctor Node)

SSH into your **Raspberry Pi 4**:
```powershell
ssh pi@100.90.160.78
# Password: edge12345
```

The service is configured as a native systemd daemon:
```bash
sudo systemctl status edge-ai
# If needed: sudo systemctl restart edge-ai
```

- **Live Operator Console (RPi4):** **[http://100.90.160.78:8080/](http://100.90.160.78:8080/)**
- **Mobile Access:** Open `http://100.90.160.78:8080/` in Chrome/Safari on any device connected to the Tailscale mesh.
- **AI Chatbot:** The chat drawer routes queries through the RPi4 gateway to LOQ Cloud `:8001/api/chat` with authenticated `X-API-Key`.

---

#### STEP 3: Start the Telemetry Collector on Patient Machine

On the monitored patient server (IdeaPad S145 or LOQ):

**For IdeaPad S145 (Remote execution from LOQ via SSH):**
```powershell
start_patient_ideapad_from_loq.bat
```

**For IdeaPad S145 (Local execution on IdeaPad):**
```powershell
start_patient_ideapad_local.bat
```

**For LOQ (Monitoring LOQ hardware):**
```powershell
start_patient_loq.bat
```

---

## 🔍 Verification & Health Checks

Run health checks from LOQ terminal:
```powershell
# 1. Check RPi4 Dashboard API
curl.exe -s http://100.90.160.78:8080/api/state

# 2. Check Cloud Inference Health
curl.exe -s http://localhost:8000/api/health

# 3. Check Cloud Chat API Health
curl.exe -s http://localhost:8001/api/health

# 4. Check TimescaleDB Ingestion
docker exec -it timescale psql -U pm_user -d predictive_maintenance -c "SELECT count(*) FROM server_telemetry;"
```

All 3 nodes are now fully active, synchronized, and streaming real-time predictive maintenance telemetry!
