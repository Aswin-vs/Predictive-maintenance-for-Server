# Edge-AI Predictive Maintenance — Project Handover

**Date**: 2026-10-01  
**Status**: DEPLOYMENT READY  
**Audit**: 28 findings — 27 resolved, 1 architectural limitation (F-005)

---

## Architecture (3-Tier)

```
IdeaPad (Patient)  ->  RPi4 (Doctor/Edge)  ->  LOQ (Hospital/Cloud)
  tag:patient            tag:edge                tag:cloud
  100.97.69.16           100.90.160.78           100.115.95.24
  Collector agent        Edge inference          Cloud GPU inference
  MQTT publisher         Dashboard :8080         Inference :8000
                         Mosquitto :1883         Chat API :8001
                                                 TimescaleDB :5432
                                                 Grafana :3000
```

All devices on Tailscale mesh (separate physical networks, connected via WireGuard).

---

## Credentials

| Device | User | Password | SSH |
|--------|------|----------|-----|
| RPi4 | pi | `<RPI4_PASSWORD>` (set in network.env) | ssh pi@100.90.160.78 |
| IdeaPad | Aswin | `<IDEAPAD_PASSWORD>` (set in network.env) | ssh Aswin@100.97.69.16 |
| TimescaleDB | pm_user | `<TSDB_PASSWORD>` (set in cloud.env) | psql -h localhost -d predictive_maintenance |
| Tailscale Admin | `<TAILSCALE_ADMIN_EMAIL>` | - | https://login.tailscale.com/admin |

---

## How to Start Services (LOQ)

```powershell
cd "c:\Users\LOQ\Documents\Antigravity\Predictive maintenance for Server"
# Inference Server (7 models + TSDB)
venv_loq_py311\Scripts\python.exe -m uvicorn server.deep_analysis.inference_server:app --host 0.0.0.0 --port 8000
# Chat API (Ollama qwen3.5:4b RAG)
venv_loq_py311\Scripts\python.exe -m uvicorn server.deep_analysis.chat_api:app --host 0.0.0.0 --port 8001
```

RPi4 auto-starts via systemd (edge-ai.service). Dashboard at http://100.90.160.78:8080

---

## Key File Paths

| Purpose | Path |
|---------|------|
| Inference server | server/deep_analysis/inference_server.py |
| Chat API | server/deep_analysis/chat_api.py |
| Cloud fusion | server/deep_analysis/cloud_fusion.py |
| Dashboard HTML | dashboard/index.html (92KB) |
| Dashboard JS | dashboard/assets/app.js (103KB) |
| Feature schema | server/training/feature_schema.py (28 features) |
| Models | server/models/ |
| RPi4 dashboard | /home/pi/predictive-maintenance/dashboard/ |
| Audit report | audit/FINAL_AUDIT_REPORT.md |
| Remediation status | implementation/STATUS.md |

---

## ML Models & Fusion Weights

| Model | Trained On | Fusion Weight |
|-------|-----------|---------------|
| 1D-CNN Classifier | AI4I synthetic | (edge tier, 75% via edge weight) |
| LSTM-AE | 70K real TSDB (retrained 2026-10-01) | 15% |
| MAD-GAN | AI4I synthetic | 5% |
| STGCN | AI4I synthetic | 5% |

---

## Only Remaining Item: F-005

Domain normalization in inference_server.py maps real values into AI4I ranges.
Fix: Retrain classifier on real labeled fault data (needs labeled anomalies across 7 classes).
Mitigated: Edge model has 75% fusion weight.

---

## Deploy Dashboard to RPi4

```python
import paramiko
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect('100.90.160.78', username='pi', password='edge12345')
sftp = ssh.open_sftp()
sftp.put('dashboard/index.html', '/home/pi/predictive-maintenance/dashboard/index.html')
sftp.put('dashboard/assets/app.js', '/home/pi/predictive-maintenance/dashboard/assets/app.js')
sftp.close(); ssh.close()
```
