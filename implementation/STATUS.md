# Remediation STATUS — Edge-AI Predictive Maintenance
Started: 2026-09-30T02:39:00+05:30
Resumed: 2026-10-01T17:41:00+05:30

## Phase 6: Accepted Risks (Confirm & Close)

| ID | Status | Notes |
|----|--------|-------|
| F-003 | CLOSED-ACCEPTED | Hardcoded TSDB password — user: "only patient folder transfers, server is my personal lap" |
| F-006 | CLOSED-ACCEPTED | Plaintext SSH passwords in network.env — user: "no need to remove password" |
| F-007 | CLOSED-ACCEPTED | API key in api_keys.json — stays on LOQ only |
| F-008 | CLOSED-ACCEPTED | Stale InfluxDB token in .env — stays on LOQ only |
| F-013 | CLOSED-ACCEPTED | SSH passwords in installer — accepted risk |
| F-025 | CLOSED-ACCEPTED | Passwords printed to console in start_loq_cloud.bat — accepted risk |

## Phase 1: ML Accuracy

| ID | Status | Evidence | Notes |
|----|--------|----------|-------|
| F-026 | DONE | `cloud_fusion.py:37-45` | Weights: madgan 0.30→0.05, stgcn 0.25→0.05, lstm_ae 0.25→0.15, edge 0.20→0.75. Comment explains mitigation. |
| F-014 | DONE | `server/models/lstm_ae.h5` (899KB), `lstm_ae_scaler.pkl` (1.2KB) | Retrained on 70,414 real healthy TSDB samples. Best val_loss=0.094945. Threshold: 0.368556→0.330935 (97th percentile). Updated in inference_server.py:276. |
| F-005 | BLOCKED-ARCHITECTURE | — | Classifier is supervised (needs labeled faults across 7 classes). Real TSDB data has NO fault labels. Domain normalization remains necessary as bridge until labeled data available. |
| F-004 | DONE | `inference_server.py:697-722` | Replaced `np.tile` with rolling window via `_stgcn_windows` TTLCache (matching existing MAD-GAN pattern). Graceful degradation when window not full. |

## Phase 2: Dashboard

| ID | Status | Evidence | Notes |
|----|--------|----------|-------|
| F-018 | DONE | `app.js:646-775` | Sensor Matrix now reads GPU/CPU/RAM/storage from active server telemetry instead of hardcoding LOQ. |
| F-019 | DONE | `app.js:277-280` | Replaced hardcoded `16874` fallback with "Awaiting telemetry data..." |
| F-022 | CLOSED-NON-ISSUE | `index.html:9` loads Inter from Google Fonts CDN | `fonts/inter.css` is an unused local fallback — harmless, not worth deleting |
| F-023 | CLOSED-NON-ISSUE | `app.js:148-149` | CSS toggle at L149 already shows `pane-models-view` before L163 guard. No render function needed — pane is static HTML. |
| DASH-001 | DONE | `index.html` (12 replacements) + `app.js` (2 replacements) | All Llama 3.1 8B references → Qwen 3.5 4B. VRAM 4.9GB → 3.4GB. |
| DASH-002 | DONE | `index.html` Cloud Fusion + Model Portfolio sections | Fusion weights 30/25/25/20 → 5/5/15/75 to match cloud_fusion.py |
| DASH-003 | DONE | RPi4:/home/pi/predictive-maintenance/dashboard/ | Files deployed via SCP, browser-verified — all 4 checks PASS |
| DASH-004 | DONE | `app.js:1881`, `chat_api.py:158`, `edge/api_server.py:584` | Added X-API-Key auth header (`pm-edge-ac0008caeb077588ba481e57e1bde8af`) to `/api/chat` fetch in `app.js`. Updated `chat_api.py` with `num_ctx: 4096`, `num_predict: 512`, thinking fallback, and forwarded header in `edge/api_server.py`. Deployed to RPi4 and verified end-to-end (15s latency, real SHAP diagnostic response). |

## Phase 3: Data Pipeline

| ID | Status | Evidence | Notes |
|----|--------|----------|-------|
| F-027 | DONE | `tsdb_writer.py:383-392` | GPU temp extraction now checks `temp_core_c` (v2 schema) before falling back to `temp`. |
| F-009 | CLOSED-NON-ISSUE | — | Not a real duplication: `edge/main.py` does domain normalization (AI4I range mapping), `inference_server.py` uses `classifier_scaler` for standard scaling. Different operations in different tiers. |
| F-010 | CLOSED-NON-ISSUE | — | Report says "theoretical race on adaptive_baselines dict (benign under GIL)". Not actionable. |
| F-011 | CLOSED-NON-ISSUE | `edge_inference.py:150-157` | Code already uses shape-based output detection (shape[-1]==7 for class, ==1 for TTF). No hardcoded index. |

## Phase 4: Operational

| ID | Status | Evidence | Notes |
|----|--------|----------|-------|
| F-016 | DONE | `:8000/api/health` → 7 models loaded, TSDB connected. `:8001/api/health` → Ollama qwen3.5:4b. | chat_api.py MODEL_NAME updated from llama3.1 to qwen3.5:4b (lower VRAM). |
| F-015 | DONE | Tailscale admin screenshots + SSH verified | User set tags via admin console. ACL updated. SSH LOQ→RPi4 + LOQ→IdeaPad confirmed working via Tailscale mesh. |

## Phase 5: Documentation

| ID | Status | Evidence | Notes |
|----|--------|----------|-------|
| F-028 | DONE | `project_viva_walkthrough.md:229` | alpha=2.0 → alpha=5.0 |
| VIVA-STALE | DONE | `VIVA_MASTER_GUIDE.md:272` | alpha=2.0 → alpha=5.0 |

## Follow-up Items (Out of Scope)
- Retrain MAD-GAN on real telemetry (requires data collection period + validation)
- Retrain STGCN on real telemetry (same)
- Retrain LSTM-AE on real telemetry (same — fixes F-014)
- Retrain 1D-CNN on real telemetry (same — fixes F-005)
- Full v1→v2 feature schema migration

## NEXT ACTION
ALL REMEDIATION + DASHBOARD FIXES COMPLETE. Deployed and verified.
F-005 is the only remaining item (BLOCKED-ARCHITECTURE: needs labeled fault data for classifier retraining).
