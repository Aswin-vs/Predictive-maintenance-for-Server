# Live Changes Ledger (APPEND-ONLY)
# Every change made to any machine outside the audit/ folder
# Project: Edge-AI Predictive Maintenance for Server Hardware
# Started: 2026-09-29T00:57:01+05:30 Session 1

[2026-09-29T19:47:00+05:30] [Session 8] [F-001 FIX] edge/api_server.py:411-415 — Added `from edge.main import adaptive_baselines` with ImportError guard before L412 usage. Prevents NameError crash on /api/thresholds/{server_id} when cloud is unreachable.
[2026-09-29T19:48:00+05:30] [Session 8] [F-002 FIX] edge/api_server.py:252-272 — Removed duplicate /api/history handler (simple proxy). The richer handler at ~L460 with component mapping and fault labels now takes effect.
[2026-09-29T19:48:30+05:30] [Session 8] [F-012 FIX] requirements_server.txt — Added torch>=2.0.0 and torchvision>=0.15.0 with CUDA install comment. Fresh deploy now gets all PyTorch cloud models.
[2026-09-29T19:49:00+05:30] [Session 8] [F-020/F-021 FIX] dashboard/assets/app.js:1897-1958 — Removed 63-line duplicate of updateCalibrationBanner() and updateComponentHealthBars() plus duplicate calForceBtn listener. Original null-safe versions at L529-629 now execute.
[2026-09-29T19:52:00+05:30] [Session 8] [F-017 FIX] dashboard/assets/app.js:573-576 — Removed hardcoded wrong API key 'edge-super-secret-key-2026' from Force Calibration fetch call. Dashboard is local to RPi4; no auth header needed for local calibrate endpoint.
[2026-09-29T19:52:30+05:30] [Session 8] [F-024 FIX] server/training/train_classifier.py:36,72 — Changed import from asymmetric_loss.py (alpha=2.0) to loss.py (alpha=5.0). Next retrain will use documented 5x overestimation penalty. Existing deployed model still uses alpha=2.0 until retrained.
