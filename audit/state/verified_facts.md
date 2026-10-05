# Verified Facts Ledger (APPEND-ONLY)
# Project: Edge-AI Predictive Maintenance for Server Hardware
# Started: 2026-09-29T00:57:01+05:30 Session 1

[2026-09-29T00:57:01+05:30] [Step 0] FACT: Agent workspace is LOQ machine at c:\Users\LOQ\Documents\Antigravity\Predictive maintenance for Server | EVIDENCE: direct filesystem access | VOLATILE: no
[2026-09-29T00:57:01+05:30] [Step 0] FACT: Git HEAD on LOQ is 2da9c959aa2ac318ee91b23d68794656061995d2 | EVIDENCE: git rev-parse HEAD output | VOLATILE: yes
[2026-09-29T00:57:01+05:30] [Step 0] FACT: Working tree has modified tracked files (.gitignore, MEMORIES.md, PROJECT_SUMMARY.md, docs/implementation_plan.md, requirements_edge.txt, requirements_server.txt) and many untracked files (AUDIT_REPORT.md, README.md, dashboard/, server/, edge/, scripts/, etc.) | EVIDENCE: git status --porcelain output | VOLATILE: yes
[2026-09-29T00:57:01+05:30] [Step 0] FACT: Tailscale on LOQ shows loq (100.111.172.121) as offline/starting; RPi4 and IdeaPad not visible in tailscale status output | EVIDENCE: tailscale status output | VOLATILE: yes
[2026-09-29T00:57:01+05:30] [Step 0] FACT: RPi4 and IdeaPad are NOT directly reachable from this session — audit is limited to LOQ filesystem | EVIDENCE: tailscale status shows only LOQ offline/starting | VOLATILE: yes
[2026-09-29T00:57:01+05:30] [Step 0] FACT: An existing audit/ folder with old manifest/benchmark files exists but has no PROGRESS.md — this is NOT a prior structured audit checkpoint | EVIDENCE: list_dir of audit/ | VOLATILE: no
[2026-09-29T00:57:01+05:30] [Step 0] FACT: An existing AUDIT_REPORT.md (26544 bytes) in project root appears to be from a prior agent session claiming 100/100 readiness — treated as unverified | EVIDENCE: file listing | VOLATILE: no
[2026-09-29T00:57:01+05:30] [Step 0] FACT: PROJECT_SUMMARY.md claims "PRODUCTION CERTIFIED — ALL 7 PHASES COMPLETE (36/36 TESTS PASSED)" — treated as unverified | EVIDENCE: PROJECT_SUMMARY.md line 5 | VOLATILE: no
[2026-09-29T01:02:00+05:30] [Step 4] FACT: BUG-01 (cleanup loop TypeError) is FIXED — parse_last_seen() at L66-79 properly converts ISO-8601 to float epoch before subtraction at L95 | EVIDENCE: edge/api_server.py L66-95 | VOLATILE: yes
[2026-09-29T01:02:00+05:30] [Step 4] FACT: BUG-02 (train_stgcn.py fit-before-split) is FIXED — chronological split at L354-358 happens before scaler.fit_transform at L362 | EVIDENCE: server/training/train_stgcn.py L354-362 | VOLATILE: yes
[2026-09-29T01:02:00+05:30] [Step 4] FACT: BUG-03 (auto_retrain.py fit-before-split) is FIXED for LSTM-AE function — split at L147-150 before scaler.fit_transform at L155. IsolationForest at L97-99 fits on full data, acceptable for unsupervised model | EVIDENCE: server/training/auto_retrain.py L147-155 | VOLATILE: yes
[2026-09-29T01:02:00+05:30] [Step 4] FACT: BUG-04 (unbounded defaultdict/deque) is FIXED — replaced with TTLCache at L197-201 (maxsize=1000, TTL-based), deques have maxlen (10, 25) | EVIDENCE: server/deep_analysis/inference_server.py L197-201,594,671 | VOLATILE: yes
[2026-09-29T01:02:00+05:30] [Step 4] FACT: BUG-05 (sort headers + UTF-8) is FIXED — charset="UTF-8" at index.html L4, sort click listeners at app.js L1565-1573 | EVIDENCE: dashboard/index.html L4, dashboard/assets/app.js L1565-1573 | VOLATILE: yes
[2026-09-29T01:02:00+05:30] [Step 4] FACT: BUG-06 (dead InfluxDB imports) is FIXED — no Influx references remain in retrain_manager.py or train_adapter.py | EVIDENCE: grep returned empty | VOLATILE: yes
[2026-09-29T01:02:00+05:30] [Step 4] FACT: NEW BUG F-001 found — edge/api_server.py L412 references adaptive_baselines without import (NameError when cloud proxy fails) | EVIDENCE: edge/api_server.py L412 vs L694,778 | VOLATILE: yes
[2026-09-29T01:02:00+05:30] [Step 4] FACT: NEW BUG F-002 found — duplicate /api/history route at L252 and L474; first wins, second is dead code | EVIDENCE: edge/api_server.py L252,474 | VOLATILE: yes
[2026-09-29T01:06:00+05:30] [Step 4] FACT: 97th percentile adaptive quantile is consistent across train_lstm_ae.py (L278), train_madgan.py (L223), and auto_retrain.py (L214) | EVIDENCE: grep results | VOLATILE: yes
[2026-09-29T01:06:00+05:30] [Step 4] FACT: Asymmetric TTF loss uses alpha=5.0 for overestimation penalty (server/training/loss.py L36) with Weibull AFT right-censoring | EVIDENCE: loss.py L27-61, L64-100 | VOLATILE: yes
[2026-09-29T01:06:00+05:30] [Step 4] FACT: Sensor reader uses smbus2 for ADXL345 and ALSA for INMP441, no old adafruit_adxl34x/mcp3xxx imports remain | EVIDENCE: edge/sensor_reader.py, grep empty | VOLATILE: yes
[2026-09-29T01:06:00+05:30] [Step 4] FACT: Schema harmonization generates (N,36) binary missingness mask, but train_stgcn.py loads it at L347 without passing it into the STGCNEnterprise model forward() — mask is NOT consumed by the model | EVIDENCE: train_stgcn.py L347,380-388 | VOLATILE: yes
[2026-09-29T01:06:00+05:30] [Step 2] FACT: Tailscale ACL policy JSON at docs/tailscale_acl_policy.json matches the design: patient->edge:1883, edge->cloud:8000/8001, admin->SSH(22) to all. Whether it's applied in Tailscale admin console is UNVERIFIABLE (Tailscale offline) | EVIDENCE: docs/tailscale_acl_policy.json | VOLATILE: yes
[2026-09-29T01:20:35+05:30] [Step 0 resume] FACT: Git HEAD unchanged at 2da9c959aa2ac318ee91b23d68794656061995d2 between sessions | EVIDENCE: git rev-parse HEAD | VOLATILE: yes
[2026-09-29T01:29:00+05:30] [Step 4] FACT: adaptive_baseline.py Z-score anomaly detection uses sigmoid mapping (L388), domain-aware min_std floors (L366-384), EMA alpha=0.001 for slow baseline drift | EVIDENCE: edge/adaptive_baseline.py | VOLATILE: yes
[2026-09-29T01:30:00+05:30] [Step 4] FACT: Dashboard SSE reconnect is self-healing: close on error -> polling fallback 2s -> retry SSE at 4s. No eval() or document.write in 2030-line app.js | EVIDENCE: dashboard/assets/app.js L179-225 | VOLATILE: yes
[2026-09-29T01:30:00+05:30] [Step 4] FACT: Auth module (server/utils/auth.py) removed PM_AUTH_DISABLED backdoor; passes through if no keys configured (dev mode) | EVIDENCE: server/utils/auth.py L39-50 | VOLATILE: yes
[2026-09-29T01:30:00+05:30] [Step 5] FACT: PyTorch IS installed in venv_loq_py311 (torch package found in site-packages) but NOT listed in requirements_server.txt (F-012 confirmed) | EVIDENCE: task-283 output | VOLATILE: yes
[2026-09-29T01:30:00+05:30] [Step 6] FACT: Project has 7 test files: test_apis_end_to_end.py, test_dashboard_revamp.py, test_e2e_pipeline.py, test_enterprise_portfolio_datasets.py, test_phase1_hardening.py, test_phase4_plugins.py, test_phase5_survival_ml.py | EVIDENCE: file listing | VOLATILE: no
[2026-09-29T09:07:00+05:30] [Step 10] FACT: RPi4 edge node is LIVE. Tailscale active (100.90.160.78, direct at 192.168.1.2). MQTT:1883 OPEN, Dashboard:8080 OPEN, SSH:22 CLOSED. Uptime 16.7 hours. | EVIDENCE: tailscale status + port scan + /api/health | VOLATILE: yes
[2026-09-29T09:07:00+05:30] [Step 10] FACT: IdeaPad-Patient (76a82f6f) is actively publishing telemetry to RPi4. Server connected, production mode, 2.75/5.88GB RAM, health 95.7%. IdeaPad NOT on Tailscale as a named peer but reachable via LAN+MQTT. | EVIDENCE: /api/state | VOLATILE: yes
[2026-09-29T09:07:00+05:30] [Step 10] FACT: IdeaPad Tailscale DNS (ideapads145.tail1ab139.ts.net) does NOT resolve. Device not appearing in tailscale status. But patient agent IS running and publishing to RPi4 via LAN/MQTT. | EVIDENCE: ping + tailscale status | VOLATILE: yes
[2026-09-29T09:08:00+05:30] [Step 8] FACT: configs/edge-ai.service runs as User=pi (correct), contradicting F-011 which was about install_patient_linux.sh running as root. The actual RPi4 service file is properly configured. | EVIDENCE: configs/edge-ai.service:9 | VOLATILE: no

### Session 7 Connectivity Verification (2026-09-29T19:25)
[2026-09-29T19:24:00+05:30] [Step 10] FACT: IdeaPad IS on Tailscale. From RPi4's tailscale status: 100.97.69.16 ideapads145 tagged-devices active via 192.168.137.137:41641. LOQ doesn't show IdeaPad because it's registered as tagged-devices (different owner). | EVIDENCE: RPi4 SSH > tailscale status | VOLATILE: yes
[2026-09-29T19:24:00+05:30] [Step 10] FACT: IdeaPad SSH:22 OPEN on 192.168.137.137 via hotspot. Hostname: ideapadS145. Python PIDs 9896 + 10456 (272 CPU). Patient package at C:\ProgramData\PredictiveMaintenance. Scheduled task PredictiveMaintenancePatient: State=Running. | EVIDENCE: SSH + tasklist | VOLATILE: yes
[2026-09-29T19:24:00+05:30] [Step 10] FACT: RPi4 SSH:22 OPEN on 192.168.137.100 (hotspot) but CLOSED on 100.90.160.78 (Tailscale). Tailscale ACL allows admin->tag:edge:22 but RPi4 is tagged-devices, NOT tag:edge. ACL tag mismatch blocks SSH over Tailscale. | EVIDENCE: port scan + ACL policy L84 | VOLATILE: no
[2026-09-29T19:25:00+05:30] [Step 10] FACT: Edge inference is offloading to cloud (LOQ). Latest data: location=CLOUD, confidence=0.145, latency=2086ms, fault_class=0 HEALTHY, ttf_days=90, offloaded=true. Adaptive baselines calibrated, 485526 samples, 43 metrics, 10 components. | EVIDENCE: /api/state | VOLATILE: yes

### Verified Fact 42 — MAD-GAN Model Performance
Source: server/models/madgan_metadata.json
Evidence: F1=0.0514 (5.1%), Precision=0.0509, Recall=0.0519. 20 epochs, 32s training on CUDA.
Impact: Near-random performance. Cloud fusion gives this model 0.30 weight (30% of health score). Effectively adds noise to cloud-tier decisions.

### Verified Fact 43 — STGCN Model Performance
Source: server/models/stgcn_metadata.json
Evidence: val_F1=0.0408 (4.1%), 36 features, 0.72s training time. 
Impact: Even worse than MAD-GAN. Gets 0.25 weight in cloud fusion (25%). Combined MAD-GAN+STGCN=55% of cloud score is near-random.

### Verified Fact 44 — Patient Package File Integrity
Source: SHA256 comparison of 4 files
Evidence: patient_package/{auto_discovery,collector_v2,mqtt_publisher_v2,server_identity}.py are all exact byte-identical copies of server/telemetry_agent/ originals. build_patient_package.bat keeps them in sync.
Impact: No code drift between deployed patient package and development source.

### Verified Fact 45 — VIVA Guide Alpha Mismatch
Source: docs/VIVA_MASTER_GUIDE.md:272
Evidence: Guide says 'alpha=2.0' but loss.py uses alpha=5.0 and F-024 fix switched train_classifier.py to use loss.py. Guide is now stale on this claim.
Impact: Viva documentation mismatches deployed codebase post-fix.

### Verified Fact 46 — Full File Coverage Achieved
Source: Session 10 inventory reconciliation
Evidence: 130+ unique project files read/audited across all components (edge, server, training, patient_package, scripts, configs, docs, tests, dashboard). Only remaining unread: data/ directory (third-party datasets), scratch/ (debug scripts), .vscode/settings.json, .pytest_cache.
Impact: Audit coverage is comprehensive across all production code.

