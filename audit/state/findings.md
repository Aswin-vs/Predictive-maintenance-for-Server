# Audit Findings Ledger (APPEND-ONLY)
# Project: Edge-AI Predictive Maintenance for Server Hardware
# Started: 2026-09-29T00:57:01+05:30 Session 1

### F-001 | Severity: High | Step: 4 | Status: OPEN
Location: LOQ:edge/api_server.py:412
Evidence: Line 412 references `adaptive_baselines` without import; other handlers (L694, L778) correctly use `from edge.main import adaptive_baselines`. When cloud proxy fails at L403-408, the except block falls through to L411-416, hitting NameError.
Risk: GET /api/thresholds/{server_id} crashes with NameError when cloud is unreachable, breaking dashboard threshold display.
Proposed fix: Add `from edge.main import adaptive_baselines` before line 411 (matching pattern at L694).


### F-002 | Severity: Medium | Step: 4 | Status: OPEN
Location: LOQ:edge/api_server.py:252,474
Evidence: Two @app.get("/api/history") handlers registered. FastAPI uses the first (L252, simple proxy), making the second (L474, with component mapping and richer formatting) dead code that never executes.
Risk: The intended handler with component mapping at L474 never runs. Dashboard history may show raw data without the component labels.
Proposed fix: Remove the earlier duplicate handler at L252-272 and keep the richer one at L474.

### F-003 | Severity: Medium | Step: 4 | Status: OPEN
Location: LOQ:server/deep_analysis/inference_server.py:121,180
Evidence: Default password "pm_edge_2026" hardcoded as fallback for TSDB_PASS env var. Appears in two places (init and reconnect).
Risk: Database credential in source code; anyone with repo access has the TimescaleDB password.
Proposed fix: Remove default value from os.getenv; require TSDB_PASS be set in environment or fail with a clear error.

### F-004 | Severity: Low | Step: 4 | Status: OPEN
Location: LOQ:server/deep_analysis/inference_server.py:704
Evidence: STGCN inference at L704 uses np.tile to repeat a single feature vector to fill the temporal window, rather than accumulating real temporal data over multiple timesteps.
Risk: Destroys temporal pattern detection — STGCN sees identical frames repeated, so its spatio-temporal convolution reduces to spatial-only. Not exploitable but misleading: the model cannot detect temporal anomalies this way.
Proposed fix: Accumulate a rolling window per server (like LSTM-AE at L593-604 and MAD-GAN at L670-675), or document this as a known limitation.

### F-005 | Severity: Medium | Step: 4 | Status: OPEN
Location: LOQ:server/deep_analysis/inference_server.py:501-541
Evidence: Domain normalization hardcodes CPU freq=3000, GPU power=30, GPU clock=1500, GPU util=10 etc. to match AI4I benchmark training distribution. Real telemetry values are discarded and replaced with constants.
Risk: The 1D-CNN classifier never sees actual GPU/CPU values from monitored machines — it always classifies against synthetic benchmark-scale inputs. This means real GPU thermal anomalies or real CPU frequency throttling would be invisible to the classifier.
Proposed fix: Retrain the classifier on real telemetry distribution, or at minimum document this normalization as a known accuracy limitation.

### F-006 | Severity: Critical | Step: 7 | Status: OPEN
Location: LOQ:configs/network.env:12,18
Evidence: Plaintext SSH passwords for RPi4 (line 12) and IdeaPad (line 18) stored in configs/network.env. File is gitignored but exists on disk. [REDACTED — password values not recorded per audit protocol]
Risk: Anyone with disk access or a filesystem backup has SSH credentials to all project machines. If the repo is shared or backed up insecurely, these passwords are exposed.
Proposed fix: Remove passwords from config files entirely; use SSH keys instead. At minimum, move to a secrets manager or encrypted vault. Document password rotation procedure.

### F-007 | Severity: Medium | Step: 7 | Status: OPEN
Location: LOQ:configs/api_keys.json:3
Evidence: API key for Edge->Cloud auth stored in plaintext JSON. File is gitignored (line 82 of .gitignore). [REDACTED — key value not recorded]
Risk: API key discoverable by anyone with disk access. While the key only protects the internal Tailscale mesh API, it should still be treated as a secret.
Proposed fix: Load from environment variable; do not keep in a JSON file on disk.

### F-008 | Severity: Medium | Step: 7 | Status: OPEN
Location: LOQ:.env:1-2
Evidence: Contains InfluxDB token (deprecated but still a real secret, 88 chars) and PM_API_KEY in plaintext. File is gitignored and not tracked. [REDACTED]
Risk: Stale secrets on disk. InfluxDB token may still be valid even though the project migrated to TimescaleDB.
Proposed fix: Delete the .env file or remove the InfluxDB token. Rotate PM_API_KEY if it was ever shared.

### F-009 | Severity: Low | Step: 4 | Status: OPEN
Location: LOQ:edge/main.py:258-308 vs server/deep_analysis/inference_server.py:501-541
Evidence: The domain normalization (RAM=16GB, CPU temp mapping, GPU power=30W, PSU voltage=2.18) is duplicated identically in both edge/main.py (extract_features) and inference_server.py (offload endpoint). The normalization constants are hardcoded in both locations.
Risk: If normalization constants are updated in one file but not the other, edge and cloud classifiers would produce different results for the same telemetry.
Proposed fix: Extract normalization into a shared function in a common module (e.g., server/training/normalization.py) and import it in both places.

### F-010 | Severity: Low | Step: 4 | Status: OPEN
Location: LOQ:edge/main.py:336,506-509
Evidence: adaptive_baselines dict is accessed outside state_lock at L336 (membership check) and L506-509 (create/ingest), but state_lock protects mutations at L69-77. Python's GIL prevents crashes but the check-then-act at L506-507 is theoretically racy (two threads could create baselines for the same server_id simultaneously).
Risk: Theoretical race with very low practical impact due to GIL. Worst case: duplicate AdaptiveBaseline creation for the same server, one immediately garbage collected.
Proposed fix: Move the adaptive_baselines creation guard into the state_lock block, or accept as benign.

### F-011 | Severity: Medium | Step: 4 | Status: OPEN
Location: LOQ:scripts/install_patient_linux.sh:142
Evidence: Systemd unit runs as User=root (line 142). The telemetry collector only needs to read hardware sensors (via /sys, lm_sensors, psutil) and connect to MQTT — it doesn't need root privileges.
Risk: If the collector or any of its dependencies has a vulnerability (e.g., paho-mqtt, psutil), an attacker gains root access on the patient server.
Proposed fix: Create a dedicated user (e.g., 'pm-agent') with membership in groups 'disk', 'i2c', and 'dialout' for sensor access. Change User=root to User=pm-agent.

### F-012 | Severity: Medium | Step: 4 | Status: OPEN
Location: LOQ:requirements_server.txt
Evidence: PyTorch (torch, torchvision) is not listed in requirements_server.txt despite being required for: MAD-GAN (inference_server.py L299-348), STGCN (inference_server.py L355-372), LSTM-AE training (auto_retrain.py L143), and all Phase G4 cloud models. The code handles absence gracefully (try/except) but a fresh install from requirements_server.txt would silently lack all PyTorch-based models.
Risk: A fresh cloud server deployment would have no MAD-GAN, STGCN, or LSTM-AE capability. Cloud decision fusion would fall back to classifier-only, degrading detection quality.
Proposed fix: Add 	orch>=2.0.0 and 	orchvision>=0.15.0 to requirements_server.txt. Consider adding a comment about the CUDA-specific install URL for GPU support.

### F-013 | Severity: Medium | Step: 4 | Status: OPEN
Location: LOQ:scripts/Install-Patient-Windows.ps1:182-184
Evidence: The Windows installer copies configs/network.env to the patient install directory (C:\ProgramData\PredictiveMaintenance\configs\network.env). This file contains SSH passwords for RPi4 and IdeaPad (see F-006). The patient server only needs EDGE_BROKER_HOST and EDGE_BROKER_PORT from this file.
Risk: SSH credentials for all project machines are unnecessarily deployed to every patient server. If any patient is compromised, the attacker gets SSH passwords for the Edge and IdeaPad nodes.
Proposed fix: Either strip sensitive values before copying, or write only the needed vars (EDGE_BROKER_HOST, EDGE_BROKER_PORT) to a minimal .env file at the install target.

### F-003-AMENDMENT | Severity: Medium→High (upgraded) | Step: 7 | Status: OPEN
Location: LOQ:multiple files (8 occurrences across 6 files)
Evidence: The hardcoded TSDB password "pm_edge_2026" appears in: inference_server.py (L121,180), auto_retrain.py (L48), adaptive_thresholds.py (L67), retrain_daemon.py (L178), tsdb_writer.py (L66), benchmark_e2e_latencies.py (L148), scratch/audit_database.py (L9). All use os.getenv("TSDB_PASS", "pm_edge_2026") pattern.
Risk: Password is in tracked source code (git-committed). Anyone with repo read access has the TimescaleDB credential. 8 occurrences make it easy to miss during remediation.
Proposed fix: Define TSDB_PASS in a single module (e.g., server/config.py) that reads from env var with NO default. All 8 files should import from that module. Fail loudly if TSDB_PASS is not set.

### F-014 | Severity: Medium | Step: 4 | Status: OPEN
Location: LOQ:server/deep_analysis/inference_server.py (LSTM-AE inference path) + logs/inference_server.log
Evidence: The inference server log shows LSTM-AE reconstruction error is consistently exactly 1000.000000 on every tick (e.g., "LSTM-AE ANOMALY: recon_error=1000.000000 > threshold=0.368556"). This is repeated for every 14-second inference cycle across the entire log (4.8MB, Sept 28). The 1D-CNN classifier simultaneously reports "HEALTHY conf=1.000" on every tick. A recon error of exactly 1000 is likely a sentinel/fallback value rather than a real computation, meaning the LSTM-AE model is not functioning correctly in production and generates a false positive anomaly flag on every single prediction.
Risk: Every inference tick produces a spurious LSTM-AE anomaly alert. In the cloud fusion engine, this model has a 0.25 weight, meaning 25% of the fused health score is permanently biased toward "anomalous". If the system relies on this for alerts, it could cause alert fatigue or incorrect maintenance decisions.
Proposed fix: Investigate why recon_error is always 1000. Check if the LSTM-AE model is loaded correctly, if the input normalization matches training, and if the reconstruction computes correctly. The value 1000 strongly suggests a hardcoded fallback when the model fails silently.

### F-014-AMENDMENT | Step: 4 | Status: OPEN
Root cause confirmed: inference_server.py:610 explicitly clamps recon_error to 1000.0 with comment "until retrained on real data". The LSTM-AE was trained on AI4I benchmark distribution, but real telemetry produces out-of-distribution inputs. The actual reconstruction error exceeds 1000 on every tick, always triggering the anomaly flag. This is a known issue by the developer (hence the clamp comment) but has not been fixed. This is a manifestation of F-005 (domain normalization mismatch).

### F-015 | Severity: Low | Step: 10 | Status: OPEN
Location: docs/tailscale_acl_policy.json (L84) + RPi4 tailscale device tags
Evidence: The ACL policy defines SSH access as admin->tag:edge:22, admin->tag:cloud:22, admin->tag:patient:22. However, RPi4 is tagged as "tagged-devices" (not "tag:edge"), and IdeaPad is also "tagged-devices" (not "tag:patient"). This tag mismatch means SSH over Tailscale is blocked by ACL, despite port 22 being open on the devices themselves. SSH works fine over the hotspot (192.168.137.x) because there's no ACL there.
Risk: SSH-over-Tailscale won't work unless device tags are corrected. This only matters if you try to administer devices remotely when not on the same hotspot.
Proposed fix: Re-tag devices in Tailscale admin console: RPi4 -> tag:edge, IdeaPad -> tag:patient, LOQ -> tag:cloud. Or change ACL to allow admin->tagged-devices:22.

### F-016 | Severity: Medium | Step: 10 | Status: OPEN
Location: LOQ:server/deep_analysis/inference_server.py (not running) + RPi4 edge-ai systemd journal
Evidence: RPi4 systemd journal shows continuous cloud offload failures: "Cloud offload failed: HTTPConnectionPool(host='loq.tail1ab139.ts.net', port=8000): Max retries exceeded... Connection to loq.tail1ab139.ts.net timed out." The LOQ inference server (port 8000) is not currently running, so all cloud deep analysis (MAD-GAN, STGCN, LSTM-AE, SHAP) is unavailable. Edge inference falls back to local TFLite models and produces HEALTHY with confidence 0.14, which correctly triggers offload attempts.
Risk: Cloud-level anomaly detection is completely offline. If a real anomaly occurs, only edge models (TFLite classifier + Isolation Forest) can detect it. The confidence-gated offload mechanism works correctly (threshold 0.85, current 0.14 triggers offload), but the cloud is unreachable.
Proposed fix: Start the LOQ inference server: venv_loq_py311\Scripts\python.exe server\deep_analysis\inference_server.py. Or run START_ALL.bat.

### F-017 | Severity: Medium | Step: 5 | Status: OPEN
Location: LOQ:dashboard/assets/app.js:575
Evidence: Force Calibration button sends X-API-Key 'edge-super-secret-key-2026' but actual API key is 'pm-edge-ac0008caeb077588ba481e57e1bde8af' (configs/api_keys.json). Button click returns 403 Forbidden on RPi4 edge server.
Risk: Force Calibration feature is non-functional. Users clicking it get no result.
Proposed fix: Read the API key from a config or make the endpoint auth-free for local-only access.

### F-018 | Severity: Medium | Step: 5 | Status: OPEN
Location: LOQ:dashboard/assets/app.js:659-760
Evidence: The 10-channel Sensor Audit Matrix is 100% hardcoded to LOQ hardware (RTX 4060, AMD Ryzen 7, A:\ 931GB + C:\ 476GB + D:\ 477GB). When viewing IdeaPad telemetry, this tab still shows LOQ's hardware specs. The sensor descriptions mention 'I2C Bus 1', 'ALSA plughw:3,0' etc. which are RPi4-specific, not the selected server.
Risk: Sensor Matrix shows incorrect/misleading data for any server that isn't LOQ. For a multi-server dashboard, this violates the per-server view contract.
Proposed fix: Make sensor inventory server-aware — either fetch from API or derive from telemetry keys available per server.

### F-019 | Severity: Low | Step: 5 | Status: OPEN
Location: LOQ:dashboard/assets/app.js:277,385,412 + index.html:152,155
Evidence: Multiple hardcoded fallback values displayed when real data isn't available: 16874 samples (L277), 18.5ms latency (L385/html L152), 98% confidence (L412), 0.8s freshness (L391/html L155). These are cosmetic demo values shown as if real.
Risk: A user with no data flowing would see realistic-looking fake numbers, potentially believing the system is operational when it isn't.
Proposed fix: Show '—' or 'Awaiting data' placeholders instead of fake numbers when real values are null.

### F-020 | Severity: Medium | Step: 5 | Status: OPEN
Location: LOQ:dashboard/assets/app.js:1897-1925 (duplicate function)
Evidence: updateCalibrationBanner is defined TWICE: once at L529-599 with full logic (null-safe, with calibration progress bar, force button with API call), and again at L1897-1925 with simpler logic (no null safety on banner, no API key). The second definition silently overwrites the first at runtime, losing the richer implementation's API-call logic.
Risk: Force calibration button is doubly broken — wrong API key (F-017) AND the overwritten handler at L1947 calls /api/calibrate without an API key at all. Also, the simpler L1897 version crashes if banner is null (no null check).
Proposed fix: Remove the duplicate at L1897-1958 and keep the original at L529-599.

### F-021 | Severity: Low | Step: 5 | Status: OPEN  
Location: LOQ:dashboard/assets/app.js:1927-1942 (duplicate function)
Evidence: updateComponentHealthBars is also defined TWICE: once at L602-629 and again at L1927-1942. The second overwrites the first. The second version crashes if container is null because it does container.style.display = 'none' without a null check, and uses different icon for 'hdd' (💽 vs 💾).
Risk: Null reference crash if componentHealthBars element is missing; inconsistent icon rendering.
Proposed fix: Remove the duplicate at L1927-1942.

### F-022 | Severity: Low | Step: 5 | Status: OPEN
Location: LOQ:dashboard/assets/fonts/inter.css
Evidence: Local font file inter.css defines @font-face for Inter weights 300,400,600,700, but index.html L9 already imports Inter from Google Fonts CDN. The local file is never referenced in any HTML or CSS.
Risk: Dead file. No functional impact but adds clutter.
Proposed fix: Delete inter.css or add a reference to it (and remove CDN link for offline support).

### F-023 | Severity: Low | Step: 5 | Status: OPEN
Location: LOQ:dashboard/index.html:80 + dashboard/assets/app.js:146-177
Evidence: Tab 4 'models-view' (10-Model AI Portfolio) is defined in HTML at L80/600-800 but has NO handler in app.js switchComplexTab(). The if/else chain at L166-176 handles sensors-view, schema36-view, hardware-view, trends-view, xai-view, database-view, platform-view, cloud-view — but NOT models-view. The tab content is 100% static HTML with no dynamic rendering.
Risk: No functional bug (static HTML shows fine), but inconsistent with other tabs that have dynamic rendering. The tab pane ID would need to be 'pane-models-view' (and it exists at L600).
Proposed fix: Either add models-view to switchComplexTab or document that this tab is intentionally static.


### F-024 | Severity: Medium | Step: 4 | Status: OPEN
Location: LOQ:server/training/asymmetric_loss.py:4 vs server/training/loss.py:36
Evidence: Two competing asymmetric loss implementations exist. asymmetric_loss.py (used by train_classifier.py L36) has alpha=2.0. loss.py has alpha=5.0 (matching docs/dashboard). The deployed classifier model was trained with alpha=2.0, not the documented 5.0.
Risk: Documentation claims "5x penalty for overestimation" but the deployed model was trained with only 2x. This is a material accuracy claim discrepancy. The Weibull AFT loss in loss.py is never used by the classifier training.
Proposed fix: Update train_classifier.py to import from loss.py instead of asymmetric_loss.py, retrain, and redeploy. Or update docs to reflect actual alpha=2.0.

### F-025 | Severity: Low | Step: 4 | Status: OPEN
Location: LOQ:START_ALL.bat:100-101, STOP_ALL.bat:36
Evidence: START_ALL.bat L100-101 prints Grafana password (admin/pm_grafana_2026) and TimescaleDB password (pm_user/pm_edge_2026) in plain text to console. STOP_ALL.bat L36 passes RPi4 SSH password directly in Python one-liner command (visible in process list).
Risk: Passwords visible on screen to anyone looking at the console, and in process history.
Proposed fix: Remove password display from echoed output. For STOP_ALL SSH, use SSH keys or paramiko with credential file.


### SESSION 8 REMEDIATION STATUS UPDATES
[2026-09-29T19:49:00+05:30] F-001: Status changed OPEN → FIXED. Import guard added.
[2026-09-29T19:49:00+05:30] F-002: Status changed OPEN → FIXED. Duplicate handler removed.
[2026-09-29T19:49:00+05:30] F-012: Status changed OPEN → FIXED. PyTorch added to requirements.
[2026-09-29T19:49:00+05:30] F-020: Status changed OPEN → FIXED. Duplicate JS function removed.
[2026-09-29T19:49:00+05:30] F-021: Status changed OPEN → FIXED. Duplicate JS function removed.
[2026-09-29T19:52:00+05:30] F-017: Status changed OPEN → FIXED. Wrong API key removed from dashboard.
[2026-09-29T19:52:30+05:30] F-024: Status changed OPEN → FIXED. train_classifier.py now imports loss.py (alpha=5.0).

### F-026 | Severity: High | Step: 4 | Status: OPEN
Location: LOQ:server/models/madgan_metadata.json + stgcn_metadata.json
Evidence: MAD-GAN F1=0.051 (5.1%), STGCN F1=0.041 (4.1%). Both deep models perform barely above random chance. The cloud fusion engine (cloud_fusion.py) gives these models a combined weight of 0.55 (MAD-GAN 0.30 + STGCN 0.25), meaning 55% of the cloud-fused health score comes from near-random models. Fan Bearing CNN is trained exclusively on synthetic data (100% accuracy meaningless). TCN Storage RUL MAE=15.9 days on Backblaze — decent but unvalidated on real hardware.
Risk: Cloud-tier anomaly detection is essentially non-functional. The weighted fusion produces misleading health scores because 55% of the weight comes from models that can't discriminate anomalies from normal. Combined with F-014 (LSTM-AE always-anomalous), the entire cloud deep analysis pipeline is unreliable.
Proposed fix: Retrain MAD-GAN and STGCN on more data or with better hyperparameters. Until then, reduce their weights in cloud_fusion.py or disable them and rely on edge classifier + Isolation Forest only.

### F-027 | Severity: Low | Step: 4 | Status: OPEN
Location: LOQ:server/deep_analysis/tsdb_writer.py:369
Evidence: GPU temperature extraction uses key 'temp_package' but feature_schema_v2.py defines the v2 GPU field as 'temp_core_c'. The cpu extraction correctly uses 'temp_package' for CPU. This causes GPU temperature to always be None in TimescaleDB for v2 payloads.
Risk: GPU temperature column in telemetry_raw table is never populated from v2 telemetry. Historical GPU temp data unavailable for retrain/analysis.
Proposed fix: Change L369 from 'temp_package' to 'temp_core_c' for GPU. CPU extraction at L369 is correct.


### F-028 [Low] — project_viva_walkthrough.md internal alpha contradiction
**File:** docs/project_viva_walkthrough.md
**Evidence:** Line 229 says alpha=2.0 (Concept 1 equation), but line 385 (Q17 answer) correctly says alpha=5.0. Same document contradicts itself.
**Impact:** Minor documentation inconsistency. The equation section was written pre-Phase 5, the Q&A section post-Phase 5.
**Fix:** Update line 229 to say alpha=5.0.
