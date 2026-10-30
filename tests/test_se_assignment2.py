
# Author: Aswin V S (Roll 21, MBT23CA021) — Module: Confidence Router BVA Tests
"""
Unit Test Suite for 23CTB40B Software Engineering — Assignment 2
================================================================
Covers Q3(a, b, c) and satisfies GitHub Actions CI requirement Q4(a).

Test Cases:
  TC01: Confidence Router - Boundary Value Analysis (BVA) On-Boundary (0.85) -> Edge Local
  TC02: Confidence Router - Boundary Value Analysis (BVA) Off-Boundary (0.849) -> Cloud Offload
  TC03: Adaptive Baseline - Equivalence Partitioning (EP) Nominal vs Anomalous Z-score
  TC04: Asymmetric TTF Loss - Equivalence Partitioning (EP) Over- vs Under-estimation Penalties
  TC05: Feature Schema v2 - Missing Sensor Diode Imputation (Feature Engineering)
  TC06: Cloud Decision Fusion - Multi-Model Weighted Ensemble Verification (60/20/10/10)
"""

import os
import sys
import pytest
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from edge.ensemble_decision import EnsembleDecisionEngine, EnsembleDecision
from edge.adaptive_baseline import AdaptiveBaseline, MetricBaseline
from server.training.loss import asymmetric_ttf_loss_np
from server.training.feature_schema_v2 import v2_to_v1_feature_vector, FEATURE_NAMES_V1
from server.deep_analysis.cloud_fusion import CloudDecisionFusion


# ============================================================================
# Q3(c) Test Case 1: Boundary Value Analysis (BVA) — Exact Boundary Threshold
# ============================================================================
def test_tc01_confidence_router_bva_on_boundary():
    """
    TC01: Boundary Value Analysis (BVA) — On-Boundary Confidence (0.85).
    Technique: Boundary Value Analysis (BVA)
    Rule: Confidence >= 0.85 must be executed locally on the Raspberry Pi 4 edge.
    Input: avg_confidence = 0.85, normal severity
    Expected Output: action == "monitor", offload == False
    """
    engine = EnsembleDecisionEngine(offload_threshold=0.85)
    decision = EnsembleDecision(health_score=95.0, severity="normal")

    # avg_confidence is exactly at threshold 0.85
    action = engine._compute_action(decision, avg_confidence=0.85)
    
    assert action != "offload", f"Expected local edge execution at boundary 0.85, got {action}"
    assert action == "monitor", f"Expected 'monitor' for healthy edge verdict, got {action}"


# ============================================================================
# Q3(c) Test Case 2: Boundary Value Analysis (BVA) — Immediate Off-Boundary
# ============================================================================
def test_tc02_confidence_router_bva_off_boundary():
    """
    TC02: Boundary Value Analysis (BVA) — Immediate Off-Boundary Confidence (0.849).
    Technique: Boundary Value Analysis (BVA)
    Rule: Confidence < 0.85 must trigger cloud offload for deep GPU analysis.
    Input: avg_confidence = 0.849 (just below boundary 0.850)
    Expected Output: action == "offload"
    """
    engine = EnsembleDecisionEngine(offload_threshold=0.85)
    decision = EnsembleDecision(health_score=95.0, severity="normal")

    # Immediate below-boundary value
    action = engine._compute_action(decision, avg_confidence=0.849)
    assert action == "offload", f"Expected cloud offload at 0.849, got {action}"

    # Secondary lower bound check
    action_low = engine._compute_action(decision, avg_confidence=0.50)
    assert action_low == "offload", f"Expected cloud offload at 0.50, got {action_low}"


# ============================================================================
# Q3(c) Test Case 3: Equivalence Partitioning (EP) — Adaptive Baseline Z-Scores
# ============================================================================
def test_tc03_adaptive_baseline_equivalence_partitioning():
    """
    TC03: Equivalence Partitioning (EP) — Nominal vs Anomalous Thermal Partitions.
    Technique: Equivalence Partitioning (EP)
    Partitions:
      - Partition 1 (Nominal): Z < 3.0 -> Healthy normal operation
      - Partition 2 (Anomalous): Z >= 3.0 -> Statistical anomaly detected
    Baseline Setup: CPU package temp mean = 50.0°C, std = 2.0°C (domain floor min_std = 6.0°C)
    """
    baseline_engine = AdaptiveBaseline(server_id="test-server-01")
    component_id = "cpu_0"
    metric_name = "temp_package_c"
    key = baseline_engine._metric_key(component_id, metric_name)

    # Establish calibrated baseline profile
    baseline_engine.baselines[key] = MetricBaseline(
        mean=50.0,
        std=2.0,
        count=150,
        min_val=42.0,
        max_val=58.0
    )

    # Partition 1: Nominal thermal value (56.0°C -> delta +6.0°C, effective std = max(2.0, 6.0) = 6.0)
    # Z = (56.0 - 50.0) / 6.0 = 1.0 (Nominal partition: Z < 3.0)
    z_nominal = baseline_engine.get_z_score(component_id, metric_name, 56.0)
    assert pytest.approx(z_nominal, 0.01) == 1.0
    assert abs(z_nominal) < 3.0, "Expected nominal partition (Z < 3.0)"

    # Partition 2: Anomalous thermal fault (86.0°C -> delta +36.0°C)
    # Z = (86.0 - 50.0) / 6.0 = 6.0 (Anomalous partition: Z >= 3.0)
    z_anomalous = baseline_engine.get_z_score(component_id, metric_name, 86.0)
    assert pytest.approx(z_anomalous, 0.01) == 6.0
    assert z_anomalous >= 3.0, "Expected anomalous fault partition (Z >= 3.0)"


# ============================================================================
# Q3(c) Test Case 4: Equivalence Partitioning (EP) — Asymmetric Survival Loss
# ============================================================================
def test_tc04_asymmetric_loss_equivalence_partitioning():
    """
    TC04: Equivalence Partitioning (EP) — Asymmetric TTF Error Partitions.
    Technique: Equivalence Partitioning (EP)
    Partitions:
      - Partition 1 (Overestimation, e = y_pred - y_true > 0): Dangerous late warning, penalized alpha = 5.0x
      - Partition 2 (Underestimation, e = y_pred - y_true <= 0): Safe early warning, penalized 1.0x
    Inputs:
      - Overestimation: y_true = 10.0, y_pred = 20.0 (error = +10.0) -> Loss = 5.0 * 10^2 = 500.0
      - Underestimation: y_true = 20.0, y_pred = 10.0 (error = -10.0) -> Loss = 1.0 * (-10)^2 = 100.0
    Expected Output: Overestimation loss / Underestimation loss == 5.0x
    """
    y_true_over = np.array([10.0])
    y_pred_over = np.array([20.0])
    loss_over = asymmetric_ttf_loss_np(y_true_over, y_pred_over, alpha=5.0)

    y_true_under = np.array([20.0])
    y_pred_under = np.array([10.0])
    loss_under = asymmetric_ttf_loss_np(y_true_under, y_pred_under, alpha=5.0)

    assert pytest.approx(loss_over, 0.01) == 500.0, f"Expected 500.0 for overestimate, got {loss_over}"
    assert pytest.approx(loss_under, 0.01) == 100.0, f"Expected 100.0 for underestimate, got {loss_under}"
    assert pytest.approx(loss_over / loss_under, 0.01) == 5.0, "Expected exact 5.0x asymmetry ratio"


# ============================================================================
# Q3(a) Test Case 5: Feature Engineering & Missing Sensor Diode Imputation
# ============================================================================
def test_tc05_schema_v2_missing_diode_imputation():
    """
    TC05: Feature Engineering — Missing Sensor Diode Imputation.
    Technique: Boundary / Robustness Testing
    Tests v2_to_v1_feature_vector() with missing sensor telemetry.
    Input: Empty/partial payload (no CPU diode temp)
    Expected Output: 28-element float vector with index 0 imputed to 45.0°C and 0 NaNs.
    """
    partial_payload = {
        "schema_version": "2.0.0",
        "server_id": "srv-test-edge",
        "cpus": [{}],  # Missing temp_package_c
        "gpus": [],
        "storage": [],
        "memory": [],
    }

    feature_vec = v2_to_v1_feature_vector(partial_payload)

    assert isinstance(feature_vec, list)
    assert len(feature_vec) == 28, f"Expected 28 features, got {len(feature_vec)}"
    # Index 0 is cpu_temp, default is 45.0°C
    assert feature_vec[0] == 45.0, f"Expected imputed cpu_temp to be 45.0, got {feature_vec[0]}"
    # Ensure no NaN or None in vector
    assert not any(np.isnan(val) for val in feature_vec if isinstance(val, (int, float)))


# ============================================================================
# Q3(a) Test Case 6: Cloud Decision Fusion Ensemble Weights (60/20/10/10)
# ============================================================================
def test_tc06_cloud_fusion_ensemble_weights():
    """
    TC06: Cloud Decision Fusion — Multi-Model Weighted Ensemble Verification.
    Technique: Functional Integration Unit Testing
    Weights: Edge Classifier 60%, LSTM-AE 20%, STGCN 10%, MAD-GAN 10%.
    Input: Nominal predictions across all models
    Expected Output: health_score >= 80.0, severity == 'HEALTHY'
    """
    fusion_engine = CloudDecisionFusion()
    
    # Verify architectural weights
    assert fusion_engine.MODEL_WEIGHTS["edge_classifier"] == 0.60
    assert fusion_engine.MODEL_WEIGHTS["lstm_ae"] == 0.20
    assert fusion_engine.MODEL_WEIGHTS["stgcn"] == 0.10
    assert fusion_engine.MODEL_WEIGHTS["madgan"] == 0.10

    # Simulate nominal multi-model responses
    edge_pred = {"class": 0, "confidence": 0.98, "ttf_days": 180.0}
    madgan_score = 0.05
    stgcn_score = 0.02
    lstm_ae_score = 0.04

    fusion_result = fusion_engine.fuse_predictions(
        server_id="hospital-cloud-srv",
        edge_prediction=edge_pred,
        madgan_score=madgan_score,
        stgcn_score=stgcn_score,
        lstm_ae_score=lstm_ae_score
    )

    assert fusion_result["health_score"] >= 80.0, f"Expected health_score >= 80, got {fusion_result['health_score']}"
    assert fusion_result["severity"] == "HEALTHY", f"Expected severity 'HEALTHY', got {fusion_result['severity']}"
    assert fusion_result["is_anomalous"] is False, "Expected false for anomaly on healthy server"
