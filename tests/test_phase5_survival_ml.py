"""
Automated PyTest Suite for Phase 5: Enterprise ML Training & Survival Analysis.
Tests:
1. Asymmetric TTF penalty loss (alpha=5.0 ratio verification)
2. Weibull AFT survival negative log-likelihood and gradients
3. Multi-task combined survival loss
4. Enterprise Schema Harmonization (36 features & missingness mask)
5. Alibaba Cloud PAKDD 2021 DRAM dataset loader
6. Alibaba cluster-trace-gpu-v2020 dataset loader
7. Backblaze Drive Stats dataset loader
8. Retraining Daemon drift detection (PSI/Wasserstein) & promotion gate
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from server.training.loss import (
    AsymmetricTTFLoss,
    WeibullAFTLoss,
    CombinedSurvivalLoss,
    asymmetric_ttf_loss_np,
    weibull_median_ttf_np
)
from server.training.datasets.schema_harmonization import (
    EnterpriseSchemaHarmonizer,
    ENTERPRISE_36_FEATURES,
    ENTERPRISE_DEFAULTS,
    FIELD_POLICIES
)
from server.training.datasets.alibaba_dram_loader import AlibabaDRAMLoader
from server.training.datasets.alibaba_gpu_trace_loader import AlibabaGPUTraceLoader
from server.training.datasets.backblaze_loader import BackblazeLoader
from server.deep_analysis.retrain_daemon import DriftDetector, RetrainingDaemon


# ─── 1. Asymmetric Loss Tests ──────────────────────────────────────────────────

def test_asymmetric_loss_penalty_ratio():
    """Verify that dangerous overestimates are penalized exactly alpha=5.0x more heavily."""
    loss_fn = AsymmetricTTFLoss(alpha=5.0, imminent_threshold=0.0)

    # Identical magnitude errors of 2.0 days
    y_true = torch.tensor([10.0])
    y_pred_over = torch.tensor([12.0])  # error = +2.0 (overestimate)
    y_pred_under = torch.tensor([8.0])  # error = -2.0 (underestimate)

    loss_over = loss_fn(y_pred_over, y_true).item()
    loss_under = loss_fn(y_pred_under, y_true).item()

    assert loss_under == pytest.approx(4.0, abs=1e-4)
    assert loss_over == pytest.approx(20.0, abs=1e-4)
    assert (loss_over / loss_under) == pytest.approx(5.0, abs=1e-4)


def test_asymmetric_loss_np():
    """Verify NumPy evaluation utility."""
    y_true = np.array([10.0, 5.0])
    y_pred = np.array([12.0, 3.0])  # error +2.0 (loss 20), error -2.0 (loss 4)
    mean_loss = asymmetric_ttf_loss_np(y_true, y_pred, alpha=5.0)
    assert mean_loss == pytest.approx(12.0, abs=1e-4)


# ─── 2. Weibull AFT Loss Tests ─────────────────────────────────────────────────

def test_weibull_aft_loss_and_gradients():
    """Verify Weibull AFT log-likelihood computation and backward gradient propagation."""
    loss_fn = WeibullAFTLoss()

    log_scale = torch.tensor([2.5, 3.0, 1.8], requires_grad=True)
    log_shape = torch.tensor([0.2, 0.4, 0.1], requires_grad=True)
    duration = torch.tensor([12.0, 25.0, 5.0])
    event = torch.tensor([1.0, 0.0, 1.0])  # 2 failures, 1 censored

    loss = loss_fn(log_scale, log_shape, duration, event)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() > 0

    loss.backward()
    assert log_scale.grad is not None
    assert log_shape.grad is not None
    assert not torch.isnan(log_scale.grad).any()
    assert not torch.isnan(log_shape.grad).any()


def test_weibull_median_ttf():
    scale = np.array([20.0, 10.0])
    shape = np.array([2.0, 1.0])
    median = weibull_median_ttf_np(scale, shape)
    # For shape=1 (exponential), median = scale * ln(2)
    assert median[1] == pytest.approx(10.0 * np.log(2.0), abs=1e-4)


# ─── 3. Combined Survival Loss Tests ──────────────────────────────────────────

def test_combined_survival_loss():
    """Verify multi-task combined loss and metrics dictionary."""
    combined_fn = CombinedSurvivalLoss(alpha_asym=5.0)

    pred_cls = torch.tensor([[0.7], [0.1]])
    pred_ttf = torch.tensor([[0.2], [0.8]])
    true_cls = torch.tensor([[1.0], [0.0]])
    true_ttf = torch.tensor([[0.1], [0.9]])
    log_scale = torch.tensor([[2.0], [3.5]])
    log_shape = torch.tensor([[0.3], [0.5]])

    total_loss, metrics = combined_fn(
        pred_cls=pred_cls,
        pred_ttf=pred_ttf,
        true_cls=true_cls,
        true_ttf=true_ttf,
        pred_log_scale=log_scale,
        pred_log_shape=log_shape
    )

    assert "loss_cls" in metrics
    assert "loss_ttf_asym" in metrics
    assert "loss_aft" in metrics
    assert "loss_total" in metrics
    assert total_loss.item() > 0


# ─── 4. Schema Harmonization Tests ─────────────────────────────────────────────

def test_schema_harmonizer_36_features():
    """Verify output contains exactly 36 features in canonical order and valid mask."""
    harmonizer = EnterpriseSchemaHarmonizer()

    raw_sample = {
        "cpu_temp": 58.2,
        "load_percent": 45.0,
        "gpu_temperature": 62.0,
        "smart_5": 14,
        "vibration": 0.045
    }

    feat_vec, mask_vec = harmonizer.harmonize_record(raw_sample)
    assert len(feat_vec) == 36
    assert len(mask_vec) == 36
    assert len(harmonizer.feature_names) == 36

    # Features present in raw_sample should be masked as 1.0 (genuine)
    assert mask_vec[harmonizer.feature_index["cpu_temp"]] == 1.0
    assert feat_vec[harmonizer.feature_index["cpu_temp"]] == pytest.approx(58.2, abs=1e-2)

    # Missing features must be imputed with physical baseline, NOT zeroed, and mask=0.0
    assert mask_vec[harmonizer.feature_index["humidity"]] == 0.0
    assert feat_vec[harmonizer.feature_index["humidity"]] == pytest.approx(ENTERPRISE_DEFAULTS["humidity"], abs=1e-2)


# ─── 5. Dataset Loaders Tests ──────────────────────────────────────────────────

def test_alibaba_dram_loader():
    """Verify Alibaba DRAM loader generates compliant enterprise records."""
    loader = AlibabaDRAMLoader()
    df_feat, df_mask, ttf, event = loader.load_or_generate(max_samples=200, synthetic_fallback=True)

    assert df_feat.shape == (200, 36)
    assert df_mask.shape == (200, 36)
    assert len(ttf) == 200
    assert len(event) == 200
    assert (ttf >= 0).all()
    assert set(event).issubset({0.0, 1.0})


def test_alibaba_gpu_trace_loader():
    """Verify Alibaba GPU trace loader outputs compliant records and hotspot metrics."""
    loader = AlibabaGPUTraceLoader()
    df_feat, df_mask, ttf, event = loader.load_or_generate(max_samples=200, synthetic_fallback=True)

    assert df_feat.shape == (200, 36)
    assert df_mask.shape == (200, 36)
    assert len(ttf) == 200
    assert len(event) == 200


def test_backblaze_loader():
    """Verify Backblaze loader ingests SMART data and produces survival TTF targets."""
    loader = BackblazeLoader()
    df_feat, df_mask, ttf, event = loader.load_or_generate(max_samples=200, synthetic_fallback=True)

    assert df_feat.shape[1] == 36
    assert df_mask.shape[1] == 36
    assert len(ttf) == len(df_feat)
    assert len(event) == len(df_feat)


# ─── 6. Retraining Daemon & Drift Tests ─────────────────────────────────────────

def test_drift_detector_psi_and_wasserstein():
    """Verify PSI and Wasserstein calculations."""
    np.random.seed(42)
    ref = np.random.normal(loc=50.0, scale=5.0, size=1000)
    same = np.random.normal(loc=50.0, scale=5.0, size=1000)
    drifted = np.random.normal(loc=65.0, scale=8.0, size=1000)

    # Identical distribution -> low PSI (< 0.10)
    psi_low = DriftDetector.calculate_psi(ref, same)
    assert psi_low < 0.10

    # Shifted distribution -> high PSI (> 0.25)
    psi_high = DriftDetector.calculate_psi(ref, drifted)
    assert psi_high > 0.25

    # Wasserstein distance on shifted data > same data
    w_same = DriftDetector.calculate_wasserstein(ref, same)
    w_drift = DriftDetector.calculate_wasserstein(ref, drifted)
    assert w_drift > w_same


def test_retraining_daemon_promotion_gate():
    """Verify promotion gate requires >= +2% improvement over baseline."""
    daemon = RetrainingDaemon(min_f1_improvement=0.02)
    daemon.state["current_production_f1"] = 0.9000

    # Candidate with +1.5% improvement should be rejected
    rejected = daemon.evaluate_and_promote_candidate(candidate_f1=0.9150, candidate_model_path="dummy.pt", dry_run=True)
    assert not rejected

    # Candidate with +2.5% improvement should be promoted
    promoted = daemon.evaluate_and_promote_candidate(candidate_f1=0.9250, candidate_model_path="dummy.pt", dry_run=True)
    assert promoted
