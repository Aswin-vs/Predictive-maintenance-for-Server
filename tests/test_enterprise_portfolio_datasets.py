"""
Unit tests for the complete Enterprise Dataset Portfolio:
1. Alibaba DCBrain SSD SMART & Failure Tags (alibaba-edu/dcbrain)
2. Delta Supercomputer NVIDIA A100 GPU XID Dataset & InfiniteHBD-Trace
3. NASA C-MAPSS Turbofan / Cooling Fan Benchmark (FD001-FD004)
4. Full Backblaze Drive Stats
"""

import os
import pytest
import numpy as np

from server.training.datasets import (
    EnterpriseSchemaHarmonizer,
    ENTERPRISE_36_FEATURES,
    AlibabaDRAMLoader,
    AlibabaGPUTraceLoader,
    AlibabaSSDLoader,
    BackblazeLoader,
    DeltaGPUXIDLoader,
    NASACMAPSSLoader,
)


def test_alibaba_ssd_loader():
    """Verify Alibaba DCBrain SSD loader ingests failure tags and SMART telemetry into 36 features."""
    loader = AlibabaSSDLoader(data_dir="data/alibaba_ssd", max_samples=50)
    X, M, y_anomaly, y_ttf = loader.to_canonical_records()

    assert X.shape[1] == 36, f"Expected 36 features, got {X.shape[1]}"
    assert M.shape == X.shape, f"Mask shape {M.shape} does not match feature shape {X.shape}"
    assert len(y_anomaly) == len(X), "Mismatch between records and anomaly labels"
    assert len(y_ttf) == len(X), "Mismatch between records and TTF targets"
    assert np.all(y_ttf >= 0.0), "TTF must be non-negative"

    # Verify storage channel attributes are non-trivial
    feat_names = ENTERPRISE_36_FEATURES
    wear_idx = feat_names.index("storage_nvme_wear_pct")
    assert np.any(X[:, wear_idx] > 0.0), "Expected non-zero wear percentage"


def test_delta_gpu_xid_loader():
    """Verify Delta Supercomputer & InfiniteHBD loader ingests GPU failure traces into 36 features."""
    loader = DeltaGPUXIDLoader(max_samples=100)
    X, M, y_anomaly, y_ttf = loader.to_canonical_records()

    assert X.shape[1] == 36, f"Expected 36 features, got {X.shape[1]}"
    assert M.shape == X.shape, "Mask shape must match feature shape"
    assert len(y_anomaly) == len(X), "Mismatch in labels"
    assert len(y_ttf) == len(X), "Mismatch in TTF"

    # Verify GPU channels are mapped
    feat_names = ENTERPRISE_36_FEATURES
    gpu_temp_idx = feat_names.index("gpu_temp")
    assert np.any(X[:, gpu_temp_idx] > 40.0), "Expected active GPU temperatures"


def test_nasa_cmapss_loader():
    """Verify NASA C-MAPSS loader ingests turbofan trajectories into 36 features."""
    loader = NASACMAPSSLoader(data_dir="data/nasa_cmapss", sub_dataset="FD001", max_samples=100)
    X, M, y_anomaly, y_ttf = loader.to_canonical_records()

    assert X.shape[1] == 36, f"Expected 36 features, got {X.shape[1]}"
    assert M.shape == X.shape, "Mask shape must match feature shape"
    assert len(y_anomaly) == len(X), "Mismatch in labels"
    assert len(y_ttf) == len(X), "Mismatch in TTF"

    # Verify vibration / fan channels are mapped
    feat_names = ENTERPRISE_36_FEATURES
    vib_idx = feat_names.index("vibration_rms")
    assert np.any(X[:, vib_idx] > 0.0), "Expected active vibration RMS"


def test_enterprise_portfolio_combined_harmonization():
    """Verify all 5 enterprise dataset sources seamlessly harmonize into a single batch."""
    ssd_loader = AlibabaSSDLoader(max_samples=20)
    gpu_loader = DeltaGPUXIDLoader(max_samples=20)
    cmapss_loader = NASACMAPSSLoader(max_samples=20)
    dram_loader = AlibabaDRAMLoader(max_samples=20)
    bb_loader = BackblazeLoader(max_samples=20)

    X_ssd, M_ssd, _, _ = ssd_loader.to_canonical_records()
    X_gpu, M_gpu, _, _ = gpu_loader.to_canonical_records()
    X_cm, M_cm, _, _ = cmapss_loader.to_canonical_records()
    X_dram, M_dram, _, _ = dram_loader.to_canonical_records()
    X_bb, M_bb, _, _ = bb_loader.to_canonical_records()

    X_all = np.vstack([X_ssd, X_gpu, X_cm, X_dram, X_bb])
    M_all = np.vstack([M_ssd, M_gpu, M_cm, M_dram, M_bb])

    assert X_all.shape[1] == 36
    assert M_all.shape == X_all.shape
    assert not np.isnan(X_all).any(), "Combined enterprise feature matrix contains NaNs"
    assert np.all((M_all == 0) | (M_all == 1)), "Mask must be strictly binary"
