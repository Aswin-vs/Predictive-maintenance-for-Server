"""
Asymmetric Survival TTF Loss and Weibull Accelerated Failure Time (AFT) Engine.
Phase 5: Enterprise ML Training & Survival Analysis.

Implements:
1. Asymmetric Penalty Loss (alpha = 5.0 for overestimating TTF vs alpha = 1.0 for underestimating)
2. Weibull Accelerated Failure Time (AFT) Negative Log-Likelihood with Right-Censoring Support
3. Multi-Task Combined Survival Loss (Classification + TTF Regression + Weibull Hazard)
4. TensorFlow & NumPy drop-in compatible variants for cross-tier edge-cloud training.
"""

import math
from typing import Optional, Tuple, Dict, Any
import numpy as np

# ─── PyTorch Implementation ───────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class AsymmetricTTFLoss(nn.Module):
        """
        Asymmetric penalty loss for Time-to-Failure (TTF) regression.

        Formula:
            L_asym(y, y_hat) = alpha * (y_hat - y)^2 if y_hat > y else (y_hat - y)^2
            alpha = 5.0 penalizes dangerous overestimation of remaining life 5x more
            heavily than conservative early inspection.
        """
        def __init__(self, alpha: float = 5.0, imminent_threshold: float = 0.2, imminent_weight: float = 2.0):
            super().__init__()
            self.alpha = float(alpha)
            self.imminent_threshold = float(imminent_threshold)
            self.imminent_weight = float(imminent_weight)

        def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            """
            Args:
                y_pred: Predicted TTF, shape (N, 1) or (N,)
                y_true: Ground truth TTF, shape matching y_pred
            """
            y_pred = y_pred.view(-1)
            y_true = y_true.view(-1)
            error = y_pred - y_true
            squared_error = torch.square(error)

            # Overestimate penalty: alpha * (y_pred - y_true)^2 when y_pred > y_true (error > 0)
            loss = torch.where(error > 0, self.alpha * squared_error, squared_error)

            # Imminent failure penalty multiplier
            if self.imminent_threshold > 0:
                weights = torch.where(y_true < self.imminent_threshold, self.imminent_weight, 1.0)
                loss = loss * weights

            return torch.mean(loss)


    class WeibullAFTLoss(nn.Module):
        """
        Weibull Accelerated Failure Time (AFT) Negative Log-Likelihood Loss.
        Supports right-censored telemetry data where a machine has not yet failed (event=0).

        Survival Function:
            S(t | x) = exp( - (t / lambda(x))^beta )

        Hazard Function:
            h(t | x) = (beta / lambda(x)) * (t / lambda(x))^(beta - 1)

        Density Function:
            f(t | x) = h(t | x) * S(t | x)

        Log-Likelihood:
            log L = sum [ event * log f(t | x) + (1 - event) * log S(t | x) ]
            Negative Log-Likelihood (NLL) = - log L
        """
        def __init__(self, eps: float = 1e-7):
            super().__init__()
            self.eps = eps

        def forward(
            self,
            log_scale: torch.Tensor,
            log_shape: torch.Tensor,
            duration: torch.Tensor,
            event: torch.Tensor
        ) -> torch.Tensor:
            """
            Args:
                log_scale: log(lambda), shape (N, 1) or (N,)
                log_shape: log(beta), shape (N, 1) or (N,)
                duration: observed time t > 0, shape (N, 1) or (N,)
                event: 1.0 if failure observed (uncensored), 0.0 if right-censored
            """
            log_scale = log_scale.view(-1)
            log_shape = log_shape.view(-1)
            duration = duration.view(-1).clamp(min=self.eps)
            event = event.view(-1)

            scale = torch.exp(log_scale).clamp(min=self.eps)
            shape = torch.exp(log_shape).clamp(min=self.eps, max=20.0)

            z = duration / scale
            z_pow = torch.pow(z, shape).clamp(max=1e10)

            # Log survival: log S(t) = - (t / lambda)^beta
            log_survival = - z_pow

            # Log hazard: log h(t) = log(beta) - log(lambda) + (beta - 1) * (log(t) - log(lambda))
            log_hazard = log_shape - log_scale + (shape - 1.0) * torch.log(z.clamp(min=self.eps))

            # Total log-likelihood
            log_likelihood = event * log_hazard + log_survival
            nll = - torch.mean(log_likelihood)

            return nll


    class CombinedSurvivalLoss(nn.Module):
        """
        Multi-Task Enterprise Loss combining:
        1. Anomaly / Fault Classification (BCE or Cross-Entropy)
        2. Asymmetric TTF Regression Loss
        3. Weibull AFT Survival Log-Likelihood
        """
        def __init__(
            self,
            alpha_asym: float = 5.0,
            weight_cls: float = 1.0,
            weight_ttf: float = 2.0,
            weight_aft: float = 1.0
        ):
            super().__init__()
            self.asym_loss = AsymmetricTTFLoss(alpha=alpha_asym)
            self.aft_loss = WeibullAFTLoss()
            self.bce_loss = nn.BCELoss()
            self.weight_cls = weight_cls
            self.weight_ttf = weight_ttf
            self.weight_aft = weight_aft

        def forward(
            self,
            pred_cls: torch.Tensor,
            pred_ttf: torch.Tensor,
            true_cls: torch.Tensor,
            true_ttf: torch.Tensor,
            pred_log_scale: Optional[torch.Tensor] = None,
            pred_log_shape: Optional[torch.Tensor] = None,
            event: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, Dict[str, float]]:
            pred_cls_clamped = torch.clamp(pred_cls.view(-1), min=1e-6, max=1.0 - 1e-6)
            true_cls_clamped = torch.clamp(true_cls.view(-1).float(), min=0.0, max=1.0)
            loss_cls = self.bce_loss(pred_cls_clamped, true_cls_clamped)
            loss_ttf = self.asym_loss(pred_ttf.view(-1), true_ttf.view(-1).float())

            loss_total = self.weight_cls * loss_cls + self.weight_ttf * loss_ttf
            metrics = {
                "loss_cls": float(loss_cls.item()),
                "loss_ttf_asym": float(loss_ttf.item()),
            }

            if pred_log_scale is not None and pred_log_shape is not None:
                if event is None:
                    event = (true_cls > 0).float()
                loss_aft = self.aft_loss(pred_log_scale, pred_log_shape, true_ttf, event)
                loss_total = loss_total + self.weight_aft * loss_aft
                metrics["loss_aft"] = float(loss_aft.item())

            metrics["loss_total"] = float(loss_total.item())
            return loss_total, metrics


# ─── TensorFlow / Keras Implementation ─────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras import backend as K
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


if TF_AVAILABLE:
    def get_asymmetric_ttf_loss_tf(alpha: float = 5.0, imminent_threshold: float = 0.2, imminent_weight: float = 2.0):
        """Keras-compatible loss function for Asymmetric TTF regression."""
        def loss_fn(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            error = y_pred - y_true
            squared_error = tf.square(error)
            asym_loss = tf.where(error > 0.0, alpha * squared_error, squared_error)
            if imminent_threshold > 0:
                weights = tf.where(y_true < imminent_threshold, imminent_weight, 1.0)
                asym_loss = asym_loss * weights
            return K.mean(asym_loss)
        return loss_fn

    def get_weibull_aft_loss_tf(eps: float = 1e-7):
        """Keras-compatible Weibull AFT survival negative log-likelihood."""
        def loss_fn(y_true, y_pred):
            # y_true: [duration, event]
            # y_pred: [log_scale, log_shape]
            duration = tf.clip_by_value(y_true[:, 0], eps, 1e8)
            event = y_true[:, 1]
            log_scale = y_pred[:, 0]
            log_shape = y_pred[:, 1]
            scale = tf.clip_by_value(tf.exp(log_scale), eps, 1e8)
            shape = tf.clip_by_value(tf.exp(log_shape), eps, 20.0)
            z = duration / scale
            z_pow = tf.clip_by_value(tf.pow(z, shape), 0.0, 1e10)
            log_survival = - z_pow
            log_hazard = log_shape - log_scale + (shape - 1.0) * tf.math.log(tf.clip_by_value(z, eps, 1e8))
            nll = - tf.reduce_mean(event * log_hazard + log_survival)
            return nll
        return loss_fn


# ─── NumPy Inference Utilities ────────────────────────────────────────────────
def asymmetric_ttf_loss_np(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 5.0) -> float:
    """NumPy implementation for offline metric evaluation."""
    y_true = np.asarray(y_true, dtype=np.float32).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float32).flatten()
    error = y_pred - y_true
    squared_error = np.square(error)
    loss = np.where(error > 0, alpha * squared_error, squared_error)
    return float(np.mean(loss))


def weibull_median_ttf_np(scale: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """Compute median Time-to-Failure from Weibull scale and shape parameters."""
    return scale * (math.log(2.0) ** (1.0 / shape))


def weibull_mean_ttf_np(scale: np.ndarray, shape: np.ndarray) -> np.ndarray:
    """Compute expected Time-to-Failure using Gamma function."""
    gamma_val = np.array([math.gamma(1.0 + 1.0 / s) for s in shape])
    return scale * gamma_val
