"""
G3.5: Synthetic Failure Cascade Generator
==========================================
Generates synthetic fan bearing degradation data for training the Fan Bearing CNN.
Simulates PWM duty cycle, actual RPM, and the correlation breakdown that occurs
when fan bearings degrade.

Healthy fan: RPM closely tracks PWM duty cycle (linear correlation)
Degrading fan: RPM drops below expected for given PWM (bearing friction)
"""

import numpy as np
import pandas as pd
import os


def generate_healthy_fan_data(n_sequences=500, sequence_length=60, noise_std=0.02):
    """
    Generate healthy fan data where RPM tracks PWM closely.
    
    Healthy behavior:
    - PWM varies between 20-100%
    - RPM = PWM * max_rpm * (1 + small_noise)
    - Ratio RPM/expected_RPM ≈ 1.0
    """
    sequences = []
    max_rpm = 3000  # Typical laptop fan max RPM
    
    for _ in range(n_sequences):
        # Random PWM profile (gradually changing)
        base_pwm = np.random.uniform(20, 90)
        pwm_drift = np.cumsum(np.random.randn(sequence_length) * 2)
        pwm = np.clip(base_pwm + pwm_drift, 15, 100)
        
        # Expected RPM (linear relationship with slight non-linearity)
        expected_rpm = (pwm / 100) * max_rpm
        
        # Healthy: actual RPM closely tracks expected
        noise = np.random.randn(sequence_length) * noise_std * max_rpm
        actual_rpm = expected_rpm + noise
        actual_rpm = np.clip(actual_rpm, 0, max_rpm * 1.05)
        
        # Ratio (should be close to 1.0 for healthy)
        ratio = actual_rpm / np.maximum(expected_rpm, 1)
        
        # Stack features: [PWM%, RPM, ratio]
        seq = np.stack([pwm, actual_rpm, ratio], axis=-1)
        sequences.append(seq)
    
    return np.array(sequences, dtype=np.float32)


def generate_degrading_fan_data(n_sequences=500, sequence_length=60, 
                                 degradation_levels=None):
    """
    Generate degrading fan data where RPM drops below expected.
    
    Degrading behavior:
    - PWM commands are normal
    - RPM response is reduced (bearing friction)
    - Intermittent stalling (RPM drops to near 0 briefly)
    - Increased noise/vibration in RPM signal
    """
    sequences = []
    max_rpm = 3000
    
    if degradation_levels is None:
        degradation_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    for _ in range(n_sequences):
        # Random degradation level
        deg_level = np.random.choice(degradation_levels)
        
        # Random PWM profile
        base_pwm = np.random.uniform(20, 90)
        pwm_drift = np.cumsum(np.random.randn(sequence_length) * 2)
        pwm = np.clip(base_pwm + pwm_drift, 15, 100)
        
        # Expected RPM
        expected_rpm = (pwm / 100) * max_rpm
        
        # Degraded: RPM is reduced by degradation factor
        rpm_reduction = 1.0 - deg_level * np.random.uniform(0.5, 1.5)
        
        # Add progressive degradation over the window
        degradation_profile = np.linspace(
            1.0 - deg_level * 0.3, 
            1.0 - deg_level, 
            sequence_length
        )
        
        # Increased noise with degradation
        noise_amp = 0.02 + deg_level * 0.15
        noise = np.random.randn(sequence_length) * noise_amp * max_rpm
        
        actual_rpm = expected_rpm * degradation_profile + noise
        
        # Intermittent stalling for severe degradation
        if deg_level > 0.4:
            n_stalls = int(deg_level * 5)
            stall_positions = np.random.randint(0, sequence_length, n_stalls)
            for pos in stall_positions:
                stall_len = np.random.randint(1, 4)
                end_pos = min(pos + stall_len, sequence_length)
                actual_rpm[pos:end_pos] *= np.random.uniform(0, 0.3)
        
        actual_rpm = np.clip(actual_rpm, 0, max_rpm * 1.05)
        
        # Ratio (should be significantly below 1.0 for degrading)
        ratio = actual_rpm / np.maximum(expected_rpm, 1)
        
        # Stack features
        seq = np.stack([pwm, actual_rpm, ratio], axis=-1)
        sequences.append(seq)
    
    return np.array(sequences, dtype=np.float32)


def generate_training_dataset(n_healthy=2000, n_degrading=2000, sequence_length=60,
                               save_dir=None):
    """
    Generate complete training dataset with balanced healthy/degrading samples.
    
    Returns:
        X: (n_samples, 60, 3) — [PWM%, RPM, ratio]
        y: (n_samples,) — 0=healthy, 1=degrading
    """
    print("Generating synthetic fan bearing data...")
    print(f"  Healthy samples: {n_healthy}")
    print(f"  Degrading samples: {n_degrading}")
    print(f"  Sequence length: {sequence_length} ticks")
    
    X_healthy = generate_healthy_fan_data(n_healthy, sequence_length)
    X_degrading = generate_degrading_fan_data(n_degrading, sequence_length)
    
    X = np.concatenate([X_healthy, X_degrading], axis=0)
    y = np.concatenate([
        np.zeros(n_healthy, dtype=np.float32),
        np.ones(n_degrading, dtype=np.float32)
    ])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    print(f"  Total samples: {len(X)}")
    print(f"  Shape: X={X.shape}, y={y.shape}")
    print(f"  Class balance: {(y==0).sum()} healthy, {(y==1).sum()} degrading")
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "fan_X.npy"), X)
        np.save(os.path.join(save_dir, "fan_y.npy"), y)
        print(f"  Saved to: {save_dir}/fan_X.npy, fan_y.npy")
    
    return X, y


if __name__ == "__main__":
    X, y = generate_training_dataset(
        n_healthy=3000, n_degrading=3000, 
        save_dir="data/synthetic_fan"
    )
    print(f"\nSample healthy ratio mean: {X[y==0][:, :, 2].mean():.3f}")
    print(f"Sample degrading ratio mean: {X[y==1][:, :, 2].mean():.3f}")
