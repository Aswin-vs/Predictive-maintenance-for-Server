"""
G4.1: MAD-GAN — Multivariate Anomaly Detection using GAN (PyTorch + CUDA)
Cloud-side model for deep multi-variate time-series anomaly detection.

Architecture:
  - Generator: LSTM(64) → LSTM(32) → Linear(n_features) with tanh
  - Discriminator: LSTM(64) → LSTM(32) → Linear(1) with sigmoid
  - Uses RTX 4060 GPU via PyTorch CUDA

Training Data: SMD (Server Machine Dataset) — 5 machines × 38 features
Output: PyTorch .pt models + anomaly threshold
"""

import os
import sys
import json
import time
import numpy as np
import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib

# ─── Config ─────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'smd')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

WINDOW = 25
LATENT = 16
BATCH = 256
EPOCHS = 20
LR = 2e-4
MAX_MACHINES = 5
MAX_WINDOWS = 15000

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ─── Models ─────────────────────────────────────────────────

class Generator(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.lstm1 = nn.LSTM(LATENT, 64, batch_first=True)
        self.ln1 = nn.LayerNorm(64)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.ln2 = nn.LayerNorm(32)
        self.fc = nn.Linear(32, n_features)

    def forward(self, z):
        x, _ = self.lstm1(z)
        x = self.ln1(x)
        x, _ = self.lstm2(x)
        x = self.ln2(x)
        return torch.tanh(self.fc(x))


class Discriminator(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, 64, batch_first=True)
        self.ln1 = nn.LayerNorm(64)
        self.drop1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True)
        self.ln2 = nn.LayerNorm(32)
        self.fc = nn.Sequential(nn.Linear(32, 16), nn.LeakyReLU(0.2), nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.drop1(self.ln1(x))
        x, (h, _) = self.lstm2(x)
        return self.fc(h.squeeze(0))


# ─── Data ────────────────────────────────────────────────────

def load_smd(data_dir, max_m=MAX_MACHINES):
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    label_dir = os.path.join(data_dir, 'test_label')
    files = sorted(glob.glob(os.path.join(train_dir, 'machine-*.txt')))[:max_m]
    
    trains, tests, labels = [], [], []
    for f in files:
        name = os.path.basename(f)
        print(f"  Loading {name}...")
        trains.append(np.loadtxt(f, delimiter=','))
        tp = os.path.join(test_dir, name)
        if os.path.exists(tp):
            tests.append(np.loadtxt(tp, delimiter=','))
        lp = os.path.join(label_dir, name)
        if os.path.exists(lp):
            labels.append(np.loadtxt(lp, delimiter=','))
    
    return np.vstack(trains), np.vstack(tests) if tests else None, np.concatenate(labels) if labels else None


def make_windows(data, w=WINDOW, stride=5):
    return np.array([data[i:i+w] for i in range(0, len(data)-w+1, stride)], dtype=np.float32)


# ─── Training ───────────────────────────────────────────────

def train():
    print("=" * 60)
    print(f"  MAD-GAN Training | Device: {DEVICE}")
    print("=" * 60)
    
    # Load
    X_train, X_test, y_test = load_smd(os.path.abspath(DATA_DIR))
    n_feat = X_train.shape[1]
    print(f"Features: {n_feat}, Train: {X_train.shape}")
    
    # Scale
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test) if X_test is not None else None
    
    # Windows
    windows = make_windows(X_tr)
    np.random.shuffle(windows)
    if len(windows) > MAX_WINDOWS:
        windows = windows[:MAX_WINDOWS]
    print(f"Windows: {windows.shape}")
    
    # DataLoader
    dataset = TensorDataset(torch.from_numpy(windows))
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, pin_memory=True)
    
    # Models
    G = Generator(n_feat).to(DEVICE)
    D = Discriminator(n_feat).to(DEVICE)
    opt_g = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
    bce = nn.BCELoss()
    
    print(f"\nTraining {EPOCHS} epochs, {len(loader)} batches/epoch...")
    t0 = time.time()
    
    for epoch in range(EPOCHS):
        d_losses, g_losses = [], []
        
        for (real,) in loader:
            real = real.to(DEVICE)
            bs = real.size(0)
            
            # --- Discriminator ---
            noise = torch.randn(bs, WINDOW, LATENT, device=DEVICE)
            fake = G(noise).detach()
            
            real_labels = torch.rand(bs, 1, device=DEVICE) * 0.15 + 0.85
            fake_labels = torch.rand(bs, 1, device=DEVICE) * 0.15
            
            d_real = D(real)
            d_fake = D(fake)
            d_loss = (bce(d_real, real_labels) + bce(d_fake, fake_labels)) / 2
            
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()
            
            # --- Generator ---
            noise = torch.randn(bs, WINDOW, LATENT, device=DEVICE)
            fake = G(noise)
            g_out = D(fake)
            g_loss = bce(g_out, torch.ones(bs, 1, device=DEVICE))
            
            # Feature matching
            feat_loss = torch.mean((real.mean(0) - fake.mean(0)) ** 2)
            g_total = g_loss + 0.1 * feat_loss
            
            opt_g.zero_grad()
            g_total.backward()
            opt_g.step()
            
            d_losses.append(d_loss.item())
            g_losses.append(g_total.item())
        
        if (epoch + 1) % 2 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:2d}/{EPOCHS} | D={np.mean(d_losses):.4f} G={np.mean(g_losses):.4f} | {elapsed:.0f}s")
    
    total_time = time.time() - t0
    print(f"\nTraining done in {total_time:.0f}s")
    
    # --- Anomaly scoring on test data ---
    precision = recall = f1 = 0.0
    threshold = 0.5
    
    if X_te is not None and y_test is not None:
        print("\nComputing anomaly scores on test data...")
        test_wins = make_windows(X_te, stride=1)
        if len(test_wins) > 20000:
            idx = np.random.choice(len(test_wins), 20000, replace=False)
            tw = test_wins[idx]
        else:
            idx = np.arange(len(test_wins))
            tw = test_wins
        
        G.eval()
        D.eval()
        scores = []
        
        with torch.no_grad():
            for i in range(0, len(tw), 256):
                batch = torch.from_numpy(tw[i:i+256]).to(DEVICE)
                disc = D(batch).cpu().numpy().flatten()
                noise = torch.randn(len(batch), WINDOW, LATENT, device=DEVICE)
                recon = G(noise).cpu().numpy()
                recon_err = np.mean((tw[i:i+256] - recon) ** 2, axis=(1, 2))
                combined = (1 - disc) * 0.6 + np.clip(recon_err / (np.max(recon_err) + 1e-8), 0, 1) * 0.4
                scores.extend(combined)
        
        scores = np.array(scores)
        wl = np.array([1.0 if np.any(y_test[i:i+WINDOW] > 0) else 0.0 for i in idx])
        
        normal_s = scores[wl == 0]
        if len(normal_s) > 0:
            threshold = float(np.percentile(normal_s, 97))
        
        pred = (scores > threshold).astype(float)
        tp = np.sum((pred == 1) & (wl == 1))
        fp = np.sum((pred == 1) & (wl == 0))
        fn = np.sum((pred == 0) & (wl == 1))
        precision = float(tp / (tp + fp + 1e-8))
        recall = float(tp / (tp + fn + 1e-8))
        f1 = float(2 * precision * recall / (precision + recall + 1e-8))
        
        print(f"  Threshold: {threshold:.4f}")
        print(f"  P={precision:.4f} R={recall:.4f} F1={f1:.4f}")
    
    # --- Save ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    gen_path = os.path.join(MODEL_DIR, 'madgan_generator.pt')
    disc_path = os.path.join(MODEL_DIR, 'madgan_discriminator.pt')
    scaler_path = os.path.join(MODEL_DIR, 'madgan_scaler.joblib')
    meta_path = os.path.join(MODEL_DIR, 'madgan_metadata.json')
    
    torch.save(G.state_dict(), gen_path)
    torch.save(D.state_dict(), disc_path)
    joblib.dump(scaler, scaler_path)
    
    with open(meta_path, 'w') as f:
        json.dump({
            "model_name": "MAD-GAN (PyTorch)",
            "phase": "G4.1",
            "framework": "pytorch",
            "device": str(DEVICE),
            "architecture": {"gen": "LSTM(64→32)→Linear→tanh", "disc": "LSTM(64→32)→Linear→sigmoid"},
            "input_shape": [WINDOW, n_feat],
            "n_features": n_feat,
            "latent_dim": LATENT,
            "anomaly_threshold": threshold,
            "precision": precision, "recall": recall, "f1": f1,
            "training_time_s": round(total_time),
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "epochs": EPOCHS,
        }, f, indent=2)
    
    print(f"\nSaved: {gen_path} ({os.path.getsize(gen_path)/1024:.1f} KB)")
    print(f"Saved: {disc_path} ({os.path.getsize(disc_path)/1024:.1f} KB)")
    print(f"\n{'='*60}")
    print(f"  MAD-GAN DONE | F1={f1:.4f} | {total_time:.0f}s on {DEVICE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    train()
