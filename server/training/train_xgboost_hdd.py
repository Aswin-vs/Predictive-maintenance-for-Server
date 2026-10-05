"""
Train XGBoost Model for HDD TTF Regression.

XGBoost is used as the specialist model for hard drive failure prediction
using SMART data attributes and delta-features. It leverages GPU acceleration.
"""

import os
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODEL_DIR = os.path.join(BASE_DIR, 'server', 'models')

# Features required for XGBoost HDD model
XGB_FEATURES = [
    'smart_5_raw', 'smart_9_raw', 'smart_187_raw', 
    'smart_188_raw', 'smart_194_raw', 'smart_197_raw', 'smart_198_raw'
    # In a full pipeline, delta-features from feature_engineering.py would also be included here
]
TARGET = 'ttf_days'

def train_xgboost(data_path=None, use_gpu=True):
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if data_path and os.path.exists(data_path):
        print(f"Loading HDD data from {data_path}...")
        df = pd.read_csv(data_path)
        # Drop rows where target is missing
        df = df.dropna(subset=[TARGET])
        
        # Filter: only train on drives with observed failures within 365 days
        df = df[df[TARGET] < 365.0]
        
        # We need a validation split
        train_df = df.sample(frac=0.8, random_state=42)
        val_df = df.drop(train_df.index)
        
        X_train = train_df[XGB_FEATURES]
        y_train = train_df[TARGET]
        X_val = val_df[XGB_FEATURES]
        y_val = val_df[TARGET]
    else:
        print("Real data not found. Generating dummy Backblaze data...")
        X_train = np.random.randint(0, 100, size=(10000, len(XGB_FEATURES)))
        y_train = np.random.uniform(0, 365, size=10000)
        X_val = np.random.randint(0, 100, size=(2000, len(XGB_FEATURES)))
        y_val = np.random.uniform(0, 365, size=2000)
        
    print(f"Training XGBoost on {len(X_train)} samples (GPU={use_gpu})...")
    
    # Configure model
    params = {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 6,
        'objective': 'reg:squarederror', # TTF is regression
        'random_state': 42
    }
    
    if use_gpu:
        try:
            # Check if GPU support is available in XGBoost
            params['tree_method'] = 'hist'
            params['device'] = 'cuda'
        except Exception:
            print("XGBoost GPU configuration failed. Falling back to CPU.")
            params['device'] = 'cpu'
            
    params['early_stopping_rounds'] = 50
    model = xgb.XGBRegressor(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=10
    )
    
    print("Training complete!")
    model_path = os.path.join(MODEL_DIR, 'xgboost_hdd.pkl')
    joblib.dump(model, model_path)
    print(f"XGBoost model saved to {model_path}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=os.path.join(DATA_DIR, 'backblaze_ttf.csv'))
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    args = parser.parse_args()
    
    train_xgboost(args.data, use_gpu=not args.cpu)
