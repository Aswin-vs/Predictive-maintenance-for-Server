import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')
BACKBLAZE_DIR = os.path.join(DATA_DIR, 'backblaze')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

def build_ttf_labels():
    print("Building TTF labels for Backblaze data...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # 1. Find all CSV files in the backblaze directory (could be nested)
    csv_files = glob.glob(os.path.join(BACKBLAZE_DIR, '**', '*.csv'), recursive=True)
    if not csv_files:
        print("No CSV files found in Backblaze directory.")
        return
    
    print(f"Found {len(csv_files)} CSV files. First pass: identifying failure dates...")
    
    # Only load columns needed to find failure dates
    usecols = ['date', 'serial_number', 'failure']
    
    # We will collect failed serial numbers and their failure dates
    failure_records = []
    
    for file in tqdm(csv_files, desc="Pass 1: Finding failures"):
        try:
            df = pd.read_csv(file, usecols=usecols)
            failed = df[df['failure'] == 1]
            if not failed.empty:
                failure_records.append(failed[['serial_number', 'date']])
        except Exception as e:
            print(f"Error reading {file}: {e}")
            
    if not failure_records:
        print("No failures found in dataset. Using right-censored labels for all.")
        failure_map = {}
    else:
        failure_df = pd.concat(failure_records)
        # In case a drive fails multiple times (rare but data errors happen), take the first failure date
        failure_map = failure_df.groupby('serial_number')['date'].min().to_dict()
        print(f"Identified {len(failure_map)} failed drives.")
    
    print("Second pass: Building labeled dataset...")
    # Now we process files and append the TTF column
    # To save memory, we'll only extract the necessary features for the HDD model
    features_cols = [
        'date', 'serial_number', 'model', 'failure',
        'smart_5_raw', 'smart_9_raw', 'smart_187_raw', 
        'smart_188_raw', 'smart_194_raw', 'smart_197_raw', 'smart_198_raw'
    ]
    
    output_file = os.path.join(PROCESSED_DIR, 'backblaze_ttf.csv')
    
    # Write header first
    header_written = False
    
    for file in tqdm(csv_files, desc="Pass 2: Applying TTF"):
        try:
            # Read all available columns to check intersection
            header = pd.read_csv(file, nrows=0).columns.tolist()
            actual_cols = [c for c in features_cols if c in header]
            
            df = pd.read_csv(file, usecols=actual_cols)
            
            # Map failure dates
            df['failure_date'] = df['serial_number'].map(failure_map)
            
            # Calculate TTF in days
            # If no failure_date, fill with a large right-censored value, e.g., 9999
            df['date'] = pd.to_datetime(df['date'])
            df['failure_date'] = pd.to_datetime(df['failure_date'])
            
            df['ttf_days'] = (df['failure_date'] - df['date']).dt.days
            
            # Fill right-censored instances
            df['ttf_days'] = df['ttf_days'].fillna(9999)
            
            # Drop the intermediate failure_date column
            df = df.drop(columns=['failure_date'])
            
            # Append to master CSV
            df.to_csv(output_file, mode='a', header=not header_written, index=False)
            header_written = True
            
        except Exception as e:
            print(f"Error processing {file}: {e}")

    print(f"TTF labeling complete. Data saved to {output_file}")

if __name__ == "__main__":
    build_ttf_labels()
