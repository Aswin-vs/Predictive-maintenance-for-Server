import os
import requests
import zipfile
from io import BytesIO

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, 'data')

AI4I_DIR = os.path.join(DATA_DIR, 'ai4i_2020')
CMAPSS_DIR = os.path.join(DATA_DIR, 'nasa_cmapss')
BACKBLAZE_DIR = os.path.join(DATA_DIR, 'backblaze')

# URLs
AI4I_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
CMAPSS_URL = "https://ti.arc.nasa.gov/m/project/prognostic-repository/CMAPSSData.zip"
BACKBLAZE_URL = "https://f001.backblazeb2.com/file/Backblaze-Hard-Drive-Data/data_Q1_2023.zip"

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")

def download_and_extract(url, extract_to):
    print(f"Downloading and extracting {url} to {extract_to}...")
    response = requests.get(url)
    response.raise_for_status()
    try:
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            z.extractall(extract_to)
        print("Extraction complete.")
    except zipfile.BadZipFile:
        print(f"Error: The downloaded content from {url} is not a valid zip file. Skipping.")

def main():
    os.makedirs(AI4I_DIR, exist_ok=True)
    os.makedirs(CMAPSS_DIR, exist_ok=True)
    os.makedirs(BACKBLAZE_DIR, exist_ok=True)
    
    # AI4I 2020
    ai4i_dest = os.path.join(AI4I_DIR, 'ai4i2020.csv')
    if not os.path.exists(ai4i_dest):
        download_file(AI4I_URL, ai4i_dest)
    else:
        print("AI4I 2020 already exists.")
        
    # NASA C-MAPSS
    cmapss_check = os.path.join(CMAPSS_DIR, 'train_FD001.txt')
    if not os.path.exists(cmapss_check):
        download_and_extract(CMAPSS_URL, CMAPSS_DIR)
    else:
        print("NASA C-MAPSS already exists.")
        
    # Backblaze
    # Q1 2023 contains multiple CSVs (one per day). We just check for one of them or if the dir is non-empty (excluding zips)
    files = [f for f in os.listdir(BACKBLAZE_DIR) if f.endswith('.csv')]
    if len(files) == 0:
        download_and_extract(BACKBLAZE_URL, BACKBLAZE_DIR)
    else:
        print("Backblaze Q1 2023 already exists.")

if __name__ == "__main__":
    main()
