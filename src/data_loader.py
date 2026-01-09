import os
import requests
import zipfile
import shutil
import pandas as pd
from tqdm import tqdm
import time

# --- Configuration Constants ---
DATASET_URL = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
ZIP_FILE = "ESC-50-master.zip"
EXTRACT_DIR = "ESC-50-master"
BASE_DATA_DIR = "data"

# --- Dataset Classification Schema ---
# Dictionary mapping project specific labels (keys) to ESC-50 source categories (values).
# Keys represent the binary classification targets for the model.
TARGET_CLASSES = {
    "danger": [
        "chainsaw",         # Primary threat: Illegal logging activity
        "hand_saw",         # Secondary threat: Manual logging noise
        "engine",           # Vehicle intrusion (Trucks/Off-road vehicles)
        "crackling_fire",   # Environmental threat: Forest fire initiation
        "fireworks",        # Acoustic proxy for gunshots (Poaching simulation)
        "helicopter",       # Aerial intrusion
        "glass_breaking"    # Vandalism indicator
    ], 
    "safe": [
        "chirping_birds",   # Natural biophony: Avian
        "wind",             # Environmental background: Wind
        "thunderstorm",     # Environmental background: Storms
        "pouring_water",    # Environmental background: Rain/Stream
        "crickets",         # Natural biophony: Nocturnal insects (High frequency)
        "insects",          # Natural biophony: General insects
        "frog",             # Natural biophony: Amphibians
        "footsteps",        # Human presence: Patrols/Hikers (False positive reduction)
        "cow",              # Livestock: Cattle
        "sheep"             # Livestock: Sheep
    ]
}

def clean_broken_downloads():
    """
    Removes potentially corrupted archives or incomplete extraction directories
    to ensure a clean state before execution.
    """
    if os.path.exists(ZIP_FILE):
        print(f"Existing archive detected: '{ZIP_FILE}'. Removing to ensure integrity...")
        os.remove(ZIP_FILE)
    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)

def create_dir_structure():
    """
    Initializes the standard directory structure for the dataset.
    Ensures idempotency by clearing existing data directories before recreation.
    """
    print(f"Initializing directory structure in '{BASE_DATA_DIR}'...")
    
    if os.path.exists(BASE_DATA_DIR):
        shutil.rmtree(BASE_DATA_DIR)
    
    for label in TARGET_CLASSES.keys():
        path = os.path.join(BASE_DATA_DIR, label)
        os.makedirs(path, exist_ok=True)
        print(f"  [Directory Created] {path}")

def download_file(url, filename, retries=3):
    """
    Downloads the dataset from the specified URL with retry logic and progress tracking.
    
    Args:
        url (str): Source URL.
        filename (str): Destination filename.
        retries (int): Number of connection attempts allowed.
    """
    for attempt in range(retries):
        try:
            print(f"Initiating download sequence (Attempt {attempt + 1}/{retries})...")
            response = requests.get(url, stream=True)
            response.raise_for_status() 
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filename, 'wb') as file, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)
            print("Download sequence completed.")
            return
        except Exception as e:
            print(f"Download Failure: {e}. Retrying in 3 seconds...")
            if os.path.exists(filename):
                os.remove(filename) 
            time.sleep(3)
    
    raise Exception("Critical Error: Failed to retrieve dataset after maximum retries.")

def process_data():
    """
    Extracts the archive, filters audio files based on target categories,
    and organizes them into the training directory structure.
    
    Returns:
        dict: Statistics regarding the number of files processed per category.
    """
    print("Extracting archive...")
    try:
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(".")
    except zipfile.BadZipFile:
        raise Exception("Error: Corrupted zip archive. Delete and restart pipeline.")
    
    # Locate metadata file
    meta_path = os.path.join(EXTRACT_DIR, "meta", "esc50.csv")
    audio_src_dir = os.path.join(EXTRACT_DIR, "audio")
    
    if not os.path.exists(meta_path):
        raise FileNotFoundError("Critical Error: Metadata CSV file missing.")

    df = pd.read_csv(meta_path)
    
    print("Filtering and organizing audio samples...")
    stats = {}
    
    for label, categories in TARGET_CLASSES.items():
        # Filter DataFrame for rows matching the current category list
        filtered_df = df[df['category'].isin(categories)]
        target_dir = os.path.join(BASE_DATA_DIR, label)
        
        count = 0
        for _, row in tqdm(filtered_df.iterrows(), total=len(filtered_df), desc=f"Processing class: {label}"):
            src_file = os.path.join(audio_src_dir, row['filename'])
            dst_file = os.path.join(target_dir, row['filename'])
            shutil.copy2(src_file, dst_file)
            count += 1
        stats[label] = count

    return stats

def cleanup():
    """
    Removes temporary artifacts and extraction cache to free up space.
    """
    print("Executing cleanup protocol...")
    if os.path.exists(EXTRACT_DIR):
        shutil.rmtree(EXTRACT_DIR)
    if os.path.exists(ZIP_FILE):
        os.remove(ZIP_FILE)
    print("Cleanup completed.")

if __name__ == "__main__":
    print("=== EcoEar Data Pipeline: Initialization ===")
    
    try:
        clean_broken_downloads() 
        create_dir_structure()
        download_file(DATASET_URL, ZIP_FILE)
        stats = process_data()
        cleanup()
        
        print("\nPipeline Execution Successful: Data Loaded.")
        print("--- Final Dataset Statistics ---")
        total_files = sum(stats.values())
        for label, count in stats.items():
            print(f"  Category [{label.upper()}]: {count} audio files")
        print(f"  Total Samples: {total_files} files ready for training.")
            
    except Exception as e:
        print(f"\nPipeline Execution Failed: {e}")