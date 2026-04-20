#!/usr/bin/env python3
"""
Download and extract UCI Household Power Consumption dataset
"""

import os
import urllib.request
import zipfile
from pathlib import Path

# Configuration
UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
DATA_DIR = Path(__file__).parent.parent  # Go up to data/ directory
RAW_DIR = DATA_DIR / "raw"
ZIP_FILE = RAW_DIR / "household_power_consumption.zip"
EXTRACTED_FILE = RAW_DIR / "household_power_consumption.txt"

def download_dataset():
    """Download the dataset from UCI repository"""
    
    print("=" * 60)
    print("UCI Household Power Consumption Dataset Download")
    print("=" * 60)
    
    # Create directories if they don't exist
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    if EXTRACTED_FILE.exists():
        print(f"✓ Dataset already exists at: {EXTRACTED_FILE}")
        print(f"  File size: {EXTRACTED_FILE.stat().st_size / (1024**2):.2f} MB")
        response = input("\nDo you want to re-download? (y/n): ")
        if response.lower() != 'y':
            print("Skipping download.")
            return
        else:
            print("Re-downloading...")
    
    # Download
    print(f"\n📥 Downloading from: {UCI_URL}")
    print(f"   Saving to: {ZIP_FILE}")
    
    try:
        # Download with progress
        def report_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded / total_size * 100, 100)
            print(f"\r   Progress: {percent:.1f}% ({downloaded/(1024**2):.2f} MB / {total_size/(1024**2):.2f} MB)", 
                  end='', flush=True)
        
        urllib.request.urlretrieve(UCI_URL, ZIP_FILE, reporthook=report_progress)
        print("\n✓ Download complete!")
        
    except Exception as e:
        print(f"\n✗ Error downloading: {e}")
        print("\nAlternative: Download manually from:")
        print("https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption")
        return
    
    # Extract
    print(f"\n📦 Extracting ZIP file...")
    try:
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR)
        print(f"✓ Extracted to: {RAW_DIR}")
        
        # Verify extracted file
        if EXTRACTED_FILE.exists():
            file_size = EXTRACTED_FILE.stat().st_size / (1024**2)
            print(f"✓ Dataset file: {EXTRACTED_FILE.name}")
            print(f"  Size: {file_size:.2f} MB")
            
            # Count lines
            print(f"\n📊 Counting records...")
            with open(EXTRACTED_FILE, 'r') as f:
                num_lines = sum(1 for _ in f) - 1  # -1 for header
            print(f"✓ Total records: {num_lines:,}")
            
        else:
            print("✗ Error: Extracted file not found!")
            
    except Exception as e:
        print(f"✗ Error extracting: {e}")
        return
    
    # Optional: Delete ZIP file to save space
    response = input(f"\n🗑️  Delete ZIP file to save space? (y/n): ")
    if response.lower() == 'y':
        ZIP_FILE.unlink()
        print("✓ ZIP file deleted")
    
    print("\n" + "=" * 60)
    print("✓ DOWNLOAD COMPLETE!")
    print("=" * 60)
    print(f"\nDataset location: {EXTRACTED_FILE}")
    print("\nNext steps:")
    print("1. Run: python data/scripts/explore_data.py")
    print("2. Run: python data/scripts/preprocess.py")
    print("=" * 60)

if __name__ == "__main__":
    download_dataset()
