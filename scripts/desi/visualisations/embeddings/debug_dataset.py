"""Debug script to check for duplicate TARGETIDs in the DESI dataset."""

import sys
sys.path.insert(0, '/home/zoubian/Workspace/AstroInfo/astroPT')

from scripts.euclid_desi_dataset.desi_spectrum_dataloader import DESISpectraDataset
import numpy as np
from collections import Counter

# Load dataset
print("Loading dataset...")
dataset = DESISpectraDataset(
    data_dir="/pbs/throng/training/astroinfo2025/data/astroPT_desi_dataset"
)

print(f"Dataset size: {len(dataset)}")

# Extract first 1000 TARGETIDs
print("\nExtracting sample TARGETIDs...")
sample_size = min(1000, len(dataset))
target_ids = []

for i in range(sample_size):
    sample = dataset.dataset[i]  # Access underlying HF dataset
    target_ids.append(sample['targetid'])
    if i < 10:
        print(f"  [{i}] TARGETID: {sample['targetid']}")

print(f"\nFirst 10 TARGETIDs: {target_ids[:10]}")
print(f"Last 10 TARGETIDs: {target_ids[-10:]}")

# Check for duplicates
target_ids_array = np.array(target_ids)
unique_ids = np.unique(target_ids_array)
print(f"\nTotal TARGETIDs sampled: {len(target_ids_array)}")
print(f"Unique TARGETIDs: {len(unique_ids)}")

# Count duplicates
counts = Counter(target_ids)
duplicates = {tid: count for tid, count in counts.items() if count > 1}

if duplicates:
    print(f"\nFound {len(duplicates)} duplicate TARGETIDs in sample:")
    for tid, count in list(duplicates.items())[:10]:
        print(f"  TARGETID {tid} appears {count} times")
else:
    print("\n✓ No duplicates found in sample")

# Check if TARGETIDs match catalog
print("\nChecking first TARGETID from catalog...")
from astropy.io import fits
catalog_path = "/pbs/throng/training/astroinfo2025/data/astroPT_desi_dataset/DESI_dataset_catalog.fits"
with fits.open(catalog_path) as hdul:
    catalog_targetids = hdul[1].data['TARGETID']
    print(f"First catalog TARGETID: {catalog_targetids[0]}")
    print(f"First dataset TARGETID: {target_ids[0]}")
    
    # Check if any dataset IDs are in catalog
    matches = np.isin(target_ids_array, catalog_targetids)
    print(f"\nTARGETIDs matching catalog: {np.sum(matches)}/{len(target_ids_array)}")
