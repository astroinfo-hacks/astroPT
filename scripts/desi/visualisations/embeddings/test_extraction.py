"""Test extraction to understand the duplication issue."""

import sys
sys.path.insert(0, './scripts')

import numpy as np
import torch
from torch.utils.data import DataLoader
from euclid_desi_dataset.desi_spectrum_dataloader import (
    DESISpectraDataset,
    spectra_collate,
)

# Load dataset
data_dir = "/pbs/throng/training/astroinfo2025/data/astroPT_desi_dataset"
print(f"Loading dataset from: {data_dir}")
dataset = DESISpectraDataset(data_dir=data_dir)
print(f"Dataset size: {len(dataset)}")

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    collate_fn=spectra_collate,
    pin_memory=False,
)

print(f"\nDataLoader will yield ~{len(dataset) // 32} batches")

# Extract TARGETIDs from first 5 batches
all_target_ids = []
for batch_idx, batch in enumerate(dataloader):
    if batch_idx >= 5:
        break
    
    target_ids = batch["targetid"]
    print(f"\nBatch {batch_idx}:")
    print(f"  Type: {type(target_ids)}")
    print(f"  Length: {len(target_ids)}")
    print(f"  First 3: {target_ids[:3]}")
    
    all_target_ids.extend(target_ids)

print(f"\n\n=== SUMMARY ===")
print(f"Total TARGETIDs collected: {len(all_target_ids)}")
print(f"Unique TARGETIDs: {len(set(all_target_ids))}")
print(f"First 10: {all_target_ids[:10]}")
print(f"Last 10: {all_target_ids[-10:]}")

# Check for duplicates
from collections import Counter
counts = Counter(all_target_ids)
duplicates = {tid: count for tid, count in counts.items() if count > 1}
if duplicates:
    print(f"\n⚠️  Found {len(duplicates)} duplicate TARGETIDs!")
    print(f"Examples: {list(duplicates.items())[:5]}")
else:
    print("\n✓ No duplicates found")
