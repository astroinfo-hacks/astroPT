"""Check what's actually in the saved embedding files."""

import numpy as np
from collections import Counter

print("Loading saved embeddings and TARGETIDs...")

# Load the files
embeddings = np.load("./scripts/desi/visualisations/embeddings/desi_embeddings.npy")
target_ids = np.load("./scripts/desi/visualisations/embeddings/desi_targetids.npy")
redshifts = np.load("./scripts/desi/visualisations/embeddings/desi_redshifts.npy")

print(f"\nEmbeddings shape: {embeddings.shape}")
print(f"TARGETIDs shape: {target_ids.shape}")
print(f"TARGETIDs dtype: {target_ids.dtype}")
print(f"Redshifts shape: {redshifts.shape}")

print(f"\nFirst 10 TARGETIDs:")
print(target_ids[:10])

print(f"\nLast 10 TARGETIDs:")
print(target_ids[-10:])

print(f"\nTotal TARGETIDs: {len(target_ids)}")
print(f"Unique TARGETIDs: {len(np.unique(target_ids))}")

# Check for duplicates
unique, counts = np.unique(target_ids, return_counts=True)
duplicates = unique[counts > 1]

if len(duplicates) > 0:
    print(f"\n⚠️  Found {len(duplicates)} TARGETIDs with duplicates!")
    print(f"\nTop 10 most duplicated TARGETIDs:")
    dup_counts = counts[counts > 1]
    top_indices = np.argsort(dup_counts)[-10:][::-1]
    for idx in top_indices:
        tid = duplicates[idx]
        count = dup_counts[idx]
        print(f"  {tid}: appears {count} times")
    
    # Check if all duplicates have the same count
    if len(set(dup_counts)) == 1:
        print(f"\n✓ All duplicates appear exactly {dup_counts[0]} times")
        print(f"  This suggests systematic duplication (e.g., {dup_counts[0]} workers/shards)")
else:
    print("\n✓ No duplicates found")

# Check a specific TARGETID from the catalog
catalog_tid = 39627322701120643
if catalog_tid in target_ids:
    print(f"\n✓ Catalog TARGETID {catalog_tid} found in embeddings")
    indices = np.where(target_ids == catalog_tid)[0]
    print(f"  Appears at indices: {indices}")
else:
    print(f"\n✗ Catalog TARGETID {catalog_tid} NOT found in embeddings")
    print(f"  Min TARGETID: {target_ids.min()}")
    print(f"  Max TARGETID: {target_ids.max()}")
