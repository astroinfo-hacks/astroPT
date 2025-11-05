"""Quick diagnostic to check TARGETID distribution in saved embeddings."""

import numpy as np
from collections import Counter
import sys

def analyze_targetids(targetid_file):
    """Analyze the TARGETID distribution in saved embeddings."""
    
    print(f"Loading TARGETIDs from: {targetid_file}")
    target_ids = np.load(targetid_file)
    
    print(f"\nBasic statistics:")
    print(f"  Total entries: {len(target_ids):,}")
    print(f"  Data type: {target_ids.dtype}")
    print(f"  First 10 TARGETIDs: {target_ids[:10]}")
    print(f"  Last 10 TARGETIDs: {target_ids[-10:]}")
    
    # Check for duplicates
    unique_ids = np.unique(target_ids)
    print(f"\nDuplication analysis:")
    print(f"  Unique TARGETIDs: {len(unique_ids):,}")
    print(f"  Duplication ratio: {len(target_ids) / len(unique_ids):.2f}x")
    
    if len(unique_ids) < len(target_ids):
        # Find most duplicated IDs
        counter = Counter(target_ids)
        most_common = counter.most_common(10)
        
        print(f"\n⚠️  DUPLICATION DETECTED!")
        print(f"\nTop 10 most duplicated TARGETIDs:")
        for tid, count in most_common:
            print(f"  TARGETID {tid}: appears {count:,} times")
        
        # Check if it's the same ID repeated
        if len(unique_ids) == 1:
            print(f"\n⚠️  CRITICAL: All entries are the SAME TARGETID!")
            print(f"  TARGETID: {unique_ids[0]}")
        
        return False
    else:
        print(f"\n✓ No duplicates - all TARGETIDs are unique")
        return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_embedding_targetids.py <targetids.npy>")
        print("\nExample:")
        print("  python check_embedding_targetids.py desi_targetids.npy")
        sys.exit(1)
    
    targetid_file = sys.argv[1]
    analyze_targetids(targetid_file)
