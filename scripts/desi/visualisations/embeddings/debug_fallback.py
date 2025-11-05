"""
Diagnostic script to test the fallback mechanism in DESISpectraDataset.

This script will request the same indices multiple times and check if
the fallback mechanism causes different indices to return the same TARGETID.
"""

import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from scripts.euclid_desi_dataset.desi_spectrum_dataloader import DESISpectraDataset
from collections import Counter


def test_fallback_issue(data_dir: str, num_samples: int = 1000):
    """Test if multiple indices return the same TARGETID due to fallback."""
    
    print(f"Loading dataset from {data_dir}")
    dataset = DESISpectraDataset(data_dir=data_dir)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"\nTesting {num_samples} random indices...\n")
    
    # Test: request specific indices and track what TARGETID we get
    index_to_targetid = {}
    targetid_to_indices = {}
    
    import random
    random.seed(42)
    test_indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    
    for idx in test_indices:
        try:
            sample = dataset[idx]
            targetid = sample["targetid"]
            
            index_to_targetid[idx] = targetid
            
            if targetid not in targetid_to_indices:
                targetid_to_indices[targetid] = []
            targetid_to_indices[targetid].append(idx)
            
        except Exception as e:
            print(f"Error at index {idx}: {e}")
            continue
    
    # Find TARGETIDs that are returned for multiple indices
    duplicated_targetids = {
        tid: indices 
        for tid, indices in targetid_to_indices.items() 
        if len(indices) > 1
    }
    
    print(f"Results:")
    print(f"  Requested indices: {len(test_indices)}")
    print(f"  Unique TARGETIDs returned: {len(targetid_to_indices)}")
    print(f"  TARGETIDs returned multiple times: {len(duplicated_targetids)}")
    
    if duplicated_targetids:
        print(f"\n⚠️  FALLBACK ISSUE CONFIRMED!")
        print(f"\nExamples of TARGETIDs returned for multiple indices:")
        for tid, indices in list(duplicated_targetids.items())[:5]:
            print(f"  TARGETID {tid}: returned for indices {indices}")
        
        # Calculate duplication ratio
        total_mappings = sum(len(indices) for indices in targetid_to_indices.values())
        unique_targetids = len(targetid_to_indices)
        duplication_ratio = total_mappings / unique_targetids if unique_targetids > 0 else 0
        
        print(f"\nDuplication statistics:")
        print(f"  Average times each TARGETID is returned: {duplication_ratio:.2f}")
        print(f"  This matches the embedding issue: {len(test_indices)} requests -> {unique_targetids} unique TARGETIDs")
        
        return False
    else:
        print(f"\n✓ No duplication detected in this sample")
        return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="/pbs/throng/training/astroinfo2025/data/astroPT_desi_dataset",
        help="Dataset directory"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of random indices to test"
    )
    
    args = parser.parse_args()
    test_fallback_issue(args.data_dir, args.num_samples)
