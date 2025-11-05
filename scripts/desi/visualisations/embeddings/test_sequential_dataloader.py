"""Test if sequential DataLoader iteration shows duplication.

This mimics what happens during embedding extraction: sequential batches
through the entire dataset with shuffle=False.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from scripts.euclid_desi_dataset.desi_spectrum_dataloader import (
    DESISpectraDataset,
    spectra_collate,
)
from torch.utils.data import DataLoader
from collections import Counter


def test_sequential_dataloader(data_dir: str, num_batches: int = 100, batch_size: int = 32):
    """Test sequential DataLoader iteration for duplicates."""
    
    print(f"Loading dataset from {data_dir}")
    dataset = DESISpectraDataset(data_dir=data_dir)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"\nTesting {num_batches} sequential batches (batch_size={batch_size})...")
    print(f"Expected unique TARGETIDs: {min(num_batches * batch_size, len(dataset))}\n")
    
    # Create DataLoader with shuffle=False (same as extraction)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Sequential access
        num_workers=0,
        collate_fn=spectra_collate,
    )
    
    all_targetids = []
    batch_count = 0
    
    for batch in dataloader:
        target_ids = batch["targetid"]
        all_targetids.extend(target_ids)
        
        batch_count += 1
        if batch_count >= num_batches:
            break
        
        if batch_count % 10 == 0:
            print(f"  Processed {batch_count} batches, {len(all_targetids)} spectra...")
    
    # Analyze results
    print(f"\nResults:")
    print(f"  Total spectra processed: {len(all_targetids)}")
    
    counter = Counter(all_targetids)
    unique_count = len(counter)
    
    print(f"  Unique TARGETIDs: {unique_count}")
    
    duplicates = {tid: count for tid, count in counter.items() if count > 1}
    
    if duplicates:
        print(f"  TARGETIDs appearing multiple times: {len(duplicates)}")
        print(f"\n⚠️  DUPLICATION DETECTED IN SEQUENTIAL ACCESS!")
        print(f"\nTop 10 most duplicated TARGETIDs:")
        for tid, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"    TARGETID {tid}: {count} times")
        
        # Check duplication pattern
        total_expected = min(num_batches * batch_size, len(dataset))
        duplication_ratio = len(all_targetids) / unique_count if unique_count > 0 else 0
        print(f"\n  Expected TARGETIDs: {total_expected}")
        print(f"  Duplication ratio: {duplication_ratio:.2f}x")
        
        return False
    else:
        print(f"\n✓ No duplication detected in sequential access")
        
        # Check for sequential pattern
        first_10 = all_targetids[:10]
        print(f"\nFirst 10 TARGETIDs: {first_10}")
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
        "--num-batches",
        type=int,
        default=100,
        help="Number of batches to test"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    
    args = parser.parse_args()
    test_sequential_dataloader(args.data_dir, args.num_batches, args.batch_size)
