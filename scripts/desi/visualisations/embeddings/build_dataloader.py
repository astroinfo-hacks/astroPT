"""Build a DataLoader for DESI spectra to extract embeddings.

This script creates a PyTorch DataLoader that yields batches of DESI spectra
from the saved HuggingFace dataset, using the same preprocessing as during training.
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.euclid_desi_dataset.desi_spectrum_dataloader import (
    DESISpectraDataset,
    spectra_collate,
)

DEFAULT_DATA_DIR = "/pbs/home/a/astroinfo08/astroinfo2025/data/astroPT_desi_dataset"


def main():
    parser = argparse.ArgumentParser(
        description="Build DataLoader for DESI spectra embeddings"
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Root directory containing the DESI dataset",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split to load (e.g., 'train', 'validation', 'test')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for the DataLoader",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers",
    )
    args = parser.parse_args()

    print(f"Loading DESI dataset from: {args.data_dir}")
    if args.split:
        print(f"Split: {args.split}")

    # Create dataset
    dataset = DESISpectraDataset(data_dir=args.data_dir, split=args.split)
    print(f"Dataset size: {len(dataset)} spectra")

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=spectra_collate,
        pin_memory=False,
    )

    # Test loading a batch
    batch = next(iter(dataloader))
    print("\nSample batch:")
    print("  Keys:", list(batch.keys()))
    print("  Flux shape:", batch["flux"].shape)
    
    if isinstance(batch["wavelength"][0], torch.Tensor):
        print("  Wavelength shape:", batch["wavelength"][0].shape)
    if isinstance(batch["ivar"][0], torch.Tensor):
        print("  Inverse variance shape:", batch["ivar"][0].shape)
    if isinstance(batch["mask"][0], torch.Tensor):
        print("  Mask shape:", batch["mask"][0].shape)
    
    print("  Target IDs (first 3):", batch["targetid"][:3])
    print("  Redshifts (first 3):", batch["redshift"][:3])
    
    print("\nDataLoader ready!")


if __name__ == "__main__":
    main()
