# Dataloader for Euclid dataset
# 1. Loads the small Euclid dataset (msiudek/astroPT_euclid_dataset) with HF Datasets.
# 2. Transforms each example into a 4-channel image tensor [VIS, Y, J, H] of shape (4, 224, 224) with a robust per-image normalization.
# 3. Builds a PyTorch DataLoader that yields batches (B, 4, 224, 224) plus a list of object_ids.

Prints a quick sanity check (shapes, dtypes, a few IDs).
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

DATASET_ID = "msiudek/astroPT_euclid_dataset"
DTYPE = torch.float32  # can switch to torch.bfloat16 later

def _to_np(img_like):
    # HF Image feature: PIL.Image or already np-like
    if hasattr(img_like, "convert"):  # PIL
        return np.array(img_like, dtype=np.float32)
    arr = np.asarray(img_like)
    return arr.astype(np.float32) if arr.dtype != np.float32 else arr

def robust_standardize(x, eps=1e-6):
    med = np.median(x)
    q1  = np.quantile(x, 0.25)
    q3  = np.quantile(x, 0.75)
    iqr = max(q3 - q1, eps)
    return (x - med) / iqr

def stack_channels(example):
    # Order: [VIS, Y, J, H]
    vis = _to_np(example["VIS_image"])
    y   = _to_np(example["NISP_Y_image"])
    j   = _to_np(example["NISP_J_image"])
    h   = _to_np(example["NISP_H_image"])

    def prep(ch):
        ch = np.nan_to_num(ch, nan=0.0, posinf=0.0, neginf=0.0)
        return robust_standardize(ch)

    img4 = np.stack([prep(vis), prep(y), prep(j), prep(h)], axis=0)  # (4, 224, 224)
    return {"pixel_values": img4.astype(np.float32),
            "object_id": example.get("object_id")}

def collate(batch):
    xs, ids = [], []
    for b in batch:
        x = b["pixel_values"]
        if isinstance(x, list):
            x = np.array(x, dtype=np.float32)
        elif not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=np.float32)
        xs.append(torch.as_tensor(x, dtype=DTYPE))
        ids.append(b["object_id"])
    return torch.stack(xs, dim=0), ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train","test"], default="test")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=0)  # start with 0 for clarity
    args = ap.parse_args()

    print(f"Loading dataset: {DATASET_ID} split={args.split}")
    ds = load_dataset(DATASET_ID, split=args.split)

    keep_cols = {"object_id"}
    remove_cols = [c for c in ds.column_names if c not in keep_cols]
    ds_proc = ds.map(stack_channels, remove_columns=remove_cols, batched=False)

    print("Building DataLoader…")
    loader = DataLoader(
        ds_proc,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=False,
    )

    xb, ids = next(iter(loader))
    print("Sample batch:")
    print("  tensor shape:", tuple(xb.shape))  # (B, 4, 224, 224)
    print("  dtype       :", xb.dtype)
    print("  first ids   :", ids[:5])
    print("OK")

if __name__ == "__main__":
    main()

