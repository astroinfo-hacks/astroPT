# extract_embeddings.py
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

from astropt.model_utils import load_astropt
import torch.nn.functional as F


# -------------------- Config defaults --------------------
MODEL_REPO   = "msiudek/astroPT_euclid_VIS_NISP_model"
MODEL_DIR    = "astropt/090M"
WEIGHTS_FILE = "ckpt.pt"

DATASET_ID   = "msiudek/astroPT_euclid_dataset"   # small dataset (train/test)
DTYPE        = torch.float32                      # CPU-friendly; switch to bfloat16 if you like
BATCH_SIZE   = 16
NUM_WORKERS  = 0                                  # simpler debugging on CPU
IQR_MIN      = 1e-3                               # clamp tiny IQRs to avoid overflow
# ---------------------------------------------------------

def _to_np(img_like):
    if hasattr(img_like, "convert"):
        return np.array(img_like, dtype=np.float32)
    arr = np.asarray(img_like)
    return arr.astype(np.float32) if arr.dtype != np.float32 else arr

def robust_standardize(x, iqr_min=IQR_MIN):
    med = np.median(x)
    q1  = np.quantile(x, 0.25)
    q3  = np.quantile(x, 0.75)
    iqr = q3 - q1
    if iqr < iqr_min:
        iqr = iqr_min
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


def infer_model_dims(model):
    """Try to read patch_size and block_size (= #tokens) from the model; fallback to Euclid defaults."""
    ps = getattr(model, "patch_size", None)
    if ps is None and hasattr(model, "config"):
        ps = getattr(model.config, "patch_size", None)
    if ps is None:
        ps = 16  # from printed args

    T = getattr(model, "block_size", None)
    if T is None and hasattr(model, "config"):
        T = getattr(model.config, "block_size", None)
    if T is None:
        T = 196  # (224/16)**2

    return int(ps), int(T)

def patchify_images(x, patch_size: int):
    """
    x: (B, C, H, W) float tensor
    returns tokens: (B, L, C*patch_size*patch_size) where L = (H/ps)*(W/ps)
    """
    # unfold makes (B, C*ps*ps, L) with stride=ps, kernel=ps
    tokens = F.unfold(x, kernel_size=patch_size, stride=patch_size)  # (B, C*ps*ps, L)
    tokens = tokens.transpose(1, 2).contiguous()                     # (B, L, C*ps*ps)
    return tokens

def find_final_norm(module):
    """
    Try to find the last normalization layer. Typical names for GPT-like models:
    'transformer.ln_f' or something ending with 'ln_f'/'final_ln'/'norm'.
    """
    candidate = None
    for name, mod in module.named_modules():
        if name.endswith(("ln_f", "final_ln", "norm")):
            candidate = mod
    return candidate
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train","test"], default="test")
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    ap.add_argument("--out_z", type=str, default="zss.npy")
    ap.add_argument("--out_idx", type=str, default="idxs.npy")
    args = ap.parse_args()

    # 1) Load model
    print("Loading model…")
    model = load_astropt(repo_id=MODEL_REPO, path=MODEL_DIR, weights_filename=WEIGHTS_FILE)
    device = "cpu"
    model.to(device).eval()

    # infer patch + token dims (for sanity)
    patch_size, block_size = infer_model_dims(model)
    print(f"patch_size={patch_size}, expected_tokens={block_size}")

    # 2) Hook final norm
    ln_final = find_final_norm(model)
    assert ln_final is not None, "Could not find a final normalization layer to hook."
    last_hidden = {}
    def hook(_, __, out):
        last_hidden["h"] = out
    handle = ln_final.register_forward_hook(hook)

    # 3) Dataset + loader (same as before)
    print(f"Loading dataset: {DATASET_ID} split={args.split}")
    ds = load_dataset(DATASET_ID, split=args.split)
    keep_cols = {"object_id"}
    remove_cols = [c for c in ds.column_names if c not in keep_cols]
    ds_proc = ds.map(stack_channels, remove_columns=remove_cols, batched=False)

    loader = DataLoader(
        ds_proc,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=False,
    )

    # 4) Iterate and embed
    all_ids, zs_list = [], []
    with torch.no_grad():
        for xb, ids in tqdm(loader, desc="Embedding"):
            xb = xb.to(device)                           # (B, 4, 224, 224)
            tokens = patchify_images(xb, patch_size)     # (B, L=196, C=4*16*16=1024)
            # Optional sanity: assert tokens.shape[1] == block_size
            # if tokens.shape[1] != block_size: print("Warning: token length mismatch", tokens.shape)

            _ = model(tokens)                            # forward on tokens; hook captures final token states
            h = last_hidden.pop("h")                     # (B, T, C_emb=768)
            z = h.mean(dim=1).cpu().to(torch.float32).numpy()  # (B, 768)
            zs_list.append(z)
            all_ids.extend(ids)

    zss = np.concatenate(zs_list, axis=0)
    idxs = np.array(all_ids)
    np.save(args.out_z, zss)
    np.save(args.out_idx, idxs)

    handle.remove()
    print(f"Saved embeddings to {args.out_z} with shape {zss.shape}")
    print(f"Saved ids to {args.out_idx} with shape {idxs.shape}")
    print("Done")
if __name__ == "__main__":
    main()

