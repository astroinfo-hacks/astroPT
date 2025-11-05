"""Clean re-extraction with diagnostics to track duplication issues."""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter

import sys
sys.path.insert(0, './scripts')

from euclid_desi_dataset.desi_spectrum_dataloader import (
    DESISpectraDataset,
    spectra_collate,
)


DEFAULT_DATA_DIR = "/pbs/throng/training/astroinfo2025/data/astroPT_desi_dataset"
DEFAULT_CHECKPOINT = "/pbs/throng/training/astroinfo2025/work/jzoubian/logs/astropt_desi_spectra_2/ckpt_best_L12_H12_E768_BS256_20251105-124945.pt"
#DEFAULT_CHECKPOINT = "/pbs/throng/training/astroinfo2025/work/jzoubian/logs/astropt_desi_spectra_2/ckpt_L12_H12_E768_BS256_20251105-124945_iter230000.pt"


def load_checkpoint(checkpoint_path: str, device: str, patch_size: int = 256):
    """Load model checkpoint from disk."""
    from astropt.model import GPT, GPTConfig, ModalityConfig, ModalityRegistry
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model_args = checkpoint.get('model_args', checkpoint.get('config', {}))
    
    if isinstance(model_args, GPTConfig):
        config = model_args
    elif isinstance(model_args, dict):
        import inspect
        valid_params = set(inspect.signature(GPTConfig.__init__).parameters.keys())
        valid_params.discard('self')
        filtered_args = {k: v for k, v in model_args.items() if k in valid_params}
        config = GPTConfig(**filtered_args)
    else:
        config = model_args
    
    modalities = [
        ModalityConfig(
            name="spectra",
            input_size=patch_size,
            patch_size=patch_size,
            pos_input_size=1,
            loss_weight=1.0,
            embed_pos=True,
        ),
    ]
    modality_registry = ModalityRegistry(modalities)
    
    model = GPT(config, modality_registry)
    state_dict = checkpoint.get('model', checkpoint)
    
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print(f"Model loaded: block_size={config.block_size}, n_embd={config.n_embd}")
    return model, config


def find_final_norm(model):
    """Find the final normalization layer."""
    candidate = None
    for name, module in model.named_modules():
        if 'ln_f' in name.lower() or 'final' in name.lower():
            candidate = module
    return candidate


def prepare_spectra_batch(batch, patch_size: int, block_size: int, device: str):
    """Prepare a batch of spectra for model input.
    
    Args:
        batch: Batch dictionary from dataloader
        patch_size: Size of each patch
        block_size: Maximum sequence length (number of patches)
        device: Device to place tensors on
    
    Returns:
        Tuple of (inputs_dict, target_ids, redshifts)
    """
    flux = batch["flux"].to(device)
    B, L = flux.shape
    
    # Pad to be divisible by patch_size
    pad = (patch_size - (L % patch_size)) % patch_size
    if pad:
        flux = F.pad(flux, (0, pad))
    
    # Reshape into patches
    patches = flux.view(B, -1, patch_size)
    num_patches = patches.size(1)
    
    # CRITICAL: Truncate to block_size to match training
    if block_size > 0 and num_patches > block_size:
        num_patches = block_size
        patches = patches[:, :num_patches]
    
    # Create position indices
    positions = torch.arange(num_patches, device=device, dtype=torch.long)
    positions = positions.unsqueeze(0).expand(B, -1)
    
    inputs = {
        "spectra": patches,
        "spectra_positions": positions,
    }
    
    target_ids = batch["targetid"]
    redshifts = batch["redshift"]
    
    return inputs, target_ids, redshifts


def extract_embeddings_with_diagnostics(
    model,
    dataloader,
    patch_size: int,
    block_size: int,
    device: str,
    max_batches: int = None,
):
    """Extract embeddings with duplication checks.
    
    Args:
        model: Trained model
        dataloader: DataLoader for spectra
        patch_size: Size of each patch
        block_size: Maximum sequence length
        device: Device to run on
        max_batches: Optional limit on number of batches
    """
    ln_final = find_final_norm(model)
    if ln_final is None:
        raise ValueError("Could not find final normalization layer in model")
    
    print(f"Hooking into layer: {ln_final}")
    
    captured_hidden = {}
    
    def hook_fn(module, input, output):
        captured_hidden['last'] = output.detach()
    
    handle = ln_final.register_forward_hook(hook_fn)
    
    all_embeddings = []
    all_target_ids = []
    all_redshifts = []
    
    # Track TARGETIDs for duplication detection
    seen_targetids = Counter()
    
    try:
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Extracting embeddings")
            for batch_idx, batch in enumerate(pbar):
                if max_batches and batch_idx >= max_batches:
                    break
                
                inputs, target_ids, redshifts = prepare_spectra_batch(
                    batch, patch_size, block_size, device
                )
                
                # Check for duplicates in this batch
                for tid in target_ids:
                    seen_targetids[tid] += 1
                
                _ = model(inputs)
                
                hidden = captured_hidden['last']
                embeddings = hidden.mean(dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
                all_target_ids.extend(target_ids)
                all_redshifts.extend(redshifts)
                
                # Report duplication status every 1000 batches
                if (batch_idx + 1) % 1000 == 0:
                    unique = len(seen_targetids)
                    total = sum(seen_targetids.values())
                    pbar.set_postfix({
                        'batch': batch_idx + 1,
                        'unique': unique,
                        'total': total,
                        'dup_rate': f'{(1 - unique/total)*100:.1f}%'
                    })
    
    finally:
        handle.remove()
    
    embeddings_array = np.concatenate(all_embeddings, axis=0)
    
    # Final diagnostics
    print(f"\n=== EXTRACTION DIAGNOSTICS ===")
    print(f"Total spectra processed: {len(all_target_ids)}")
    print(f"Unique TARGETIDs: {len(seen_targetids)}")
    print(f"Embedding shape: {embeddings_array.shape}")
    
    # Check for duplicates
    duplicates = {tid: count for tid, count in seen_targetids.items() if count > 1}
    if duplicates:
        print(f"\n⚠️  WARNING: Found {len(duplicates)} duplicate TARGETIDs!")
        top_dups = sorted(duplicates.items(), key=lambda x: x[1], reverse=True)[:5]
        print("Top 5 duplicates:")
        for tid, count in top_dups:
            print(f"  {tid}: {count} times")
        print("\n❌ DUPLICATION DETECTED - DO NOT USE THESE EMBEDDINGS")
        print("Something is wrong with the dataset or DataLoader")
        return None, None, None
    else:
        print("\n✓ No duplicates found - embeddings are valid!")
    
    return embeddings_array, all_target_ids, all_redshifts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)  # Changed default to 0!
    parser.add_argument("--patch-size", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--out-dir", default="./scripts/desi/visualisations/embeddings/")
    args = parser.parse_args()

    print("="*60)
    print("CLEAN EMBEDDING EXTRACTION WITH DIAGNOSTICS")
    print("="*60)
    
    # Load model
    model, config = load_checkpoint(args.checkpoint, args.device, args.patch_size)
    
    # Create dataset
    print(f"\nLoading dataset from: {args.data_dir}")
    dataset = DESISpectraDataset(data_dir=args.data_dir)
    print(f"Dataset size: {len(dataset)} spectra")
    
    # IMPORTANT: Start with num_workers=0 to avoid multiprocessing issues
    print(f"\nCreating DataLoader (num_workers={args.num_workers})...")
    if args.num_workers > 0:
        print("⚠️  WARNING: Using num_workers > 0 may cause duplication issues")
        print("   If you see duplicates, try running with --num-workers=0")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=spectra_collate,
        pin_memory=(args.device == "cuda"),
    )
    
    # Extract embeddings with diagnostics
    print("\nExtracting embeddings...")
    print(f"Using block_size={config.block_size} (from checkpoint)")
    embeddings, target_ids, redshifts = extract_embeddings_with_diagnostics(
        model,
        dataloader,
        args.patch_size,
        config.block_size,
        args.device,
        args.max_batches,
    )
    
    # Only save if extraction was successful
    if embeddings is not None:
        import os
        os.makedirs(args.out_dir, exist_ok=True)
        
        out_embeddings = os.path.join(args.out_dir, "desi_embeddings_230k.npy")
        out_targetids = os.path.join(args.out_dir, "desi_targetids_230k.npy")
        out_redshifts = os.path.join(args.out_dir, "desi_redshifts_230k.npy")
        
        print(f"\nSaving embeddings to: {out_embeddings}")
        np.save(out_embeddings, embeddings)
        
        print(f"Saving target IDs to: {out_targetids}")
        np.save(out_targetids, np.array(target_ids, dtype=np.int64))
        
        print(f"Saving redshifts to: {out_redshifts}")
        np.save(out_redshifts, np.array(redshifts, dtype=np.float32))
        
        print("\n✓ Extraction complete - embeddings are valid!")
    else:
        print("\n❌ Extraction FAILED due to duplicates - files NOT saved")
        print("Please investigate the dataset or try with --num-workers=0")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
