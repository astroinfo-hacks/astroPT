"""Extract embeddings from DESI spectra using a trained AstroPT model.

This script loads a trained checkpoint, processes DESI spectra through the model,
and extracts the final hidden states (embeddings) before the output layer.
The embeddings are saved to disk for downstream visualization and analysis.
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.euclid_desi_dataset.desi_spectrum_dataloader import (
    DESISpectraDataset,
    spectra_collate,
)


DEFAULT_DATA_DIR = "/pbs/throng/training/astroinfo2025/data/astroPT_desi_dataset/"
DEFAULT_CHECKPOINT = "/pbs/throng/training/astroinfo2025/work/jzoubian/logs/astropt_desi_spectra/ckpt.pt"


def load_checkpoint(checkpoint_path: str, device: str, patch_size: int = 256):
    """Load model checkpoint from disk.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on
        patch_size: Patch size used during training (needed for modality config)
    
    Returns:
        Tuple of (model, config)
    """
    from astropt.model import GPT, GPTConfig, ModalityConfig, ModalityRegistry
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model configuration
    model_args = checkpoint.get('model_args', checkpoint.get('config', {}))
    
    # If model_args is already a GPTConfig object, use it directly
    if isinstance(model_args, GPTConfig):
        config = model_args
    elif isinstance(model_args, dict):
        # Filter out non-model config keys (training-specific parameters)
        # Get valid GPTConfig parameters by inspecting the class
        import inspect
        valid_params = set(inspect.signature(GPTConfig.__init__).parameters.keys())
        valid_params.discard('self')
        
        # Filter model_args to only include valid GPTConfig parameters
        filtered_args = {k: v for k, v in model_args.items() if k in valid_params}
        config = GPTConfig(**filtered_args)
    else:
        config = model_args
    
    # Create modality registry for DESI spectra (same as in training script)
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
    
    # Initialize model with config and modality registry
    model = GPT(config, modality_registry)
    state_dict = checkpoint.get('model', checkpoint)
    
    # Remove module prefix if present (from DDP or torch.compile)
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
    """Find the final normalization layer in the transformer.
    
    Typical names: 'transformer.ln_f', 'ln_f', 'final_ln', etc.
    """
    candidate = None
    for name, module in model.named_modules():
        if 'ln_f' in name.lower() or 'final' in name.lower():
            candidate = module
    return candidate


def prepare_spectra_batch(batch, patch_size: int, block_size: int, device: str):
    """Prepare a batch of spectra for model input.
    
    Args:
        batch: Dictionary from spectra_collate with 'flux', 'targetid', etc.
        patch_size: Patch size for tokenization
        block_size: Maximum sequence length (number of patches)
        device: Device to move tensors to
    
    Returns:
        Tuple of (inputs_dict, target_ids, redshifts)
    """
    flux = batch["flux"].to(device)  # (B, L)
    B, L = flux.shape
    
    # Pad flux to be divisible by patch_size
    pad = (patch_size - (L % patch_size)) % patch_size
    if pad:
        flux = F.pad(flux, (0, pad))
    
    # Reshape into patches: (B, L) -> (B, num_patches, patch_size)
    patches = flux.view(B, -1, patch_size)
    num_patches = patches.size(1)
    
    # CRITICAL: Truncate to block_size to match training
    if block_size > 0 and num_patches > block_size:
        num_patches = block_size
        patches = patches[:, :num_patches]
    
    # Create position indices for each patch
    positions = torch.arange(num_patches, device=device, dtype=torch.long)
    positions = positions.unsqueeze(0).expand(B, -1)  # (B, num_patches)
    
    # Prepare inputs dict as expected by the model
    inputs = {
        "spectra": patches,  # (B, num_patches, patch_size)
        "spectra_positions": positions,  # (B, num_patches)
    }
    
    # Get metadata
    target_ids = batch["targetid"]
    redshifts = batch["redshift"]
    
    return inputs, target_ids, redshifts


def extract_embeddings(
    model,
    dataloader,
    patch_size: int,
    block_size: int,
    device: str,
    max_batches: int = None,
):
    """Extract embeddings from all spectra in the dataloader.
    
    Args:
        model: Trained AstroPT model
        dataloader: DataLoader yielding spectra batches
        patch_size: Patch size for tokenization
        block_size: Maximum sequence length (number of patches)
        device: Device for inference
        max_batches: Maximum number of batches to process (None = all)
    
    Returns:
        Tuple of (embeddings, target_ids, redshifts)
    """
    # Find and hook the final normalization layer
    ln_final = find_final_norm(model)
    if ln_final is None:
        raise ValueError("Could not find final normalization layer in model")
    
    print(f"Hooking into layer: {ln_final}")
    
    # Container to capture hidden states
    captured_hidden = {}
    
    def hook_fn(module, input, output):
        # Store the output of the final norm (the embeddings before lm_head)
        captured_hidden['last'] = output.detach()
    
    handle = ln_final.register_forward_hook(hook_fn)
    
    all_embeddings = []
    all_target_ids = []
    all_redshifts = []
    
    try:
        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Extracting embeddings")
            for batch_idx, batch in enumerate(pbar):
                if max_batches and batch_idx >= max_batches:
                    break
                
                # Prepare input
                inputs, target_ids, redshifts = prepare_spectra_batch(
                    batch, patch_size, block_size, device
                )
                
                # Forward pass (triggers hook)
                _ = model(inputs)
                
                # Extract embeddings from hook
                hidden = captured_hidden['last']  # (B, num_tokens, n_embd)
                
                # Pool across tokens (mean pooling)
                embeddings = hidden.mean(dim=1)  # (B, n_embd)
                
                # Store results
                all_embeddings.append(embeddings.cpu().numpy())
                all_target_ids.extend(target_ids)
                all_redshifts.extend(redshifts)
                
                pbar.set_postfix({'batch_size': len(target_ids)})
    
    finally:
        handle.remove()
    
    # Concatenate all embeddings
    embeddings_array = np.concatenate(all_embeddings, axis=0)
    
    print(f"\nExtracted {len(embeddings_array)} embeddings")
    print(f"Embedding shape: {embeddings_array.shape}")
    
    return embeddings_array, all_target_ids, all_redshifts


def main():
    parser = argparse.ArgumentParser(
        description="Extract embeddings from DESI spectra using trained AstroPT"
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Path to model checkpoint file",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Root directory containing DESI dataset",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split to process (e.g., 'train', 'validation', 'test')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Patch size for tokenization (should match training)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for inference",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Maximum number of batches to process (for testing)",
    )
    parser.add_argument(
        "--out-embeddings",
        default="./scripts/desi/visualisations/embeddings/desi_embeddings.npy",
        help="Output file for embeddings",
    )
    parser.add_argument(
        "--out-targetids",
        default="./scripts/desi/visualisations/embeddings/desi_targetids.npy",
        help="Output file for target IDs",
    )
    parser.add_argument(
        "--out-redshifts",
        default="./scripts/desi/visualisations/embeddings/desi_redshifts.npy",
        help="Output file for redshifts",
    )
    args = parser.parse_args()

    # Load model
    model, config = load_checkpoint(args.checkpoint, args.device, args.patch_size)
    
    # Create dataset and dataloader
    print(f"\nLoading dataset from: {args.data_dir}")
    dataset = DESISpectraDataset(data_dir=args.data_dir, split=args.split)
    print(f"Dataset size: {len(dataset)} spectra")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=spectra_collate,
        pin_memory=(args.device == "cuda"),
    )
    
    # Extract embeddings
    print("\nExtracting embeddings...")
    print(f"Using block_size={config.block_size} (from checkpoint)")
    embeddings, target_ids, redshifts = extract_embeddings(
        model,
        dataloader,
        args.patch_size,
        config.block_size,
        args.device,
        args.max_batches,
    )
    
    # Save to disk
    print(f"\nSaving embeddings to: {args.out_embeddings}")
    np.save(args.out_embeddings, embeddings)
    
    print(f"Saving target IDs to: {args.out_targetids}")
    # CRITICAL: Save as int64 to preserve full precision of TARGETIDs
    np.save(args.out_targetids, np.array(target_ids, dtype=np.int64))
    
    print(f"Saving redshifts to: {args.out_redshifts}")
    np.save(args.out_redshifts, np.array(redshifts, dtype=np.float32))
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()
