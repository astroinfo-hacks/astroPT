"""Load a trained AstroPT model checkpoint for DESI spectra embeddings."""

import torch
from pathlib import Path
from astropt.model import GPT, GPTConfig


def load_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """Load a trained model checkpoint from disk.
    
    Args:
        checkpoint_path: Path to the checkpoint file (e.g., 'ckpt.pt')
        device: Device to load the model on ('cpu' or 'cuda')
    
    Returns:
        tuple: (model, checkpoint_dict) where checkpoint_dict contains training info
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model configuration from checkpoint
    model_args = checkpoint.get('model_args', checkpoint.get('config', {}))
    
    # Create GPTConfig from model_args
    if isinstance(model_args, dict):
        config = GPTConfig(**model_args)
    else:
        config = model_args
    
    # Initialize model
    model = GPT(config)
    
    # Load state dict
    state_dict = checkpoint.get('model', checkpoint)
    
    # Remove 'module.' prefix if present (from DDP training)
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    print(f"Model type: {type(model).__name__}")
    print(f"Block size: {config.block_size}")
    print(f"Embedding dim: {config.n_embd}")
    print(f"Layers: {config.n_layer}")
    print(f"Heads: {config.n_head}")
    
    return model, checkpoint


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load AstroPT DESI checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (e.g., ckpt.pt)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to load model on",
    )
    args = parser.parse_args()
    
    model, ckpt = load_checkpoint(args.checkpoint, args.device)
    
    # Print additional checkpoint info
    if 'iter_num' in ckpt:
        print(f"Training iteration: {ckpt['iter_num']}")
    if 'best_val_loss' in ckpt:
        print(f"Best validation loss: {ckpt['best_val_loss']:.4f}")
    
    print("\nModel ready for inference!")
