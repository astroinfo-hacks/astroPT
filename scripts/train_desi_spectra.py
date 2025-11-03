"""
Distributed training entry-point for AstroPT on DESI spectra.

This script mirrors the structure of the generic AstroPT training loop while
replacing all imaging components with the DESI spectrum loader introduced in
scripts/euclid_desi_dataset/desi_spectrum_dataloader.py. It supports single GPU
debug runs and multi-GPU distributed data parallel (DDP) launches via torchrun.
"""

from __future__ import annotations

import argparse
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    import wandb

    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _WANDB_AVAILABLE = False

from astropt.model import GPT, GPTConfig, ModalityConfig, ModalityRegistry
from scripts.euclid_desi_dataset.desi_spectrum_dataloader import (
    DESISpectraDataset,
    spectra_collate,
)


# ---------------------------------------------------------------------------
# Helper dataclass for runtime configuration.
# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    """Container that gathers together the main hyperparameters."""

    out_dir: str = "logs/astropt_desi_spectra"
    data_dir: str = "/pbs/home/a/astroinfo08/astroinfo2025/data/astroPT_desi_dataset"
    train_split: str | None = None
    val_split: str | None = None
    eval_interval: int = 500
    eval_iters: int = 100
    log_interval: int = 50
    checkpoint_interval: int = 2000
    always_save_checkpoint: bool = False
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_workers: int = 8
    block_size: int = 256
    patch_size: int = 256
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    decay_lr: bool = True
    warmup_iters: int = 2000
    lr_decay_iters: int = 30000
    min_lr: float = 6e-5
    max_iters: int = 30000
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = True
    log_via_wandb: bool = False
    wandb_project: str | None = None
    wandb_run_name: str | None = None


# ---------------------------------------------------------------------------
# Utility functions.
# ---------------------------------------------------------------------------
def parse_args() -> TrainingConfig:
    """Collect CLI overrides for the training configuration."""

    parser = argparse.ArgumentParser(
        description="Train AstroPT on DESI spectra with DDP support."
    )
    parser.add_argument("--out-dir", default=TrainingConfig.out_dir)
    parser.add_argument("--data-dir", default=TrainingConfig.data_dir)
    parser.add_argument("--train-split", default=None)
    parser.add_argument("--val-split", default=None)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=TrainingConfig.gradient_accumulation_steps,
        help="Gradient accumulation steps per optimization step.",
    )
    parser.add_argument("--num-workers", type=int, default=TrainingConfig.num_workers)
    parser.add_argument("--block-size", type=int, default=TrainingConfig.block_size)
    parser.add_argument("--patch-size", type=int, default=TrainingConfig.patch_size)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--no-compile", dest="compile", action="store_false")
    parser.add_argument(
        "--log-wandb",
        action="store_true",
        default=TrainingConfig.log_via_wandb,
        help="Enable Weights & Biases logging if available.",
    )
    parser.add_argument("--wandb-project", default=TrainingConfig.wandb_project)
    parser.add_argument("--wandb-run-name", default=TrainingConfig.wandb_run_name)
    args = parser.parse_args()

    config = TrainingConfig()
    config.out_dir = args.out_dir
    config.data_dir = args.data_dir
    config.train_split = args.train_split
    config.val_split = args.val_split
    config.batch_size = args.batch_size
    config.gradient_accumulation_steps = args.grad_accum
    config.num_workers = args.num_workers
    config.block_size = args.block_size
    config.patch_size = args.patch_size
    config.compile = args.compile
    config.log_via_wandb = args.log_wandb and _WANDB_AVAILABLE
    config.wandb_project = args.wandb_project
    config.wandb_run_name = args.wandb_run_name
    return config


def maybe_init_wandb(config: TrainingConfig, run_name: str | None) -> None:
    """Kick off a Weights & Biases session if requested."""
    if not config.log_via_wandb or not _WANDB_AVAILABLE:
        return
    wandb.init(
        project=config.wandb_project,
        name=run_name,
        config=config.__dict__,
    )


def cleanup_wandb() -> None:
    """Ensure WANDB run is closed cleanly."""
    if _WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()


def setup_ddp(config: TrainingConfig) -> tuple[bool, int, int, torch.device]:
    """Initialise DDP process group metadata."""
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = torch.device(f"cuda:{ddp_local_rank}")
        torch.cuda.set_device(device)
    else:
        ddp_rank = 0
        ddp_world_size = 1
        device = torch.device(config.device)

    return ddp, ddp_rank, ddp_world_size, device


def prepare_spectra_batch(
    batch: dict[str, Any],
    patch_size: int,
    block_size: int,
    device: torch.device,
    target_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """
    Convert the collated DESI batch into the tensors expected by the model.

    We chunk each spectrum into consecutive windows of `patch_size` pixels,
    pad incomplete windows, and trim to `block_size` tokens so that the GPT
    model receives aligned inputs and position indices.
    """
    flux = batch["flux"].to(device=device, dtype=torch.float32)
    B, L = flux.shape
    pad = (patch_size - (L % patch_size)) % patch_size
    if pad:
        flux = F.pad(flux, (0, pad))

    patches = flux.view(B, -1, patch_size)
    positions = torch.arange(patches.size(1), device=device, dtype=torch.long)
    positions = positions.unsqueeze(0).expand(B, -1)

    if block_size > 0 and patches.size(1) > block_size:
        patches = patches[:, :block_size]
        positions = positions[:, :block_size]

    spectra_tokens = patches.to(dtype=target_dtype)
    batch_dict = {
        "spectra": spectra_tokens,
        "spectra_positions": positions,
    }
    return batch_dict


def create_dataloaders(
    config: TrainingConfig, ddp: bool, world_size: int, rank: int
) -> tuple[DataLoader, DataLoader | None]:
    """Initialise training and validation dataloaders."""
    train_dataset = DESISpectraDataset(
        data_dir=config.data_dir,
        split=config.train_split,
    )
    val_dataset = (
        DESISpectraDataset(
            data_dir=config.data_dir,
            split=config.val_split,
        )
        if config.val_split is not None
        else None
    )

    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        if ddp
        else None
    )

    val_sampler = (
        DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )
        if ddp and val_dataset is not None
        else None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(not ddp),
        num_workers=config.num_workers,
        collate_fn=spectra_collate,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=spectra_collate,
            pin_memory=True,
            drop_last=False,
            sampler=val_sampler,
        )
        if val_dataset is not None
        else None
    )
    return train_loader, val_loader


def estimate_tokens_per_iter(
    config: TrainingConfig, world_size: int, modalities: list[ModalityConfig]
) -> int:
    """Compute logging-friendly estimate for tokens processed each step."""
    tokens_per_iter = (
        config.gradient_accumulation_steps
        * world_size
        * config.batch_size
        * config.block_size
        * len(modalities)
    )
    return tokens_per_iter


def main() -> None:
    """Entry point."""
    config = parse_args()

    ddp, ddp_rank, world_size, device = setup_ddp(config)
    master_process = ddp_rank == 0

    # Seed management for reproducibility across ranks.
    torch.manual_seed(1337 + ddp_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Construct modality registry for spectra-only training.
    modalities = [
        ModalityConfig(
            name="spectra",
            input_size=config.patch_size,
            patch_size=config.patch_size,
            pos_input_size=1,
            loss_weight=1.0,
            embed_pos=False,
        ),
    ]
    modality_registry = ModalityRegistry(modalities)

    if ddp:
        # When running with DDP every rank performs gradient accumulation locally.
        # We divide by world size so the effective number of micro-steps matches
        # the single-GPU configuration.
        assert config.gradient_accumulation_steps % world_size == 0, (
            "Gradient accumulation steps must be divisible by the world size."
        )
        config.gradient_accumulation_steps //= world_size

    # Build the PyTorch dataloaders, wiring in DistributedSampler instances when
    # the script is launched under torchrun so that each rank sees unique data.
    train_loader, val_loader = create_dataloaders(config, ddp, world_size, ddp_rank)
    tokens_per_iter = estimate_tokens_per_iter(config, world_size, modalities)

    if master_process:
        os.makedirs(config.out_dir, exist_ok=True)
        print(f"Training logs will be written to: {config.out_dir}")
        print(f"Estimated tokens per iteration: {tokens_per_iter:,}")

    # Configure model and optimiser.
    gpt_config = GPTConfig(
        block_size=config.block_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        bias=config.bias,
        dropout=config.dropout,
        attn_type="causal",
    )
    # Instantiate the AstroPT transformer and move it onto the worker's device.
    model = GPT(gpt_config, modality_registry)
    model = model.to(device)
    if config.compile:
        # torch.compile can fuse kernels for sizeable speed-ups on modern GPUs.
        model = torch.compile(model)  # type: ignore[attr-defined]

    if ddp:
        # Wrap the model with DDP to synchronise gradients across ranks.
        model = DDP(model, device_ids=[device], find_unused_parameters=False)

    # Configure the fused AdamW optimiser that ships with AstroPT.
    base_model = model.module if isinstance(model, DDP) else model
    optimizer = base_model.configure_optimizers(
        weight_decay=config.weight_decay,
        learning_rate=config.learning_rate,
        betas=(config.beta1, config.beta2),
        device_type=device.type,
    )

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    target_dtype = dtype_map.get(config.dtype, torch.float32)
    use_amp = target_dtype in {torch.bfloat16, torch.float16}
    # GradScaler only kicks in for float16; bfloat16 tolerates large gradients.
    scalar = torch.cuda.amp.GradScaler(enabled=(target_dtype == torch.float16))
    autocast_ctx = (
        torch.amp.autocast(device_type=device.type, dtype=target_dtype)
        if use_amp
        else nullcontext()
    )

    maybe_init_wandb(config, config.wandb_run_name)

    best_val_loss = float("inf")
    iter_num = 0
    leftover_tokens = 0
    loss_meter = 0.0
    start_time = time.time()

    # We track a logical epoch counter so DDP samplers reshuffle between passes.
    epoch = 0
    while iter_num < config.max_iters:
        if ddp:
            assert isinstance(train_loader.sampler, DistributedSampler)
            train_loader.sampler.set_epoch(epoch)
            if val_loader is not None and isinstance(val_loader.sampler, DistributedSampler):
                val_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            model.train()
            # Convert the raw DESI spectra into modality-aware tensors for AstroPT.
            batch_tokens = prepare_spectra_batch(
                batch,
                patch_size=config.patch_size,
                block_size=config.block_size,
                device=device,
                target_dtype=target_dtype,
            )

            with autocast_ctx:
                # Forward pass returns a dictionary; we focus on the loss scalar.
                outputs = model(batch_tokens)
                loss = outputs["loss"]

            # Gradient accumulation splits the effective batch across micro-steps.
            loss = loss / config.gradient_accumulation_steps
            scalar.scale(loss).backward()

            leftover_tokens += 1
            iter_num += 1
            loss_meter += loss.item() * config.gradient_accumulation_steps

            if leftover_tokens % config.gradient_accumulation_steps == 0:
                # Optionally clip gradients before the optimiser step to stabilise training.
                if config.grad_clip > 0:
                    scalar.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scalar.step(optimizer)
                scalar.update()
                optimizer.zero_grad(set_to_none=True)
                leftover_tokens = 0

            if master_process and iter_num % config.log_interval == 0:
                # Emit console (and optional WandB) logs so we can monitor convergence.
                elapsed = time.time() - start_time
                tokens = iter_num * config.batch_size * config.block_size
                speed = tokens / elapsed if elapsed > 0 else 0.0
                print(
                    f"iter {iter_num:6d} | loss {loss_meter / config.log_interval:.4f} "
                    f"| tokens/sec {speed:,.0f}"
                )
                if config.log_via_wandb:
                    wandb.log(
                        {
                            "train/loss": loss_meter / config.log_interval,
                            "train/learning_rate": optimizer.param_groups[0]["lr"],
                            "train/tokens_per_sec": speed,
                        },
                        step=iter_num,
                    )
                loss_meter = 0.0

            if iter_num % config.eval_interval == 0:
                def run_eval(loader: DataLoader | None) -> float:
                    if loader is None:
                        return float("inf")
                    model.eval()
                    losses = []
                    with torch.no_grad():
                        loader_iter = iter(loader)
                        for _ in range(config.eval_iters):
                            try:
                                eval_batch = next(loader_iter)
                            except StopIteration:
                                loader_iter = iter(loader)
                                eval_batch = next(loader_iter)

                            eval_tokens = prepare_spectra_batch(
                                eval_batch,
                                patch_size=config.patch_size,
                                block_size=config.block_size,
                                device=device,
                                target_dtype=target_dtype,
                            )
                            with autocast_ctx:
                                outputs = model(eval_tokens)
                                losses.append(outputs["loss"].item())
                    model.train()
                    return float(sum(losses) / len(losses))

                val_loss = run_eval(val_loader)
                if master_process:
                    print(f"[eval] iter {iter_num} | val loss {val_loss:.4f}")
                    if config.log_via_wandb:
                        wandb.log({"val/loss": val_loss}, step=iter_num)
                    if val_loss < best_val_loss or config.always_save_checkpoint:
                        best_val_loss = val_loss
                        checkpoint = {
                            "model": base_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "iter_num": iter_num,
                            "config": config.__dict__,
                        }
                        ckpt_path = os.path.join(config.out_dir, "ckpt.pt")
                        torch.save(checkpoint, ckpt_path)
                        print(f"Saved checkpoint to {ckpt_path}")

            if iter_num >= config.max_iters:
                break

        epoch += 1

    if ddp:
        destroy_process_group()
    cleanup_wandb()


if __name__ == "__main__":
    main()
