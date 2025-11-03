"""Utility script to inspect DESI spectroscopic data stored in HuggingFace format.

The dataset lives on a remote filesystem that already contains the Arrow shards
exported from HuggingFace. Each record contains DESI spectral information
(`wavelength`, `flux`, `ivar`, `mask`, `targetid`, `redshift`) with no imaging
fields.
"""

from __future__ import annotations

import argparse
import glob
import math
import os

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for remote execution
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import Dataset, concatenate_datasets, load_from_disk
from torch.utils.data import DataLoader


DEFAULT_DATA_DIR = "/pbs/home/a/astroinfo08/astroinfo2025/data/astroPT_desi_dataset"


def _load_local_hf_dataset(data_dir: str, split: str | None) -> Dataset:
    """Load a HuggingFace dataset that has been saved to disk already."""
    candidate_path = data_dir
    if split:
        split_path = os.path.join(data_dir, split)
        if os.path.isdir(split_path):
            candidate_path = split_path

    if not os.path.exists(candidate_path):
        raise FileNotFoundError(
            f"Dataset path '{candidate_path}' does not exist. "
            f"Check the value of --data-dir (current default: {DEFAULT_DATA_DIR})."
        )

    try:
        return load_from_disk(candidate_path)
    except Exception as err:
        # Fall back to manual Arrow concatenation when load_from_disk is not viable.
        arrow_glob = os.path.join(candidate_path, "data-*.arrow")
        arrow_files = sorted(glob.glob(arrow_glob))

        if not arrow_files and split:
            split_pattern = os.path.join(data_dir, f"{split}-*.arrow")
            arrow_files = sorted(glob.glob(split_pattern))

        if not arrow_files:
            arrow_glob = os.path.join(data_dir, "data-*.arrow")
            arrow_files = sorted(glob.glob(arrow_glob))

        if not arrow_files:
            raise RuntimeError(
                f"Unable to discover Arrow shards under '{candidate_path}'."
            ) from err

        datasets = [Dataset.from_file(path) for path in arrow_files]
        return concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]


def _to_tensor(array) -> torch.Tensor | None:
    """Convert feature arrays to float tensors while keeping None values."""
    if array is None:
        return None
    if isinstance(array, torch.Tensor):
        return array.to(torch.float32)
    np_array = np.asarray(array)
    return torch.from_numpy(np_array).to(torch.float32)


class DESISpectraDataset(torch.utils.data.Dataset):
    """PyTorch wrapper for the DESI spectra dataset."""

    def __init__(
        self,
        data_dir: str = DEFAULT_DATA_DIR,
        split: str | None = None,
        hf_dataset: Dataset | None = None,
    ):
        self.data_dir = data_dir
        self.split = split
        self.dataset = hf_dataset or _load_local_hf_dataset(data_dir, split)

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _normalise_spectrum(
        flux: torch.Tensor,
        ivar: torch.Tensor | None,
        wavelength: torch.Tensor | None,
        mask: torch.Tensor | None,
        redshift: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, float] | None:
        """Normalise a spectrum using the median flux in a rest-frame window."""
        if flux is None or wavelength is None:
            return None

        flux = flux.clone()
        if ivar is not None:
            ivar = ivar.clone()
        if mask is not None:
            mask = mask.clone()

        flux = torch.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
        if ivar is not None:
            ivar = torch.nan_to_num(ivar, nan=0.0, posinf=0.0, neginf=0.0)
        wavelength = torch.nan_to_num(wavelength, nan=0.0, posinf=0.0, neginf=0.0)

        wave_rest = wavelength
        valid = torch.isfinite(flux) & torch.isfinite(wave_rest)
        if ivar is not None:
            valid &= ivar > 0
        if mask is not None:
            valid &= mask == 0

        window = (wave_rest > 5300) & (wave_rest < 5850)
        sel = valid & window
        if not torch.any(sel):
            return None

        norm_tensor = torch.nanmedian(flux[sel])
        norm_value = float(norm_tensor) if norm_tensor.numel() > 0 else float("nan")
        if not math.isfinite(norm_value) or norm_value == 0:
            return None

        flux = flux / norm_value
        if ivar is not None:
            ivar = ivar * (norm_value**2)

        flux = torch.where(valid, flux, torch.zeros_like(flux))
        if ivar is not None:
            ivar = torch.where(valid, ivar, torch.zeros_like(ivar))

        return flux, ivar, norm_value

    def __getitem__(self, idx: int) -> dict:
        attempts = 0
        data_len = len(self.dataset)
        current_idx = idx % data_len

        while attempts < data_len:
            sample = self.dataset[current_idx]
            flux = _to_tensor(sample.get("flux"))
            wavelength = _to_tensor(sample.get("wavelength"))
            ivar = _to_tensor(sample.get("ivar"))
            mask = _to_tensor(sample.get("mask"))
            redshift = sample.get("redshift")

            if flux is None:
                attempts += 1
                current_idx = (current_idx + 1) % data_len
                continue

            normalised = self._normalise_spectrum(flux, ivar, wavelength, mask, redshift)
            if normalised is None:
                attempts += 1
                current_idx = (current_idx + 1) % data_len
                continue

            norm_flux, norm_ivar, norm_value = normalised

            payload = {
                "flux": norm_flux,
                "wavelength": wavelength,
                "ivar": norm_ivar,
                "mask": mask,
                "targetid": sample.get("targetid"),
                "redshift": redshift,
                "norm": norm_value,
            }
            return payload

        raise RuntimeError("Unable to normalise any spectrum in the dataset.")


def spectra_collate(batch: list[dict]) -> dict:
    """Custom collate function that keeps metadata as lists and stacks tensors."""
    # Filter out any None entries (should not occur, but keeps collate robust).
    batch = [item for item in batch if item is not None]
    if not batch:
        raise ValueError("All samples in batch were invalid after normalisation.")

    collated: dict[str, list | torch.Tensor | None] = {}
    keys = batch[0].keys()
    for key in keys:
        values = [item[key] for item in batch]
        first = values[0]
        if isinstance(first, torch.Tensor):
            collated[key] = torch.stack(values)
        elif isinstance(first, (int, float)) and all(
            isinstance(v, (int, float)) for v in values
        ):
            collated[key] = torch.tensor(values, dtype=torch.float32)
        else:
            collated[key] = values
    return collated


def visualise_sample(batch: dict, output_path: str) -> None:
    """Save a flux-versus-wavelength plot for the first item in the batch."""
    flux = batch["flux"][0].cpu().numpy()
    wavelength_tensor = batch["wavelength"][0]
    wavelength = (
        wavelength_tensor.cpu().numpy()
        if isinstance(wavelength_tensor, torch.Tensor)
        else np.arange(len(flux))
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(wavelength, flux, color="tab:blue", linewidth=0.8, label="flux")

    if isinstance(batch["ivar"][0], torch.Tensor):
        ivar = batch["ivar"][0].cpu().numpy()
        sigma = np.where(ivar > 0, np.sqrt(1.0 / ivar), 0.0)
        ax.fill_between(
            wavelength,
            flux - sigma,
            flux + sigma,
            color="tab:orange",
            alpha=0.2,
            label="±1σ (ivar)",
        )

    title_id = batch["targetid"][0] or "unknown"
    redshift = batch["redshift"][0]
    title = f"DESI Spectrum {title_id}"
    if redshift is not None:
        title += f" (z={redshift:.4f})"

    ax.set_title(title)
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test DESI spectroscopic dataloader.")
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help="Root directory containing the Arrow shards or HF dataset export.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Optional split subdirectory (e.g. 'train', 'validation'). "
        "Leave unset to use the root dataset.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Loader batch size.")
    parser.add_argument(
        "--num-workers", type=int, default=0, help="Number of DataLoader workers."
    )
    parser.add_argument(
        "--plot-path",
        default="desi_spectrum_batch.png",
        help="Output path for the diagnostic spectrum plot.",
    )
    args = parser.parse_args()

    dataset = DESISpectraDataset(data_dir=args.data_dir, split=args.split)
    print(f"Loaded {len(dataset)} spectra from '{args.data_dir}'.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=spectra_collate,
    )

    batch = next(iter(dataloader))
    print("Batch keys:", list(batch.keys()))
    print("Flux tensor shape:", batch["flux"].shape)
    if isinstance(batch["wavelength"][0], torch.Tensor):
        print("Wavelength tensor shape:", batch["wavelength"].shape)
    if isinstance(batch["ivar"][0], torch.Tensor):
        print("Inverse variance tensor shape:", batch["ivar"].shape)
    if isinstance(batch["mask"][0], torch.Tensor):
        print("Mask tensor shape:", batch["mask"].shape)

    print("Example target IDs:", batch["targetid"])
    print("Example redshifts:", batch["redshift"])
    print("Normalisation factors:", batch["norm"])

    visualise_sample(batch, args.plot_path)
    print(f"Saved diagnostic plot to '{args.plot_path}'.")


if __name__ == "__main__":
    main()
