# DESI Spectra Embeddings Visualization

This directory contains tools for extracting and visualizing embeddings from DESI spectra using trained AstroPT models.

## Overview

The workflow consists of three main steps:
1. **Load Model**: Load a trained AstroPT checkpoint
2. **Extract Embeddings**: Process DESI spectra through the model and extract embeddings
3. **Visualize**: Use dimensionality reduction to visualize the embedding space

## Files

- `load_model.py`: Script to load a trained AstroPT checkpoint
- `build_dataloader.py`: Build a PyTorch DataLoader for DESI spectra
- `extract_embeddings.py`: Extract embeddings from DESI spectra
- `visualize_embeddings.ipynb`: Jupyter notebook for visualizing embeddings

## Quick Start

### 1. Extract Embeddings

Extract embeddings from your trained model checkpoint:

```bash
python scripts/desi/visualisations/embeddings/extract_embeddings.py \
    --checkpoint /path/to/your/checkpoint/ckpt.pt \
    --data-dir /path/to/desi/dataset \
    --split train \
    --batch-size 32 \
    --patch-size 256 \
    --device cuda \
    --out-embeddings desi_embeddings.npy \
    --out-targetids desi_targetids.npy \
    --out-redshifts desi_redshifts.npy
```

**Arguments:**
- `--checkpoint`: Path to the trained model checkpoint (e.g., `ckpt.pt`)
- `--data-dir`: Root directory of the DESI dataset
- `--split`: Which split to process (`train`, `validation`, `test`, or `None` for all)
- `--batch-size`: Batch size for inference (default: 32)
- `--patch-size`: Patch size for tokenization - should match training config (default: 256)
- `--device`: Device for inference (`cuda` or `cpu`)
- `--out-embeddings`: Output file for embeddings array
- `--out-targetids`: Output file for DESI target IDs
- `--out-redshifts`: Output file for redshift values
- `--max-batches`: (Optional) Process only first N batches for testing

### 2. Visualize Embeddings

Open and run the Jupyter notebook:

```bash
jupyter notebook scripts/desi/visualisations/embeddings/visualize_embeddings.ipynb
```

The notebook will:
- Load the extracted embeddings
- Perform UMAP and t-SNE dimensionality reduction
- Create various visualizations colored by redshift
- Show density plots and redshift bin comparisons
- Save publication-quality figures

**Required packages:**
```bash
pip install umap-learn scikit-learn matplotlib seaborn
```

## Example Usage

### Testing on a Small Subset

First, test the extraction pipeline on a small subset:

```bash
python scripts/desi/visualisations/embeddings/extract_embeddings.py \
    --checkpoint /pbs/throng/training/astroinfo2025/work/jzoubian/logs/astropt_desi_spectra_2/ckpt.pt \
    --data-dir /pbs/throng/training/astroinfo2025/data/astroPT_desi_dataset/ \
    --batch-size 16 \
    --max-batches 10 \
    --device cuda \
    --out-embeddings test_embeddings.npy \
    --out-targetids test_targetids.npy \
    --out-redshifts test_redshifts.npy
```

### Full Dataset Extraction

Process the entire dataset (this may take hours):

```bash
python scripts/desi/visualisations/embeddings/extract_embeddings.py \
    --checkpoint /pbs/throng/training/astroinfo2025/work/jzoubian/logs/astropt_desi_spectra_2/ckpt.pt \
    --data-dir /pbs/throng/training/astroinfo2025/data/astroPT_desi_dataset/ \
    --split train \
    --batch-size 64 \
    --num-workers 8 \
    --device cuda \
    --out-embeddings desi_train_embeddings.npy \
    --out-targetids desi_train_targetids.npy \
    --out-redshifts desi_train_redshifts.npy
```

## Output Files

After running the extraction script, you'll have:

1. **Embeddings** (`*.npy`): NumPy array of shape `(N, embedding_dim)` containing the extracted embeddings
2. **Target IDs** (`*.npy`): NumPy array of shape `(N,)` containing DESI target IDs
3. **Redshifts** (`*.npy`): NumPy array of shape `(N,)` containing spectroscopic redshifts

The visualization notebook will generate:
- `umap_desi_embeddings_redshift.png`: UMAP projection colored by redshift
- `umap_desi_embeddings_bins.png`: UMAP projection with redshift bins
- `tsne_desi_embeddings_redshift.png`: t-SNE projection colored by redshift
- `density_desi_embeddings.png`: Density plots for both projections
- `desi_embeddings_umap2d.npy`: 2D UMAP coordinates
- `desi_embeddings_tsne2d.npy`: 2D t-SNE coordinates
- `desi_embeddings_with_metadata.npz`: Combined file with all data

## Troubleshooting

### Memory Issues

If you encounter out-of-memory errors:
- Reduce `--batch-size`
- Use `--max-batches` to process a subset
- Process splits separately (`--split train`, `--split validation`, etc.)

### Import Errors

The extraction script imports from the training codebase. Make sure you're running from the repository root:

```bash
cd /home/zoubian/Workspace/AstroInfo/astroPT
python scripts/desi/visualisations/embeddings/extract_embeddings.py ...
```

Or add the repo to your PYTHONPATH:

```bash
export PYTHONPATH=/home/zoubian/Workspace/AstroInfo/astroPT:$PYTHONPATH
```

### Model Loading Issues

Ensure the checkpoint path is correct and contains:
- Model state dict (`model` key)
- Model configuration (`model_args` or `config` key)

You can inspect a checkpoint with:

```python
import torch
ckpt = torch.load('ckpt.pt', map_location='cpu')
print(ckpt.keys())
```

## Interpretation

The visualizations help you understand what the model has learned:

- **Clustering**: Do similar spectra cluster together?
- **Redshift structure**: Is there a smooth progression along redshift?
- **Separability**: Are different object types separable in embedding space?
- **Outliers**: Are there unusual spectra that stand out?

Good embeddings should show:
- Smooth transitions along physical parameters (redshift, spectral type)
- Clear separation of different object classes
- Compact clusters for similar objects
- Minimal outliers (unless scientifically interesting)

## Related Scripts

This pipeline is adapted from the Euclid image embeddings workflow in:
- `scripts/euclid/visualisations/embeddings/`

The main differences:
- Processes 1D spectra instead of 2D images
- Uses spectral patchification instead of image patches
- Includes DESI-specific metadata (target IDs, redshifts)

## Citation

If you use these tools in your research, please cite the AstroPT paper and the DESI survey.
