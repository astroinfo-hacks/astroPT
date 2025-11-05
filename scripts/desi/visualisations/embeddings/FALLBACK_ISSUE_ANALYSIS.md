# DESI Embedding Extraction Issue: Root Cause Analysis

## Problem Summary

**Symptom:** 658 unique TARGETIDs in 754,064 embedding entries (massive duplication)

**Root Cause:** Fallback mechanism in `DESISpectraDataset.__getitem__()`

## Detailed Analysis

### How the Fallback Works

In `scripts/euclid_desi_dataset/desi_spectrum_dataloader.py`, lines 148-177:

```python
def __getitem__(self, idx: int) -> dict:
    attempts = 0
    data_len = len(self.dataset)
    current_idx = idx % data_len

    while attempts < data_len:
        sample = self.dataset[current_idx]
        # ... try to normalize spectrum ...
        
        normalised = self._normalise_spectrum(flux, ivar, wavelength, mask, redshift)
        if normalised is None:
            attempts += 1
            current_idx = (current_idx + 1) % data_len  # ← FALLBACK TO NEXT INDEX
            continue
        
        # Return spectrum from current_idx (which may differ from requested idx!)
        return payload
```

### Why This Causes Duplication

1. **Training scenario (works fine):**
   - Training only cares about flux values for reconstruction loss
   - TARGETIDs are only used for logging/visualization
   - Fallback doesn't affect training quality

2. **Embedding extraction scenario (BROKEN):**
   - Extraction processes dataset sequentially: indices 0, 1, 2, 3, ...
   - Suppose spectra at indices 0, 1, 2 fail normalization, but index 3 succeeds
   - Result:
     - `dataset[0]` → returns spectrum from index 3 (TARGETID_3)
     - `dataset[1]` → returns spectrum from index 3 (TARGETID_3)
     - `dataset[2]` → returns spectrum from index 3 (TARGETID_3)
     - `dataset[3]` → returns spectrum from index 3 (TARGETID_3)
   - **Same spectrum returned 4 times!**

3. **Why no catalog matches:**
   - The 658 unique TARGETIDs don't correspond to a contiguous range
   - They're the TARGETIDs of spectra that succeeded normalization
   - But they appear at random positions in the embedding array
   - Catalog matching assumes embedding[i] corresponds to catalog[i]

## Why Previous Diagnostics Didn't Reveal This

✅ **Dataset check:** Dataset itself has 754,064 unique TARGETIDs → CORRECT
✅ **DataLoader check:** Small sample (160 spectra) showed no duplicates → Didn't hit enough failed normalizations
✅ **Collate function:** Preserves TARGETIDs correctly → CORRECT
❌ **Extraction:** Full dataset has ~753,406 failed normalizations (99.9% failure rate!)

## Solutions

### Option 1: Strict Mode (Recommended for Extraction)
Remove fallback during extraction to detect which spectra fail:

```python
def __getitem__(self, idx: int) -> dict:
    sample = self.dataset[idx]
    # ... normalize ...
    
    if normalised is None:
        raise ValueError(f"Spectrum at index {idx} failed normalization")
    
    return payload
```

Then filter dataset to only include valid spectra before extraction.

### Option 2: Pre-filter Dataset
Create a filtered dataset of only normalizable spectra:

```python
valid_indices = []
for idx in range(len(dataset)):
    try:
        _ = dataset[idx]
        valid_indices.append(idx)
    except:
        continue

# Create subset
valid_dataset = torch.utils.data.Subset(dataset, valid_indices)
```

### Option 3: Track True Index in Return
Modify `__getitem__` to return which index was actually used:

```python
payload = {
    "flux": norm_flux,
    "targetid": sample.get("targetid"),
    "true_index": current_idx,  # ← Add this
    "requested_index": idx,
}
```

## Recommended Fix

**For immediate extraction:**
1. Use Option 2: Pre-filter dataset to create `valid_indices.npy`
2. Extract embeddings only for valid indices
3. Save embeddings with true TARGETID mapping

**For long-term training:**
- Keep current fallback (it's fine for training)
- Add warning in documentation about extraction requirements

## Testing the Fix

Run `debug_fallback.py` to confirm the issue:
```bash
python scripts/desi/visualisations/embeddings/debug_fallback.py \
    --data-dir /pbs/throng/training/astroinfo2025/data/astroPT_desi_dataset \
    --num-samples 1000
```

Expected output with bug:
```
Requested indices: 1000
Unique TARGETIDs returned: ~1
TARGETIDs returned multiple times: ~1
⚠️  FALLBACK ISSUE CONFIRMED!
```
