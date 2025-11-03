import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless servers
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from PIL import Image
import pandas as pd

def test_dataset_loading():
    """Test loading and visualizing data from HuggingFace dataset."""
    
    # Load the dataset
    print("Loading dataset from HuggingFace...")
    try:
        # Load the first available split (train_batch_1)
        dataset = load_dataset("msiudek/astroPT_euclid_desi_dataset", split="train_batch_1")
        print(f"Successfully loaded dataset with {len(dataset)} samples")
        
        # Print dataset info
        print(f"Dataset features: {dataset.features}")
        print(f"Column names: {list(dataset.features.keys())}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Get a sample to test
    sample_idx = 0
    sample = dataset[sample_idx]
    
    print(f"\n=== Sample {sample_idx} ===")
    print(f"Object ID: {sample['object_id']}")
    print(f"Target ID: {sample['targetid']}")
    print(f"Redshift: {sample['redshift']}")
    
    # Check what data is available
    has_rgb = 'RGB_image' in sample and sample['RGB_image'] is not None
    has_spectrum = 'spectrum' in sample and sample['spectrum'] is not None
    has_sed = 'sed_data' in sample and sample['sed_data'] is not None
    has_vis = 'VIS_image' in sample and sample['VIS_image'] is not None
    has_nisp_y = 'NISP_Y_image' in sample and sample['NISP_Y_image'] is not None
    has_nisp_j = 'NISP_J_image' in sample and sample['NISP_J_image'] is not None
    has_nisp_h = 'NISP_H_image' in sample and sample['NISP_H_image'] is not None
    
    print(f"Available data: RGB={has_rgb}, Spectrum={has_spectrum}, SED={has_sed}")
    print(f"Individual bands: VIS={has_vis}, Y={has_nisp_y}, J={has_nisp_j}, H={has_nisp_h}")
    
    # Create a comprehensive figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"Dataset Sample {sample_idx}: Object {sample['object_id']} (z={sample['redshift']:.4f})", fontsize=16)
    
    # Plot RGB image
    if has_rgb:
        axes[0, 0].imshow(sample['RGB_image'])
        axes[0, 0].set_title('RGB Composite Image')
        axes[0, 0].axis('off')
    else:
        axes[0, 0].text(0.5, 0.5, 'No RGB Image', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('RGB Image (Missing)')
        axes[0, 0].axis('off')
    
    # Plot spectrum if available
    if has_spectrum and sample['spectrum']['wavelength'] is not None and sample['spectrum']['flux'] is not None:
        wavelength = np.array(sample['spectrum']['wavelength'])
        flux = np.array(sample['spectrum']['flux'])
        
        axes[0, 1].plot(wavelength, flux, 'b-', linewidth=0.8)
        axes[0, 1].set_xlabel('Wavelength (Å)')
        axes[0, 1].set_ylabel('Flux')
        axes[0, 1].set_title('DESI Spectrum')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot error if available
        if sample['spectrum']['error'] is not None:
            error = np.array(sample['spectrum']['error'])
            axes[0, 1].fill_between(wavelength, flux - error, flux + error, alpha=0.3, color='gray', label='±1σ error')
            axes[0, 1].legend()
    else:
        axes[0, 1].text(0.5, 0.5, 'No spectrum data', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Spectrum (Missing)')
    
    # Plot SED data if available
    if has_sed:
        sed_data = sample['sed_data']
        # Extract flux values and filter names
        flux_filters = [k for k in sed_data.keys() if k.startswith('flux_')]
        flux_values = [sed_data[k] for k in flux_filters]
        filter_names = [f.replace('flux_', '').replace('_ext_', '_').replace('_1fwhm_aper', '') for f in flux_filters]
        
        # Create a bar plot of fluxes
        bars = axes[0, 2].bar(range(len(flux_values)), flux_values)
        axes[0, 2].set_xlabel('Filter')
        axes[0, 2].set_ylabel('Flux')
        axes[0, 2].set_title(f'SED Photometry ({len(flux_values)} bands)')
        axes[0, 2].set_xticks(range(len(flux_values)))
        axes[0, 2].set_xticklabels([f[:8] for f in filter_names], rotation=45, ha='right', fontsize=8)
        
        # Print some SED info
        print(f"SED filters available: {len(flux_filters)}")
        print(f"Sample filters: {flux_filters[:3]}...")
    else:
        axes[0, 2].text(0.5, 0.5, 'No SED data', ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_title('SED Data (Missing)')
    
    # Show summary statistics
    axes[0, 3].axis('off')
    summary_text = f"""Dataset Summary:
    
Total samples: {len(dataset)}
Sample ID: {sample_idx}
Object ID: {sample['object_id']}
Target ID: {sample['targetid']}
Redshift: {sample['redshift']:.4f}

Data Available:
✓ RGB Image: {has_rgb}
✓ Spectrum: {has_spectrum}
✓ SED Data: {has_sed}
✓ VIS Band: {has_vis}
✓ NIR-Y Band: {has_nisp_y}
✓ NIR-J Band: {has_nisp_j}
✓ NIR-H Band: {has_nisp_h}"""
    
    axes[0, 3].text(0.05, 0.95, summary_text, transform=axes[0, 3].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    axes[0, 3].set_title('Dataset Summary')
    
    # Plot individual band images
    band_images = [
        (sample.get('VIS_image'), 'VIS Band', has_vis),
        (sample.get('NISP_Y_image'), 'NIR-Y Band', has_nisp_y),
        (sample.get('NISP_J_image'), 'NIR-J Band', has_nisp_j),
        (sample.get('NISP_H_image'), 'NIR-H Band', has_nisp_h)
    ]
    
    for i, (band_data, title, available) in enumerate(band_images):
        if available and band_data is not None:
            image_data = np.array(band_data)
            im = axes[1, i].imshow(image_data, cmap='viridis')
            axes[1, i].set_title(title)
            axes[1, i].axis('off')
            plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)
        else:
            axes[1, i].text(0.5, 0.5, f'No {title}', ha='center', va='center', transform=axes[1, i].transAxes)
            axes[1, i].set_title(f'{title} (Missing)')
            axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # Save the figure explicitly
    output_file = 'dataset_test_sample.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved as '{output_file}'")
    
    # Close the figure to free memory
    plt.close(fig)
    
    # Test multiple samples
    print(f"\n=== Testing multiple samples ===")
    n_test = min(5, len(dataset))
    for i in range(n_test):
        sample = dataset[i]
        has_rgb = 'RGB_image' in sample and sample['RGB_image'] is not None
        has_spectrum = 'spectrum' in sample and sample['spectrum'] is not None
        has_sed = 'sed_data' in sample and sample['sed_data'] is not None
        has_vis = 'VIS_image' in sample and sample['VIS_image'] is not None
        
        print(f"Sample {i}: Object {sample['object_id']}, z={sample['redshift']:.4f}, "
              f"RGB={'✓' if has_rgb else '✗'}, "
              f"Spectrum={'✓' if has_spectrum else '✗'}, "
              f"SED={'✓' if has_sed else '✗'}, "
              f"VIS={'✓' if has_vis else '✗'}")

def check_all_splits():
    """Check what splits are available in the dataset."""
    try:
        from huggingface_hub import list_repo_files
        files = list_repo_files("msiudek/astroPT_euclid_desi_dataset", repo_type="dataset")
        
        # Find parquet files which indicate splits
        parquet_files = [f for f in files if f.endswith('.parquet')]
        splits = set()
        for f in parquet_files:
            # Extract split name from file path like "train_batch_1/data-00000-of-00001.parquet"
            if '/' in f:
                split_name = f.split('/')[0]
                splits.add(split_name)
        
        print(f"Available splits: {sorted(splits)}")
        return sorted(splits)
        
    except Exception as e:
        print(f"Error checking splits: {e}")
        return []

if __name__ == "__main__":
    print("=== HuggingFace Dataset Test ===")
    
    # Skip API check due to SSL issues, test directly
    print("Testing with default split: train_batch_1")
    test_dataset_loading()
        
    print("\nTest complete!")
    print("Check 'dataset_test_sample.png' for the visualization!")