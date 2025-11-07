"""
Unified Data Preparation Pipeline
✅ FIXED: Single source of truth for ALL data processing
✅ FIXED: Stacks unlabeled data to .npy (no more .tif bottleneck)
✅ FIXED: Creates comprehensive metadata.json
"""

import numpy as np
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm
import json
import logging
from typing import Dict, List, Optional
import tifffile

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare SELMA3D dataset (labeled + unlabeled)'
    )
    
    # Input/Output
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Path to raw SELMA3D data')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Path to save processed data')
    parser.add_argument('--data_type', type=str, required=True,
                       choices=['unlabeled', 'labeled'],
                       help='Type of data to process')
    
    # Processing options
    parser.add_argument('--downsample_xy', type=float, default=0.5,
                       help='XY downsampling factor')
    parser.add_argument('--downsample_z', type=float, default=1.0,
                       help='Z downsampling factor')
    parser.add_argument('--normalize', action='store_true', default=True,
                       help='Normalize intensities')
    
    # Memory optimization
    parser.add_argument('--max_slices_in_memory', type=int, default=1000,
                       help='Max slices to load at once (for large brains)')
    
    return parser.parse_args()


def load_brain_volume_from_tifs(
    brain_folder: Path,
    downsample_xy: float = 1.0,
    downsample_z: float = 1.0,
    normalize: bool = True,
    max_slices: Optional[int] = None
) -> Optional[np.ndarray]:
    """
    ✅ UNIFIED FUNCTION: Load all 2D slices and stack into 3D volume
    
    Args:
        brain_folder: Path to folder containing .tif slices
        downsample_xy: XY downsampling factor
        downsample_z: Z downsampling factor
        normalize: Whether to normalize intensities
        max_slices: Max slices to load (for memory constraints)
    
    Returns:
        volume: (D, H, W) numpy array or None if no slices found
    """
    # Find all .tif/.tiff files
    slice_files = sorted(
        list(brain_folder.glob("*.tif")) + list(brain_folder.glob("*.tiff")),
        key=lambda x: int(''.join(filter(str.isdigit, x.stem))) if any(c.isdigit() for c in x.stem) else 0
    )
    
    if len(slice_files) == 0:
        logger.warning(f"No .tif files found in {brain_folder}")
        return None
    
    # Limit slices if specified
    if max_slices and len(slice_files) > max_slices:
        logger.info(f"  Limiting to {max_slices} slices (out of {len(slice_files)})")
        slice_files = slice_files[:max_slices]
    
    logger.info(f"  Loading {len(slice_files)} slices...")
    
    # Load first slice to get dimensions
    first_slice = np.array(Image.open(slice_files[0]))
    H, W = first_slice.shape
    D = len(slice_files)
    
    # Calculate output dimensions
    if downsample_z != 1.0:
        D_out = max(1, int(D * downsample_z))
        z_indices = np.linspace(0, D - 1, D_out, dtype=int)
    else:
        D_out = D
        z_indices = range(D)
    
    if downsample_xy != 1.0:
        H_out = max(1, int(H * downsample_xy))
        W_out = max(1, int(W * downsample_xy))
    else:
        H_out, W_out = H, W
    
    logger.info(f"  Original shape: ({D}, {H}, {W})")
    logger.info(f"  Output shape: ({D_out}, {H_out}, {W_out})")
    
    # Preallocate output volume
    volume = np.zeros((D_out, H_out, W_out), dtype=np.float32)
    
    # Load and process slices
    for i, z_idx in enumerate(tqdm(z_indices, desc="  Stacking", leave=False)):
        try:
            slice_img = np.array(Image.open(slice_files[z_idx])).astype(np.float32)
            
            # Downsample XY if needed
            if downsample_xy != 1.0:
                from scipy.ndimage import zoom
                slice_img = zoom(slice_img, downsample_xy, order=1)
            
            # Normalize if requested
            if normalize:
                slice_min, slice_max = slice_img.min(), slice_img.max()
                if slice_max > slice_min:
                    slice_img = (slice_img - slice_min) / (slice_max - slice_min)
            
            volume[i] = slice_img
            
        except Exception as e:
            logger.error(f"  Failed to load slice {z_idx}: {e}")
            raise
    
    return volume


def process_unlabeled_data(input_dir: Path, output_dir: Path, args) -> List[Dict]:
    """
    ✅ FIXED: Process unlabeled data (SSL) - now stacks to .npy!
    
    Expected structure:
    input_dir/
        cFos/
            brain_001/
                slice_0000.tif
                ...
        vessel/
            brain_001/
        ...
    
    Output structure:
    output_dir/
        cFos/
            brain_001.npy
            brain_002.npy
        vessel/
            brain_001.npy
        ...
        metadata.json
    """
    logger.info("\n" + "="*80)
    logger.info("PROCESSING UNLABELED DATA (SSL)")
    logger.info("="*80)
    
    all_metadata = []
    
    # Marker type mapping (for weak supervision)
    marker_map = {
        'cfos': 0,
        'vessel': 1,
        'nucleus': 2,
        'ab_plaque': 3,
        'ad_plaque': 3,  # Alternative naming
        'microglia': 4   # For validation/test
    }
    
    # Find all marker directories
    marker_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    if len(marker_dirs) == 0:
        raise ValueError(f"No marker directories found in {input_dir}")
    
    logger.info(f"Found marker types: {[d.name for d in marker_dirs]}\n")
    
    for marker_dir in marker_dirs:
        marker_name = marker_dir.name
        marker_name_lower = marker_name.lower()
        
        # Get marker label
        marker_label = marker_map.get(marker_name_lower, 0)
        
        logger.info(f"📁 Processing marker: {marker_name} (label={marker_label})")
        
        # Find all brain folders
        brain_folders = [d for d in marker_dir.iterdir() if d.is_dir()]
        
        if len(brain_folders) == 0:
            logger.warning(f"  No brain folders found in {marker_name}, skipping")
            continue
        
        logger.info(f"  Found {len(brain_folders)} brains")
        
        # Create output directory
        output_marker_dir = output_dir / marker_name
        output_marker_dir.mkdir(parents=True, exist_ok=True)
        
        for brain_folder in brain_folders:
            brain_name = brain_folder.name
            logger.info(f"\n  Processing: {brain_name}")
            
            # ✅ CRITICAL FIX: Stack slices to 3D volume
            volume = load_brain_volume_from_tifs(
                brain_folder,
                downsample_xy=args.downsample_xy,
                downsample_z=args.downsample_z,
                normalize=args.normalize,
                max_slices=args.max_slices_in_memory
            )
            
            if volume is None:
                logger.warning(f"  Skipping {brain_name} (no valid slices)")
                continue
            
            # Save as single .npy file
            output_file = output_marker_dir / f"{brain_name}.npy"
            np.save(output_file, volume)
            
            file_size_mb = output_file.stat().st_size / 1e6
            logger.info(f"  ✅ Saved: {output_file.name} ({file_size_mb:.1f} MB)")
            
            # Store metadata
            metadata = {
                'brain_name': brain_name,
                'marker_type': marker_name,
                'marker_label': marker_label,
                'shape': volume.shape,
                'file_path': str(output_file.relative_to(output_dir)),
                'file_size_mb': file_size_mb,
                'processing': {
                    'downsample_xy': args.downsample_xy,
                    'downsample_z': args.downsample_z,
                    'normalized': args.normalize
                }
            }
            all_metadata.append(metadata)
    
    # Save metadata
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump({
            'num_volumes': len(all_metadata),
            'marker_types': list(set(m['marker_type'] for m in all_metadata)),
            'total_size_mb': sum(m['file_size_mb'] for m in all_metadata),
            'volumes': all_metadata
        }, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ Processed {len(all_metadata)} unlabeled volumes")
    logger.info(f"✅ Total size: {sum(m['file_size_mb'] for m in all_metadata):.1f} MB")
    logger.info(f"✅ Metadata saved: {metadata_path}")
    logger.info(f"{'='*80}")
    
    return all_metadata


def process_labeled_data(input_dir: Path, output_dir: Path, args) -> List[Dict]:
    """
    ✅ ENHANCED: Process labeled data with improved error handling
    
    Expected structure:
    input_dir/
        cFos/
            patch_001_img.tif
            patch_001_mask.tif
        ...
    
    Output structure:
    output_dir/
        train/
            cFos_001_img.npy
            cFos_001_mask.npy
        val/
        test/
        metadata.json
    """
    logger.info("\n" + "="*80)
    logger.info("PROCESSING LABELED DATA (FINE-TUNING)")
    logger.info("="*80)
    
    all_samples = []
    
    # Find all marker directories
    marker_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    if len(marker_dirs) == 0:
        raise ValueError(f"No marker directories found in {input_dir}")
    
    logger.info(f"Found marker types: {[d.name for d in marker_dirs]}\n")
    
    for marker_dir in marker_dirs:
        marker_name = marker_dir.name
        logger.info(f"📁 Processing marker: {marker_name}")
        
        # Find all image-mask pairs
        img_files = sorted(marker_dir.glob("*_img.tif"))
        
        if len(img_files) == 0:
            logger.warning(f"  No *_img.tif files found, skipping")
            continue
        
        logger.info(f"  Found {len(img_files)} image files")
        
        for img_file in tqdm(img_files, desc=f"  Processing {marker_name}"):
            mask_file = img_file.parent / img_file.name.replace('_img.', '_mask.')
            
            if not mask_file.exists():
                logger.warning(f"  Missing mask for {img_file.name}, skipping")
                continue
            
            try:
                # Load image and mask
                # img = np.array(Image.open(img_file)).astype(np.float32)
                # mask = np.array(Image.open(mask_file)).astype(np.int64)
                # --- NEW 3D LOADING ---
                img = tifffile.imread(img_file).astype(np.float32)
                mask = tifffile.imread(mask_file).astype(np.int64)
                
                # Validate shapes
                if img.shape != mask.shape:
                    logger.warning(
                        f"  Shape mismatch: {img_file.name} {img.shape} vs "
                        f"{mask_file.name} {mask.shape}, skipping"
                    )
                    continue
                
                # Downsample if requested
                if args.downsample_xy != 1.0:
                    from scipy.ndimage import zoom
                    img = zoom(img, args.downsample_xy, order=1)
                    mask = zoom(mask, args.downsample_xy, order=0)
                
                # Normalize image
                if args.normalize:
                    img_min, img_max = img.min(), img.max()
                    if img_max > img_min:
                        img = (img - img_min) / (img_max - img_min)
                
                # Store sample info
                all_samples.append({
                    'img': img,
                    'mask': mask,
                    'marker_type': marker_name,
                    'filename': img_file.stem.replace('_img', ''),
                    'shape': img.shape
                })
                
            except Exception as e:
                logger.error(f"  Failed to process {img_file.name}: {e}")
                continue
    
    if len(all_samples) == 0:
        raise ValueError("No valid samples found!")
    
    logger.info(f"\nTotal samples: {len(all_samples)}")
    
    # ✅ ENHANCED: Stratified split by marker type
    from sklearn.model_selection import train_test_split
    
    # Extract marker types for stratification
    marker_types = [s['marker_type'] for s in all_samples]
    
    # Train/val/test split (80/10/10)
    train_samples, temp_samples = train_test_split(
        all_samples,
        test_size=0.2,
        stratify=marker_types,
        random_state=42
    )
    
    temp_markers = [s['marker_type'] for s in temp_samples]
    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=0.5,
        stratify=temp_markers,
        random_state=42
    )
    
    logger.info(f"Split: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
    
    # Save splits
    split_metadata = {}
    
    for split_name, split_samples in [
        ('train', train_samples),
        ('val', val_samples),
        ('test', test_samples)
    ]:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        split_meta = []
        
        for idx, sample in enumerate(tqdm(split_samples, desc=f"Saving {split_name}")):
            # Create unique filename
            base_name = f"{sample['marker_type']}_{idx:04d}"
            
            # Save image and mask
            np.save(split_dir / f"{base_name}_img.npy", sample['img'])
            np.save(split_dir / f"{base_name}_mask.npy", sample['mask'])
            
            split_meta.append({
                'filename': base_name,
                'marker_type': sample['marker_type'],
                'shape': sample['shape'],
                'original_name': sample['filename']
            })
        
        split_metadata[split_name] = split_meta
    
    # Save metadata
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump({
            'num_samples': len(all_samples),
            'splits': {
                'train': len(train_samples),
                'val': len(val_samples),
                'test': len(test_samples)
            },
            'marker_types': list(set(s['marker_type'] for s in all_samples)),
            'data': split_metadata
        }, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ Processed {len(all_samples)} labeled samples")
    logger.info(f"✅ Metadata saved: {metadata_path}")
    logger.info(f"{'='*80}")
    
    return all_samples


def main():
    args = parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    logger.info("="*80)
    logger.info("SELMA3D DATA PREPARATION (UNIFIED PIPELINE)")
    logger.info("="*80)
    logger.info(f"Input:  {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Type:   {args.data_type}")
    logger.info(f"XY Downsample: {args.downsample_xy}x")
    logger.info(f"Z Downsample:  {args.downsample_z}x")
    logger.info(f"Normalize: {args.normalize}")
    
    # Validate input
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process based on data type
    if args.data_type == 'unlabeled':
        process_unlabeled_data(input_dir, output_dir, args)
    elif args.data_type == 'labeled':
        process_labeled_data(input_dir, output_dir, args)
    else:
        raise ValueError(f"Invalid data_type: {args.data_type}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ DATA PREPARATION COMPLETE!")
    logger.info("="*80)


if __name__ == '__main__':
    main()