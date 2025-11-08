"""
Unified Data Preparation Pipeline - Format Agnostic
✅ FIXED: Supports TIFF (.tif) and NIfTI (.nii.gz) formats
✅ FIXED: Auto-detects file format
✅ FIXED: Preserves metadata (spacing, orientation)
✅ FIXED: Handles multiple naming conventions
"""

import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import logging
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import warnings

# Suppress PIL warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# SUPPORTED FILE FORMATS
# =============================================================================

class FileFormat(Enum):
    """Supported medical imaging formats"""
    TIFF = "tiff"
    NIFTI = "nifti"
    UNKNOWN = "unknown"


def detect_file_format(file_path: Path) -> FileFormat:
    """
    Auto-detect file format from extension
    
    Args:
        file_path: Path to file
    
    Returns:
        FileFormat enum
    """
    suffix = file_path.suffix.lower()
    
    if suffix in ['.tif', '.tiff']:
        return FileFormat.TIFF
    elif suffix in ['.nii', '.gz']:
        # Check for .nii.gz
        if file_path.name.endswith('.nii.gz'):
            return FileFormat.NIFTI
        elif suffix == '.nii':
            return FileFormat.NIFTI
        else:
            return FileFormat.UNKNOWN
    else:
        return FileFormat.UNKNOWN


# =============================================================================
# UNIFIED VOLUME LOADER
# =============================================================================

class VolumeLoader:
    """
    ✅ PERMANENT FIX: Format-agnostic 3D volume loader
    
    Supports:
    - TIFF: 2D slices or 3D stacks
    - NIfTI: 3D volumes (.nii or .nii.gz)
    
    Preserves metadata (spacing, orientation)
    """
    
    @staticmethod
    def load_volume(
        file_or_folder: Path,
        downsample_xy: float = 1.0,
        downsample_z: float = 1.0,
        normalize: bool = True,
        max_slices: Optional[int] = None
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Load 3D volume from TIFF or NIfTI
        
        Args:
            file_or_folder: Path to single file (.nii.gz) or folder (2D .tif slices)
            downsample_xy: XY downsampling factor
            downsample_z: Z downsampling factor
            normalize: Whether to normalize intensities
            max_slices: Max slices to load (for memory constraints)
        
        Returns:
            volume: (D, H, W) numpy array or None if loading failed
            metadata: Dict with spacing, orientation, etc.
        """
        if file_or_folder.is_file():
            # Single file: Could be NIfTI or multi-page TIFF
            file_format = detect_file_format(file_or_folder)
            
            if file_format == FileFormat.NIFTI:
                return VolumeLoader._load_nifti(
                    file_or_folder, downsample_xy, downsample_z, normalize
                )
            elif file_format == FileFormat.TIFF:
                return VolumeLoader._load_tiff_stack(
                    file_or_folder, downsample_xy, downsample_z, normalize
                )
            else:
                logger.error(f"Unknown file format: {file_or_folder}")
                return None, None
        
        elif file_or_folder.is_dir():
            # Folder of 2D TIFF slices (original SELMA3D format)
            return VolumeLoader._load_tiff_slices(
                file_or_folder, downsample_xy, downsample_z, normalize, max_slices
            )
        
        else:
            logger.error(f"Path does not exist: {file_or_folder}")
            return None, None
    
    @staticmethod
    def _load_nifti(
        nifti_path: Path,
        downsample_xy: float = 1.0,
        downsample_z: float = 1.0,
        normalize: bool = True
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Load NIfTI file (.nii or .nii.gz)
        
        Returns:
            volume: (D, H, W) array
            metadata: Dict with spacing, affine, etc.
        """
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError(
                "nibabel is required to load NIfTI files.\n"
                "Install with: pip install nibabel"
            )
        
        logger.info(f"  Loading NIfTI: {nifti_path.name}")
        
        try:
            # Load NIfTI file
            nii = nib.load(str(nifti_path))
            volume = nii.get_fdata().astype(np.float32)
            
            # Get spacing from header
            spacing = nii.header.get_zooms()[:3]  # (z, y, x) in mm
            
            logger.info(f"  Original shape: {volume.shape}")
            logger.info(f"  Spacing: {spacing} mm")
            
            # NIfTI is (X, Y, Z) but we want (D, H, W) = (Z, Y, X)
            volume = np.transpose(volume, (2, 1, 0))
            
            # Downsample if needed
            if downsample_xy != 1.0 or downsample_z != 1.0:
                from scipy.ndimage import zoom
                zoom_factors = (downsample_z, downsample_xy, downsample_xy)
                volume = zoom(volume, zoom_factors, order=1)
                logger.info(f"  Downsampled to: {volume.shape}")
            
            # Normalize
            if normalize:
                vol_min, vol_max = volume.min(), volume.max()
                if vol_max > vol_min:
                    volume = (volume - vol_min) / (vol_max - vol_min)
            
            # Metadata
            metadata = {
                'format': 'nifti',
                'original_shape': nii.shape,
                'spacing': tuple(spacing),
                'affine': nii.affine.tolist(),
                'orientation': nib.aff2axcodes(nii.affine)
            }
            
            return volume, metadata
        
        except Exception as e:
            logger.error(f"Failed to load NIfTI {nifti_path}: {e}")
            return None, None
    
    @staticmethod
    def _load_tiff_stack(
        tiff_path: Path,
        downsample_xy: float = 1.0,
        downsample_z: float = 1.0,
        normalize: bool = True
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Load multi-page TIFF stack
        
        Returns:
            volume: (D, H, W) array
            metadata: Dict with spacing
        """
        try:
            import tifffile
        except ImportError:
            raise ImportError(
                "tifffile is required to load TIFF files.\n"
                "Install with: pip install tifffile"
            )
        
        logger.info(f"  Loading TIFF stack: {tiff_path.name}")
        
        try:
            volume = tifffile.imread(str(tiff_path)).astype(np.float32)
            
            logger.info(f"  Original shape: {volume.shape}")
            
            # Ensure 3D
            if volume.ndim == 2:
                volume = volume[np.newaxis, ...]
            
            # Downsample
            if downsample_xy != 1.0 or downsample_z != 1.0:
                from scipy.ndimage import zoom
                zoom_factors = (downsample_z, downsample_xy, downsample_xy)
                volume = zoom(volume, zoom_factors, order=1)
                logger.info(f"  Downsampled to: {volume.shape}")
            
            # Normalize
            if normalize:
                vol_min, vol_max = volume.min(), volume.max()
                if vol_max > vol_min:
                    volume = (volume - vol_min) / (vol_max - vol_min)
            
            metadata = {
                'format': 'tiff_stack',
                'spacing': (2.0, 1.0, 1.0)  # Default LSM spacing
            }
            
            return volume, metadata
        
        except Exception as e:
            logger.error(f"Failed to load TIFF stack {tiff_path}: {e}")
            return None, None
    
    @staticmethod
    def _load_tiff_slices(
        folder: Path,
        downsample_xy: float = 1.0,
        downsample_z: float = 1.0,
        normalize: bool = True,
        max_slices: Optional[int] = None
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Load folder of 2D TIFF slices (original SELMA3D format)
        
        Returns:
            volume: (D, H, W) array
            metadata: Dict with spacing
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError(
                "Pillow is required to load TIFF slices.\n"
                "Install with: pip install Pillow"
            )
        
        # Find all TIFF files
        slice_files = sorted(
            list(folder.glob("*.tif")) + list(folder.glob("*.tiff")),
            key=lambda x: int(''.join(filter(str.isdigit, x.stem))) 
                         if any(c.isdigit() for c in x.stem) else 0
        )
        
        if len(slice_files) == 0:
            logger.warning(f"No .tif files found in {folder}")
            return None, None
        
        # Limit slices if specified
        if max_slices and len(slice_files) > max_slices:
            logger.info(f"  Limiting to {max_slices} slices (out of {len(slice_files)})")
            slice_files = slice_files[:max_slices]
        
        logger.info(f"  Loading {len(slice_files)} TIFF slices...")
        
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
        
        # Preallocate volume
        volume = np.zeros((D_out, H_out, W_out), dtype=np.float32)
        
        # Load and process slices
        for i, z_idx in enumerate(tqdm(z_indices, desc="  Stacking", leave=False)):
            try:
                slice_img = np.array(Image.open(slice_files[z_idx])).astype(np.float32)
                
                # Downsample XY
                if downsample_xy != 1.0:
                    from scipy.ndimage import zoom
                    slice_img = zoom(slice_img, downsample_xy, order=1)
                
                # Normalize
                if normalize:
                    slice_min, slice_max = slice_img.min(), slice_img.max()
                    if slice_max > slice_min:
                        slice_img = (slice_img - slice_min) / (slice_max - slice_min)
                
                volume[i] = slice_img
            
            except Exception as e:
                logger.error(f"  Failed to load slice {z_idx}: {e}")
                raise
        
        metadata = {
            'format': 'tiff_slices',
            'spacing': (2.0, 1.0, 1.0)
        }
        
        return volume, metadata


# =============================================================================
# DATA DISCOVERY: FLEXIBLE NAMING CONVENTIONS
# =============================================================================

class DatasetDiscovery:
    """
    ✅ PERMANENT FIX: Flexible data discovery for multiple naming conventions
    
    Supports:
    - SELMA3D Challenge: marker_folders/brain_folders/*.tif
    - EBI BioStudies: RAW/*.nii.gz + GT/*.nii.gz
    - Custom formats
    """
    
    @staticmethod
    def discover_unlabeled_data(input_dir: Path) -> List[Dict]:
        """
        Discover unlabeled data with flexible naming
        
        Supports:
        1. Folder-based: marker_type/brain_name/*.tif
        2. File-based: marker_type/brain_name.nii.gz
        
        Returns:
            List of dicts with 'marker_type', 'brain_name', 'path'
        """
        discovered = []
        
        # Find all subdirectories (marker types)
        marker_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
        
        for marker_dir in marker_dirs:
            marker_name = marker_dir.name
            
            # Check for NIfTI files
            nifti_files = list(marker_dir.glob("*.nii.gz")) + list(marker_dir.glob("*.nii"))
            
            if nifti_files:
                # File-based format (e.g., EBI)
                for nifti_file in nifti_files:
                    discovered.append({
                        'marker_type': marker_name,
                        'brain_name': nifti_file.stem.replace('.nii', ''),
                        'path': nifti_file,
                        'is_file': True
                    })
            else:
                # Folder-based format (e.g., SELMA3D)
                brain_folders = [d for d in marker_dir.iterdir() if d.is_dir()]
                
                for brain_folder in brain_folders:
                    discovered.append({
                        'marker_type': marker_name,
                        'brain_name': brain_folder.name,
                        'path': brain_folder,
                        'is_file': False
                    })
        
        return discovered
    
    @staticmethod
    def discover_labeled_data(input_dir: Path) -> List[Dict]:
        """
        Discover labeled data with flexible naming
        
        Supports:
        1. SELMA3D: marker/patch_XXX_img.tif + patch_XXX_mask.tif
        2. EBI: RAW/patchvolume_XXX.nii.gz + GT/patchvolume_XXX.nii.gz
        
        Returns:
            List of dicts with 'marker_type', 'sample_name', 'img_path', 'mask_path'
        """
        discovered = []
        
        # Check for EBI-style structure (RAW + GT folders)
        raw_dir = input_dir / 'RAW'
        gt_dir = input_dir / 'GT'
        
        if raw_dir.exists() and gt_dir.exists():
            # EBI format
            logger.info("Detected EBI BioStudies format (RAW + GT)")
            
            img_files = sorted(raw_dir.glob("*.nii.gz"))
            
            for img_file in img_files:
                # Find corresponding mask in GT folder
                mask_file = gt_dir / img_file.name
                
                if mask_file.exists():
                    discovered.append({
                        'marker_type': 'unknown',  # No marker type in EBI format
                        'sample_name': img_file.stem.replace('.nii', ''),
                        'img_path': img_file,
                        'mask_path': mask_file
                    })
                else:
                    logger.warning(f"Missing mask for {img_file.name}")
        
        else:
            # SELMA3D format
            logger.info("Detected SELMA3D Challenge format (marker folders)")
            
            marker_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
            
            for marker_dir in marker_dirs:
                marker_name = marker_dir.name
                
                # Check for TIFF pairs
                img_files = sorted(marker_dir.glob("*_img.tif"))
                
                for img_file in img_files:
                    mask_file = img_file.parent / img_file.name.replace('_img.', '_mask.')
                    
                    if mask_file.exists():
                        discovered.append({
                            'marker_type': marker_name,
                            'sample_name': img_file.stem.replace('_img', ''),
                            'img_path': img_file,
                            'mask_path': mask_file
                        })
                    else:
                        logger.warning(f"Missing mask for {img_file.name}")
        
        return discovered


# =============================================================================
# MAIN PROCESSING FUNCTIONS (Refactored)
# =============================================================================

def process_unlabeled_data(input_dir: Path, output_dir: Path, args) -> List[Dict]:
    """
    ✅ REFACTORED: Format-agnostic unlabeled data processing
    """
    logger.info("\n" + "="*80)
    logger.info("PROCESSING UNLABELED DATA (SSL)")
    logger.info("="*80)
    
    all_metadata = []
    
    # Marker type mapping
    marker_map = {
        'cfos': 0,
        'vessel': 1,
        'nucleus': 2,
        'ab_plaque': 3,
        'ad_plaque': 3,
        'microglia': 4,
        'unknown': 5
    }
    
    # ✅ CRITICAL: Flexible data discovery
    discovered_data = DatasetDiscovery.discover_unlabeled_data(input_dir)
    
    if len(discovered_data) == 0:
        raise ValueError(f"No data found in {input_dir}")
    
    logger.info(f"Discovered {len(discovered_data)} volumes\n")
    
    # Group by marker type for logging
    from collections import defaultdict
    by_marker = defaultdict(list)
    for item in discovered_data:
        by_marker[item['marker_type']].append(item)
    
    for marker_name, items in by_marker.items():
        marker_label = marker_map.get(marker_name.lower(), 5)
        
        logger.info(f"📁 Processing marker: {marker_name} (label={marker_label})")
        logger.info(f"  Found {len(items)} volumes")
        
        # Create output directory
        output_marker_dir = output_dir / marker_name
        output_marker_dir.mkdir(parents=True, exist_ok=True)
        
        for item in items:
            brain_name = item['brain_name']
            logger.info(f"\n  Processing: {brain_name}")
            
            # ✅ CRITICAL: Use unified loader
            volume, metadata = VolumeLoader.load_volume(
                item['path'],
                downsample_xy=args.downsample_xy,
                downsample_z=args.downsample_z,
                normalize=args.normalize,
                max_slices=args.max_slices_in_memory
            )
            
            if volume is None:
                logger.warning(f"  Skipping {brain_name} (loading failed)")
                continue
            
            # Save as .npy
            output_file = output_marker_dir / f"{brain_name}.npy"
            np.save(output_file, volume)
            
            file_size_mb = output_file.stat().st_size / 1e6
            logger.info(f"  ✅ Saved: {output_file.name} ({file_size_mb:.1f} MB)")
            
            # Store metadata
            meta = {
                'brain_name': brain_name,
                'marker_type': marker_name,
                'marker_label': marker_label,
                'shape': list(volume.shape),
                'file_path': str(output_file.relative_to(output_dir)),
                'file_size_mb': file_size_mb,
                'original_format': metadata.get('format', 'unknown'),
                'spacing': metadata.get('spacing', (2.0, 1.0, 1.0)),
                'processing': {
                    'downsample_xy': args.downsample_xy,
                    'downsample_z': args.downsample_z,
                    'normalized': args.normalize
                }
            }
            all_metadata.append(meta)
    
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
    ✅ REFACTORED: Format-agnostic labeled data processing
    """
    logger.info("\n" + "="*80)
    logger.info("PROCESSING LABELED DATA (FINE-TUNING)")
    logger.info("="*80)
    
    all_samples = []
    
    # ✅ CRITICAL: Flexible data discovery
    discovered_pairs = DatasetDiscovery.discover_labeled_data(input_dir)
    
    if len(discovered_pairs) == 0:
        raise ValueError(f"No labeled pairs found in {input_dir}")
    
    logger.info(f"Discovered {len(discovered_pairs)} image-mask pairs\n")
    
    # Process each pair
    for pair in tqdm(discovered_pairs, desc="Processing samples"):
        try:
            # Load image
            img, img_meta = VolumeLoader.load_volume(
                pair['img_path'],
                downsample_xy=args.downsample_xy,
                downsample_z=args.downsample_z,
                normalize=args.normalize
            )
            
            # Load mask
            mask, mask_meta = VolumeLoader.load_volume(
                pair['mask_path'],
                downsample_xy=args.downsample_xy,
                downsample_z=args.downsample_z,
                normalize=False  # Don't normalize masks!
            )
            
            if img is None or mask is None:
                logger.warning(f"  Failed to load {pair['sample_name']}, skipping")
                continue
            
            # Ensure mask is integer
            mask = mask.astype(np.int64)
            
            # Validate shapes
            if img.shape != mask.shape:
                logger.warning(
                    f"  Shape mismatch: {pair['sample_name']} "
                    f"img {img.shape} vs mask {mask.shape}, skipping"
                )
                continue
            
            # Store
            all_samples.append({
                'img': img,
                'mask': mask,
                'marker_type': pair['marker_type'],
                'filename': pair['sample_name'],
                'shape': img.shape,
                'format': img_meta.get('format', 'unknown')
            })
        
        except Exception as e:
            logger.error(f"  Failed to process {pair['sample_name']}: {e}")
            continue
    
    if len(all_samples) == 0:
        raise ValueError("No valid samples processed!")
    
    logger.info(f"\nTotal valid samples: {len(all_samples)}")
    
    # Stratified split
    from sklearn.model_selection import train_test_split
    
    marker_types = [s['marker_type'] for s in all_samples]
    
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
                'shape': list(sample['shape']),
                'original_name': sample['filename'],
                'original_format': sample['format']
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


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare SELMA3D dataset (labeled + unlabeled) - Format Agnostic',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process SELMA3D Challenge unlabeled data (TIFF slices)
  python prepare_data.py \\
      --input_dir data/raw/train_unlabeled \\
      --output_dir data/processed/volumes_ssl \\
      --data_type unlabeled
  
  # Process EBI BioStudies labeled data (NIfTI)
  python prepare_data.py \\
      --input_dir data/raw/S-BIAD1196/Files \\
      --output_dir data/processed/volumes_labeled \\
      --data_type labeled
  
  # With downsampling
  python prepare_data.py \\
      --input_dir data/raw/train_labeled \\
      --output_dir data/processed/volumes_labeled \\
      --data_type labeled \\
      --downsample_xy 0.5 \\
      --downsample_z 0.5
        """
    )
    
    # Input/Output
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Path to raw data (any format)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Path to save processed data')
    parser.add_argument('--data_type', type=str, required=True,
                       choices=['unlabeled', 'labeled'],
                       help='Type of data to process')
    
    # Processing options
    parser.add_argument('--downsample_xy', type=float, default=1.0,
                       help='XY downsampling factor (default: 1.0 = no downsampling)')
    parser.add_argument('--downsample_z', type=float, default=1.0,
                       help='Z downsampling factor (default: 1.0 = no downsampling)')
    parser.add_argument('--normalize', action='store_true', default=True,
                       help='Normalize intensities to [0, 1]')
    parser.add_argument('--no-normalize', dest='normalize', action='store_false',
                       help='Disable normalization')
    
    # Memory optimization
    parser.add_argument('--max_slices_in_memory', type=int, default=None,
                       help='Max slices to load at once (for large TIFF stacks)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    logger.info("="*80)
    logger.info("SELMA3D DATA PREPARATION - FORMAT AGNOSTIC")
    logger.info("="*80)
    logger.info(f"Input:  {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Type:   {args.data_type}")
    logger.info(f"XY Downsample: {args.downsample_xy}x")
    logger.info(f"Z Downsample:  {args.downsample_z}x")
    logger.info(f"Normalize: {args.normalize}")
    logger.info("")
    logger.info("Supported formats:")
    logger.info("  ✅ TIFF (.tif, .tiff) - 2D slices or 3D stacks")
    logger.info("  ✅ NIfTI (.nii, .nii.gz) - 3D volumes")
    logger.info("="*80)
    
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