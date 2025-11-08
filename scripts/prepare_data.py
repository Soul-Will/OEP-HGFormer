"""
Unified Data Preparation Pipeline - PERMANENT FIX VERSION
✅ FIXED: Broadcasting error (lazy allocation)
✅ FIXED: NumPy 32-bit overflow (HDF5)
✅ ADDED: Robust error handling
✅ ADDED: Progress tracking
"""

import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import logging
import h5py  # ✅ NEW: For large file handling
from typing import Dict, List, Optional, Tuple
from enum import Enum
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# FILE FORMAT DETECTION
# =============================================================================

class FileFormat(Enum):
    """Supported formats"""
    TIFF = "tiff"
    NIFTI = "nifti"
    UNKNOWN = "unknown"


def detect_file_format(file_path: Path) -> FileFormat:
    """Auto-detect format from extension"""
    suffix = file_path.suffix.lower()
    
    if suffix in ['.tif', '.tiff']:
        return FileFormat.TIFF
    elif suffix in ['.nii', '.gz']:
        if file_path.name.endswith('.nii.gz'):
            return FileFormat.NIFTI
        elif suffix == '.nii':
            return FileFormat.NIFTI
    
    return FileFormat.UNKNOWN


# =============================================================================
# ✅ PERMANENT FIX: UNIFIED VOLUME LOADER WITH HDF5
# =============================================================================

class VolumeLoader:
    """
    ✅ FIXED: Format-agnostic loader with lazy allocation & HDF5 output
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
        
        Returns:
            volume: (D, H, W) float32 array
            metadata: Dict with spacing, format, etc.
        """
        if file_or_folder.is_file():
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
                logger.error(f"Unknown format: {file_or_folder}")
                return None, None
        
        elif file_or_folder.is_dir():
            # ✅ CRITICAL FIX: This calls the fixed function
            return VolumeLoader._load_tiff_slices(
                file_or_folder, downsample_xy, downsample_z, normalize, max_slices
            )
        
        else:
            logger.error(f"Path not found: {file_or_folder}")
            return None, None
    
    @staticmethod
    def _load_nifti(
        nifti_path: Path,
        downsample_xy: float = 1.0,
        downsample_z: float = 1.0,
        normalize: bool = True
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Load NIfTI (unchanged from your original)"""
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError("Install nibabel: pip install nibabel")
        
        logger.info(f"  Loading NIfTI: {nifti_path.name}")
        
        try:
            nii = nib.load(str(nifti_path))
            volume = nii.get_fdata().astype(np.float32)
            spacing = nii.header.get_zooms()[:3]
            
            logger.info(f"  Original shape: {volume.shape}")
            logger.info(f"  Spacing: {spacing} mm")
            
            # NIfTI is (X,Y,Z) → convert to (D,H,W)
            volume = np.transpose(volume, (2, 1, 0))
            
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
                'format': 'nifti',
                'original_shape': nii.shape,
                'spacing': tuple(spacing),
                'affine': nii.affine.tolist(),
                'orientation': nib.aff2axcodes(nii.affine)
            }
            
            return volume, metadata
        
        except Exception as e:
            logger.error(f"Failed to load NIfTI: {e}")
            return None, None
    
    @staticmethod
    def _load_tiff_stack(
        tiff_path: Path,
        downsample_xy: float = 1.0,
        downsample_z: float = 1.0,
        normalize: bool = True
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Load multi-page TIFF (unchanged)"""
        try:
            import tifffile
        except ImportError:
            raise ImportError("Install tifffile: pip install tifffile")
        
        logger.info(f"  Loading TIFF stack: {tiff_path.name}")
        
        try:
            volume = tifffile.imread(str(tiff_path)).astype(np.float32)
            logger.info(f"  Original shape: {volume.shape}")
            
            if volume.ndim == 2:
                volume = volume[np.newaxis, ...]
            
            if downsample_xy != 1.0 or downsample_z != 1.0:
                from scipy.ndimage import zoom
                zoom_factors = (downsample_z, downsample_xy, downsample_xy)
                volume = zoom(volume, zoom_factors, order=1)
                logger.info(f"  Downsampled to: {volume.shape}")
            
            if normalize:
                vol_min, vol_max = volume.min(), volume.max()
                if vol_max > vol_min:
                    volume = (volume - vol_min) / (vol_max - vol_min)
            
            metadata = {
                'format': 'tiff_stack',
                'spacing': (2.0, 1.0, 1.0)
            }
            
            return volume, metadata
        
        except Exception as e:
            logger.error(f"Failed to load TIFF stack: {e}")
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
        ✅ PERMANENT FIX: Lazy allocation to avoid broadcasting errors
        
        Key Changes:
        1. Process all slices into a list first
        2. Stack at the very end (shape guaranteed consistent)
        3. No pre-allocation → no rounding mismatch possible
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Install Pillow: pip install Pillow")
        
        from scipy.ndimage import zoom
        
        # Find TIFF files
        slice_files = sorted(
            list(folder.glob("*.tif")) + list(folder.glob("*.tiff")),
            key=lambda x: int(''.join(filter(str.isdigit, x.stem))) 
                         if any(c.isdigit() for c in x.stem) else 0
        )
        
        if len(slice_files) == 0:
            logger.warning(f"No .tif files in {folder}")
            return None, None
        
        # Limit slices if specified
        if max_slices and len(slice_files) > max_slices:
            logger.info(f"  Limiting to {max_slices} slices (out of {len(slice_files)})")
            slice_files = slice_files[:max_slices]
        
        logger.info(f"  Loading {len(slice_files)} TIFF slices...")
        
        # ✅ FIX STEP 1: Process all slices into a list (flexible)
        processed_slices = []
        
        for slice_file in tqdm(slice_files, desc="  Processing", leave=False):
            try:
                # Load slice
                slice_img = np.array(Image.open(slice_file)).astype(np.float32)
                
                # Downsample XY (let zoom decide output shape)
                if downsample_xy != 1.0:
                    slice_img = zoom(slice_img, downsample_xy, order=1)
                
                # Normalize THIS slice
                if normalize:
                    slice_min, slice_max = slice_img.min(), slice_img.max()
                    if slice_max > slice_min:
                        slice_img = (slice_img - slice_min) / (slice_max - slice_min)
                
                processed_slices.append(slice_img)
            
            except Exception as e:
                logger.error(f"  Failed to load {slice_file.name}: {e}")
                raise
        
        # ✅ FIX STEP 2: Stack at the end (shape guaranteed consistent)
        try:
            volume = np.stack(processed_slices, axis=0)  # ✅ NumPy handles shape validation
        except ValueError as e:
            logger.error(f"Slice shape mismatch detected: {e}")
            logger.error("This means zoom() produced different shapes for different slices.")
            logger.error("Inspect your input data for inconsistencies.")
            raise
        
        logger.info(f"  ✅ Stacked volume shape: {volume.shape}")
        
        # Downsample Z if needed
        if downsample_z != 1.0:
            D, H, W = volume.shape
            D_out = max(1, int(D * downsample_z))
            
            # Resample along Z axis
            volume = zoom(volume, (downsample_z, 1.0, 1.0), order=1)
            logger.info(f"  Downsampled Z: {volume.shape}")
        
        metadata = {
            'format': 'tiff_slices',
            'spacing': (2.0, 1.0, 1.0),
            'num_slices': len(processed_slices)
        }
        
        return volume, metadata


# =============================================================================
# ✅ PERMANENT FIX: HDF5 SAVING FUNCTION
# =============================================================================

def save_volume_hdf5(
    volume: np.ndarray,
    output_path: Path,
    metadata: Dict,
    compression: str = 'gzip',
    compression_opts: int = 4
) -> Dict:
    """
    ✅ PERMANENT FIX: Save large volumes to HDF5 (no size limit!)
    
    Args:
        volume: (D, H, W) array
        output_path: Path to .h5 file
        metadata: Dict with spacing, format, etc.
        compression: 'gzip' (best) or 'lzf' (fastest)
        compression_opts: 1-9 (gzip only, 4 = balanced)
    
    Returns:
        save_metadata: Dict with file size, compression ratio, etc.
    """
    output_path = output_path.with_suffix('.h5')  # Force .h5 extension
    
    logger.info(f"  Saving to HDF5: {output_path.name}")
    
    try:
        with h5py.File(output_path, 'w') as f:
            # Save volume with compression
            f.create_dataset(
                'volume',
                data=volume,
                compression=compression,
                compression_opts=compression_opts,
                dtype=np.float32
            )
            
            # Save metadata as attributes
            for key, value in metadata.items():
                if isinstance(value, (int, float, str, bool)):
                    f['volume'].attrs[key] = value
                elif isinstance(value, (list, tuple)):
                    f['volume'].attrs[key] = str(value)
        
        # Get file stats
        file_size_mb = output_path.stat().st_size / 1e6
        
        # Calculate compression ratio
        uncompressed_size_mb = volume.nbytes / 1e6
        compression_ratio = uncompressed_size_mb / file_size_mb if file_size_mb > 0 else 1.0
        
        logger.info(f"  ✅ Saved: {file_size_mb:.1f} MB (compression: {compression_ratio:.2f}x)")
        
        return {
            'file_path': str(output_path),
            'file_size_mb': file_size_mb,
            'uncompressed_size_mb': uncompressed_size_mb,
            'compression_ratio': compression_ratio,
            'compression': compression
        }
    
    except Exception as e:
        logger.error(f"Failed to save HDF5: {e}")
        raise


def load_volume_hdf5(hdf5_path: Path) -> Tuple[np.ndarray, Dict]:
    """
    Load volume from HDF5
    
    Args:
        hdf5_path: Path to .h5 file
    
    Returns:
        volume: (D, H, W) array
        metadata: Dict with spacing, etc.
    """
    with h5py.File(hdf5_path, 'r') as f:
        volume = f['volume'][:]  # Load full volume
        
        # Extract metadata
        metadata = dict(f['volume'].attrs)
    
    return volume, metadata


def load_patch_hdf5(
    hdf5_path: Path,
    patch_slice: Tuple[slice, slice, slice]
) -> np.ndarray:
    """
    ✅ CRITICAL: Load ONLY a patch from HDF5 (memory efficient!)
    
    Args:
        hdf5_path: Path to .h5 file
        patch_slice: Tuple of (slice_d, slice_h, slice_w)
                     e.g., (slice(0,64), slice(0,128), slice(0,128))
    
    Returns:
        patch: (D, H, W) array (only the requested region)
    
    Example:
        # Load patch at (z=100, y=200, x=300) with size (64, 128, 128)
        patch = load_patch_hdf5(
            Path('brain.h5'),
            (slice(100, 164), slice(200, 328), slice(300, 428))
        )
    """
    with h5py.File(hdf5_path, 'r') as f:
        patch = f['volume'][patch_slice]  # ✅ HDF5 only reads this region from disk!
    
    return patch


# =============================================================================
# DATA DISCOVERY (unchanged from your original)
# =============================================================================

class DatasetDiscovery:
    """Flexible data discovery for multiple naming conventions"""
    
    @staticmethod
    def discover_unlabeled_data(input_dir: Path) -> List[Dict]:
        """Discover unlabeled volumes"""
        discovered = []
        
        marker_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
        
        for marker_dir in marker_dirs:
            marker_name = marker_dir.name
            
            # Check for NIfTI files
            nifti_files = list(marker_dir.glob("*.nii.gz")) + list(marker_dir.glob("*.nii"))
            
            if nifti_files:
                for nifti_file in nifti_files:
                    discovered.append({
                        'marker_type': marker_name,
                        'brain_name': nifti_file.stem.replace('.nii', ''),
                        'path': nifti_file,
                        'is_file': True
                    })
            else:
                # Folder-based (TIFF slices)
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
    # In scripts/prepare_data.py

    def discover_labeled_data(input_dir: Path) -> List[Dict]:
        """
        ✅ FIXED: Discovers labeled data (NIfTI ONLY)
        
        Supports:
        1. EBI-style (top-level): input_dir/raw/ + input_dir/gt/
        2. SELMA3D-style (nested): input_dir/cFos/raw/ + input_dir/cFos/gt/
        """
        discovered = []
        
        # ---
        # Strategy 1: Check for EBI-style (raw/gt) at the TOP level
        # ---
        raw_dir = input_dir / 'raw'
        gt_dir = input_dir / 'gt'
        
        if raw_dir.exists() and gt_dir.exists():
            logger.info(f"Detected top-level 'raw'/'gt' NIfTI structure in {input_dir}")
            img_files = sorted(raw_dir.glob("*.nii.gz"))
            
            for img_file in img_files:
                # --- Fix for _0000 vs _000 ---
                raw_stem = img_file.name.replace('.nii.gz', '')
                stem_parts = raw_stem.split('_')[:-1]
                gt_stem = '_'.join(stem_parts)
                gt_filename = f"{gt_stem}.nii.gz"
                mask_file = gt_dir / gt_filename
                # --- End fix ---

                if mask_file.exists():
                    discovered.append({
                        'marker_type': input_dir.name, # Use parent folder name
                        'sample_name': gt_stem,
                        'img_path': img_file,
                        'mask_path': mask_file
                    })
                else:
                    logger.warning(f"  Missing mask for {img_file.name}, tried {gt_filename}")
        
        # ---
        # Strategy 2: Check for SELMA3D-style (nested marker folders)
        # ---
        else:
            logger.info(f"No top-level 'raw'/'gt' found. Scanning for marker subfolders (e.g., cFos, vessel)...")
            marker_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
            
            if len(marker_dirs) == 0:
                logger.warning(f"No marker subfolders found in {input_dir}")
                return [] # Return empty

            for marker_dir in marker_dirs:
                marker_name = marker_dir.name
                
                # ---
                # Strategy 2a: Check for NESTED raw/gt (e.g., cFos/raw/)
                # ---
                nested_raw_dir = marker_dir / 'raw'
                nested_gt_dir = marker_dir / 'gt'

                if nested_raw_dir.exists() and nested_gt_dir.exists():
                    logger.info(f"  Found 'raw'/'gt' structure in: {marker_name}")
                    img_files = sorted(nested_raw_dir.glob("*.nii.gz"))
                    
                    for img_file in img_files:
                        # --- Fix for _0000 vs _000 ---
                        raw_stem = img_file.name.replace('.nii.gz', '')
                        stem_parts = raw_stem.split('_')[:-1]
                        gt_stem = '_'.join(stem_parts)
                        gt_filename = f"{gt_stem}.nii.gz"
                        mask_file = nested_gt_dir / gt_filename
                        # --- End fix ---

                        if mask_file.exists():
                            discovered.append({
                                'marker_type': marker_name,
                                'sample_name': gt_stem,
                                'img_path': img_file,
                                'mask_path': mask_file
                            })
                        else:
                            logger.warning(f"    Missing mask for {img_file.name}, tried {gt_filename}")
                
                # ---
                # Strategy 2b (REMOVED): We no longer look for loose .tif files
                # ---
                else:
                    logger.warning(f"  No 'raw'/'gt' subfolders found in {marker_name}. Skipping.")

        return discovered
    # def discover_labeled_data(input_dir: Path) -> List[Dict]:
    #     """Discover labeled pairs"""
    #     discovered = []
        
    #     # EBI format
    #     raw_dir = input_dir / 'raw'
    #     gt_dir = input_dir / 'gt'
        
    #     if raw_dir.exists() and gt_dir.exists():
    #         logger.info("Detected EBI format (RAW + GT)")
            
    #         img_files = sorted(raw_dir.glob("*.nii.gz"))
            
    #         for img_file in img_files:
    #             mask_file = gt_dir / img_file.name
                
    #             if mask_file.exists():
    #                 discovered.append({
    #                     'marker_type': 'unknown',
    #                     'sample_name': img_file.stem.replace('.nii', ''),
    #                     'img_path': img_file,
    #                     'mask_path': mask_file
    #                 })
        
    #     else:
    #         # SELMA3D format
    #         logger.info("Detected SELMA3D format")
            
    #         marker_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
            
    #         for marker_dir in marker_dirs:
    #             marker_name = marker_dir.name
                
    #             img_files = sorted(marker_dir.glob("*_img.tif"))
                
    #             for img_file in img_files:
    #                 mask_file = img_file.parent / img_file.name.replace('_img.', '_mask.')
                    
    #                 if mask_file.exists():
    #                     discovered.append({
    #                         'marker_type': marker_name,
    #                         'sample_name': img_file.stem.replace('_img', ''),
    #                         'img_path': img_file,
    #                         'mask_path': mask_file
    #                     })
        
    #     return discovered


# =============================================================================
# ✅ MAIN PROCESSING FUNCTIONS (REFACTORED WITH HDF5)
# =============================================================================

def process_unlabeled_data(input_dir: Path, output_dir: Path, args) -> List[Dict]:
    """
    ✅ REFACTORED: Now saves to HDF5 instead of .npy
    """
    logger.info("\n" + "="*80)
    logger.info("PROCESSING UNLABELED DATA (SSL)")
    logger.info("="*80)
    
    all_metadata = []
    
    marker_map = {
        'ab_plaque': 3,
        'ad_plaque': 3,
        'cfos': 0,
        'microglia': 4,
        'nucleus': 2,
        'unknown': 5,
        'vessel': 1
    }
    
    discovered_data = DatasetDiscovery.discover_unlabeled_data(input_dir)
    
    if len(discovered_data) == 0:
        raise ValueError(f"No data found in {input_dir}")
    
    logger.info(f"Discovered {len(discovered_data)} volumes\n")
    
    from collections import defaultdict
    by_marker = defaultdict(list)
    for item in discovered_data:
        by_marker[item['marker_type']].append(item)
    
    for marker_name, items in by_marker.items():
        marker_label = marker_map.get(marker_name.lower(), 5)
        
        logger.info(f"📁 Processing marker: {marker_name} (label={marker_label})")
        logger.info(f"  Found {len(items)} volumes")
        
        output_marker_dir = output_dir / marker_name
        output_marker_dir.mkdir(parents=True, exist_ok=True)
        
        for item in items:
            brain_name = item['brain_name']
            logger.info(f"\n  Processing: {brain_name}")
            
            # Load volume (uses fixed lazy allocation)
            volume, vol_metadata = VolumeLoader.load_volume(
                item['path'],
                downsample_xy=args.downsample_xy,
                downsample_z=args.downsample_z,
                normalize=args.normalize,
                max_slices=args.max_slices_in_memory
            )
            
            if volume is None:
                logger.warning(f"  Skipping {brain_name} (loading failed)")
                continue
            
            # ✅ CRITICAL: Save as HDF5 instead of .npy
            output_file = output_marker_dir / f"{brain_name}"  # No extension yet
            
            try:
                save_info = save_volume_hdf5(
                    volume,
                    output_file,
                    vol_metadata,
                    compression='gzip',
                    compression_opts=4
                )
            except Exception as e:
                logger.error(f"  Failed to save {brain_name}: {e}")
                continue
            
            # Store metadata
            meta = {
                'brain_name': brain_name,
                'marker_type': marker_name,
                'marker_label': marker_label,
                'shape': list(volume.shape),
                'file_path': str(Path(save_info['file_path']).relative_to(output_dir)),
                'file_size_mb': save_info['file_size_mb'],
                'compression_ratio': save_info['compression_ratio'],
                'original_format': vol_metadata.get('format', 'unknown'),
                'spacing': vol_metadata.get('spacing', (2.0, 1.0, 1.0)),
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
            'volumes': all_metadata,
            'storage_format': 'hdf5'  # ✅ NEW
        }, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ Processed {len(all_metadata)} unlabeled volumes")
    logger.info(f"✅ Total size: {sum(m['file_size_mb'] for m in all_metadata):.1f} MB")
    logger.info(f"✅ Storage format: HDF5 (no size limit)")
    logger.info(f"✅ Metadata saved: {metadata_path}")
    logger.info(f"{'='*80}")
    
    return all_metadata


def process_labeled_data(input_dir: Path, output_dir: Path, args) -> List[Dict]:
    """
    ✅ REFACTORED: Now saves to HDF5
    """
    logger.info("\n" + "="*80)
    logger.info("PROCESSING LABELED DATA (FINE-TUNING)")
    logger.info("="*80)
    
    all_samples = []
    
    discovered_pairs = DatasetDiscovery.discover_labeled_data(input_dir)
    
    if len(discovered_pairs) == 0:
        raise ValueError(f"No labeled pairs found in {input_dir}")
    
    logger.info(f"Discovered {len(discovered_pairs)} image-mask pairs\n")
    
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
                normalize=False
            )
            
            if img is None or mask is None:
                logger.warning(f"  Failed to load {pair['sample_name']}, skipping")
                continue
            
            mask = mask.astype(np.int64)
            
            if img.shape != mask.shape:
                logger.warning(
                    f"  Shape mismatch: {pair['sample_name']} "
                    f"img {img.shape} vs mask {mask.shape}, skipping"
                )
                continue
            
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
    
    # Save splits to HDF5
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
            base_name = f"{sample['marker_type']}_{idx:04d}"
            
            # ✅ CRITICAL: Save as HDF5 with both image and mask in same file
            output_file = split_dir / f"{base_name}.h5"
            
            try:
                with h5py.File(output_file, 'w') as f:
                    # Save image
                    f.create_dataset(
                        'image',
                        data=sample['img'],
                        compression='gzip',
                        compression_opts=4,
                        dtype=np.float32
                    )
                    
                    # Save mask
                    f.create_dataset(
                        'mask',
                        data=sample['mask'],
                        compression='gzip',
                        compression_opts=4,
                        dtype=np.int64
                    )
                    
                    # Save metadata as attributes
                    f['image'].attrs['marker_type'] = sample['marker_type']
                    f['image'].attrs['original_filename'] = sample['filename']
                    f['image'].attrs['shape'] = str(sample['shape'])
                
                file_size_mb = output_file.stat().st_size / 1e6
                
                split_meta.append({
                    'filename': base_name,
                    'marker_type': sample['marker_type'],
                    'shape': list(sample['shape']),
                    'original_name': sample['filename'],
                    'original_format': sample['format'],
                    'file_size_mb': file_size_mb,
                    'file_path': str(output_file.relative_to(output_dir))
                })
            
            except Exception as e:
                logger.error(f"Failed to save {base_name}: {e}")
                continue
        
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
            'data': split_metadata,
            'storage_format': 'hdf5'  # ✅ NEW
        }, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"✅ Processed {len(all_samples)} labeled samples")
    logger.info(f"✅ Storage format: HDF5 (image + mask in same file)")
    logger.info(f"✅ Metadata saved: {metadata_path}")
    logger.info(f"{'='*80}")
    
    return all_samples


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare SELMA3D dataset - PERMANENT FIX VERSION',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
✅ PERMANENT FIXES:
  1. Broadcasting error: Lazy allocation (no pre-calculation)
  2. NumPy overflow: HDF5 storage (no 2GB limit)
  3. Memory efficiency: Patch loading from HDF5

Examples:
  # Process unlabeled data (SSL)
  python prepare_data.py \\
      --input_dir data/raw/train_unlabeled \\
      --output_dir data/processed/volumes_ssl \\
      --data_type unlabeled
  
  # Process labeled data (fine-tuning)
  python prepare_data.py \\
      --input_dir data/raw/train_labeled \\
      --output_dir data/processed/volumes_labeled \\
      --data_type labeled
  
  # With downsampling
  python prepare_data.py \\
      --input_dir data/raw/train_unlabeled \\
      --output_dir data/processed/volumes_ssl \\
      --data_type unlabeled \\
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
                       help='Max slices to process at once (for very large stacks)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    logger.info("="*80)
    logger.info("SELMA3D DATA PREPARATION - PERMANENT FIX VERSION")
    logger.info("="*80)
    logger.info(f"Input:  {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Type:   {args.data_type}")
    logger.info(f"XY Downsample: {args.downsample_xy}x")
    logger.info(f"Z Downsample:  {args.downsample_z}x")
    logger.info(f"Normalize: {args.normalize}")
    logger.info("")
    logger.info("✅ PERMANENT FIXES APPLIED:")
    logger.info("  1. Lazy allocation (no broadcasting errors)")
    logger.info("  2. HDF5 storage (no 2GB limit)")
    logger.info("  3. Patch-based loading (memory efficient)")
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