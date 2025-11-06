"""
Unified Data Loaders with Metadata-Driven Discovery
✅ FIXED: Reads from metadata.json (no string parsing!)
✅ FIXED: Fast .npy loading (no .tif bottleneck!)
✅ FIXED: Dynamic patch sampling (no overfitting!)
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import json
import random
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class VolumeDataset3D(Dataset):
    """
    ✅ COMPLETELY FIXED: SSL Dataset with Metadata-Driven Discovery
    
    Key Features:
    - Reads metadata.json for volume paths and labels (NO STRING PARSING!)
    - Loads fast .npy files (NOT slow .tif slices!)
    - Dynamic random patch sampling (different patches each epoch!)
    - Memory-efficient (loads volumes on-demand)
    
    Usage:
        dataset = VolumeDataset3D(
            data_dir="data/processed/volumes_ssl",
            patch_size=(64, 128, 128),
            num_patches_per_epoch=1000
        )
    """
    
    def __init__(
        self,
        data_dir: str,
        patch_size: Tuple[int, int, int] = (64, 128, 128),
        num_patches_per_epoch: int = 1000,
        transform: Optional[callable] = None,
        config: Optional[Dict] = None,
        preload: bool = False
    ):
        """
        Args:
            data_dir: Root directory with processed volumes and metadata.json
            patch_size: (D, H, W) patch dimensions
            num_patches_per_epoch: Virtual dataset size (patches per epoch)
            transform: Optional transform (e.g., torchio.Compose)
            config: Optional configuration dict
            preload: If True, load all volumes into RAM (only for small datasets!)
        """
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.num_patches_per_epoch = num_patches_per_epoch
        self.transform = transform
        self.config = config or {}
        
        # ✅ CRITICAL FIX: Load metadata.json
        metadata_path = self.data_dir / 'metadata.json'
        
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"metadata.json not found in {self.data_dir}\n"
                f"Did you run: python scripts/prepare_data.py --data_type unlabeled?"
            )
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Extract volume information
        self.volumes = []
        for vol_meta in metadata['volumes']:
            # Construct full path
            vol_path = self.data_dir / vol_meta['file_path']
            
            if not vol_path.exists():
                logger.warning(f"Volume not found: {vol_path}, skipping")
                continue
            
            self.volumes.append({
                'path': vol_path,
                'marker_label': vol_meta['marker_label'],
                'marker_type': vol_meta['marker_type'],
                'shape': tuple(vol_meta['shape']),
                'brain_name': vol_meta['brain_name']
            })
        
        if len(self.volumes) == 0:
            raise ValueError(f"No valid volumes found in {self.data_dir}")
        
        logger.info(f"VolumeDataset3D initialized:")
        logger.info(f"  Volumes: {len(self.volumes)}")
        logger.info(f"  Patch size: {patch_size}")
        logger.info(f"  Patches per epoch: {num_patches_per_epoch}")
        
        # Count samples per marker type
        marker_counts = {}
        for vol in self.volumes:
            marker = vol['marker_type']
            marker_counts[marker] = marker_counts.get(marker, 0) + 1
        
        logger.info(f"  Marker distribution: {marker_counts}")
        
        # Optionally preload (only for small datasets)
        if preload:
            logger.warning("Preloading all volumes into RAM...")
            self.volume_cache = {}
            for vol_info in self.volumes:
                self.volume_cache[str(vol_info['path'])] = np.load(vol_info['path'])
        else:
            self.volume_cache = None
    
    def __len__(self) -> int:
        """
        ✅ FIXED: Return virtual length for dynamic sampling
        """
        return self.num_patches_per_epoch
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ✅ FIXED: Dynamic random patch sampling
        
        Returns:
            patch: (1, D, H, W) float32 tensor
            marker_label: int64 tensor (for weak supervision)
        """
        # 1. Randomly select a volume
        vol_info = random.choice(self.volumes)
        
        # 2. Load volume (from cache or disk)
        if self.volume_cache is not None:
            volume = self.volume_cache[str(vol_info['path'])]
        else:
            try:
                volume = np.load(vol_info['path']).astype(np.float32)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load volume {vol_info['path']}: {e}\n"
                    f"Idx: {idx}, Volume info: {vol_info}"
                )
        
        # 3. Validate volume shape
        D, H, W = volume.shape
        pd, ph, pw = self.patch_size
        
        if D < pd or H < ph or W < pw:
            raise ValueError(
                f"Patch size {self.patch_size} too large for volume "
                f"{volume.shape} from {vol_info['brain_name']}"
            )
        
        # 4. ✅ CRITICAL: Random patch coordinates (changes every call!)
        d_start = random.randint(0, D - pd)
        h_start = random.randint(0, H - ph)
        w_start = random.randint(0, W - pw)
        
        # Extract patch
        patch = volume[
            d_start:d_start + pd,
            h_start:h_start + ph,
            w_start:w_start + pw
        ].copy()  # Copy to avoid memory issues
        
        # 5. Normalize
        patch_mean = patch.mean()
        patch_std = patch.std()
        if patch_std > 1e-8:
            patch = (patch - patch_mean) / patch_std
        
        # 6. Convert to tensor
        patch = torch.from_numpy(patch).unsqueeze(0)  # (1, D, H, W)
        
        # 7. Apply transforms if provided
        if self.transform is not None:
            try:
                patch = self.transform(patch)
            except Exception as e:
                logger.error(f"Transform failed: {e}")
                # Continue without transform rather than crashing
        
        # 8. Get marker label
        marker_label = torch.tensor(vol_info['marker_label'], dtype=torch.long)
        
        return patch, marker_label


class SELMA3DDataset(Dataset):
    """
    ✅ ENHANCED: Fine-tuning Dataset with Metadata-Driven Discovery
    
    Key Features:
    - Reads metadata.json for sample paths
    - Proper synchronized augmentations (uses torchio)
    - Extensive validation
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[callable] = None,
        config: Optional[Dict] = None
    ):
        """
        Args:
            data_dir: Root directory containing train/val/test splits
            split: 'train', 'val', or 'test'
            transform: torchio.Compose transform (handles image+mask sync)
            config: Configuration dict
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.config = config or {}
        
        # ✅ CRITICAL FIX: Load metadata.json
        metadata_path = self.data_dir / 'metadata.json'
        
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"metadata.json not found in {self.data_dir}\n"
                f"Did you run: python scripts/prepare_data.py --data_type labeled?"
            )
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Validate split exists
        if split not in metadata['data']:
            raise ValueError(
                f"Split '{split}' not found in metadata. "
                f"Available splits: {list(metadata['data'].keys())}"
            )
        
        # Extract samples for this split
        split_data = metadata['data'][split]
        
        self.samples = []
        split_dir = self.data_dir / split
        
        for sample_meta in split_data:
            img_path = split_dir / f"{sample_meta['filename']}_img.npy"
            mask_path = split_dir / f"{sample_meta['filename']}_mask.npy"
            
            # Validate files exist
            if not img_path.exists() or not mask_path.exists():
                logger.warning(f"Files not found for {sample_meta['filename']}, skipping")
                continue
            
            self.samples.append({
                'image': img_path,
                'mask': mask_path,
                'filename': sample_meta['filename'],
                'marker_type': sample_meta['marker_type'],
                'shape': tuple(sample_meta['shape'])
            })
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {split} split!")
        
        logger.info(f"SELMA3DDataset ({split}) initialized:")
        logger.info(f"  Samples: {len(self.samples)}")
        
        # Count samples per marker type
        marker_counts = {}
        for sample in self.samples:
            marker = sample['marker_type']
            marker_counts[marker] = marker_counts.get(marker, 0) + 1
        
        logger.info(f"  Marker distribution: {marker_counts}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Returns:
            image: (1, D, H, W) float32 tensor
            mask: (D, H, W) int64 tensor
            metadata: Dict with filename, marker_type, etc.
        """
        sample_info = self.samples[idx]
        
        # Load image and mask
        try:
            image = np.load(sample_info['image']).astype(np.float32)
            mask = np.load(sample_info['mask']).astype(np.int64)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load sample {idx} ({sample_info['filename']}): {e}"
            )
        
        # Validate shapes
        if image.shape != mask.shape:
            raise ValueError(
                f"Shape mismatch in {sample_info['filename']}: "
                f"image {image.shape} vs mask {mask.shape}"
            )
        
        # Store original shape
        original_shape = image.shape
        
        # ✅ CRITICAL: Use torchio.Subject for synchronized transforms
        if self.transform is not None:
            try:
                import torchio as tio
                
                subject = tio.Subject(
                    image=tio.ScalarImage(tensor=image[None, ...]),  # (1, D, H, W)
                    mask=tio.LabelMap(tensor=mask[None, ...])
                )
                
                transformed = self.transform(subject)
                
                image = transformed.image.data  # (1, D, H, W)
                mask = transformed.mask.data.squeeze(0).long()  # (D, H, W)
                
            except Exception as e:
                raise RuntimeError(
                    f"Transform failed on sample {idx}: {e}\n"
                    f"Image shape: {image.shape}, Mask shape: {mask.shape}"
                )
        else:
            # No transforms: normalize and convert to tensor
            image_mean = image.mean()
            image_std = image.std()
            if image_std > 1e-8:
                image = (image - image_mean) / image_std
            
            image = torch.from_numpy(image).unsqueeze(0)  # (1, D, H, W)
            mask = torch.from_numpy(mask).long()  # (D, H, W)
        
        # Metadata
        metadata = {
            'filename': sample_info['filename'],
            'marker_type': sample_info['marker_type'],
            'original_shape': original_shape,
            'augmented_shape': tuple(image.shape),
            'index': idx
        }
        
        # Final validation
        assert image.ndim == 4, f"Image should be (1,D,H,W), got {image.shape}"
        assert mask.ndim == 3, f"Mask should be (D,H,W), got {mask.shape}"
        assert image.shape[1:] == mask.shape, \
            f"Spatial dims don't match: {image.shape[1:]} vs {mask.shape}"
        
        return image, mask, metadata


# ============================================================================
# UTILITY FUNCTIONS FOR VALIDATION
# ============================================================================

def validate_metadata_format(data_dir: Path, data_type: str):
    """
    Validate metadata.json format
    
    Args:
        data_dir: Directory containing metadata.json
        data_type: 'unlabeled' or 'labeled'
    """
    metadata_path = data_dir / 'metadata.json'
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {data_dir}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    if data_type == 'unlabeled':
        # Check required fields
        required_fields = ['num_volumes', 'marker_types', 'volumes']
        for field in required_fields:
            if field not in metadata:
                raise ValueError(f"Missing field '{field}' in metadata.json")
        
        # Validate each volume entry
        for vol in metadata['volumes']:
            required_vol_fields = ['brain_name', 'marker_type', 'marker_label', 
                                   'shape', 'file_path']
            for field in required_vol_fields:
                if field not in vol:
                    raise ValueError(
                        f"Missing field '{field}' in volume entry: {vol.get('brain_name', 'unknown')}"
                    )
    
    elif data_type == 'labeled':
        # Check required fields
        required_fields = ['num_samples', 'splits', 'data']
        for field in required_fields:
            if field not in metadata:
                raise ValueError(f"Missing field '{field}' in metadata.json")
        
        # Validate splits
        required_splits = ['train', 'val', 'test']
        for split in required_splits:
            if split not in metadata['data']:
                raise ValueError(f"Missing split '{split}' in metadata")
    
    else:
        raise ValueError(f"Invalid data_type: {data_type}")
    
    logger.info(f"✅ Metadata validation passed for {data_type} data")


def test_dataset_loading():
    """Test that datasets can load properly"""
    logger.info("\n" + "="*80)
    logger.info("TESTING DATASET LOADING")
    logger.info("="*80)
    
    # Test VolumeDataset3D
    try:
        ssl_dir = Path("data/processed/volumes_ssl")
        if ssl_dir.exists() and (ssl_dir / 'metadata.json').exists():
            logger.info("\n1. Testing VolumeDataset3D (SSL)...")
            
            dataset = VolumeDataset3D(
                data_dir=str(ssl_dir),
                patch_size=(64, 128, 128),
                num_patches_per_epoch=10
            )
            
            # Load a few samples
            for i in range(min(3, len(dataset))):
                patch, label = dataset[i]
                logger.info(f"  Sample {i}: patch {patch.shape}, label {label.item()}")
            
            logger.info("  ✅ VolumeDataset3D working!")
        else:
            logger.warning("  ⚠️  SSL data not found, skipping test")
    
    except Exception as e:
        logger.error(f"  ❌ VolumeDataset3D failed: {e}")
    
    # Test SELMA3DDataset
    try:
        labeled_dir = Path("data/processed/volumes_labeled")
        if labeled_dir.exists() and (labeled_dir / 'metadata.json').exists():
            logger.info("\n2. Testing SELMA3DDataset (Fine-tuning)...")
            
            dataset = SELMA3DDataset(
                data_dir=str(labeled_dir),
                split='train'
            )
            
            # Load a sample
            image, mask, metadata = dataset[0]
            logger.info(f"  Sample 0: {metadata['filename']}")
            logger.info(f"    Image: {image.shape}")
            logger.info(f"    Mask: {mask.shape}")
            logger.info(f"    Marker: {metadata['marker_type']}")
            
            logger.info("  ✅ SELMA3DDataset working!")
        else:
            logger.warning("  ⚠️  Labeled data not found, skipping test")
    
    except Exception as e:
        logger.error(f"  ❌ SELMA3DDataset failed: {e}")
    
    logger.info("\n" + "="*80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_dataset_loading()