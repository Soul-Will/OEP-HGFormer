"""
Data Loaders with HDF5 Support
✅ FIXED: Memory-efficient patch loading from HDF5
✅ FIXED: Handles multi-GB volumes without loading all into RAM
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import json
import random
from typing import Dict, List, Tuple, Optional
import logging
import h5py  # ✅ NEW

logger = logging.getLogger(__name__)


class VolumeDataset3D(Dataset):
    """
    ✅ UPDATED: SSL Dataset with HDF5 support
    
    Key Features:
    - Loads patches directly from HDF5 (no full volume in RAM!)
    - Metadata-driven discovery
    - Dynamic random patch sampling
    """
    
    def __init__(
        self,
        data_dir: str,
        patch_size: Tuple[int, int, int] = (64, 128, 128),
        num_patches_per_epoch: int = 1000,
        transform: Optional[callable] = None,
        config: Optional[Dict] = None,
        preload: bool = False  # ⚠️ Set to False for HDF5!
    ):
        """
        Args:
            data_dir: Directory with HDF5 files and metadata.json
            patch_size: (D, H, W) patch dimensions
            num_patches_per_epoch: Virtual dataset size
            transform: Optional torchio transform
            config: Optional config dict
            preload: DON'T USE with HDF5 (defeats the purpose!)
        """
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.num_patches_per_epoch = num_patches_per_epoch
        self.transform = transform
        self.config = config or {}
        
        # Load metadata
        metadata_path = self.data_dir / 'metadata.json'
        
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"metadata.json not found in {self.data_dir}\n"
                f"Did you run prepare_data.py?"
            )
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Check storage format
        storage_format = metadata.get('storage_format', 'numpy')
        if storage_format != 'hdf5':
            logger.warning(
                f"Metadata indicates storage format: {storage_format}\n"
                f"This loader expects HDF5 files. If you have .npy files, "
                f"please re-run prepare_data.py to convert to HDF5."
            )
        
        # Extract volume information
        self.volumes = []
        for vol_meta in metadata['volumes']:
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
        logger.info(f"  Storage: HDF5 (memory-efficient patch loading)")
        logger.info(f"  Patch size: {patch_size}")
        logger.info(f"  Patches per epoch: {num_patches_per_epoch}")
        
        # Optionally preload (NOT RECOMMENDED for HDF5!)
        if preload:
            logger.warning(
                "⚠️ WARNING: preload=True defeats HDF5's memory efficiency!\n"
                "You're loading multi-GB files into RAM. Set preload=False."
            )
            self.volume_cache = {}
            for vol_info in self.volumes:
                with h5py.File(vol_info['path'], 'r') as f:
                    self.volume_cache[str(vol_info['path'])] = f['volume'][:]
        else:
            self.volume_cache = None
    
    def __len__(self) -> int:
        return self.num_patches_per_epoch
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ✅ CRITICAL: Load ONLY a patch from HDF5 (not the full volume!)
        
        Returns:
            patch: (1, D, H, W) tensor
            marker_label: int64 tensor
        """
        # Randomly select a volume
        vol_info = random.choice(self.volumes)
        
        # Get volume shape
        D, H, W = vol_info['shape']
        pd, ph, pw = self.patch_size
        
        # Validate patch size
        if D < pd or H < ph or W < pw:
            raise ValueError(
                f"Patch size {self.patch_size} too large for volume "
                f"{vol_info['shape']} from {vol_info['brain_name']}"
            )
        
        # Random patch coordinates
        d_start = random.randint(0, D - pd)
        h_start = random.randint(0, H - ph)
        w_start = random.randint(0, W - pw)
        
        # ✅ CRITICAL: Load ONLY the patch from HDF5
        if self.volume_cache is not None:
            # Preloaded (not recommended)
            volume = self.volume_cache[str(vol_info['path'])]
            patch = volume[
                d_start:d_start + pd,
                h_start:h_start + ph,
                w_start:w_start + pw
            ].copy()
        else:
            # ✅ Memory-efficient: Load only the patch!
            with h5py.File(vol_info['path'], 'r') as f:
                patch = f['volume'][
                    d_start:d_start + pd,
                    h_start:h_start + ph,
                    w_start:w_start + pw
                ]
        
        # Normalize
        patch_mean = patch.mean()
        patch_std = patch.std()
        if patch_std > 1e-8:
            patch = (patch - patch_mean) / patch_std
        
        # Convert to tensor
        patch = torch.from_numpy(patch).unsqueeze(0)  # (1, D, H, W)
        
        # Apply transforms
        if self.transform is not None:
            try:
                patch = self.transform(patch)
            except Exception as e:
                logger.error(f"Transform failed: {e}")
        
        # Get marker label
        marker_label = torch.tensor(vol_info['marker_label'], dtype=torch.long)
        
        return patch, marker_label


class SELMA3DDataset(Dataset):
    """
    ✅ UPDATED: Fine-tuning dataset with HDF5 support
    
    Key Feature: Image and mask stored in SAME HDF5 file!
    """
    
    def __init__(
        self,
        data_dir: str,
        patch_size: Tuple[int, int, int] = (64, 128, 128),
        samples_per_volume: int = 10,
        transform: Optional[callable] = None,
        config: Optional[Dict] = None
    ):
        """
        Args:
            data_dir: Directory with train/val/test splits (HDF5 files)
            patch_size: (D, H, W) patch size
            samples_per_volume: Patches per volume per epoch
            transform: torchio.Compose transform
            config: Config dict
        """
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.samples_per_volume = samples_per_volume
        self.transform = transform
        self.config = config or {}
        
        # Load metadata
        metadata_path = self.data_dir / 'metadata.json'
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {self.data_dir}")
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Infer split from data_dir name
        split = self.data_dir.name
        if split not in ['train', 'val', 'test']:
            # Try parent directory
            split = 'train'  # Default
            if 'train' in str(self.data_dir):
                split = 'train'
            elif 'val' in str(self.data_dir):
                split = 'val'
            elif 'test' in str(self.data_dir):
                split = 'test'
        
        logger.info(f"Loading split: {split}")
        
        # Get split data
        if split not in metadata['data']:
            raise ValueError(
                f"Split '{split}' not found in metadata. "
                f"Available: {list(metadata['data'].keys())}"
            )
        
        split_data = metadata['data'][split]
        
        # Load volume paths
        self.volumes = []
        
        for sample_meta in split_data:
            # HDF5 file contains BOTH image and mask
            hdf5_path = self.data_dir / f"{sample_meta['filename']}.h5"
            
            if not hdf5_path.exists():
                logger.warning(f"File not found: {hdf5_path}, skipping")
                continue
            
            self.volumes.append({
                'path': hdf5_path,
                'filename': sample_meta['filename'],
                'marker_type': sample_meta['marker_type'],
                'shape': tuple(sample_meta['shape'])
            })
        
        if len(self.volumes) == 0:
            raise ValueError(f"No valid volumes in {split} split!")
        
        logger.info(f"SELMA3DDataset ({split}) initialized:")
        logger.info(f"  Volumes: {len(self.volumes)}")
        logger.info(f"  Storage: HDF5 (image + mask in same file)")
        logger.info(f"  Patch size: {patch_size}")
        logger.info(f"  Samples per volume: {samples_per_volume}")
        logger.info(f"  Total virtual samples: {len(self)}")
    
    def __len__(self) -> int:
        return len(self.volumes) * self.samples_per_volume
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        ✅ CRITICAL: Load a random patch from HDF5 (memory-efficient!)
        
        Returns:
            image: (1, D, H, W) tensor (patch, not full volume)
            mask: (D, H, W) tensor (patch)
            metadata: Dict
        """
        # Map index to volume
        volume_idx = idx // self.samples_per_volume
        sample_idx = idx % self.samples_per_volume
        
        vol_info = self.volumes[volume_idx]
        
        # Get shape from metadata
        D, H, W = vol_info['shape']
        pd, ph, pw = self.patch_size
        
        # Validate
        if D < pd or H < ph or W < pw:
            raise ValueError(
                f"Patch size {self.patch_size} too large for "
                f"{vol_info['filename']} shape {vol_info['shape']}"
            )
        
        # Random patch coordinates
        d_start = random.randint(0, D - pd)
        h_start = random.randint(0, H - ph)
        w_start = random.randint(0, W - pw)
        
        # ✅ CRITICAL: Load ONLY the patch from HDF5
        try:
            with h5py.File(vol_info['path'], 'r') as f:
                # Load image patch
                image_patch = f['image'][
                    d_start:d_start + pd,
                    h_start:h_start + ph,
                    w_start:w_start + pw
                ].astype(np.float32)
                
                # Load mask patch
                mask_patch = f['mask'][
                    d_start:d_start + pd,
                    h_start:h_start + ph,
                    w_start:w_start + pw
                ].astype(np.int64)
        
        except Exception as e:
            raise RuntimeError(
                f"Failed to load patch from {vol_info['filename']}: {e}"
            )
        
        # Convert to tensors
        image = torch.from_numpy(image_patch).unsqueeze(0)  # (1, D, H, W)
        mask = torch.from_numpy(mask_patch)  # (D, H, W)
        
        # Apply transforms (synchronized for image + mask)
        if self.transform is not None:
            try:
                import torchio as tio
                subject = tio.Subject(
                    image=tio.ScalarImage(tensor=image),
                    mask=tio.LabelMap(tensor=mask.unsqueeze(0))
                )
                transformed = self.transform(subject)
                
                image = transformed.image.data
                mask = transformed.mask.data.squeeze(0).long()
            
            except Exception as e:
                raise RuntimeError(f"Transform failed: {e}")
        
        # Metadata
        metadata = {
            'filename': vol_info['filename'],
            'marker_type': vol_info['marker_type'],
            'original_shape': vol_info['shape'],
            'patch_shape': tuple(image.shape),
            'volume_idx': volume_idx,
            'sample_idx': sample_idx,
            'patch_coords': (d_start, h_start, w_start)
        }
        
        # Validation
        assert image.ndim == 4, f"Image should be (1,D,H,W), got {image.shape}"
        assert mask.ndim == 3, f"Mask should be (D,H,W), got {mask.shape}"
        assert image.shape[1:] == mask.shape, \
            f"Spatial dims don't match: {image.shape[1:]} vs {mask.shape}"
        
        return image, mask, metadata


# ============================================================================
# UTILITY: VALIDATE METADATA
# ============================================================================

def validate_metadata_format(data_dir: Path, data_type: str):
    """Validate metadata.json format"""
    metadata_path = data_dir / 'metadata.json'
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {data_dir}")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Check storage format
    storage_format = metadata.get('storage_format', 'unknown')
    if storage_format != 'hdf5':
        logger.warning(
            f"⚠️ Storage format is '{storage_format}', expected 'hdf5'.\n"
            f"Please re-run prepare_data.py to convert to HDF5."
        )
    
    if data_type == 'unlabeled':
        required_fields = ['num_volumes', 'marker_types', 'volumes', 'storage_format']
        for field in required_fields:
            if field not in metadata:
                raise ValueError(f"Missing field '{field}' in metadata.json")
        
        for vol in metadata['volumes']:
            required_vol_fields = ['brain_name', 'marker_type', 'marker_label', 
                                   'shape', 'file_path']
            for field in required_vol_fields:
                if field not in vol:
                    raise ValueError(
                        f"Missing field '{field}' in volume: {vol.get('brain_name', 'unknown')}"
                    )
    
    elif data_type == 'labeled':
        required_fields = ['num_samples', 'splits', 'data', 'storage_format']
        for field in required_fields:
            if field not in metadata:
                raise ValueError(f"Missing field '{field}' in metadata.json")
        
        required_splits = ['train', 'val', 'test']
        for split in required_splits:
            if split not in metadata['data']:
                raise ValueError(f"Missing split '{split}' in metadata")
    
    else:
        raise ValueError(f"Invalid data_type: {data_type}")
    
    logger.info(f"✅ Metadata validation passed for {data_type} data")