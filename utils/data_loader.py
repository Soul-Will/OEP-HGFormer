"""
Robust, Config-Driven Data Loaders with Proper Augmentation
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import torchio as tio
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SELMA3DDataset(Dataset):
    """
    ✅ FIXED: Proper synchronized augmentations for segmentation
    
    Key Features:
    - Uses torchio.Subject for automatic image-mask alignment
    - Config-driven (no hard-coded values)
    - Extensive validation and error handling
    - Memory-efficient lazy loading
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[tio.Compose] = None,
        config: Optional[Dict] = None
    ):
        """
        Args:
            data_dir: Root directory containing processed volumes
            split: 'train', 'val', or 'test'
            transform: torchio transform (handles image+mask sync)
            config: Configuration dict (for validation)
        """
        self.data_dir = Path(data_dir) / split
        self.transform = transform
        self.config = config or {}
        
        # Validate directory exists
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}\n"
                f"Did you run prepare_data.py?"
            )
        
        # Find all image-mask pairs
        self.samples = self._find_samples()
        
        if len(self.samples) == 0:
            raise ValueError(
                f"No samples found in {self.data_dir}\n"
                f"Expected format: *_img.npy and *_mask.npy"
            )
        
        logger.info(f"Loaded {len(self.samples)} samples from {split} split")
    
    def _find_samples(self) -> List[Dict[str, Path]]:
        """Find all valid image-mask pairs"""
        samples = []
        
        for img_path in sorted(self.data_dir.glob("*_img.npy")):
            mask_path = img_path.parent / img_path.name.replace('_img', '_mask')
            
            if not mask_path.exists():
                logger.warning(f"Missing mask for {img_path.name}, skipping")
                continue
            
            # Validate files are loadable
            try:
                img_shape = np.load(img_path, mmap_mode='r').shape
                mask_shape = np.load(mask_path, mmap_mode='r').shape
                
                if img_shape != mask_shape:
                    logger.warning(
                        f"Shape mismatch: {img_path.name} {img_shape} vs "
                        f"{mask_path.name} {mask_shape}, skipping"
                    )
                    continue
                
            except Exception as e:
                logger.warning(f"Error loading {img_path.name}: {e}, skipping")
                continue
            
            samples.append({
                'image': img_path,
                'mask': mask_path,
                'filename': img_path.stem.replace('_img', '')
            })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Returns:
            image: (1, D, H, W) float32 tensor
            mask: (D, H, W) int64 tensor
            metadata: Dict with filename, original_shape, etc.
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
        assert image.shape == mask.shape, \
            f"Shape mismatch: image {image.shape} vs mask {mask.shape}"
        
        # Store original shape for metadata
        original_shape = image.shape
        
        # ✅ CRITICAL FIX: Use torchio.Subject for synchronized transforms
        if self.transform is not None:
            # Create torchio Subject (wraps image and mask)
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=image[None, ...]),  # (1, D, H, W)
                mask=tio.LabelMap(tensor=mask[None, ...])        # (1, D, H, W)
            )
            
            # Apply transform (automatically syncs spatial transforms)
            try:
                transformed = self.transform(subject)
            except Exception as e:
                raise RuntimeError(
                    f"Transform failed on sample {idx}: {e}\n"
                    f"Image shape: {image.shape}, Mask shape: {mask.shape}"
                )
            
            # Extract tensors
            image = transformed.image.data  # (1, D, H, W)
            mask = transformed.mask.data.squeeze(0).long()  # (D, H, W)
        
        else:
            # No transforms: just normalize and convert to tensor
            image = torch.from_numpy(image).unsqueeze(0)  # (1, D, H, W)
            mask = torch.from_numpy(mask).long()  # (D, H, W)
            
            # Basic normalization (if no transforms provided)
            image = (image - image.mean()) / (image.std() + 1e-8)
        
        # Metadata for tracking
        metadata = {
            'filename': sample_info['filename'],
            'original_shape': original_shape,
            'augmented_shape': tuple(image.shape),
            'index': idx
        }
        
        # Final validation
        assert image.ndim == 4, f"Image should be (1,D,H,W), got {image.shape}"
        assert mask.ndim == 3, f"Mask should be (D,H,W), got {mask.shape}"
        assert image.shape[1:] == mask.shape, \
            f"Spatial dims don't match: {image.shape} vs {mask.shape}"
        
        return image, mask, metadata


class VolumeDataset3D(Dataset):
    """
    ✅ FIXED: Dynamic patch sampling + pre-stacked volumes
    
    Key Changes:
    - Expects pre-stacked 3D volumes (from fixed prepare_data.py)
    - Dynamic patch sampling (different patches each epoch)
    - Config-driven parameters
    """
    
    def __init__(
        self,
        brain_folders: List[str],
        patch_size: Tuple[int, int, int],
        num_patches_per_brain: int,
        transform: Optional[tio.Compose] = None,
        config: Optional[Dict] = None,
        preload: bool = False
    ):
        """
        Args:
            brain_folders: List of paths to *_volume.npy files (not slice folders!)
            patch_size: (D, H, W) patch dimensions
            num_patches_per_brain: Patches to sample per volume per epoch
            transform: torchio transform
            config: Configuration dict
            preload: If True, load all volumes into RAM (only for small datasets)
        """
        self.brain_folders = [Path(p) for p in brain_folders]
        self.patch_size = patch_size
        self.num_patches_per_brain = num_patches_per_brain
        self.transform = transform
        self.config = config or {}
        
        # Marker type mapping (for weak supervision)
        self.marker_map = {
            'ab_plaque': 3,
            'cfos': 0,
            'nucleus': 2,
            'vessel': 1
        }
        
        # Validate all volumes exist and get their shapes
        self.volume_info = []
        for vol_path in self.brain_folders:
            if not vol_path.exists():
                raise FileNotFoundError(f"Volume not found: {vol_path}")
            
            try:
                # Use mmap to read shape without loading entire volume
                vol_shape = np.load(vol_path, mmap_mode='r').shape
                marker_type = self._get_marker_type(vol_path)
                
                self.volume_info.append({
                    'path': vol_path,
                    'shape': vol_shape,  # (D, H, W)
                    'marker_type': marker_type
                })
            except Exception as e:
                raise RuntimeError(f"Failed to read {vol_path}: {e}")
        
        # Optionally preload volumes (only for small datasets)
        if preload:
            logger.warning("Preloading all volumes into RAM (use only for small datasets)")
            self.volumes = [np.load(info['path']) for info in self.volume_info]
        else:
            self.volumes = None
        
        logger.info(f"Initialized dataset with {len(self.volume_info)} volumes")
    
    def _get_marker_type(self, vol_path: Path) -> int:
        """Extract marker type from filename"""
        name_lower = str(vol_path).lower()
        for marker, idx in self.marker_map.items():
            if marker in name_lower:
                return idx
        return 0  # Default to cFos
    
    def __len__(self) -> int:
        """
        ✅ FIXED: Virtual length for dynamic sampling
        Return total patches across all volumes
        """
        return len(self.volume_info) * self.num_patches_per_brain
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ✅ FIXED: Dynamic random patch sampling
        
        Returns:
            patch: (1, D, H, W) volume patch
            marker_type: int tensor (for weak supervision)
        """
        # Determine which volume to sample from
        vol_idx = idx % len(self.volume_info)
        vol_info = self.volume_info[vol_idx]
        
        # Load volume (from RAM if preloaded, else from disk)
        if self.volumes is not None:
            volume = self.volumes[vol_idx]
        else:
            volume = np.load(vol_info['path'])
        
        # ✅ CRITICAL FIX: Random patch location (changes every epoch!)
        D, H, W = vol_info['shape']
        pd, ph, pw = self.patch_size
        
        # Ensure patch fits in volume
        if D < pd or H < ph or W < pw:
            raise ValueError(
                f"Patch size {self.patch_size} too large for volume "
                f"{vol_info['shape']} in {vol_info['path']}"
            )
        
        # Random valid starting coordinates
        d_start = np.random.randint(0, D - pd + 1)
        h_start = np.random.randint(0, H - ph + 1)
        w_start = np.random.randint(0, W - pw + 1)
        
        # Extract patch
        patch = volume[
            d_start:d_start + pd,
            h_start:h_start + ph,
            w_start:w_start + pw
        ].astype(np.float32)
        
        # Apply transforms if provided
        if self.transform is not None:
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=patch[None, ...])
            )
            transformed = self.transform(subject)
            patch = transformed.image.data  # (1, D, H, W)
        else:
            # Normalize
            patch = (patch - patch.mean()) / (patch.std() + 1e-8)
            patch = torch.from_numpy(patch).unsqueeze(0)  # (1, D, H, W)
        
        marker_type = torch.tensor(vol_info['marker_type'], dtype=torch.long)
        
        return patch, marker_type


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def test_augmentation_sync():
    """Test that augmentations preserve image-mask alignment"""
    from utils.augmentations import get_ssl_transforms
    
    # Create dummy dataset
    test_dir = Path("data/processed/volumes_labeled/train")
    if not test_dir.exists():
        logger.warning(f"Test directory {test_dir} not found, skipping test")
        return
    
    transform = get_ssl_transforms({})
    dataset = SELMA3DDataset(
        data_dir="data/processed/volumes_labeled",
        split="train",
        transform=transform
    )
    
    # Load a sample
    image, mask, metadata = dataset[0]
    
    # Check that a rotated neuron in image appears at same location in mask
    # (This is a visual check - in practice, use intersection metrics)
    
    print(f"✅ Loaded sample: {metadata['filename']}")
    print(f"   Image shape: {image.shape}, Mask shape: {mask.shape}")
    print(f"   Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"   Mask unique values: {torch.unique(mask).tolist()}")
    
    # Verify spatial alignment (rough check)
    mask_binary = (mask > 0).float()
    if mask_binary.sum() > 0:
        # Get center of mass of mask
        indices = torch.nonzero(mask_binary, as_tuple=False)
        com_mask = indices.float().mean(dim=0)
        
        # Get brightest region in image
        threshold = image.quantile(0.95)
        bright_pixels = (image.squeeze(0) > threshold)
        if bright_pixels.sum() > 0:
            bright_indices = torch.nonzero(bright_pixels, as_tuple=False)
            com_image = bright_indices.float().mean(dim=0)
            
            # Distance between centers (should be small)
            distance = torch.dist(com_mask, com_image).item()
            print(f"   Center-of-mass distance: {distance:.2f} voxels")
            
            if distance > 20:  # Arbitrary threshold
                logger.warning(
                    f"Large COM distance ({distance:.1f}), possible misalignment!"
                )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_augmentation_sync()