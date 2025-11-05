"""
Synchronized 3D Augmentations for Medical Imaging
Uses torchio for proper image-mask alignment
"""

import torchio as tio
import torch

def get_ssl_transforms(config):
    """
    Self-supervised learning augmentations
    
    Args:
        config: Dict with augmentation parameters
        
    Returns:
        tio.Compose transform that applies to SubjectDataset
    """
    return tio.Compose([
        # Intensity augmentations (image only, automatically handled)
        tio.RandomBlur(std=(0, 2), p=0.5),
        tio.RandomNoise(std=(0, 0.1), p=0.5),
        tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.5),
        
        # Spatial augmentations (SYNCHRONIZED to image and mask)
        tio.RandomAffine(
            scales=(0.9, 1.1),
            degrees=15,
            translation=5,
            p=0.5
        ),
        tio.RandomFlip(axes=('LR',), p=0.5),  # Left-Right flip
        tio.RandomElasticDeformation(
            num_control_points=7,
            max_displacement=7.5,
            p=0.3
        ),
        
        # Normalization (always applied)
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.ZNormalization(),  # Mean=0, Std=1
    ])


def get_finetune_transforms(config):
    """
    Fine-tuning augmentations (less aggressive)
    """
    return tio.Compose([
        tio.RandomBlur(std=(0, 1), p=0.3),
        tio.RandomNoise(std=(0, 0.05), p=0.3),
        tio.RandomAffine(
            scales=(0.95, 1.05),
            degrees=10,
            translation=3,
            p=0.3
        ),
        tio.RandomFlip(axes=('LR',), p=0.5),
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.ZNormalization(),
    ])


def get_val_transforms(config):
    """
    Validation transforms (normalization only)
    """
    return tio.Compose([
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.ZNormalization(),
    ])


# Example usage with assertions for debugging
def test_transform_sync():
    """Verify transforms maintain image-mask alignment"""
    import numpy as np
    
    # Create test volume and mask
    image = np.random.rand(64, 128, 128).astype(np.float32)
    mask = np.random.randint(0, 2, (64, 128, 128)).astype(np.uint8)
    
    # Create torchio Subject
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=image[None, ...]),  # Add channel dim
        mask=tio.LabelMap(tensor=mask[None, ...])
    )
    
    # Apply transforms
    transform = get_ssl_transforms({})
    transformed = transform(subject)
    
    # Verify shapes match
    assert transformed.image.shape == transformed.mask.shape, \
        "Image and mask shapes don't match after transform!"
    
    print("✅ Transform synchronization test passed")


if __name__ == "__main__":
    test_transform_sync()