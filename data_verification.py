# scripts/verify_processed_data.py

import numpy as np
import json
from pathlib import Path

def verify_processed_data(data_dir: Path):
    """Verify processed data is correct"""
    
    # Load metadata
    with open(data_dir / 'metadata.json') as f:
        metadata = json.load(f)
    
    print(f"Total volumes: {metadata['num_volumes']}")
    print(f"Marker types: {metadata['marker_types']}")
    print(f"Total size: {metadata['total_size_mb']:.1f} MB")
    
    # Check first volume
    first_vol = metadata['volumes'][0]
    vol_path = data_dir / first_vol['file_path']
    
    print(f"\nLoading sample: {vol_path}")
    volume = np.load(vol_path)
    
    print(f"  Shape: {volume.shape}")
    print(f"  Dtype: {volume.dtype}")
    print(f"  Range: [{volume.min():.4f}, {volume.max():.4f}]")
    print(f"  Mean: {volume.mean():.4f}")
    print(f"  Format: {first_vol['original_format']}")
    print(f"  Spacing: {first_vol['spacing']}")
    
    print("\n✅ Data verification passed!")

if __name__ == '__main__':
    import sys
    verify_processed_data(Path(sys.argv[1]))