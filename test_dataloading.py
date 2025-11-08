from utils.data_loader import VolumeDataset3D
   
dataset = VolumeDataset3D(
    data_dir='data/processed/volume_ssl',
    patch_size=(64, 128, 128),
    num_patches_per_epoch=10
)
   
# Load a patch
patch, label = dataset[0]
print(f"Patch shape: {patch.shape}, Label: {label}")