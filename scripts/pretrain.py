"""
Self-Supervised Pretraining Script
✅ FIXED: Removed get_brain_folders() - uses metadata.json instead!
✅ FIXED: Trusts prepare_data.py output
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm
import logging

import sys
sys.path.append('..')

from models import HGFormer3D
from models.losses import ssl_total_loss
from utils.data_loader import VolumeDataset3D, validate_metadata_format
from utils.augmentations import get_ssl_transforms

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='SSL Pretraining for HGFormer3D')
    parser.add_argument('--config', type=str, required=True, 
                       help='Path to config YAML file')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory with processed volumes and metadata.json')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save checkpoints')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Save config to output dir
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # ✅ CRITICAL FIX: Validate metadata exists and is correct format
    data_dir = Path(args.data_dir)
    logger.info(f"Validating metadata in {data_dir}...")
    
    try:
        validate_metadata_format(data_dir, data_type='unlabeled')
    except Exception as e:
        logger.error(f"Metadata validation failed: {e}")
        logger.error(f"\n⚠️  Did you run prepare_data.py first?")
        logger.error(f"    python scripts/prepare_data.py \\")
        logger.error(f"        --input_dir data/raw/train_unlabeled \\")
        logger.error(f"        --output_dir {data_dir} \\")
        logger.error(f"        --data_type unlabeled")
        raise
    
    logger.info("✅ Metadata validation passed")
    
    # Create model
    logger.info("Creating model...")
    model = HGFormer3D(**config['model']).to(device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model created with {num_params:.2f}M parameters")
    
    # ✅ FIXED: No more get_brain_folders()! Just pass data_dir to dataset
    logger.info("Creating dataset...")
    
    # Get transforms
    train_transform = get_ssl_transforms(config.get('augmentation', {}))
    
    # Create dataset (it will read metadata.json internally)
    train_dataset = VolumeDataset3D(
        data_dir=str(data_dir),
        patch_size=tuple(config['data']['patch_size']),
        num_patches_per_epoch=config['data'].get('num_patches_per_epoch', 1000),
        transform=train_transform,
        config=config,
        preload=False
    )
    
    logger.info(f"Dataset size: {len(train_dataset)} patches per epoch")
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    # Optimizer
    logger.info("Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['epochs'],
        eta_min=config['training'].get('min_lr', 1e-6)
    )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        logger.info(f"Resuming from epoch {start_epoch}, best loss: {best_loss:.4f}")
    
    # Training loop
    logger.info("\n" + "="*80)
    logger.info("Starting SSL pretraining...")
    logger.info("="*80)
    
    for epoch in range(start_epoch, config['training']['epochs']):
        model.train()
        epoch_losses = {
            'total': [],
            'inpaint': [],
            'rotation': [],
            'contrastive': [],
            'label': []
        }
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{config["training"]["epochs"]}')
        
        for batch_idx, (volumes, marker_types) in enumerate(pbar):
            volumes = volumes.to(device)
            marker_types = marker_types.to(device)
            
            # Forward pass
            loss, loss_dict = ssl_total_loss(
                model, volumes, marker_types,
                **config['ssl_losses']
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                config['training'].get('gradient_clip', 1.0)
            )
            
            optimizer.step()
            
            # Log losses
            for key in epoch_losses.keys():
                epoch_losses[key].append(loss_dict[key])
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # Step scheduler
        scheduler.step()
        
        # Calculate average losses
        avg_losses = {k: sum(v)/len(v) for k, v in epoch_losses.items()}
        
        # Print epoch summary
        logger.info(f"\nEpoch {epoch} Summary:")
        logger.info(f"  Total Loss: {avg_losses['total']:.4f}")
        logger.info(f"  Inpaint: {avg_losses['inpaint']:.4f}")
        logger.info(f"  Rotation: {avg_losses['rotation']:.4f}")
        logger.info(f"  Contrastive: {avg_losses['contrastive']:.4f}")
        logger.info(f"  Label: {avg_losses['label']:.4f}")
        logger.info(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % config['training'].get('save_freq', 10) == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_losses['total'],
                'config': config
            }, checkpoint_path)
            logger.info(f"  Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if avg_losses['total'] < best_loss:
            best_loss = avg_losses['total']
            best_path = output_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'config': config
            }, best_path)
            logger.info(f"  ✅ New best model saved: {best_path} (loss: {best_loss:.4f})")
        
        logger.info("")
    
    logger.info("="*80)
    logger.info("SSL Pretraining completed!")
    logger.info("="*80)
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Checkpoints saved to: {output_dir}")


if __name__ == '__main__':
    main()