"""
Fine-tuning Script for Segmentation
✅ FIXED: Uses metadata-driven SELMA3DDataset
✅ FIXED: No manual file discovery
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import yaml
from tqdm import tqdm
import numpy as np
import logging

import sys
sys.path.append('..')

from models import HGFormer3D, HGFormer3D_ForSegmentation
from models.losses import segmentation_loss
from utils.data_loader import SELMA3DDataset, validate_metadata_format
from utils.augmentations import get_finetune_transforms, get_val_transforms
from utils.metrics import dice_coefficient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune HGFormer3D for segmentation')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('--pretrained_checkpoint', type=str, default=None,
                       help='Path to pretrained checkpoint (overrides config)')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory with processed labeled data and metadata.json')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (overrides config)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    return parser.parse_args()


def load_pretrained_encoder(checkpoint_path, config, device):
    """Load pretrained encoder from SSL checkpoint"""
    logger.info(f"Loading pretrained encoder from: {checkpoint_path}")
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config from checkpoint if available
    if 'config' in checkpoint:
        model_config = checkpoint['config']['model']
    else:
        # Use default config
        model_config = config['model']
    
    # Create encoder
    encoder = HGFormer3D(**model_config)
    
    # Load weights
    encoder.load_state_dict(checkpoint['model_state_dict'])
    logger.info("✅ Pretrained weights loaded successfully")
    
    if 'epoch' in checkpoint:
        logger.info(f"  Pretrained for {checkpoint['epoch']} epochs")
    if 'loss' in checkpoint:
        logger.info(f"  Pretraining loss: {checkpoint['loss']:.4f}")
    
    return encoder


def validate(model, val_loader, device):
    """Compute validation metrics"""
    model.eval()
    
    dice_scores = []
    
    with torch.no_grad():
        for volumes, masks, metadata in tqdm(val_loader, desc='Validation', leave=False):
            volumes = volumes.to(device)
            masks = masks.to(device)
            
            # Forward pass
            logits = model(volumes)
            
            # Get predictions
            preds = torch.argmax(logits, dim=1)
            
            # Compute Dice for each sample
            for i in range(preds.shape[0]):
                pred_np = preds[i].cpu().numpy()
                mask_np = masks[i].cpu().numpy()
                
                dice = dice_coefficient(pred_np, mask_np)
                dice_scores.append(dice)
    
    avg_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)
    
    return {
        'dice_mean': avg_dice,
        'dice_std': std_dice,
        'dice_scores': dice_scores
    }


def main():
    args = parse_args()
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Override config with command line args
    if args.pretrained_checkpoint:
        config['pretrained_checkpoint'] = args.pretrained_checkpoint
    if args.output_dir:
        config['paths']['output_dir'] = args.output_dir
    
    # Create output directories
    output_dir = Path(config['paths']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'finetune_config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # ✅ CRITICAL FIX: Validate metadata
    data_dir = Path(args.data_dir)
    logger.info(f"Validating metadata in {data_dir}...")
    
    try:
        validate_metadata_format(data_dir, data_type='labeled')
    except Exception as e:
        logger.error(f"Metadata validation failed: {e}")
        logger.error(f"\n⚠️  Did you run prepare_data.py first?")
        logger.error(f"    python scripts/prepare_data.py \\")
        logger.error(f"        --input_dir data/raw/train_labeled \\")
        logger.error(f"        --output_dir {data_dir} \\")
        logger.error(f"        --data_type labeled")
        raise
    
    logger.info("✅ Metadata validation passed")
    
    # Load pretrained encoder
    encoder = load_pretrained_encoder(
        config['pretrained_checkpoint'],
        config,
        device
    )
    
    # Create segmentation model
    logger.info("Creating segmentation model...")
    model = HGFormer3D_ForSegmentation(
        pretrained_encoder=encoder,
        num_classes=config['model']['num_classes'],
        freeze_encoder=config['model']['freeze_encoder'],
        decoder_channels=config['decoder']['decoder_channels']
    ).to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"Model created:")
    logger.info(f"  Total parameters: {total_params/1e6:.2f}M")
    logger.info(f"  Trainable parameters: {trainable_params/1e6:.2f}M")
    logger.info(f"  Frozen parameters: {(total_params-trainable_params)/1e6:.2f}M")
    
    # ✅ FIXED: Create datasets (they read metadata.json internally)
    logger.info("\nCreating datasets...")
    
    train_transform = get_finetune_transforms(config.get('augmentation', {}))
    val_transform = get_val_transforms(config.get('augmentation', {}))
    
    train_dataset = SELMA3DDataset(
        data_dir=str(data_dir),
        split='train',
        transform=train_transform,
        config=config
    )
    
    val_dataset = SELMA3DDataset(
        data_dir=str(data_dir),
        split='val',
        transform=val_transform,
        config=config
    )
    
    logger.info(f"  Train dataset: {len(train_dataset)} samples")
    logger.info(f"  Val dataset: {len(val_dataset)} samples")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )
    
    # Optimizer
    logger.info("\nSetting up optimizer...")
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params_list,
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Scheduler
    if config['training']['scheduler'] == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=config['training']['lr_factor'],
            patience=config['training']['patience'],
            verbose=True,
            min_lr=config['training']['min_lr']
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=config['training']['min_lr']
        )
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_dice = 0.0
    patience_counter = 0
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', 0.0)
        logger.info(f"Resuming from epoch {start_epoch}, best Dice: {best_dice:.4f}")
    
    # Training loop
    logger.info("\n" + "="*80)
    logger.info("Starting fine-tuning...")
    logger.info("="*80)
    
    for epoch in range(start_epoch, config['training']['epochs']):
        
        # Unfreeze encoder if specified
        if epoch == config['model']['unfreeze_epoch'] and config['model']['freeze_encoder']:
            logger.info(f"\n{'='*80}")
            logger.info(f"Unfreezing encoder at epoch {epoch}")
            logger.info(f"{'='*80}")
            model.unfreeze_encoder()
            
            # Update optimizer with all parameters
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config['training']['learning_rate'] * 0.1,
                weight_decay=config['training']['weight_decay']
            )
        
        # Training
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{config["training"]["epochs"]}')
        
        for batch_idx, (volumes, masks, metadata) in enumerate(pbar):
            volumes = volumes.to(device)
            masks = masks.to(device)
            
            # Forward pass
            logits = model(volumes)
            
            # Compute loss
            loss, loss_dict = segmentation_loss(
                logits, masks,
                alpha_start=config['loss']['alpha_start'],
                epoch=epoch,
                max_epochs=config['training']['epochs']
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config['training']['gradient_clip']
            )
            
            optimizer.step()
            
            train_losses.append(loss_dict)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dice': f"{loss_dict['dice']:.4f}"
            })
        
        # Calculate average losses
        avg_train_loss = {
            k: np.mean([d[k] for d in train_losses]) 
            for k in train_losses[0].keys()
        }
        
        # Validation
        if (epoch + 1) % config['training']['val_freq'] == 0:
            logger.info("\n  Running validation...")
            val_metrics = validate(model, val_loader, device)
            
            logger.info(f"\n  Epoch {epoch} Summary:")
            logger.info(f"    Train Loss: {avg_train_loss['total']:.4f}")
            logger.info(f"    Train Dice Loss: {avg_train_loss['dice']:.4f}")
            logger.info(f"    Val Dice: {val_metrics['dice_mean']:.4f} ± {val_metrics['dice_std']:.4f}")
            logger.info(f"    Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Step scheduler
            if config['training']['scheduler'] == 'ReduceLROnPlateau':
                scheduler.step(val_metrics['dice_mean'])
            else:
                scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % config['training']['save_freq'] == 0:
                checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': avg_train_loss['total'],
                    'val_dice': val_metrics['dice_mean'],
                    'best_dice': best_dice,
                    'config': config
                }, checkpoint_path)
                logger.info(f"    Checkpoint saved: {checkpoint_path}")
            
            # Save best model
            if val_metrics['dice_mean'] > best_dice:
                best_dice = val_metrics['dice_mean']
                patience_counter = 0
                
                best_path = output_dir / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_dice': best_dice,
                    'config': config
                }, best_path)
                logger.info(f"    ✅ New best model saved! Dice: {best_dice:.4f}")
            else:
                patience_counter += 1
                logger.info(f"    No improvement. Patience: {patience_counter}/{config['training']['early_stopping_patience']}")
            
            # Early stopping
            if config['training']['early_stopping']:
                if patience_counter >= config['training']['early_stopping_patience']:
                    logger.info(f"\n  Early stopping triggered after {epoch + 1} epochs")
                    break
        
        logger.info("")
    
    logger.info("="*80)
    logger.info("Fine-tuning completed!")
    logger.info("="*80)
    logger.info(f"Best validation Dice: {best_dice:.4f}")
    logger.info(f"Checkpoints saved to: {output_dir}")


if __name__ == '__main__':
    main()