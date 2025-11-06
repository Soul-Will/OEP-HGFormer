"""
Unified Integration Test for HGFormer3D
✅ FIXED: Assertion in TEST 2 now correctly checks spatial dims and class
"""

import torch
import logging
import sys

# ---
# Setup Logging
# ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---
# Add models to path
# ---
try:
    from models.hgformer import HGFormer3D
    from models.decoder import HGFormer3D_ForSegmentation
except ImportError:
    logger.error("Could not import models. Make sure you run this script from the root directory:")
    logger.error("Example: python test_model.py")
    sys.exit(1)

# ---
# Global Test Parameters
# ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

logger.info(f"Running tests on device: {DEVICE}")

def test_encoder_standalone():
    """Test that the encoder can run by itself"""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: Encoder Standalone (hgformer.py)")
    logger.info("="*80)
    
    try:
        model = HGFormer3D(
            in_channels=1,
            base_channels=32,
            depths=[1, 2, 4, 2],
            stem_stride=(2, 4, 4),
            stage_strides=[(2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 1, 1)]
        ).to(DEVICE)
        
        x = torch.randn(1, 1, 64, 128, 128).to(DEVICE)
        logger.info(f"Input shape: {x.shape}")
        
        features = model(x, return_features=True)
        
        assert len(features) == 4, f"Expected 4 feature maps, got {len(features)}"
        
        for i, feat in enumerate(features):
            logger.info(f"  Stage {i} output: {feat.shape}")
        
        logger.info("✅ PASSED: Encoder standalone test")
        return True
    
    except Exception as e:
        logger.error(f"❌ FAILED: Encoder standalone test", exc_info=True)
        return False

def test_integration_and_alignment():
    """
    Test E2E model integration and spatial alignment.
    This also implicitly tests the `topk` bug fix by using a small input.
    """
    logger.info("\n" + "="*80)
    logger.info("TEST 2: E2E Integration & Spatial Alignment (decoder.py)")
    logger.info("="*80)
    
    try:
        # 1. Create Encoder
        encoder = HGFormer3D(
            in_channels=1,
            base_channels=32,
            depths=[1, 2, 4, 2],
            stem_stride=(2, 4, 4),
            stage_strides=[(2, 2, 2), (2, 2, 2), (2, 2, 2), (1, 1, 1)]
        )
        
        # 2. Create Full Segmentation Model
        NUM_CLASSES = 2
        model = HGFormer3D_ForSegmentation(
            pretrained_encoder=encoder,
            num_classes=NUM_CLASSES,
            freeze_encoder=True
        ).to(DEVICE)
        
        logger.info("Model created successfully. Decoder auto-aligned.")
        
        # 3. Test various input sizes
        test_sizes = [
            (64, 128, 128),  # Standard patch
            (48, 96, 96),    # Odd patch size
            (32, 64, 64)     # The critical `topk` bug test
        ]
        
        for size in test_sizes:
            logger.info(f"\nTesting size: {size}...")
            x = torch.randn(1, 1, *size).to(DEVICE)
            
            # This forward pass will crash if `topk` bug exists
            logits = model(x)
            
            logger.info(f"  Input shape:  {x.shape}")
            logger.info(f"  Output shape: {logits.shape}")
            
            # ✅ BUG FIX: The assertion is now correct.
            # We check that spatial dimensions match the input.
            assert logits.shape[2:] == x.shape[2:], \
                f"SPATIAL mismatch! Input {x.shape[2:]} != Output {logits.shape[2:]}"
            
            # We check that channel dimensions match num_classes.
            assert logits.shape[1] == NUM_CLASSES, \
                f"CLASS mismatch! Expected {NUM_CLASSES} classes, got {logits.shape[1]}"
            
            logger.info(f"  ✅ Size {size} aligned perfectly!")
        
        logger.info("\n✅ PASSED: E2E Integration and Alignment (all sizes)")
        return True
        
    except Exception as e:
        logger.error(f"❌ FAILED: E2E Integration test", exc_info=True)
        return False

def test_unfreeze_encoder():
    """Test the unfreeze_encoder() helper method"""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Encoder Freezing/Unfreezing")
    logger.info("="*80)
    
    try:
        encoder = HGFormer3D()
        model = HGFormer3D_ForSegmentation(
            pretrained_encoder=encoder,
            freeze_encoder=True
        )
        
        # Check that it is frozen
        frozen_param = model.encoder.stages[0].blocks[0].mlp[0].weight
        assert not frozen_param.requires_grad, "Encoder was not frozen!"
        logger.info("  Encoder is frozen by default.")
        
        # Unfreeze
        model.unfreeze_encoder()
        assert frozen_param.requires_grad, "Encoder did not unfreeze!"
        logger.info("  Encoder unfreezing successful.")
        
        logger.info("✅ PASSED: Encoder freezing/unfreezing test")
        return True
        
    except Exception as e:
        logger.error(f"❌ FAILED: Encoder freezing/unfreezing test", exc_info=True)
        return False

# ---
# Run All Tests
# ---
if __name__ == "__main__":
    logger.info("="*80)
    logger.info("Starting Full Model Integration Test Suite")
    logger.info("="*80)
    
    results = {
        "Encoder Standalone": test_encoder_standalone(),
        "E2E Integration/Alignment": test_integration_and_alignment(),
        "Encoder Unfreeze Logic": test_unfreeze_encoder(),
    }
    
    logger.info("\n" + "="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    all_passed = True
    for test_name, result in results.items():
        if result:
            logger.info(f"  [PASS] {test_name}")
        else:
            logger.info(f"  [FAIL] {test_name}")
            all_passed = False
    
    logger.info("="*80)
    if all_passed:
        logger.info("🎉 All tests passed successfully! Your model is integrated and robust.")
    else:
        logger.error("🔥 One or more tests failed. Please review the logs above.")
    logger.info("="*80)