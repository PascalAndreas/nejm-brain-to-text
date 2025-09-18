#!/usr/bin/env python3
"""
Example script demonstrating how to use the new temperature calibration system.

This shows the post-training workflow for calibrating temperature scaling:
1. Load trained model
2. Prepare validation dataloader 
3. Run temperature calibration
4. Save calibrated model
5. Re-tune decoder hyperparameters

Usage:
    python calibration_example.py --checkpoint path/to/model.ckpt --data_dir path/to/data
"""

import torch
import argparse
from pathlib import Path

from models.lightning_module import BrainToTextLightningModule
from models.blocks.calibrator import FitCTCTemperature
from dataset.dataset import BrainToTextDataset
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="Temperature calibration example")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to data directory")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device for calibration (auto, cuda, mps, cpu)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of calibration epochs (temperature converges fast)")
    parser.add_argument("--lr", type=float, default=0.05,
                        help="Learning rate for temperature optimization")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for calibration")
    args = parser.parse_args()

    # 1. Load trained model
    print(f"Loading model from {args.checkpoint}")
    model = BrainToTextLightningModule.load_from_checkpoint(args.checkpoint)
    model.eval()
    
    # Handle device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            import os
            if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1":
                print("✅ MPS detected with fallback enabled - will use MPS with CPU fallback for CTC")
                device = "mps"
            else:
                print("⚠️  MPS detected but CTC requires CPU fallback. Using CPU instead.")
                print("   To use MPS: set PYTORCH_ENABLE_MPS_FALLBACK=1 before running")
                device = "cpu"
        else:
            device = "cpu"
        print(f"Auto-detected device: {device}")
    else:
        device = args.device
        # Warn if user explicitly chose MPS without fallback
        if device == "mps":
            import os
            if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "1":
                print("⚠️  Warning: MPS requires fallback for CTC Loss.")
                print("   Set PYTORCH_ENABLE_MPS_FALLBACK=1 or use CPU/CUDA")
    
    model = model.to(device)
    
    # 2. Prepare validation dataloader
    print("Preparing validation dataset...")
    # Note: You'll need to adapt this to your specific dataset setup
    val_dataset = BrainToTextDataset(
        data_dir=args.data_dir,
        split="val",  # or however you specify validation data
        # Add other dataset parameters as needed
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 3. Run temperature calibration
    print("Starting temperature calibration...")
    print(f"Initial temperature: {model.model.ctc_head.temperature():.4f}")
    
    calibrator = FitCTCTemperature(blank_idx=0)  # Assuming blank_idx=0
    optimal_temp = calibrator.run(
        model,
        val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        use_amp=(device == "cuda")  # Enable AMP only on CUDA
    )
    
    print(f"Calibration complete! Optimal temperature: {optimal_temp:.4f}")
    
    # 4. Save calibrated model
    output_path = Path(args.checkpoint).parent / "calibrated_model.ckpt"
    print(f"Saving calibrated model to {output_path}")
    
    # Save the full Lightning module with calibrated temperature
    torch.save({
        'state_dict': model.state_dict(),
        'hyper_parameters': model.hparams,
        'temperature': optimal_temp,
        'calibrated': True
    }, output_path)
    
    # 5. Important reminder about decoder tuning
    print("\n" + "="*60)
    print("🚨 IMPORTANT: Re-tune decoder hyperparameters!")
    print("="*60)
    print(f"Temperature scaling changed acoustic scores by factor 1/{optimal_temp:.4f}")
    print("You must re-tune the following decoder parameters:")
    print("- lm_weight (language model weight)")
    print("- word_insertion_penalty")
    print("- beam_threshold (if using beam search)")
    print("\nRecommended approach:")
    print("1. Run a small grid search on ~100-500 validation samples")
    print("2. Use Bayesian optimization for efficient hyperparameter search")
    print("3. Focus on lm_weight first, then fine-tune other parameters")
    print("="*60)


if __name__ == "__main__":
    main()
