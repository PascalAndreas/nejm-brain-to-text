#!/usr/bin/env python
"""Training script for brain-to-text neural encoder models.

This script provides the main entry point for training models using
PyTorch Lightning with comprehensive logging and checkpointing.
"""

import os
import sys
from pathlib import Path
import argparse
from typing import Dict, Any, Optional
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
    RichProgressBar,
    RichModelSummary
)
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from dataset import BrainToTextDataset, collate_fn
from models.lightning_module import BrainToTextLightningModule
from models import build_encoder


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloaders(config: Dict[str, Any]) -> Dict[str, DataLoader]:
    """Create data loaders for training and validation.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with 'train' and 'val' dataloaders
    """
    dataset_config = config['dataset']
    
    dataloaders = {}
    
    for split in ['train', 'val']:
        if split not in dataset_config.get('splits', ['train', 'val']):
            continue
        
        dataset = BrainToTextDataset(
            data_root=dataset_config['data_root'],
            split=split,
            corpus_filter=dataset_config.get('corpus_filter'),
            bad_trials_dict=dataset_config.get('bad_trials_dict'),
            random_seed=dataset_config.get('random_seed', -1)
        )
        
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=dataset_config['batch_size'],
            shuffle=(split == 'train'),
            num_workers=dataset_config.get('num_workers', 4),
            collate_fn=collate_fn,
            pin_memory=dataset_config.get('pin_memory', True),
            persistent_workers=dataset_config.get('num_workers', 4) > 0
        )
    
    return dataloaders


def create_callbacks(config: Dict[str, Any]) -> list:
    """Create Lightning callbacks for training.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of callbacks
    """
    callbacks = []
    
    # Model checkpointing
    checkpoint_config = config.get('checkpointing', {})
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_config.get('save_dir', 'models/checkpoints'),
        filename='{epoch:02d}-{val_per:.4f}',
        monitor=checkpoint_config.get('monitor', 'val/per'),
        mode=checkpoint_config.get('mode', 'min'),
        save_top_k=checkpoint_config.get('save_top_k', 3),
        save_last=checkpoint_config.get('save_last', True),
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Early stopping (optional)
    if config.get('training', {}).get('early_stopping'):
        early_stop = EarlyStopping(
            monitor='val/per',
            patience=10,
            mode='min',
            verbose=True
        )
        callbacks.append(early_stop)
    
    # Rich progress bar for better visualization
    progress_bar = RichProgressBar()
    callbacks.append(progress_bar)
    
    # Model summary
    model_summary = RichModelSummary(max_depth=2)
    callbacks.append(model_summary)
    
    return callbacks


def create_logger(config: Dict[str, Any]):
    """Create Weights & Biases logger for experiment tracking.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        WandbLogger instance
    """
    logging_config = config.get('logging', {})
    
    logger = WandbLogger(
        project=logging_config.get('project', 'brain-to-text'),
        name=logging_config.get('name', 'experiment'),
        save_dir=logging_config.get('save_dir', 'logs'),
        log_model=logging_config.get('log_model', False),
        tags=config.get('experiment', {}).get('tags'),
        notes=config.get('experiment', {}).get('notes')
    )
    
    return logger


def main(args):
    """Main training function.
    
    Args:
        args: Command line arguments
    """
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.batch_size:
        config['dataset']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['optimizer']['lr'] = args.lr
    if args.epochs:
        config['training']['max_epochs'] = args.epochs
    
    # Set random seeds
    if config.get('experiment', {}).get('deterministic'):
        pl.seed_everything(config.get('experiment', {}).get('seed', 42))
    
    # Create data loaders
    dataloaders = create_dataloaders(config)
    
    # Create model
    model = BrainToTextLightningModule(
        model_config=config['model']['params'],
        optimizer_config=config['training'].get('optimizer'),
        scheduler_config=config['training'].get('scheduler'),
        training_config=config['training'],
        use_ema=config['training'].get('use_ema', True),
        ema_decay=config['training'].get('ema_decay', 0.999),
        ema_bias_correction=config['training'].get('ema_bias_correction', True),
        ema_warmup_steps=config['training'].get('ema_warmup_steps', 10)
    )
    
    # Create callbacks
    callbacks = create_callbacks(config)
    
    # Create logger
    logger = create_logger(config)
    
    # Create trainer
    hardware_config = config.get('hardware', {})
    trainer = Trainer(
        max_epochs=config['training'].get('max_epochs', 100),
        accelerator=hardware_config.get('accelerator', 'auto'),
        devices=hardware_config.get('devices', 1),
        strategy=hardware_config.get('strategy', 'auto'),
        precision=config['training'].get('precision', 32),
        gradient_clip_val=config['training'].get('gradient_clip_val', 1.0),
        accumulate_grad_batches=config['training'].get('accumulate_grad_batches', 1),
        val_check_interval=config['training'].get('val_check_interval', 1.0),
        log_every_n_steps=config['training'].get('log_every_n_steps', 50),
        callbacks=callbacks,
        logger=logger,
        deterministic=config.get('experiment', {}).get('deterministic', False),
        benchmark=config.get('experiment', {}).get('benchmark', False),
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    # Resume from checkpoint if specified
    ckpt_path = args.resume if args.resume else None
    
    # Train model
    trainer.fit(
        model,
        train_dataloaders=dataloaders.get('train'),
        val_dataloaders=dataloaders.get('val'),
        ckpt_path=ckpt_path
    )
    
    # Temperature calibration on validation set
    if not args.skip_calibration:
        print("\nPerforming temperature calibration...")
        model.model.eval_calibrate(dataloaders['val'])
        
        # Save calibrated model
        calibrated_path = Path(config['checkpointing']['save_dir']) / 'calibrated_model.pt'
        torch.save({
            'model_state_dict': model.model.state_dict(),
            'model_config': model.model.export_config(),
            'ema_state': model.ema.state_dict() if model.ema else None
        }, calibrated_path)
        print(f"Saved calibrated model to {calibrated_path}")
    
    # Test model if test set is available
    if 'test' in config['dataset'].get('splits', []) and not args.skip_test:
        test_dataset = BrainToTextDataset(
            data_root=config['dataset']['data_root'],
            split='test',
            corpus_filter=config['dataset'].get('corpus_filter'),
            bad_trials_dict=config['dataset'].get('bad_trials_dict')
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config['dataset']['batch_size'],
            shuffle=False,
            num_workers=config['dataset'].get('num_workers', 4),
            collate_fn=collate_fn,
            pin_memory=config['dataset'].get('pin_memory', True)
        )
        
        print("\nEvaluating on test set...")
        trainer.test(model, dataloaders=test_dataloader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train brain-to-text neural encoder')
    parser.add_argument(
        '--config',
        type=str,
        default='pipeline/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override batch size from config'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Override learning rate from config'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override number of epochs from config'
    )
    parser.add_argument(
        '--skip-calibration',
        action='store_true',
        help='Skip temperature calibration'
    )
    parser.add_argument(
        '--skip-test',
        action='store_true',
        help='Skip test set evaluation'
    )
    
    args = parser.parse_args()
    main(args)
