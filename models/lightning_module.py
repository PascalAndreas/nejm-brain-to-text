"""PyTorch Lightning module for training neural encoder models.

This module provides a Lightning wrapper for training with:
- Exponential Moving Average (EMA) of weights with bias correction
- Automatic mixed precision
- Learning rate scheduling
- Auxiliary loss scheduling
- Comprehensive logging with Weights & Biases
- Validation metrics
"""

from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import pytorch_lightning as pl
from torchmetrics import Accuracy, CharErrorRate, WordErrorRate
import numpy as np
from copy import deepcopy

from .base import Batch, Emissions, NeuralEncoder
from .gru_ctc import GRUCTC
from . import build_encoder
from .schedulers import build_lr_scheduler, build_aux_scheduler, AuxiliaryLossScheduler


class EMA:
    """Exponential Moving Average of model parameters with bias correction.
    
    This class maintains a moving average of model parameters which often
    provides better generalization than the raw trained weights. Includes
    bias correction and adaptive update scheduling.
    
    Args:
        model: The model to track
        decay: Base EMA decay rate (e.g., 0.999)
        use_bias_correction: Whether to use bias correction
        warmup_steps: Number of warmup steps for bias correction
        device: Device for EMA parameters
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        use_bias_correction: bool = True,
        warmup_steps: int = 10,
        device: str = 'cpu'
    ):
        self.model = model
        self.base_decay = decay
        self.use_bias_correction = use_bias_correction
        self.warmup_steps = warmup_steps
        self.device = device
        self.updates = 0
        
        # Create EMA parameters
        self.shadow = {}
        self.backup = {}
        
        # Initialize with model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(device)
    
    def get_decay(self) -> float:
        """Get effective decay rate with bias correction.
        
        Returns:
            Effective decay rate
        """
        if self.use_bias_correction:
            # Bias-corrected decay that starts low and increases to base_decay
            effective_decay = min(
                self.base_decay,
                (1 + self.updates) / (self.warmup_steps + self.updates)
            )
            return effective_decay
        return self.base_decay
    
    @torch.no_grad()
    def update(self):
        """Update EMA parameters with current model weights."""
        decay = self.get_decay()
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    decay * self.shadow[name] +
                    (1.0 - decay) * param.data
                )
        
        self.updates += 1
    
    def should_update(self, step: int) -> bool:
        """Determine if EMA should be updated at this step.
        
        Uses adaptive scheduling: more frequent updates early in training,
        less frequent later.
        
        Args:
            step: Current training step
            
        Returns:
            True if EMA should be updated
        """
        # More frequent early (every step), less frequent later (every 50 steps)
        update_freq = min(50, max(1, step // 1000))
        return step % update_freq == 0
    
    def apply_shadow(self):
        """Apply EMA parameters to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self) -> Dict:
        """Get EMA state for checkpointing."""
        return {
            'shadow': self.shadow,
            'base_decay': self.base_decay,
            'updates': self.updates,
            'use_bias_correction': self.use_bias_correction,
            'warmup_steps': self.warmup_steps
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load EMA state from checkpoint."""
        self.shadow = state_dict['shadow']
        self.base_decay = state_dict.get('base_decay', self.base_decay)
        self.updates = state_dict.get('updates', 0)
        self.use_bias_correction = state_dict.get('use_bias_correction', True)
        self.warmup_steps = state_dict.get('warmup_steps', 10)
    
    def get_stats(self) -> Dict[str, float]:
        """Get EMA statistics for logging.
        
        Returns:
            Dictionary with EMA stats
        """
        return {
            'ema_decay': self.get_decay(),
            'ema_updates': self.updates,
            'ema_update_freq': min(50, max(1, self.updates // 1000))
        }


class BrainToTextLightningModule(pl.LightningModule):
    """Lightning module for brain-to-text neural encoder training.
    
    Args:
        model_config: Configuration for the neural encoder
        optimizer_config: Configuration for optimizer
        scheduler_config: Configuration for learning rate scheduler
        aux_loss_config: Configuration for auxiliary loss
        training_config: Additional training configuration
        use_ema: Whether to use EMA of weights
        ema_decay: Base EMA decay rate
        ema_bias_correction: Whether to use bias correction for EMA
        ema_warmup_steps: Number of warmup steps for EMA bias correction
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        optimizer_config: Optional[Dict[str, Any]] = None,
        scheduler_config: Optional[Dict[str, Any]] = None,
        aux_loss_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        ema_bias_correction: bool = True,
        ema_warmup_steps: int = 10
    ):
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Default configurations
        self.optimizer_config = optimizer_config or {
            'lr': 3e-4,
            'weight_decay': 1e-2,
            'betas': (0.9, 0.999),
            'eps': 1e-8
        }
        
        self.scheduler_config = scheduler_config or {
            'type': 'cosine',
            'warmup_steps': 5000,
            'max_lr': 3e-4,
            'min_lr': 1e-6
        }
        
        self.aux_loss_config = aux_loss_config or {}
        
        self.training_config = training_config or {
            'gradient_clip_val': 1.0,
            'accumulate_grad_batches': 1,
            'val_check_interval': 1000,
            'log_every_n_steps': 100
        }
        
        # Extract auxiliary layer from model config if present
        if 'aux_layer' in model_config:
            aux_layer = model_config.pop('aux_layer')
        else:
            aux_layer = self.aux_loss_config.get('layer', None)
        
        # Create model
        if 'name' in model_config:
            # Add aux_layer to params if needed
            params = model_config.get('params', {})
            params['aux_layer'] = aux_layer
            self.model = build_encoder(
                model_config['name'],
                params
            )
        else:
            # Default to GRU-CTC
            model_config['aux_layer'] = aux_layer
            self.model = GRUCTC(**model_config)
        
        # EMA setup with bias correction
        self.use_ema = use_ema
        self.ema = None
        if use_ema:
            self.ema = EMA(
                self.model,
                decay=ema_decay,
                use_bias_correction=ema_bias_correction,
                warmup_steps=ema_warmup_steps,
                device=self.device
            )
        
        # Auxiliary loss scheduler
        self.aux_scheduler = build_aux_scheduler(self.aux_loss_config)
        
        # Loss function
        self.ctc_loss = nn.CTCLoss(
            blank=self.model.blank_idx if hasattr(self.model, 'blank_idx') else 0,
            reduction='mean',
            zero_infinity=True
        )
        
        # Metrics
        self.train_metrics = {
            'loss': [],
            'ctc_loss': [],
            'aux_loss': [],
            'reg_loss': []
        }
        
        self.val_metrics = {
            'loss': [],
            'per': CharErrorRate(),  # Phoneme error rate
            'cer': CharErrorRate(),  # Character error rate (if decoded)
            'wer': WordErrorRate()   # Word error rate (if decoded)
        }
        
        # For tracking best model
        self.best_val_loss = float('inf')
        self.best_val_per = float('inf')
    
    def forward(self, batch: Batch) -> Emissions:
        """Forward pass through the model.
        
        Args:
            batch: Input batch
            
        Returns:
            Model emissions
        """
        return self.model(batch)
    
    def compute_loss(
        self,
        batch: Batch,
        emissions: Emissions
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute total loss and individual loss components.
        
        Args:
            batch: Input batch with targets
            emissions: Model outputs
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        losses = {}
        
        # Main CTC loss
        if batch.y is not None and batch.y_lens is not None:
            # Transpose for CTC (expects [T, B, V])
            log_probs = emissions.log_probs.transpose(0, 1)
            
            ctc_loss = self.ctc_loss(
                log_probs,
                batch.y,
                emissions.out_lens,
                batch.y_lens
            )
            losses['ctc_loss'] = ctc_loss
            total_loss = ctc_loss
        else:
            # No targets available (inference mode)
            return torch.tensor(0.0), losses
        
        # Auxiliary CTC loss if available
        if emissions.aux and 'aux_log_probs' in emissions.aux:
            aux_log_probs = emissions.aux['aux_log_probs'].transpose(0, 1)
            aux_loss = self.ctc_loss(
                aux_log_probs,
                batch.y,
                emissions.out_lens,
                batch.y_lens
            )
            
            # Get weight from scheduler
            if self.aux_scheduler is not None:
                aux_weight = self.aux_scheduler.get_weight(self.global_step)
            else:
                aux_weight = 0.0
            
            losses['aux_ctc_loss'] = aux_loss
            losses['aux_weight'] = torch.tensor(aux_weight)
            total_loss = total_loss + aux_weight * aux_loss
        
        # FiLM regularization loss
        if emissions.aux and 'film_reg_loss' in emissions.aux:
            reg_loss = emissions.aux['film_reg_loss']
            losses['film_reg_loss'] = reg_loss
            total_loss = total_loss + reg_loss
        
        losses['total_loss'] = total_loss
        return total_loss, losses
    
    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Training step.
        
        Args:
            batch: Raw batch from dataloader
            batch_idx: Batch index
            
        Returns:
            Loss value for backpropagation
        """
        # Convert to standard batch format
        batch = Batch.from_dataset_batch(batch)
        
        # Forward pass
        emissions = self.forward(batch)
        
        # Compute loss
        loss, loss_dict = self.compute_loss(batch, emissions)
        
        # Log losses
        for key, value in loss_dict.items():
            self.log(f'train/{key}', value, on_step=True, on_epoch=True, prog_bar=(key == 'total_loss'))
        
        # Update EMA with adaptive scheduling
        if self.use_ema and self.ema is not None:
            if self.ema.should_update(self.global_step):
                self.ema.update()
                
                # Log EMA stats periodically
                if self.global_step % 100 == 0:
                    ema_stats = self.ema.get_stats()
                    for key, value in ema_stats.items():
                        self.log(f'train/{key}', value)
        
        # Step auxiliary scheduler if present
        if self.aux_scheduler is not None:
            self.aux_scheduler.step()
        
        return loss
    
    def validation_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        """Validation step.
        
        Args:
            batch: Raw batch from dataloader
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        # Convert to standard batch format
        batch = Batch.from_dataset_batch(batch)
        
        # Apply EMA weights for validation
        if self.use_ema and self.ema is not None:
            self.ema.apply_shadow()
        
        # Forward pass
        emissions = self.forward(batch)
        
        # Compute loss
        loss, loss_dict = self.compute_loss(batch, emissions)
        
        # Restore original weights
        if self.use_ema and self.ema is not None:
            self.ema.restore()
        
        # Log losses
        for key, value in loss_dict.items():
            self.log(f'val/{key}', value, on_step=False, on_epoch=True, prog_bar=(key == 'total_loss'))
        
        # Compute phoneme error rate (simplified - actual implementation would use greedy decoding)
        if batch.y is not None:
            # Greedy decoding
            predictions = torch.argmax(emissions.log_probs, dim=-1)
            
            # Simple PER calculation (you'd want proper CTC decoding here)
            per = self.compute_per(predictions, batch.y, emissions.out_lens, batch.y_lens)
            self.log('val/per', per, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def compute_per(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        pred_lens: torch.Tensor,
        target_lens: torch.Tensor
    ) -> float:
        """Compute phoneme error rate.
        
        Args:
            predictions: Predicted phoneme sequences
            targets: Target phoneme sequences
            pred_lens: Prediction lengths
            target_lens: Target lengths
            
        Returns:
            Average PER across batch
        """
        batch_size = predictions.shape[0]
        total_errors = 0
        total_length = 0
        
        for i in range(batch_size):
            pred_seq = predictions[i, :pred_lens[i]]
            target_seq = targets[i, :target_lens[i]]
            
            # Remove blanks and repetitions (simplified CTC collapse)
            pred_seq = self._ctc_collapse(pred_seq)
            
            # Compute edit distance
            errors = self._edit_distance(pred_seq, target_seq)
            total_errors += errors
            total_length += len(target_seq)
        
        return total_errors / max(total_length, 1)
    
    def _ctc_collapse(self, sequence: torch.Tensor) -> torch.Tensor:
        """Collapse CTC sequence by removing blanks and repetitions.
        
        Args:
            sequence: Input sequence with blanks and repetitions
            
        Returns:
            Collapsed sequence
        """
        # Remove consecutive duplicates and blanks (assuming blank=0)
        collapsed = []
        prev = -1
        
        for token in sequence:
            if token != 0 and token != prev:  # Not blank and not repetition
                collapsed.append(token)
            prev = token
        
        return torch.tensor(collapsed)
    
    def _edit_distance(self, seq1: torch.Tensor, seq2: torch.Tensor) -> int:
        """Compute edit distance between two sequences.
        
        Args:
            seq1: First sequence
            seq2: Second sequence
            
        Returns:
            Edit distance
        """
        # Convert to numpy for easier manipulation
        seq1 = seq1.cpu().numpy() if torch.is_tensor(seq1) else seq1
        seq2 = seq2.cpu().numpy() if torch.is_tensor(seq2) else seq2
        
        # Dynamic programming for edit distance
        m, n = len(seq1), len(seq2)
        dp = np.zeros((m + 1, n + 1))
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return int(dp[m][n])
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler.
        
        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        # Create optimizer
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.optimizer_config['lr'],
            weight_decay=self.optimizer_config['weight_decay'],
            betas=self.optimizer_config.get('betas', (0.9, 0.999)),
            eps=self.optimizer_config.get('eps', 1e-8)
        )
        
        # Create learning rate scheduler
        scheduler = build_lr_scheduler(optimizer, self.scheduler_config)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
    
    def on_save_checkpoint(self, checkpoint: Dict) -> None:
        """Save EMA and auxiliary scheduler state in checkpoint.
        
        Args:
            checkpoint: Checkpoint dictionary
        """
        if self.use_ema and self.ema is not None:
            checkpoint['ema_state'] = self.ema.state_dict()
        
        if self.aux_scheduler is not None:
            checkpoint['aux_scheduler_state'] = self.aux_scheduler.state_dict()
    
    def on_load_checkpoint(self, checkpoint: Dict) -> None:
        """Load EMA and auxiliary scheduler state from checkpoint.
        
        Args:
            checkpoint: Checkpoint dictionary
        """
        if self.use_ema and 'ema_state' in checkpoint:
            if self.ema is None:
                self.ema = EMA(
                    self.model,
                    decay=self.hparams.ema_decay,
                    use_bias_correction=self.hparams.get('ema_bias_correction', True),
                    warmup_steps=self.hparams.get('ema_warmup_steps', 10)
                )
            self.ema.load_state_dict(checkpoint['ema_state'])
        
        if self.aux_scheduler is not None and 'aux_scheduler_state' in checkpoint:
            self.aux_scheduler.load_state_dict(checkpoint['aux_scheduler_state'])