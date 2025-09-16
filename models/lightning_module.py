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
        
        # Initialize with model parameters - will move to correct device on first update/apply
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Store on CPU initially to avoid device mismatch during Lightning setup
                self.shadow[name] = param.data.clone().cpu()
    
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
                # Ensure shadow tensor is on the same device as parameter
                if self.shadow[name].device != param.device:
                    self.shadow[name] = self.shadow[name].to(param.device, non_blocking=True)
                    
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
                # Ensure shadow tensor is on the same device as parameter
                param.data = self.shadow[name].to(param.device, non_blocking=True)
    
    def restore(self):
        """Restore original parameters after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self) -> Dict:
        """Get EMA state for checkpointing."""
        # Store shadow tensors on CPU to avoid large GPU checkpoints
        shadow_cpu = {k: v.cpu() for k, v in self.shadow.items()}
        return {
            'shadow': shadow_cpu,
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
            # Import locally to avoid circular import
            from . import build_encoder
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
        
        # Metrics - remove unused TorchMetrics objects since we use custom PER computation
        self.train_metrics = {
            'loss': [],
            'ctc_loss': [],
            'aux_loss': [],
            'reg_loss': []
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
            
            # Handle precision issues on MPS by casting to fp32 for CTC loss
            if log_probs.device.type == 'mps' and log_probs.dtype == torch.float16:
                log_probs = log_probs.float()
            
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
            
            # Handle precision issues on MPS by casting to fp32 for CTC loss
            if aux_log_probs.device.type == 'mps' and aux_log_probs.dtype == torch.float16:
                aux_log_probs = aux_log_probs.float()
            
            # Use aux_lengths if available, otherwise fall back to main output lengths
            aux_out_lens = emissions.aux.get('aux_lengths', emissions.out_lens)
            
            aux_loss = self.ctc_loss(
                aux_log_probs,
                batch.y,
                aux_out_lens,
                batch.y_lens
            )
            
            # Get weight from scheduler (base weight × schedule factor)
            base_weight = self.aux_loss_config.get('weight', 0.0)
            if self.aux_scheduler is not None:
                schedule_factor = self.aux_scheduler.get_weight(self.global_step)
            else:
                schedule_factor = 0.0
            
            aux_weight = base_weight * schedule_factor
            
            losses['aux_ctc_loss'] = aux_loss
            losses['aux_weight'] = torch.tensor(aux_weight)
            losses['aux_schedule_factor'] = torch.tensor(schedule_factor)
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
        
        # EMA update will be moved to optimizer_step to happen after weight updates
        # Log EMA stats periodically
        if self.use_ema and self.ema is not None and self.global_step % 100 == 0:
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
            
            # Compute PER with proper CTC decoding
            blank_idx = self.model.blank_idx if hasattr(self.model, 'blank_idx') else 0
            per = self.compute_per(predictions, batch.y, emissions.out_lens, batch.y_lens, blank_idx)
            self.log('val/per', per, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def compute_per(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        pred_lens: torch.Tensor,
        target_lens: torch.Tensor,
        blank_idx: int = 0
    ) -> float:
        """Compute phoneme error rate with proper CTC decoding.
        
        Args:
            predictions: Predicted phoneme sequences [B, T]
            targets: Target phoneme sequences [B, T] 
            pred_lens: Prediction lengths [B]
            target_lens: Target lengths [B]
            blank_idx: CTC blank token index
            
        Returns:
            Average PER across batch
        """
        batch_size = predictions.shape[0]
        total_errors = 0
        total_length = 0
        
        for i in range(batch_size):
            # Extract valid sequences
            pred_seq = predictions[i, :pred_lens[i]]
            target_seq = targets[i, :target_lens[i]]
            
            # Collapse CTC predictions (remove blanks and consecutive duplicates)
            pred_collapsed = self._ctc_collapse(pred_seq, blank_idx)
            
            # Compute edit distance
            errors = self._edit_distance(pred_collapsed, target_seq)
            total_errors += errors
            total_length += len(target_seq)
        
        return total_errors / max(total_length, 1)
    
    def _ctc_collapse(self, sequence: torch.Tensor, blank_idx: int = 0) -> torch.Tensor:
        """Collapse CTC sequence by removing blanks and consecutive duplicates.
        
        Args:
            sequence: Input sequence with blanks and repetitions
            blank_idx: Index of blank token
            
        Returns:
            Collapsed sequence
        """
        collapsed = []
        prev_token = None
        
        for token in sequence:
            token_val = token.item() if torch.is_tensor(token) else token
            # Skip blanks and consecutive duplicates
            if token_val != blank_idx and token_val != prev_token:
                collapsed.append(token_val)
            prev_token = token_val
        
        return torch.tensor(collapsed, dtype=sequence.dtype, device=sequence.device)
    
    def _edit_distance(self, seq1: torch.Tensor, seq2: torch.Tensor) -> int:
        """Compute Levenshtein edit distance between two sequences.
        
        Args:
            seq1: First sequence
            seq2: Second sequence
            
        Returns:
            Edit distance (number of insertions, deletions, substitutions)
        """
        # Convert to lists for easier manipulation
        if torch.is_tensor(seq1):
            seq1 = seq1.cpu().tolist()
        if torch.is_tensor(seq2):
            seq2 = seq2.cpu().tolist()
        
        m, n = len(seq1), len(seq2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i  # Deletions
        for j in range(n + 1):
            dp[0][j] = j  # Insertions
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]  # No operation needed
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # Deletion
                        dp[i][j-1],    # Insertion
                        dp[i-1][j-1]   # Substitution
                    )
        
        return dp[m][n]
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """Override optimizer step to update EMA after weight updates.
        
        Args:
            epoch: Current epoch
            batch_idx: Current batch index
            optimizer: Optimizer instance
            optimizer_closure: Closure for optimizer step
        """
        # Perform the actual optimizer step
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        
        # Update EMA after weights have been updated
        if self.use_ema and self.ema is not None:
            if self.ema.should_update(self.global_step):
                self.ema.update()

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