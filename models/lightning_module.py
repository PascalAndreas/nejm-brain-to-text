"""PyTorch Lightning module for training neural encoder models.

This module provides a Lightning wrapper for training with:
- Exponential Moving Average (EMA) of weights with bias correction
- Automatic mixed precision
- Learning rate scheduling
- Auxiliary loss scheduling
- Comprehensive logging with Weights & Biases
- Validation metrics
"""

from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
from torch.optim import AdamW
import pytorch_lightning as pl

from .base import Batch, Emissions, NeuralEncoder
from .gru_ctc import GRUCTC
from .schedulers import build_lr_scheduler, build_loss_scheduler, LossScheduler
from .blocks.utils import flatten_ctc_targets


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
        """Get decay rate (now fixed, bias correction removed for stability).
        
        Returns:
            Fixed decay rate
        """
        # Use fixed decay for stability - dynamic decay can hurt training
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
        less frequent later to save compute.
        
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
        
        # Store configs with defaults
        self.optimizer_config = {
            'lr': 3e-4,
            'weight_decay': 1e-2,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            **(optimizer_config or {})
        }
        
        self.scheduler_config = {
            'type': 'cosine',
            'warmup_steps': 5000,
            'max_lr': 3e-4,
            'min_lr': 1e-6,
            **(scheduler_config or {})
        }
        
        self.aux_loss_config = aux_loss_config or {}
        self.film_reg_config = (training_config or {}).get('film_reg', {'weight': 0.0})
        
        # Create model (simplified - always use GRUCTC for now)
        aux_layer = model_config.pop('aux_layer', self.aux_loss_config.get('layer', None))
        model_config['aux_layer'] = aux_layer
        self.model = GRUCTC(**model_config)
        
        # EMA setup (simplified with fixed decay)
        self.use_ema = use_ema
        self.ema = None
        if use_ema:
            self.ema = EMA(
                self.model,
                decay=ema_decay,
                use_bias_correction=False,  # Disabled for stability
                warmup_steps=0  # Not used with fixed decay
            )
        
        # Get total steps from scheduler config for fractional specifications
        total_steps = self.scheduler_config.get('T_max', None)
        
        # Unified loss schedulers
        self.aux_scheduler = build_loss_scheduler(self.aux_loss_config, total_steps)
        self.reg_scheduler = build_loss_scheduler(self.film_reg_config, total_steps)
        
        # Loss function
        self.ctc_loss = nn.CTCLoss(
            blank=self.model.blank_idx if hasattr(self.model, 'blank_idx') else 0,
            reduction='mean',
            zero_infinity=True
        )
        
        # Removed unused train_metrics dict
        
        # Removed unused best model tracking (Lightning handles this)
    
    def forward(self, batch: Batch) -> Emissions:
        """Forward pass through the model.
        
        Args:
            batch: Input batch
            
        Returns:
            Model emissions
        """
        return self.model(batch)
    
    def forward_for_calibration(self, batch):
        """Forward pass for temperature calibration.
        
        Args:
            batch: Input batch
            
        Returns:
            Dict with logits, input_lengths, targets, target_lengths
        """
        return self.model.forward_for_calibration(batch)
    
    def compute_loss(
        self,
        batch: Batch,
        emissions: Emissions
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute total loss with scale-stable supervised loss mixing and additive FiLM penalty.
        
        Args:
            batch: Input batch with targets
            emissions: Model outputs
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        losses: Dict[str, torch.Tensor] = {}

        # --- main CTC ---
        if batch.y is None or batch.y_lens is None:
            return torch.tensor(0.0, device=self.device), losses
            
        log_probs = emissions.log_probs.transpose(0, 1)  # [T,B,V]
        if log_probs.device.type == 'mps' and log_probs.dtype == torch.float16:
            log_probs = log_probs.float()

        # Flatten targets once (robust to any dataset padding)
        targets_1d = flatten_ctc_targets(batch.y, batch.y_lens)
        ctc_loss = self.ctc_loss(log_probs, targets_1d, emissions.out_lens, batch.y_lens)
        losses['ctc_loss'] = ctc_loss
        sup_loss = ctc_loss  # will be normalized mixture below

        # --- auxiliary CTC (optional) ---
        has_aux = bool(emissions.aux and 'aux_log_probs' in emissions.aux)
        w_aux_base = float(self.aux_loss_config.get('weight', 0.0))
        if self.aux_scheduler:
            total_steps = getattr(self.trainer, 'estimated_stepping_batches', None) if self.trainer else None
            w_aux_sched = float(self.aux_scheduler.get_weight(self.global_step, total_steps))
        else:
            w_aux_sched = 1.0
        w_aux = w_aux_base * w_aux_sched

        if has_aux and w_aux > 0:
            aux_lp = emissions.aux['aux_log_probs'].transpose(0, 1)  # [T,B,V]
            if aux_lp.device.type == 'mps' and aux_lp.dtype == torch.float16:
                aux_lp = aux_lp.float()
            aux_lens = emissions.aux.get('aux_lengths', emissions.out_lens)
            aux_ctc = self.ctc_loss(aux_lp, targets_1d, aux_lens, batch.y_lens)
            losses['aux_ctc_loss'] = aux_ctc

            # scale-stable mixture (supervised only)
            sup_den = 1.0 + w_aux
            sup_loss = (ctc_loss + w_aux * aux_ctc) / sup_den

            # log weights/factors
            losses['aux_weight'] = torch.tensor(w_aux, device=self.device)
            losses['aux_schedule_factor'] = torch.tensor(w_aux_sched, device=self.device)
        else:
            # still log weights for completeness
            losses['aux_weight'] = torch.tensor(w_aux, device=self.device)
            losses['aux_schedule_factor'] = torch.tensor(w_aux_sched, device=self.device)

        # --- FiLM regularizer (optional, additive penalty) ---
        reg_term = emissions.aux.get('film_reg_loss', None) if (emissions.aux is not None) else None
        
        if reg_term is not None:
            reg_term = reg_term.to(self.device)
            losses['film_reg_loss'] = reg_term
            
            # Get effective weight with scheduler and fraction guard
            if self.reg_scheduler is not None:
                total_steps = getattr(self.trainer, 'estimated_stepping_batches', None) if self.trainer else None
                w_reg_eff = self.reg_scheduler.get_effective_weight(
                    self.global_step, 
                    reg_term.item(), 
                    sup_loss.detach().item(),
                    total_steps
                )
            else:
                w_reg_eff = 0.0
            
            losses['film_reg_weight'] = torch.tensor(w_reg_eff, device=self.device)
            
            if w_reg_eff > 0:
                total_loss = sup_loss + w_reg_eff * reg_term
            else:
                total_loss = sup_loss
        else:
            total_loss = sup_loss
            # Log zero weight for completeness
            losses['film_reg_weight'] = torch.tensor(0.0, device=self.device)

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
        
        # Log regularization to supervised ratio for sanity checking
        if 'film_reg_loss' in loss_dict and 'film_reg_weight' in loss_dict and 'ctc_loss' in loss_dict:
            reg_contrib = loss_dict['film_reg_weight'] * loss_dict['film_reg_loss']
            sup_base = loss_dict['ctc_loss']
            if 'aux_ctc_loss' in loss_dict and 'aux_weight' in loss_dict:
                # Include aux in supervised base for ratio calculation
                aux_contrib = loss_dict['aux_weight'] * loss_dict['aux_ctc_loss']
                sup_base = (sup_base + aux_contrib) / (1.0 + loss_dict['aux_weight'])
            
            reg_to_sup_ratio = reg_contrib / (sup_base + 1e-12)
            self.log('train/reg_to_sup_ratio', reg_to_sup_ratio, on_step=True, on_epoch=True)
        
        # Log learning rate and EMA stats periodically
        if self.global_step % 100 == 0:
            # Log current learning rate
            if self.lr_schedulers():
                current_lr = self.lr_schedulers().get_last_lr()[0]
                self.log('train/lr', current_lr)
            
            # Log EMA stats
            if self.use_ema and self.ema is not None:
                ema_stats = self.ema.get_stats()
                for key, value in ema_stats.items():
                    self.log(f'train/{key}', value)
        
        # Step auxiliary scheduler if present
        if self.aux_scheduler is not None:
            self.aux_scheduler.step()
        
        return loss
    
    def on_validation_epoch_start(self):
        """Apply EMA weights at start of validation epoch."""
        if self.use_ema and self.ema is not None:
            self.ema.apply_shadow()
    
    def on_validation_epoch_end(self):
        """Restore original weights after validation epoch."""
        if self.use_ema and self.ema is not None:
            self.ema.restore()
    
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
        
        # Forward pass (EMA weights already applied at epoch start)
        emissions = self.forward(batch)
        
        # Compute loss
        loss, loss_dict = self.compute_loss(batch, emissions)
        
        # Log losses
        for key, value in loss_dict.items():
            self.log(f'val/{key}', value, on_step=False, on_epoch=True, prog_bar=(key == 'total_loss'))
        
        # Compute phoneme error rate
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
    
    def calibrate_temperature(self, val_dataloader) -> float:
        """Calibrate temperature scaling on validation data.
        
        This should be called after training is complete to fit the temperature
        parameter that improves calibration.
        
        Args:
            val_dataloader: Validation dataloader
            
        Returns:
            Optimal temperature value
        """
        if not hasattr(self.model, 'eval_calibrate'):
            print("Warning: Model does not support temperature calibration")
            return 1.0
            
        # Use the same device as the model
        device = str(next(self.model.parameters()).device)
        return self.model.eval_calibrate(val_dataloader, device=device)
    
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
        """Save EMA and loss scheduler states in checkpoint.
        
        Args:
            checkpoint: Checkpoint dictionary
        """
        if self.use_ema and self.ema is not None:
            checkpoint['ema_state'] = self.ema.state_dict()
        
        if self.aux_scheduler is not None:
            checkpoint['aux_scheduler_state'] = self.aux_scheduler.state_dict()
            
        if self.reg_scheduler is not None:
            checkpoint['reg_scheduler_state'] = self.reg_scheduler.state_dict()
    
    def on_load_checkpoint(self, checkpoint: Dict) -> None:
        """Load EMA and loss scheduler states from checkpoint.
        
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
            
        if self.reg_scheduler is not None and 'reg_scheduler_state' in checkpoint:
            self.reg_scheduler.load_state_dict(checkpoint['reg_scheduler_state'])