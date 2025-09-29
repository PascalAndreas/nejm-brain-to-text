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
        
        # Numerically stable accumulation in fp32
        self.shadow_dtype = torch.float32
        
        # Create EMA parameters
        self.shadow = {}
        self.backup = {}
        
        # Initialize with model parameters - cast to fp32 and store on CPU initially
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Store in fp32 on CPU initially to avoid device mismatch during Lightning setup
                self.shadow[name] = param.detach().to(self.shadow_dtype).cpu()
    
    def get_decay(self) -> float:
        """Get decay rate with optional bias correction.
        
        Returns:
            Decay rate (potentially bias-corrected)
        """
        if self.use_bias_correction and self.updates < self.warmup_steps:
            # Linear warmup from 0 to base_decay over warmup_steps
            warmup_progress = self.updates / max(1, self.warmup_steps)
            decay = self.base_decay * warmup_progress
            # Clamp to valid range
            return float(max(0.0, min(self.base_decay, decay)))
        else:
            # Use fixed decay after warmup or if bias correction disabled
            return float(self.base_decay)
    
    @torch.no_grad()
    def update(self):
        """Update EMA parameters with current model weights."""
        decay = self.get_decay()
        
        # Collect tensors for batch operations
        dst_tensors = []
        src_tensors = []
        
        for name, param in self.model.named_parameters():
            if not (param.requires_grad and name in self.shadow):
                continue
                
            # Ensure shadow is on correct device and dtype (cache after first move)
            shadow = self.shadow[name]
            if shadow.device != param.device or shadow.dtype != self.shadow_dtype:
                shadow = shadow.to(param.device, dtype=self.shadow_dtype, non_blocking=True)
                self.shadow[name] = shadow
            
            dst_tensors.append(shadow)
            # Cast source to fp32 for stable accumulation
            src_tensors.append(param.detach().to(self.shadow_dtype))
        
        # Batch EMA update using foreach ops when available
        if dst_tensors:
            if hasattr(torch, '_foreach_mul_') and hasattr(torch, '_foreach_add_'):
                torch._foreach_mul_(dst_tensors, decay)
                torch._foreach_add_(dst_tensors, src_tensors, alpha=(1.0 - decay))
            else:
                # Fallback to individual updates
                for dst, src in zip(dst_tensors, src_tensors):
                    dst.mul_(decay).add_(src, alpha=(1.0 - decay))
        
        self.updates += 1
    
    
    def apply_shadow(self):
        """Apply EMA parameters to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.detach().clone()
                # Use in-place copy to avoid confusing optimizer state
                param.detach().copy_(
                    self.shadow[name].to(param.device, dtype=param.dtype, non_blocking=True)
                )
    
    def restore(self):
        """Restore original parameters after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                # Use in-place copy to avoid confusing optimizer state
                param.detach().copy_(self.backup[name])
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
            'ema_updates': self.updates
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
        ema_warmup_steps: int = 10,
        reset_schedulers_on_resume: bool = False
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
        
        # EMA setup with bias correction to prevent early training haunting
        self.use_ema = use_ema
        self.ema = None
        if use_ema:
            self.ema = EMA(
                self.model,
                decay=ema_decay,
                use_bias_correction=ema_bias_correction,
                warmup_steps=ema_warmup_steps
            )
        
        # Loss schedulers - will be properly initialized in configure_optimizers
        # when trainer and total_steps are available
        self.aux_scheduler = None
        self.reg_scheduler = None
        
        # Loss function
        self.ctc_loss = nn.CTCLoss(
            blank=self.model.blank_idx if hasattr(self.model, 'blank_idx') else 0,
            reduction='mean',
            zero_infinity=True
        )
        
        # Removed unused train_metrics dict
        
        # Removed unused best model tracking (Lightning handles this)
    
    def forward(self, batch: Batch, aux_weight: float = 1.0) -> Emissions:
        """Forward pass through the model.
        
        Args:
            batch: Input batch
            aux_weight: Current auxiliary loss weight (0.0 skips aux computation)
            
        Returns:
            Model emissions
        """
        return self.model(batch, aux_weight=aux_weight)
    
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
            w_aux_sched = float(self.aux_scheduler.get_weight(self.global_step))
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
                w_reg_eff = self.reg_scheduler.get_effective_weight(
                    self.global_step, 
                    reg_term.item(), 
                    sup_loss.detach().item()
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
        
        # Calculate auxiliary weight for conditional computation
        w_aux_base = float(self.aux_loss_config.get('weight', 0.0))
        if self.aux_scheduler:
            w_aux_sched = float(self.aux_scheduler.get_weight(self.global_step))
        else:
            w_aux_sched = 1.0
        w_aux = w_aux_base * w_aux_sched
        
        # Forward pass with aux weight for conditional computation
        emissions = self.forward(batch, aux_weight=w_aux)
        
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
        # Skip aux computation during validation for efficiency
        emissions = self.forward(batch, aux_weight=0.0)
        
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
        # Use PyTorch's optimized method instead of Python loop (6-75x faster!)
        unique_seq = torch.unique_consecutive(sequence, dim=-1)
        # Remove blanks
        no_blanks = unique_seq[unique_seq != blank_idx]
        return no_blanks
    
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
        
        # Get total training steps from trainer (now available during configure_optimizers)
        total_steps = None
        try:
            if hasattr(self, 'trainer') and self.trainer:
                total_steps = getattr(self.trainer, 'estimated_stepping_batches', None)
        except (RuntimeError, AttributeError):
            # Fallback: this should not happen in normal Lightning flow
            total_steps = None
        
        # Initialize loss schedulers now that we have total_steps
        if self.aux_scheduler is None:
            self.aux_scheduler = build_loss_scheduler(self.aux_loss_config, total_steps=total_steps)
        if self.reg_scheduler is None:
            self.reg_scheduler = build_loss_scheduler(self.film_reg_config, total_steps=total_steps)
        
        # Create learning rate scheduler with total_steps
        scheduler = build_lr_scheduler(optimizer, self.scheduler_config, total_steps)
        
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
        
        # Load scheduler states unless reset_schedulers_on_resume is True
        if not getattr(self.hparams, 'reset_schedulers_on_resume', False):
            if self.aux_scheduler is not None and 'aux_scheduler_state' in checkpoint:
                self.aux_scheduler.load_state_dict(checkpoint['aux_scheduler_state'])
                
            if self.reg_scheduler is not None and 'reg_scheduler_state' in checkpoint:
                self.reg_scheduler.load_state_dict(checkpoint['reg_scheduler_state'])
        else:
            # Schedulers will be freshly initialized with new total_steps in configure_optimizers
            print("Resetting schedulers - will use new schedule configuration")