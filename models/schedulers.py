"""Learning rate and auxiliary loss schedulers for training.

This module provides scheduler implementations for various training
parameters including learning rates and auxiliary loss weights.
"""

from typing import Dict, Any, Optional
import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR, OneCycleLR, LambdaLR


class WarmupCosineScheduler(_LRScheduler):
    """Cosine annealing with linear warmup.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        min_lr: Minimum learning rate
        last_epoch: The index of last epoch
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate learning rate for current step."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * self.last_epoch / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return [
                self.min_lr + (base_lr - self.min_lr) * 
                0.5 * (1 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]


class LossScheduler:
    """Unified scheduler for auxiliary and regularization loss weights.
    
    Supports various scheduling patterns with fractional step specifications only.
    Can handle both auxiliary CTC loss and FiLM regularization scheduling.
    
    Args:
        base_weight: Base weight for the loss component
        schedule_type: Type of schedule ('constant', 'warmup', 'warmup_decay', 'warmup_hold_decay')
        warmup_fraction: Fraction of total steps to warm up (0 to base_weight)
        hold_fraction: Fraction of total steps to hold at base_weight (for warmup_hold_decay)
        decay_fraction: Fraction of total steps for final decay
        total_steps: Total training steps
        max_fraction_of_supervised: Optional fraction guard (for regularization)
    """
    
    def __init__(
        self,
        base_weight: float = 1.0,
        schedule_type: str = 'constant',
        warmup_fraction: float = 0.05,
        hold_fraction: Optional[float] = None,
        decay_fraction: Optional[float] = None,
        total_steps: Optional[int] = None,
        max_fraction_of_supervised: Optional[float] = None
    ):
        self.base_weight = base_weight
        self.schedule_type = schedule_type
        self.warmup_fraction = warmup_fraction
        self.hold_fraction = hold_fraction
        self.decay_fraction = decay_fraction
        self.total_steps = total_steps
        self.max_fraction_of_supervised = max_fraction_of_supervised
        self.current_step = 0
    
    def step(self) -> float:
        """Get weight for current step and increment counter."""
        weight = self.get_weight(self.current_step)
        self.current_step += 1
        return weight
    
    def get_weight(self, step: int, total_steps: Optional[int] = None) -> float:
        """Get loss weight for a given step.
        
        Args:
            step: Training step
            total_steps: Total steps (overrides instance total_steps if provided)
            
        Returns:
            Loss weight
        """
        T = total_steps or self.total_steps
        if T is None and self.schedule_type != 'constant':
            raise ValueError(f"LossScheduler with schedule_type '{self.schedule_type}' requires total_steps to be set")
        
        if self.schedule_type == 'constant':
            return self.base_weight
        
        elif self.schedule_type == 'warmup':
            # Linear warmup only
            W = int(self.warmup_fraction * T)
            if step < W:
                return self.base_weight * step / max(1, W)
            else:
                return self.base_weight
        
        elif self.schedule_type == 'warmup_decay':
            # Warmup then immediate cosine decay (for regularization)
            W = int(self.warmup_fraction * T)
            D = int(self.decay_fraction * T) if self.decay_fraction else int(0.2 * T)
            
            if step < W:
                # Linear warmup from 0 to base_weight
                return self.base_weight * step / max(1, W)
            elif step < (T - D):
                # Hold at base_weight
                return self.base_weight
            else:
                # Cosine decay to ~0
                t = step - (T - D)
                return self.base_weight * 0.5 * (1.0 + math.cos(math.pi * t / max(1, D)))
        
        elif self.schedule_type == 'warmup_hold_decay':
            # Warmup, hold, then decay (for auxiliary loss)
            W = int(self.warmup_fraction * T)
            H = int(self.hold_fraction * T) if self.hold_fraction else int(0.7 * T)
            D = int(self.decay_fraction * T) if self.decay_fraction else int(0.2 * T)
            
            if step < W:
                # Linear warmup
                return self.base_weight * step / max(1, W)
            elif step < (W + H):
                # Hold constant
                return self.base_weight
            else:
                # Cosine decay to 0
                decay_start = W + H
                decay_progress = min(1.0, (step - decay_start) / max(1, D))
                return self.base_weight * 0.5 * (1 + math.cos(decay_progress * math.pi))
        
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
    
    def get_effective_weight(
        self, 
        step: int, 
        loss_value: float, 
        supervised_loss: float
    ) -> float:
        """Get effective weight with optional fraction-of-supervised guard.
        
        Args:
            step: Current training step
            loss_value: Current loss component value
            supervised_loss: Current supervised loss value
            
        Returns:
            Effective weight (potentially clamped by fraction guard)
        """
        base_weight = self.get_weight(step)
        
        if (self.max_fraction_of_supervised is None or 
            self.max_fraction_of_supervised <= 0 or 
            loss_value <= 0):
            return base_weight
        
        # Compute fraction guard: λ_eff * L ≤ α * L_sup
        max_allowed = self.max_fraction_of_supervised * supervised_loss / (loss_value + 1e-12)
        return min(base_weight, max_allowed)
    
    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state for checkpointing."""
        return {
            'current_step': self.current_step,
            'base_weight': self.base_weight,
            'schedule_type': self.schedule_type,
            'warmup_fraction': self.warmup_fraction,
            'hold_fraction': self.hold_fraction,
            'decay_fraction': self.decay_fraction,
            'total_steps': self.total_steps,
            'max_fraction_of_supervised': self.max_fraction_of_supervised
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load scheduler state from checkpoint."""
        self.current_step = state_dict.get('current_step', 0)
        self.base_weight = state_dict.get('base_weight', self.base_weight)
        self.schedule_type = state_dict.get('schedule_type', self.schedule_type)
        self.warmup_fraction = state_dict.get('warmup_fraction', self.warmup_fraction)
        self.hold_fraction = state_dict.get('hold_fraction', self.hold_fraction)
        self.decay_fraction = state_dict.get('decay_fraction', self.decay_fraction)
        self.total_steps = state_dict.get('total_steps', self.total_steps)
        self.max_fraction_of_supervised = state_dict.get('max_fraction_of_supervised', self.max_fraction_of_supervised)


def build_lr_scheduler(
    optimizer: Optimizer,
    config: Dict[str, Any],
    total_steps: Optional[int] = None
) -> _LRScheduler:
    """Build learning rate scheduler from configuration.
    
    Args:
        optimizer: Optimizer to wrap
        config: Scheduler configuration
        total_steps: Total training steps (for some schedulers)
        
    Returns:
        Learning rate scheduler
    """
    scheduler_type = config.get('type', 'cosine')
    
    if scheduler_type == 'cosine':
        # Use T_max from config, or total_steps if available, otherwise raise error
        if 'T_max' in config:
            T_max = config['T_max']
        elif total_steps is not None:
            T_max = total_steps
        else:
            raise ValueError("CosineAnnealingLR requires either 'T_max' in config or total_steps to be provided")
        
        return CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=config.get('min_lr', 1e-6)
        )
    
    elif scheduler_type == 'warmup_cosine':
        # Calculate warmup_steps from warmup_fraction if provided
        if 'warmup_fraction' in config and total_steps:
            warmup_steps = int(config['warmup_fraction'] * total_steps)
        elif 'warmup_steps' in config:
            warmup_steps = config['warmup_steps']
        else:
            raise ValueError("WarmupCosineScheduler requires either 'warmup_fraction' with total_steps or 'warmup_steps' in config")
        
        # Get total_steps from parameter or config
        if total_steps is not None:
            scheduler_total_steps = total_steps
        elif 'total_steps' in config:
            scheduler_total_steps = config['total_steps']
        else:
            raise ValueError("WarmupCosineScheduler requires total_steps to be provided or 'total_steps' in config")
            
        return WarmupCosineScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=scheduler_total_steps,
            min_lr=config.get('min_lr', 1e-6)
        )
    
    elif scheduler_type == 'onecycle':
        # Get total_steps from parameter or config
        if total_steps is not None:
            scheduler_total_steps = total_steps
        elif 'total_steps' in config:
            scheduler_total_steps = config['total_steps']
        else:
            raise ValueError("OneCycleLR requires total_steps to be provided or 'total_steps' in config")
            
        return OneCycleLR(
            optimizer,
            max_lr=config.get('max_lr', 3e-4),
            total_steps=scheduler_total_steps,
            pct_start=config.get('pct_start', 0.1),
            anneal_strategy=config.get('anneal_strategy', 'cos'),
            div_factor=config.get('div_factor', 25.0),
            final_div_factor=config.get('final_div_factor', 10000.0)
        )
    
    elif scheduler_type == 'linear':
        # Simple linear decay - requires total_steps
        if total_steps is None and 'total_steps' not in config:
            raise ValueError("Linear scheduler requires total_steps to be provided or 'total_steps' in config")
        
        scheduler_total_steps = total_steps or config['total_steps']
        
        def lr_lambda(step):
            return max(0, 1 - step / scheduler_total_steps)
        
        return LambdaLR(optimizer, lr_lambda)
    
    elif scheduler_type == 'constant':
        # No scheduling
        return LambdaLR(optimizer, lambda step: 1.0)
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def build_loss_scheduler(
    config: Dict[str, Any], 
    total_steps: Optional[int] = None
) -> Optional[LossScheduler]:
    """Build loss scheduler from configuration.
    
    Args:
        config: Loss configuration
        total_steps: Total training steps
        
    Returns:
        Loss scheduler or None if not configured
    """
    if not config:
        return None
    
    # For auxiliary loss, check if enabled
    if 'enabled' in config and not config.get('enabled', False):
        return None
    
    # For regularization, check if weight > 0
    if 'enabled' not in config and config.get('weight', 0.0) <= 0:
        return None
    
    return LossScheduler(
        base_weight=config.get('weight', 1.0),
        schedule_type=config.get('type', 'constant'),
        warmup_fraction=config.get('warmup_fraction', 0.05),
        hold_fraction=config.get('hold_fraction'),
        decay_fraction=config.get('decay_fraction'),
        total_steps=total_steps,
        max_fraction_of_supervised=config.get('max_fraction_of_supervised')
    )
