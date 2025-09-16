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


class AuxiliaryLossScheduler:
    """Scheduler for auxiliary loss weight.
    
    Supports various scheduling strategies including constant,
    linear warmup, and warmup-hold-decay patterns.
    
    Args:
        base_weight: Base weight for auxiliary loss
        schedule_type: Type of schedule ('constant', 'warmup', 'warmup_decay')
        warmup_steps: Steps to warm up to base weight
        hold_steps: Steps to hold at base weight (for warmup_decay)
        decay_steps: Steps to decay to 0 (for warmup_decay)
    """
    
    def __init__(
        self,
        base_weight: float = 0.25,
        schedule_type: str = 'constant',
        warmup_steps: int = 5000,
        hold_steps: int = 50000,
        decay_steps: int = 30000
    ):
        self.base_weight = base_weight
        self.schedule_type = schedule_type
        self.warmup_steps = warmup_steps
        self.hold_steps = hold_steps
        self.decay_steps = decay_steps
        self.current_step = 0
    
    def step(self) -> float:
        """Get weight for current step and increment counter.
        
        Returns:
            Auxiliary loss weight for current step
        """
        weight = self.get_weight(self.current_step)
        self.current_step += 1
        return weight
    
    def get_weight(self, step: Optional[int] = None) -> float:
        """Get auxiliary loss weight for a given step.
        
        Args:
            step: Training step (uses internal counter if None)
            
        Returns:
            Auxiliary loss weight
        """
        if step is None:
            step = self.current_step
        
        if self.schedule_type == 'constant':
            return self.base_weight
        
        elif self.schedule_type == 'warmup':
            # Linear warmup only
            if step < self.warmup_steps:
                return self.base_weight * (step / self.warmup_steps)
            else:
                return self.base_weight
        
        elif self.schedule_type == 'warmup_decay':
            # Warmup, hold, then decay to 0
            if step < self.warmup_steps:
                # Linear warmup
                return self.base_weight * (step / self.warmup_steps)
            elif step < self.warmup_steps + self.hold_steps:
                # Hold constant
                return self.base_weight
            else:
                # Cosine decay to 0
                decay_progress = min(1.0, (step - self.warmup_steps - self.hold_steps) / self.decay_steps)
                return self.base_weight * 0.5 * (1 + math.cos(decay_progress * math.pi))
        
        elif self.schedule_type == 'linear_decay':
            # Linear decay from start
            if step < self.decay_steps:
                return self.base_weight * (1 - step / self.decay_steps)
            else:
                return 0.0
        
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")
    
    def state_dict(self) -> Dict[str, Any]:
        """Get scheduler state for checkpointing."""
        return {
            'current_step': self.current_step,
            'base_weight': self.base_weight,
            'schedule_type': self.schedule_type,
            'warmup_steps': self.warmup_steps,
            'hold_steps': self.hold_steps,
            'decay_steps': self.decay_steps
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load scheduler state from checkpoint."""
        self.current_step = state_dict['current_step']
        self.base_weight = state_dict.get('base_weight', self.base_weight)
        self.schedule_type = state_dict.get('schedule_type', self.schedule_type)
        self.warmup_steps = state_dict.get('warmup_steps', self.warmup_steps)
        self.hold_steps = state_dict.get('hold_steps', self.hold_steps)
        self.decay_steps = state_dict.get('decay_steps', self.decay_steps)


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
        return CosineAnnealingLR(
            optimizer,
            T_max=config.get('T_max', total_steps or 100000),
            eta_min=config.get('min_lr', 1e-6)
        )
    
    elif scheduler_type == 'warmup_cosine':
        return WarmupCosineScheduler(
            optimizer,
            warmup_steps=config.get('warmup_steps', 5000),
            total_steps=total_steps or config.get('total_steps', 100000),
            min_lr=config.get('min_lr', 1e-6)
        )
    
    elif scheduler_type == 'onecycle':
        return OneCycleLR(
            optimizer,
            max_lr=config.get('max_lr', 3e-4),
            total_steps=total_steps or config.get('total_steps', 100000),
            pct_start=config.get('pct_start', 0.1),
            anneal_strategy=config.get('anneal_strategy', 'cos'),
            div_factor=config.get('div_factor', 25.0),
            final_div_factor=config.get('final_div_factor', 10000.0)
        )
    
    elif scheduler_type == 'linear':
        # Simple linear decay
        def lr_lambda(step):
            if total_steps:
                return max(0, 1 - step / total_steps)
            else:
                return 1.0
        
        return LambdaLR(optimizer, lr_lambda)
    
    elif scheduler_type == 'constant':
        # No scheduling
        return LambdaLR(optimizer, lambda step: 1.0)
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def build_aux_scheduler(config: Dict[str, Any]) -> Optional[AuxiliaryLossScheduler]:
    """Build auxiliary loss scheduler from configuration.
    
    Args:
        config: Auxiliary loss configuration
        
    Returns:
        Auxiliary loss scheduler or None if not configured
    """
    if not config or not config.get('enabled', False):
        return None
    
    schedule_config = config.get('schedule', {})
    
    return AuxiliaryLossScheduler(
        base_weight=config.get('weight', 0.25),
        schedule_type=schedule_config.get('type', 'constant'),
        warmup_steps=schedule_config.get('warmup_steps', 5000),
        hold_steps=schedule_config.get('hold_steps', 50000),
        decay_steps=schedule_config.get('decay_steps', 30000)
    )
