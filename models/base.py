"""Base classes and contracts for neural encoder models.

This module defines the core interfaces and data structures used throughout
the models package.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
import torch
from torch import nn
from abc import ABC, abstractmethod


@dataclass
class Batch:
    """Standard batch format for neural encoder models.
    
    This wrapper provides a consistent interface regardless of the underlying
    dataset implementation.
    
    Attributes:
        x: Neural features [B, T, F]
        x_lens: True lengths before padding [B]
        y: Target phoneme IDs (optional) [B, L_p]  
        y_lens: True label lengths (optional) [B]
        day_id: Recording day indices [B]
        utt_id: Utterance identifiers
        meta: Additional metadata (sessions, blocks, etc.)
    """
    x: torch.FloatTensor  # [B, T, F]
    x_lens: torch.LongTensor  # [B]
    y: Optional[torch.LongTensor] = None  # [B, L_p]
    y_lens: Optional[torch.LongTensor] = None  # [B]
    day_id: Optional[torch.LongTensor] = None  # [B]
    utt_id: Optional[List[str]] = None
    meta: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_dataset_batch(cls, batch: Dict[str, Any]) -> 'Batch':
        """Convert dataset collate output to standard Batch format.
        
        Args:
            batch: Dictionary from dataset collate_fn
            
        Returns:
            Standardized Batch object
        """
        # Generate utterance IDs from metadata
        utt_ids = None
        if all(k in batch for k in ['sessions', 'block_nums', 'trial_nums']):
            utt_ids = [
                f"{sess}_b{block}_t{trial}"
                for sess, block, trial in zip(
                    batch['sessions'],
                    batch['block_nums'].tolist(),
                    batch['trial_nums'].tolist()
                )
            ]
        
        # Collect metadata
        meta = {
            'sessions': batch.get('sessions'),
            'block_nums': batch.get('block_nums'),
            'trial_nums': batch.get('trial_nums'),
            'corpora': batch.get('corpora'),
        }
        
        return cls(
            x=batch['input_features'],
            x_lens=batch['n_time_steps'],
            y=batch.get('seq_class_ids'),
            y_lens=batch.get('phone_seq_lens'),
            day_id=batch.get('day_indices'),
            utt_id=utt_ids,
            meta=meta
        )
    
    def to(self, device: Union[str, torch.device]) -> 'Batch':
        """Move batch to specified device."""
        return Batch(
            x=self.x.to(device),
            x_lens=self.x_lens.to(device),
            y=self.y.to(device) if self.y is not None else None,
            y_lens=self.y_lens.to(device) if self.y_lens is not None else None,
            day_id=self.day_id.to(device) if self.day_id is not None else None,
            utt_id=self.utt_id,
            meta=self.meta
        )
    
    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self.x.shape[0]
    
    @property 
    def max_len(self) -> int:
        """Get maximum sequence length in batch."""
        return self.x.shape[1]
    
    @property
    def feature_dim(self) -> int:
        """Get feature dimension."""
        return self.x.shape[2]


@dataclass  
class Emissions:
    """Standard output format from neural encoders.
    
    Attributes:
        log_probs: Log-softmax probabilities over vocabulary [B, T_out, V]
        out_lens: True emission lengths after time reduction [B]
        aux: Optional auxiliary outputs (e.g., intermediate representations)
    """
    log_probs: torch.FloatTensor  # [B, T_out, V]
    out_lens: torch.LongTensor  # [B]
    aux: Optional[Dict[str, Any]] = None
    
    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self.log_probs.shape[0]
    
    @property
    def max_len(self) -> int:
        """Get maximum emission length."""
        return self.log_probs.shape[1]
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size (including blank)."""
        return self.log_probs.shape[2]
    
    def to(self, device: torch.device) -> 'Emissions':
        """Move emissions to specified device.
        
        Args:
            device: Target device
            
        Returns:
            New Emissions instance on target device
        """
        return Emissions(
            log_probs=self.log_probs.to(device),
            out_lens=self.out_lens.to(device),
            aux=self.aux  # aux dict is not moved (contains arbitrary objects)
        )
    
    def cpu(self) -> 'Emissions':
        """Move emissions to CPU.
        
        Returns:
            New Emissions instance on CPU
        """
        return self.to(torch.device('cpu'))


class NeuralEncoder(nn.Module, ABC):
    """Abstract base class for neural encoder models.
    
    All encoder implementations must inherit from this class and implement
    the required abstract methods.
    """
    
    @abstractmethod
    def forward(self, batch: Batch) -> Emissions:
        """Forward pass through the encoder.
        
        Args:
            batch: Standardized input batch
            
        Returns:
            Emissions containing log probabilities and lengths
        """
        pass
    
    @abstractmethod
    def time_reduction(self) -> int:
        """Get the overall time reduction factor.
        
        Returns:
            Integer stride factor (1 if no reduction)
        """
        pass
    
    def supports_packed(self) -> bool:
        """Check if model supports packed sequences (for RNNs).
        
        Returns:
            True if model can use packed sequences
        """
        return False
    
    def eval_calibrate(self, dev_loader: torch.utils.data.DataLoader) -> None:
        """Calibrate temperature scaling on validation data.
        
        This method fits a temperature parameter to minimize NLL on the
        validation set. Should be called after training.
        
        Args:
            dev_loader: Validation data loader
        """
        pass
    
    def export_config(self) -> Dict[str, Any]:
        """Export model configuration for reproducibility.
        
        Returns:
            Dictionary containing architecture and hyperparameters
        """
        return {
            'class': self.__class__.__name__,
            'time_reduction': self.time_reduction(),
            'supports_packed': self.supports_packed(),
        }
    
    def get_num_params(self, trainable_only: bool = True) -> int:
        """Get number of parameters in the model.
        
        Args:
            trainable_only: If True, count only trainable parameters
            
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
