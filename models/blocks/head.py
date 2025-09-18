"""Output projection heads for neural encoder models.

This module provides linear projection heads that map from hidden
representations to output vocabulary, including support for auxiliary
CTC heads for deep supervision.
"""

from typing import Optional, Tuple, Dict, Any, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Linear projection head for CTC output.
    
    Maps from hidden dimension to vocabulary size for CTC loss computation.
    
    Args:
        input_size: Size of input features
        vocab_size: Size of output vocabulary (including CTC blank)
        dropout: Dropout probability before projection
        bias: If True, use bias in linear layer
        init_scale: Scale factor for weight initialization (σ = init_scale/√H)
                   where H is input_size. Default 1.0 gives fan-in scaling.
    """
    
    def __init__(
        self,
        input_size: int,
        vocab_size: int,
        dropout: float = 0.0,
        bias: bool = True,
        init_scale: float = 1.0
    ):
        super().__init__()
        
        self.input_size = input_size
        self.vocab_size = vocab_size
        self.dropout_rate = dropout
        
        # Dropout before projection
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Linear projection
        self.projection = nn.Linear(input_size, vocab_size, bias=bias)
        
        # Initialize weights for CTC stability
        # Use fan-in std (~1/√H) instead of xavier for better CTC convergence
        with torch.no_grad():
            nn.init.normal_(self.projection.weight, mean=0.0, std=(init_scale / math.sqrt(self.input_size)))
            if bias:
                nn.init.zeros_(self.projection.bias)
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Project hidden states to vocabulary logits.
        
        Args:
            x: Input tensor [B, T, H]
            
        Returns:
            Logits tensor [B, T, V]
        """
        x = self.dropout(x)
        logits = self.projection(x)
        return logits
    
    def extra_repr(self) -> str:
        return f'input_size={self.input_size}, vocab_size={self.vocab_size}, dropout={self.dropout_rate}'


class AuxiliaryHead(nn.Module):
    """Auxiliary CTC head for mid-layer supervision.
    
    This head attaches to intermediate layers to provide additional
    supervision signal through auxiliary CTC loss.
    
    Args:
        input_size: Size of input features from intermediate layer
        vocab_size: Size of output vocabulary
        hidden_size: Size of intermediate projection (if None, direct projection)
        dropout: Dropout probability
        layer_norm: If True, apply layer normalization
    """
    
    def __init__(
        self,
        input_size: int,
        vocab_size: int,
        hidden_size: Optional[int] = None,
        dropout: float = 0.1,
        layer_norm: bool = True
    ):
        super().__init__()
        
        self.input_size = input_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        layers = []
        
        # Optional layer normalization
        if layer_norm:
            layers.append(nn.LayerNorm(input_size))
        
        # Dropout
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # Projection layers
        if hidden_size is not None:
            # Two-layer projection with nonlinearity
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_size, vocab_size))
        else:
            # Direct projection
            layers.append(nn.Linear(input_size, vocab_size))
        
        self.projection = nn.Sequential(*layers)
        
        # Initialize all linear layers
        for m in self.projection:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Project intermediate representations to vocabulary logits.
        
        Args:
            x: Input tensor from intermediate layer [B, T, H]
            
        Returns:
            Logits tensor [B, T, V]
        """
        return self.projection(x)
    
    def extra_repr(self) -> str:
        s = f'input_size={self.input_size}, vocab_size={self.vocab_size}'
        if self.hidden_size is not None:
            s += f', hidden_size={self.hidden_size}'
        return s


class MultiHeadProjection(nn.Module):
    """Multiple projection heads for ensemble or multi-task learning.
    
    This module manages multiple projection heads that can be used for:
    - Ensemble predictions (average multiple heads)
    - Multi-task learning (different vocabularies)
    - Uncertainty estimation (variance across heads)
    
    Args:
        input_size: Size of input features
        vocab_size: Size of output vocabulary
        num_heads: Number of projection heads
        dropout: Dropout probability for each head
        ensemble_method: How to combine heads ('mean', 'weighted', 'separate')
    """
    
    def __init__(
        self,
        input_size: int,
        vocab_size: int,
        num_heads: int = 3,
        dropout: float = 0.1,
        ensemble_method: str = 'mean'
    ):
        super().__init__()
        
        self.input_size = input_size
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.ensemble_method = ensemble_method
        
        # Create multiple projection heads
        self.heads = nn.ModuleList([
            ProjectionHead(input_size, vocab_size, dropout=dropout)
            for _ in range(num_heads)
        ])
        
        # Learnable weights for weighted ensemble
        if ensemble_method == 'weighted':
            self.ensemble_weights = nn.Parameter(torch.ones(num_heads) / num_heads)
    
    def forward(
        self,
        x: torch.FloatTensor,
        return_all: bool = False
    ) -> Union[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """Project through multiple heads.
        
        Args:
            x: Input tensor [B, T, H]
            return_all: If True, return all head outputs
            
        Returns:
            If return_all is False:
                Combined logits tensor [B, T, V]
            If return_all is True:
                Tuple of (combined_logits, all_logits)
                where all_logits is [num_heads, B, T, V]
        """
        # Get predictions from all heads
        all_logits = torch.stack([head(x) for head in self.heads])  # [num_heads, B, T, V]
        
        # Combine predictions based on method
        if self.ensemble_method == 'mean':
            combined = all_logits.mean(dim=0)
        elif self.ensemble_method == 'weighted':
            weights = F.softmax(self.ensemble_weights, dim=0)
            combined = (all_logits * weights.view(-1, 1, 1, 1)).sum(dim=0)
        elif self.ensemble_method == 'separate':
            combined = all_logits  # Return all separately
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
        
        if return_all:
            return combined, all_logits
        return combined
    
    def get_uncertainty(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Compute prediction uncertainty as variance across heads.
        
        Args:
            x: Input tensor [B, T, H]
            
        Returns:
            Uncertainty scores [B, T]
        """
        all_logits = torch.stack([head(x) for head in self.heads])
        
        # Use float32 for numerical stability in mixed precision
        all_logits_32 = all_logits.float()
        all_probs = F.softmax(all_logits_32, dim=-1)
        
        # Compute variance across heads
        mean_probs = all_probs.mean(dim=0)
        variance = ((all_probs - mean_probs.unsqueeze(0)) ** 2).mean(dim=0)
        
        # Average variance across vocabulary dimension
        uncertainty = variance.mean(dim=-1)
        
        # Convert back to original dtype
        return uncertainty.to(all_logits.dtype)


class CTCHead(nn.Module):
    """Complete CTC head with projection and log-softmax.
    
    This combines projection and log-softmax for CTC loss computation.
    
    Args:
        input_size: Size of input features
        vocab_size: Size of output vocabulary
        blank_idx: Index of CTC blank token (default: 0)
        dropout: Dropout probability
        init_temp: Initial temperature value
        classwise_temp: If True, use per-class temperature scaling
    """
    
    def __init__(
        self,
        input_size: int,
        vocab_size: int,
        blank_idx: int = 0,
        dropout: float = 0.0,
        init_temp: float = 1.0,
        classwise_temp: bool = False,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.vocab_size = vocab_size
        self.blank_idx = blank_idx
        self.dropout_rate = dropout
        
        # Projection head
        self.projection = ProjectionHead(input_size, vocab_size, dropout=dropout)
        
        # τ = exp(ρ) for positivity. Off by default during training (no grads).
        self.classwise_temp = classwise_temp
        self._rho = nn.Parameter(torch.tensor(math.log(init_temp)), requires_grad=False)
        self._tau_vec = None  # lazily created if classwise
    
    # --- temperature utilities ---
    def enable_temp_training(self, enabled: bool = True):
        """Enable/disable temperature parameter training."""
        self._rho.requires_grad = enabled
        if self._tau_vec is not None:
            self._tau_vec.requires_grad = enabled

    def set_temperature(self, tau: float):
        """Set temperature value."""
        with torch.no_grad():
            tau = max(float(tau), 1e-6)
            self._rho.copy_(torch.tensor(math.log(tau), device=self._rho.device))

    def temperature(self) -> float:
        """Get current temperature value."""
        if self.classwise_temp and (self._tau_vec is not None):
            return float(self._tau_vec.detach().mean().item())
        return float(self._rho.detach().exp().item())

    def _apply_temperature(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        if self.classwise_temp:
            V = logits.size(-1)
            if (self._tau_vec is None) or (self._tau_vec.numel() != V):
                tau0 = self._rho.detach().exp().item()
                self._tau_vec = nn.Parameter(
                    torch.full((V,), tau0, device=logits.device),
                    requires_grad=False,
                )
            tau = self._tau_vec.view(1, 1, -1)
        else:
            tau = self._rho.exp()
        return logits / tau.clamp_min(1e-6)
    # -----------------------------
    
    def forward(
        self,
        x: torch.FloatTensor,
        lengths: Optional[torch.LongTensor] = None,
        temperature: Optional[float] = None
    ) -> torch.FloatTensor:
        """Compute log probabilities for CTC.
        
        Args:
            x: Input tensor [B, T, H]
            lengths: Sequence lengths for masking [B]
            temperature: Temperature override for ablations
            
        Returns:
            Log probabilities [B, T, V]
        """
        # Project to logits
        logits = self.projection(x)
        
        # Prefer internal temp; allow override for ablations
        if temperature is not None and temperature != 1.0:
            logits = logits / max(temperature, 1e-6)
        else:
            logits = self._apply_temperature(logits)
        
        # Safe masking: force blank on padded frames to avoid NaNs
        if lengths is not None:
            from .utils import force_blank_on_pad_
            force_blank_on_pad_(logits, lengths, self.blank_idx)
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        
        return log_probs
    
    def get_logits(
        self,
        x: torch.FloatTensor,
        lengths: Optional[torch.LongTensor] = None,
        apply_temperature: bool = False
    ) -> torch.FloatTensor:
        """Get raw logits without log-softmax.
        
        Args:
            x: Input tensor [B, T, H]
            lengths: Sequence lengths for masking [B]
            apply_temperature: If True, apply temperature scaling
            
        Returns:
            Logits tensor [B, T, V]
        """
        logits = self.projection(x)
        
        # Optional safe masking for decoding (before temperature scaling)
        if lengths is not None:
            from .utils import force_blank_on_pad_
            force_blank_on_pad_(logits, lengths, self.blank_idx)
        
        # Apply temperature if requested (after masking)
        if apply_temperature:
            logits = self._apply_temperature(logits)
        
        return logits
    
    def extra_repr(self) -> str:
        return (f'input_size={self.input_size}, vocab_size={self.vocab_size}, '
                f'blank_idx={self.blank_idx}')
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.extra_repr()})'
