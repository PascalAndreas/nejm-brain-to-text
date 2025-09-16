"""Pre-processing network components for neural encoder models.

This module contains preprocessing blocks that are applied to raw neural
signals before the main encoder backbone (excluding smoothing).
"""

from typing import Optional, Literal, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .utils import apply_patch_embedding, compute_output_lengths, get_activation_fn


class DayAdapter(nn.Module):
    """Feature-wise Linear Modulation (FiLM) for day-specific adaptation.
    
    Applies affine transformation: y = γ * x + β
    where γ and β are learned per recording day.
    
    Args:
        num_features: Number of input features
        num_days: Number of recording days
        grouping: How to group features ('full', 'blocks8', or int)
        identity_init: If True, initialize γ=1, β=0
        l2_tether: L2 regularization strength for deviation from identity
        l1_reg: L1 regularization strength for sparsity
    """
    
    def __init__(
        self,
        num_features: int,
        num_days: int,
        grouping: Literal['full', 'blocks8'] | int = 'full',
        identity_init: bool = True,
        l2_tether: float = 1e-4,
        l1_reg: float = 0.0
    ):
        super().__init__()
        
        self.num_features = num_features
        self.num_days = num_days
        self.grouping = grouping
        self.l2_tether = l2_tether
        self.l1_reg = l1_reg
        
        # Determine number of groups
        if grouping == 'full':
            self.num_groups = 1
            self.group_size = num_features
        elif grouping == 'blocks8':
            # 8 predefined blocks of 64 channels each (assumes 512 features)
            self.num_groups = 8
            self.group_size = num_features // 8
        elif isinstance(grouping, int):
            self.num_groups = grouping
            self.group_size = num_features // grouping
        else:
            raise ValueError(f"Invalid grouping: {grouping}")
        
        # Ensure features divide evenly into groups
        assert num_features % self.num_groups == 0, \
            f"Features {num_features} must be divisible by groups {self.num_groups}"
        
        # Create parameters for each day - vectorized for efficient lookup
        # Use delta parameterization: gamma = 1 + delta_gamma for better identity bias
        self.delta_gammas = nn.Parameter(torch.zeros(num_days, self.num_groups, self.group_size))
        self.betas = nn.Parameter(torch.zeros(num_days, self.num_groups, self.group_size))
        
        if not identity_init:
            # Initialize with small random values
            nn.init.normal_(self.delta_gammas, mean=0.0, std=0.01)
            nn.init.normal_(self.betas, mean=0.0, std=0.01)
    
    def forward(
        self,
        x: torch.FloatTensor,
        day_indices: torch.LongTensor
    ) -> torch.FloatTensor:
        """Apply day-specific FiLM transformation.
        
        Args:
            x: Input tensor [B, T, F]
            day_indices: Day index for each sample [B]
            
        Returns:
            Transformed tensor [B, T, F]
        """
        batch_size, seq_len, feat_dim = x.shape
        
        # Gather parameters for each sample's day - vectorized lookup
        delta_gamma_batch = self.delta_gammas[day_indices]  # [B, G, F/G]
        gamma_batch = 1.0 + delta_gamma_batch  # [B, G, F/G]
        beta_batch = self.betas[day_indices]   # [B, G, F/G]
        
        # Reshape to match feature dimension
        gamma_batch = gamma_batch.view(batch_size, feat_dim)  # [B, F]
        beta_batch = beta_batch.view(batch_size, feat_dim)    # [B, F]
        
        # Expand for time dimension
        gamma_batch = gamma_batch.unsqueeze(1)  # [B, 1, F]
        beta_batch = beta_batch.unsqueeze(1)    # [B, 1, F]
        
        # Apply FiLM
        x_adapted = gamma_batch * x + beta_batch
        
        return x_adapted
    
    def regularization_loss(self) -> torch.Tensor:
        """Compute regularization loss for FiLM parameters.
        
        Returns:
            Regularization loss scalar
        """
        device = self.delta_gammas.device
        loss = torch.zeros((), device=device)
        
        if self.l2_tether > 0:
            # L2 tether to identity transformation - use means for scale invariance
            # With delta parameterization, regularize delta_gammas directly (gamma = 1 + delta_gamma)
            loss += self.l2_tether * (self.delta_gammas.pow(2).mean() + self.betas.pow(2).mean())
        
        if self.l1_reg > 0:
            # L1 regularization for sparsity - use means for scale invariance
            loss += self.l1_reg * (torch.abs(self.delta_gammas).mean() + torch.abs(self.betas).mean())
        
        return loss
    
    def get_day_stats(self) -> Dict[int, Dict[str, float]]:
        """Get statistics about FiLM parameters for each day.
        
        Returns:
            Dictionary mapping day index to parameter statistics
        """
        stats = {}
        
        for day_idx in range(self.num_days):
            delta_gamma = self.delta_gammas[day_idx]  # [G, F/G]
            gamma = 1.0 + delta_gamma  # [G, F/G]
            beta = self.betas[day_idx]  # [G, F/G]
            
            stats[day_idx] = {
                'gamma_mean': gamma.mean().item(),
                'gamma_std': gamma.std().item(),
                'gamma_norm': torch.norm(delta_gamma).item(),  # Use delta_gamma norm for deviation from identity
                'delta_gamma_mean': delta_gamma.mean().item(),
                'delta_gamma_std': delta_gamma.std().item(),
                'beta_mean': beta.mean().item(),
                'beta_std': beta.std().item(),
                'beta_norm': torch.norm(beta).item(),
            }
        
        return stats
    
    def extra_repr(self) -> str:
        return (f'num_features={self.num_features}, num_days={self.num_days}, '
                f'grouping={self.grouping}, l2_tether={self.l2_tether}, l1_reg={self.l1_reg}')


class PreNet(nn.Module):
    """Complete preprocessing network combining multiple components.
    
    This module chains together day adaptation, activation,
    and patching operations. Smoothing is now handled separately.
    
    Args:
        input_dim: Number of input features
        num_days: Number of recording days
        day_adapter_config: Configuration for day adapter
        patch_config: Configuration for patching
        activation: Activation function name
    """
    
    def __init__(
        self,
        input_dim: int,
        num_days: int,
        day_adapter_config: Optional[Dict[str, Any]] = None,
        patch_config: Optional[Dict[str, Any]] = None,
        activation: str = 'softsign'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = input_dim  # Will be updated if patching is used
        
        # Patching configuration
        self.patch_config = patch_config or {}
        self.patch_size = self.patch_config.get('size', 1)
        self.patch_stride = self.patch_config.get('stride', self.patch_size)
        
        # Update output dimension if patching
        if self.patch_size > 1:
            self.output_dim = input_dim * self.patch_size
        
        # Day adapter (FiLM) - operates on original features before patching
        self.day_adapter = None
        if day_adapter_config is not None and num_days > 0:
            adapter_config = day_adapter_config.copy()
            adapter_config['num_features'] = input_dim  # Use input_dim, not output_dim
            adapter_config['num_days'] = num_days
            self.day_adapter = DayAdapter(**adapter_config)
        
        # Activation
        self.activation_name = activation
        self.activation = get_activation_fn(activation)
    
    def forward(
        self,
        x: torch.FloatTensor,
        lengths: torch.LongTensor,
        day_indices: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """Apply preprocessing to input features.
        
        Note: Smoothing is now handled separately before PreNet.
        
        Args:
            x: Input tensor [B, T, F]
            lengths: Sequence lengths [B]
            day_indices: Day indices for each sample [B]
            
        Returns:
            Tuple of:
            - Processed tensor [B, T_out, F_out]
            - Updated lengths [B]
        """
        # Day-specific adaptation
        if self.day_adapter is not None and day_indices is not None:
            x = self.day_adapter(x, day_indices)
        
        # Activation (before patching)
        x = self.activation(x)
        
        # Patching (frame concatenation)
        if self.patch_size > 1:
            x, lengths = apply_patch_embedding(
                x, self.patch_size, self.patch_stride, lengths
            )
        
        return x, lengths
    
    def get_ops_config(self) -> List[Dict[str, Any]]:
        """Get configuration of time-reducing operations.
        
        Returns:
            List of operation configurations for length computation
        """
        ops = []
        if self.patch_size > 1:
            ops.append({
                'type': 'patch',
                'size': self.patch_size,
                'stride': self.patch_stride
            })
        return ops
    
    def regularization_loss(self) -> torch.Tensor:
        """Get regularization loss from day adapter.
        
        Returns:
            Regularization loss scalar
        """
        if self.day_adapter is not None:
            return self.day_adapter.regularization_loss()
        # Return device-safe zero tensor
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device('cpu')
        return torch.zeros((), device=device)
    
    def extra_repr(self) -> str:
        parts = []
        if self.patch_size > 1:
            parts.append(f'patch_size={self.patch_size}, patch_stride={self.patch_stride}')
        if self.day_adapter:
            parts.append(f'day_adapter=True')
        parts.append(f'activation={self.activation_name}')
        return ', '.join(parts)