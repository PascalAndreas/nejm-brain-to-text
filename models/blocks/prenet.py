"""Pre-processing network components for neural encoder models.

This module contains preprocessing blocks that are applied to raw neural
signals before the main encoder backbone.
"""

from typing import Optional, Literal, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .utils import apply_patch_embedding, compute_output_lengths, get_activation_fn


class GaussianSmoother(nn.Module):
    """Gaussian smoothing for temporal signals.
    
    Applies 1D Gaussian smoothing along the time dimension independently
    for each feature channel. This is implemented as a depthwise convolution
    with fixed Gaussian weights.
    
    Args:
        kernel_size: Size of the Gaussian kernel (should be odd)
        std: Standard deviation of the Gaussian
        trainable: If True, kernel parameters are trainable
    """
    
    def __init__(
        self,
        kernel_size: int = 100,
        std: float = 2.0,
        trainable: bool = False
    ):
        super().__init__()
        
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
            
        self.kernel_size = kernel_size
        self.std = std
        self.trainable = trainable
        
        # Create Gaussian kernel
        kernel = self._create_gaussian_kernel(kernel_size, std)
        
        if trainable:
            self.kernel = nn.Parameter(kernel)
        else:
            self.register_buffer('kernel', kernel)
    
    def _create_gaussian_kernel(self, size: int, std: float) -> torch.Tensor:
        """Create 1D Gaussian kernel."""
        coords = torch.arange(size, dtype=torch.float32)
        coords -= (size - 1) / 2.0
        
        kernel = torch.exp(-(coords ** 2) / (2 * std ** 2))
        kernel = kernel / kernel.sum()
        
        return kernel
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Apply Gaussian smoothing.
        
        Args:
            x: Input tensor [B, T, F]
            
        Returns:
            Smoothed tensor [B, T, F]
        """
        batch_size, seq_len, feat_dim = x.shape
        
        # Reshape for depthwise convolution: [B, F, T]
        x = x.transpose(1, 2)
        
        # Expand kernel for all feature channels: [F, 1, kernel_size]
        kernel = self.kernel.unsqueeze(0).unsqueeze(0)
        kernel = kernel.expand(feat_dim, 1, -1)
        
        # Apply depthwise convolution with same padding
        padding = self.kernel_size // 2
        x_smoothed = F.conv1d(
            x,
            kernel,
            groups=feat_dim,
            padding=padding
        )
        
        # Restore shape: [B, T, F]
        x_smoothed = x_smoothed.transpose(1, 2)
        
        return x_smoothed
    
    def extra_repr(self) -> str:
        return f'kernel_size={self.kernel_size}, std={self.std}, trainable={self.trainable}'


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
        
        # Create parameters for each day
        self.gammas = nn.ParameterList([
            nn.Parameter(torch.ones(self.num_groups, self.group_size))
            for _ in range(num_days)
        ])
        
        self.betas = nn.ParameterList([
            nn.Parameter(torch.zeros(self.num_groups, self.group_size))
            for _ in range(num_days)
        ])
        
        if not identity_init:
            # Initialize with small random values
            for gamma in self.gammas:
                nn.init.normal_(gamma, mean=1.0, std=0.01)
            for beta in self.betas:
                nn.init.normal_(beta, mean=0.0, std=0.01)
    
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
        
        # Gather parameters for each sample's day
        gamma_batch = torch.stack([self.gammas[idx] for idx in day_indices])  # [B, G, F/G]
        beta_batch = torch.stack([self.betas[idx] for idx in day_indices])    # [B, G, F/G]
        
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
        loss = 0.0
        
        if self.l2_tether > 0:
            # L2 tether to identity transformation
            for gamma in self.gammas:
                loss += self.l2_tether * torch.sum((gamma - 1) ** 2)
            for beta in self.betas:
                loss += self.l2_tether * torch.sum(beta ** 2)
        
        if self.l1_reg > 0:
            # L1 regularization for sparsity
            for gamma in self.gammas:
                loss += self.l1_reg * torch.sum(torch.abs(gamma - 1))
            for beta in self.betas:
                loss += self.l1_reg * torch.sum(torch.abs(beta))
        
        return loss
    
    def get_day_stats(self) -> Dict[int, Dict[str, float]]:
        """Get statistics about FiLM parameters for each day.
        
        Returns:
            Dictionary mapping day index to parameter statistics
        """
        stats = {}
        
        for day_idx in range(self.num_days):
            gamma = self.gammas[day_idx]
            beta = self.betas[day_idx]
            
            stats[day_idx] = {
                'gamma_mean': gamma.mean().item(),
                'gamma_std': gamma.std().item(),
                'gamma_norm': torch.norm(gamma - 1).item(),
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
    
    This module chains together smoothing, patching, day adaptation,
    and activation functions.
    
    Args:
        input_dim: Number of input features
        num_days: Number of recording days
        smoother_config: Configuration for Gaussian smoother
        day_adapter_config: Configuration for day adapter
        patch_config: Configuration for patching
        activation: Activation function name
    """
    
    def __init__(
        self,
        input_dim: int,
        num_days: int,
        smoother_config: Optional[Dict[str, Any]] = None,
        day_adapter_config: Optional[Dict[str, Any]] = None,
        patch_config: Optional[Dict[str, Any]] = None,
        activation: str = 'softsign'
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = input_dim  # Will be updated if patching is used
        
        # Gaussian smoother
        self.smoother = None
        if smoother_config is not None:
            self.smoother = GaussianSmoother(**smoother_config)
        
        # Patching configuration
        self.patch_config = patch_config or {}
        self.patch_size = self.patch_config.get('size', 1)
        self.patch_stride = self.patch_config.get('stride', self.patch_size)
        
        # Update output dimension if patching
        if self.patch_size > 1:
            self.output_dim = input_dim * self.patch_size
        
        # Day adapter (FiLM)
        self.day_adapter = None
        if day_adapter_config is not None and num_days > 0:
            adapter_config = day_adapter_config.copy()
            adapter_config['num_features'] = self.output_dim
            adapter_config['num_days'] = num_days
            self.day_adapter = DayAdapter(**adapter_config)
        
        # Activation
        self.activation = get_activation_fn(activation)
    
    def forward(
        self,
        x: torch.FloatTensor,
        lengths: torch.LongTensor,
        day_indices: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """Apply preprocessing to input features.
        
        Args:
            x: Input tensor [B, T, F]
            lengths: Sequence lengths [B]
            day_indices: Day indices for each sample [B]
            
        Returns:
            Tuple of:
            - Processed tensor [B, T_out, F_out]
            - Updated lengths [B]
        """
        # Gaussian smoothing
        if self.smoother is not None:
            x = self.smoother(x)
        
        # Patching (frame concatenation)
        if self.patch_size > 1:
            x, lengths = apply_patch_embedding(
                x, self.patch_size, self.patch_stride, lengths
            )
        
        # Day-specific adaptation
        if self.day_adapter is not None and day_indices is not None:
            x = self.day_adapter(x, day_indices)
        
        # Activation
        x = self.activation(x)
        
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
        return torch.tensor(0.0)
    
    def extra_repr(self) -> str:
        parts = []
        if self.smoother:
            parts.append(f'smoother={self.smoother}')
        if self.patch_size > 1:
            parts.append(f'patch_size={self.patch_size}, patch_stride={self.patch_stride}')
        if self.day_adapter:
            parts.append(f'day_adapter={self.day_adapter}')
        parts.append(f'activation={self.activation}')
        return ', '.join(parts)
