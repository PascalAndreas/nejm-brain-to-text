"""Pre-processing network components for neural encoder models.

This module contains preprocessing blocks that are applied to raw neural
signals before the main encoder backbone.
"""

from typing import Optional, Literal, Tuple, Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .utils import apply_patch_embedding, compute_output_lengths, get_activation_fn


class DepthwiseCausalSmoother(nn.Module):
    """Learnable depthwise causal convolution for temporal smoothing.
    
    Applies causal 1D convolution independently per feature channel with
    learnable kernels constrained to be valid smoothing filters (sum to 1).
    Uses softmax parameterization for numerically stable normalization.
    
    Args:
        num_features: Number of input feature channels
        kernel_size: Size of convolution kernel (should be odd)
        residual_gate: If True, use gated residual connection y = (1-α)x + α·conv(x)
        init_std: Standard deviation for Gaussian initialization (in frames)
    """
    
    def __init__(
        self,
        num_features: int,
        kernel_size: int = 11,
        residual_gate: bool = True,
        init_std: float = 2.0
    ):
        super().__init__()
        
        self.num_features = num_features
        self.kernel_size = kernel_size
        self.residual_gate = residual_gate
        
        # Learnable kernel parameters (log-space, transformed with softmax)
        self.kernel_params = nn.Parameter(
            torch.zeros(num_features, 1, kernel_size)
        )
        
        # Initialize to exact Gaussian
        self._init_gaussian_kernel(init_std)
        
        # Residual gate parameters
        if residual_gate:
            # Gate parameter α (will be transformed with sigmoid)
            self.gate_param = nn.Parameter(torch.full((1,), -1.386))  # sigmoid(-1.386) ≈ 0.2
        
        # No bias in convolution
        self.padding = kernel_size - 1  # For causal padding
    
    def _init_gaussian_kernel(self, std: float):
        """Initialize kernel parameters to exact Gaussian."""
        with torch.no_grad():
            # Create causal Gaussian kernel
            positions = torch.arange(self.kernel_size, dtype=torch.float32)
            positions = self.kernel_size - 1 - positions  # Reverse for causality
            
            # Gaussian values (normalized to sum=1)
            gaussian = torch.exp(-(positions ** 2) / (2 * std ** 2))
            gaussian = gaussian / gaussian.sum()
            
            # For softmax parameterization: φ = log(h + ε)
            # This ensures softmax(φ) = h exactly
            eps = 1e-8
            log_gaussian = torch.log(gaussian + eps)
            
            # Set for all features
            self.kernel_params.data = log_gaussian.unsqueeze(0).unsqueeze(0).expand_as(self.kernel_params)
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Apply depthwise causal smoothing.
        
        Args:
            x: Input tensor [B, T, F]
            
        Returns:
            Smoothed tensor [B, T, F]
        """
        batch_size, seq_len, feat_dim = x.shape
        
        # Transform kernel parameters to valid weights using softmax
        # This ensures weights ≥ 0 and sum = 1 by construction
        kernel_weights = F.softmax(self.kernel_params, dim=-1)  # [F, 1, K]
        
        # Reshape for depthwise convolution: [B, F, T]
        x_conv = x.transpose(1, 2)
        
        # Apply causal padding (left padding = kernel_size - 1)
        x_padded = F.pad(x_conv, (self.padding, 0), mode='constant', value=0)
        
        # Depthwise convolution
        x_smoothed = F.conv1d(
            x_padded,
            kernel_weights,
            groups=feat_dim,
            stride=1
        )
        
        # Restore shape: [B, T, F]
        x_smoothed = x_smoothed.transpose(1, 2)
        
        # Apply residual gate if enabled: y = (1-α)x + α·conv(x)
        if self.residual_gate:
            alpha = torch.sigmoid(self.gate_param)
            x_smoothed = (1 - alpha) * x + alpha * x_smoothed
        
        return x_smoothed
    
    def get_effective_kernels(self) -> torch.Tensor:
        """Get the effective smoothing kernels after softmax normalization.
        
        Returns:
            Normalized kernel weights [F, K]
        """
        with torch.no_grad():
            weights = F.softmax(self.kernel_params, dim=-1).squeeze(1)  # [F, K]
        return weights
    
    def extra_repr(self) -> str:
        return f'num_features={self.num_features}, kernel_size={self.kernel_size}, residual_gate={self.residual_gate}'


class GaussianSmoother(nn.Module):
    """Fixed Gaussian smoothing for temporal signals (legacy).
    
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
    
    This module chains together smoothing, day adaptation, activation,
    and patching operations.
    
    Args:
        input_dim: Number of input features
        num_days: Number of recording days
        smoother_config: Configuration for smoother (Gaussian or Depthwise)
        day_adapter_config: Configuration for day adapter
        patch_config: Configuration for patching
        activation: Activation function name
        use_depthwise_smoother: If True, use learnable depthwise smoother
    """
    
    def __init__(
        self,
        input_dim: int,
        num_days: int,
        smoother_config: Optional[Dict[str, Any]] = None,
        day_adapter_config: Optional[Dict[str, Any]] = None,
        patch_config: Optional[Dict[str, Any]] = None,
        activation: str = 'softsign',
        use_depthwise_smoother: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = input_dim  # Will be updated if patching is used
        
        # Smoother (depthwise causal or Gaussian)
        self.smoother = None
        if smoother_config is not None:
            if use_depthwise_smoother:
                # Use learnable depthwise causal smoother
                self.smoother = DepthwiseCausalSmoother(
                    num_features=input_dim,
                    kernel_size=smoother_config.get('kernel_size', 11),
                    residual_gate=smoother_config.get('residual_gate', True),
                    init_std=smoother_config.get('std', 2.0)
                )
            else:
                # Use fixed Gaussian smoother (legacy)
                self.smoother = GaussianSmoother(**smoother_config)
        
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
        
        Args:
            x: Input tensor [B, T, F]
            lengths: Sequence lengths [B]
            day_indices: Day indices for each sample [B]
            
        Returns:
            Tuple of:
            - Processed tensor [B, T_out, F_out]
            - Updated lengths [B]
        """
        # Smoothing (depthwise causal or Gaussian)
        if self.smoother is not None:
            x = self.smoother(x)
        
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
        if self.smoother:
            parts.append(f'smoother={self.smoother.__class__.__name__}')
        if self.patch_size > 1:
            parts.append(f'patch_size={self.patch_size}, patch_stride={self.patch_stride}')
        if self.day_adapter:
            parts.append(f'day_adapter=True')
        parts.append(f'activation={self.activation_name}')
        return ', '.join(parts)