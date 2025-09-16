"""Temporal smoothing components for neural encoder models.

This module provides various smoothing strategies for preprocessing
neural signals, all with causal constraints and DC gain preservation.
"""

from typing import Optional, Tuple, Dict, Any
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SmootherBase(ABC, nn.Module):
    """Abstract base class for temporal smoothers.
    
    All smoothers must:
    - Be causal (no future information)
    - Preserve sequence length (stride=1)
    - Maintain DC gain = 1 (preserve mean)
    - Support variable-length sequences
    """
    
    @abstractmethod
    def forward(
        self,
        x: torch.FloatTensor,
        lengths: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """Apply smoothing to input sequences.
        
        Args:
            x: Input tensor [B, T, F]
            lengths: Sequence lengths [B]
            
        Returns:
            Tuple of (smoothed tensor, unchanged lengths)
        """
        pass
    
    @abstractmethod
    def receptive_field(self) -> int:
        """Get the receptive field size in frames.
        
        Returns:
            Number of input frames that influence each output frame
        """
        pass
    
    @abstractmethod
    def export_config(self) -> Dict[str, Any]:
        """Export smoother configuration.
        
        Returns:
            Dictionary with smoother parameters
        """
        pass


class DepthwiseCausalSmoother(SmootherBase):
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
    
    def forward(
        self,
        x: torch.FloatTensor,
        lengths: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """Apply depthwise causal smoothing.
        
        Args:
            x: Input tensor [B, T, F]
            lengths: Sequence lengths [B]
            
        Returns:
            Tuple of (smoothed tensor, unchanged lengths)
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
        
        return x_smoothed, lengths
    
    def receptive_field(self) -> int:
        """Get the receptive field size."""
        return self.kernel_size
    
    def get_effective_kernels(self) -> torch.Tensor:
        """Get the effective smoothing kernels after softmax normalization.
        
        Returns:
            Normalized kernel weights [F, K]
        """
        with torch.no_grad():
            weights = F.softmax(self.kernel_params, dim=-1).squeeze(1)  # [F, K]
        return weights
    
    def export_config(self) -> Dict[str, Any]:
        """Export smoother configuration."""
        return {
            'type': 'depthwise_causal',
            'num_features': self.num_features,
            'kernel_size': self.kernel_size,
            'residual_gate': self.residual_gate,
            'gate_value': torch.sigmoid(self.gate_param).item() if self.residual_gate else None
        }
    
    def extra_repr(self) -> str:
        return f'num_features={self.num_features}, kernel_size={self.kernel_size}, residual_gate={self.residual_gate}'


class GaussianSmoother(SmootherBase):
    """Fixed Gaussian smoothing for temporal signals.
    
    Applies 1D Gaussian smoothing along the time dimension independently
    for each feature channel. This is implemented as a depthwise convolution
    with fixed Gaussian weights.
    
    Args:
        num_features: Number of input feature channels
        kernel_size: Size of the Gaussian kernel (should be odd)
        std: Standard deviation of the Gaussian
        trainable: If True, kernel parameters are trainable
    """
    
    def __init__(
        self,
        num_features: int,
        kernel_size: int = 100,
        std: float = 2.0,
        trainable: bool = False
    ):
        super().__init__()
        
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
            
        self.num_features = num_features
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
    
    def forward(
        self,
        x: torch.FloatTensor,
        lengths: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """Apply Gaussian smoothing.
        
        Args:
            x: Input tensor [B, T, F]
            lengths: Sequence lengths [B]
            
        Returns:
            Tuple of (smoothed tensor, unchanged lengths)
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
        
        return x_smoothed, lengths
    
    def receptive_field(self) -> int:
        """Get the receptive field size."""
        return self.kernel_size
    
    def export_config(self) -> Dict[str, Any]:
        """Export smoother configuration."""
        return {
            'type': 'gaussian',
            'num_features': self.num_features,
            'kernel_size': self.kernel_size,
            'std': self.std,
            'trainable': self.trainable
        }
    
    def extra_repr(self) -> str:
        return f'num_features={self.num_features}, kernel_size={self.kernel_size}, std={self.std}, trainable={self.trainable}'


class EMASmoother(SmootherBase):
    """Exponential Moving Average (IIR) smoother.
    
    Implements causal IIR filtering: y_t = α·y_{t-1} + (1-α)·x_t
    where α is learned per-channel via sigmoid parameterization.
    
    This is extremely efficient (no convolution) and strictly causal,
    with an infinite but exponentially decaying receptive field.
    
    Args:
        num_features: Number of input feature channels
        init_alpha: Initial smoothing factor (0 = no smoothing, 1 = infinite smoothing)
        residual_gate: If True, use gated residual connection
    """
    
    def __init__(
        self,
        num_features: int,
        init_alpha: float = 0.2,
        residual_gate: bool = True
    ):
        super().__init__()
        
        self.num_features = num_features
        self.residual_gate = residual_gate
        
        # Alpha parameters (one per feature, transformed with sigmoid)
        # Initialize to achieve desired alpha via inverse sigmoid
        init_logit = math.log(init_alpha / (1 - init_alpha)) if init_alpha > 0 and init_alpha < 1 else 0
        self.alpha_logits = nn.Parameter(torch.full((num_features,), init_logit))
        
        # Residual gate
        if residual_gate:
            self.gate_param = nn.Parameter(torch.tensor(-1.386))  # sigmoid(-1.386) ≈ 0.2
    
    def forward(
        self,
        x: torch.FloatTensor,
        lengths: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """Apply EMA smoothing.
        
        Args:
            x: Input tensor [B, T, F]
            lengths: Sequence lengths [B]
            
        Returns:
            Tuple of (smoothed tensor, unchanged lengths)
        """
        batch_size, seq_len, feat_dim = x.shape
        device = x.device
        
        # Get smoothing factors
        alphas = torch.sigmoid(self.alpha_logits)  # [F]
        
        # Initialize hidden state
        y = torch.zeros_like(x)
        h = torch.zeros(batch_size, feat_dim, device=device)  # [B, F]
        
        # Apply IIR filter causally
        for t in range(seq_len):
            h = alphas * h + (1 - alphas) * x[:, t, :]  # [B, F]
            y[:, t, :] = h
        
        # Apply residual gate if enabled
        if self.residual_gate:
            gate = torch.sigmoid(self.gate_param)
            y = (1 - gate) * x + gate * y
        
        return y, lengths
    
    def receptive_field(self) -> int:
        """Get approximate receptive field (95% of weight mass)."""
        # For EMA with alpha, effective window ≈ -log(0.05) / -log(alpha)
        with torch.no_grad():
            avg_alpha = torch.sigmoid(self.alpha_logits).mean().item()
            if avg_alpha > 0.99:
                return 1000  # Very long
            elif avg_alpha < 0.01:
                return 1  # No smoothing
            else:
                return int(-math.log(0.05) / -math.log(avg_alpha))
    
    def export_config(self) -> Dict[str, Any]:
        """Export smoother configuration."""
        with torch.no_grad():
            alphas = torch.sigmoid(self.alpha_logits)
        
        return {
            'type': 'ema',
            'num_features': self.num_features,
            'alpha_mean': alphas.mean().item(),
            'alpha_std': alphas.std().item(),
            'residual_gate': self.residual_gate,
            'gate_value': torch.sigmoid(self.gate_param).item() if self.residual_gate else None
        }
    
    def extra_repr(self) -> str:
        with torch.no_grad():
            alpha_mean = torch.sigmoid(self.alpha_logits).mean().item()
        return f'num_features={self.num_features}, alpha_mean={alpha_mean:.3f}, residual_gate={self.residual_gate}'


class IdentitySmoother(SmootherBase):
    """Identity smoother (no smoothing).
    
    This is useful for ablation studies and debugging.
    """
    
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
    
    def forward(
        self,
        x: torch.FloatTensor,
        lengths: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """Pass through without smoothing."""
        return x, lengths
    
    def receptive_field(self) -> int:
        """No receptive field for identity."""
        return 1
    
    def export_config(self) -> Dict[str, Any]:
        """Export configuration."""
        return {'type': 'identity', 'num_features': self.num_features}


# Smoother registry for easy instantiation
SMOOTHERS = {
    'gaussian': GaussianSmoother,
    'depthwise_causal': DepthwiseCausalSmoother,
    'dw_causal': DepthwiseCausalSmoother,  # Alias
    'ema': EMASmoother,
    'identity': IdentitySmoother,
    'none': IdentitySmoother,  # Alias
}


def build_smoother(
    smoother_type: str,
    num_features: int,
    config: Optional[Dict[str, Any]] = None
) -> SmootherBase:
    """Build a smoother from configuration.
    
    Args:
        smoother_type: Type of smoother ('gaussian', 'depthwise_causal', 'ema', 'identity')
        num_features: Number of input features
        config: Additional configuration parameters
        
    Returns:
        Instantiated smoother
        
    Raises:
        ValueError: If smoother type is not recognized
    """
    if smoother_type not in SMOOTHERS:
        raise ValueError(f"Unknown smoother type: {smoother_type}. Available: {list(SMOOTHERS.keys())}")
    
    config = config or {}
    smoother_class = SMOOTHERS[smoother_type]
    
    # Add num_features to config
    config['num_features'] = num_features
    
    return smoother_class(**config)
