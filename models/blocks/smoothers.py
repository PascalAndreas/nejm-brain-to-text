"""Temporal smoothing components for neural encoder models.

This module provides various smoothing strategies for preprocessing
neural signals, with both causal and non-causal options, all designed
for DC gain preservation and robust handling of variable-length sequences.
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
    - May be causal or non-causal depending on implementation
    - Preserve sequence length (stride=1)
    - Should preserve DC gain on constant inputs (including near boundaries and with padding)
    - Support variable-length sequences robustly
    - Handle zero-padded batches without bias from padding
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


class DepthwiseSmoother(SmootherBase):
    """Learnable depthwise convolution for temporal smoothing.
    
    - One kernel per feature channel, normalized with softmax (sum=1).
    - Supports both causal and non-causal modes.
    - Length-aware masked renormalization to preserve DC gain near edges
      and avoid bias from zero padding in batched variable-length inputs.
    
    Args:
        num_features: Number of input feature channels
        kernel_size: Size of convolution kernel (forced to be odd for symmetric padding)
        residual_gate: If True, use gated residual connection y = (1-α)x + α·conv(x)
        init_std: Standard deviation for Gaussian initialization (in frames)
        eps: Small epsilon for numerical stability in renormalization
        causal: If True, use causal convolution (no future leakage)
    """

    def __init__(
        self,
        num_features: int,
        kernel_size: int = 11,
        residual_gate: bool = True,
        init_std: float = 2.0,
        eps: float = 1e-8,
        causal: bool = False,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1  # ensure odd for symmetric padding

        self.num_features = num_features
        self.kernel_size = kernel_size
        self.residual_gate = residual_gate
        self.eps = eps
        self.causal = causal

        # Learnable per-channel kernels in log-space -> softmax along K
        self.kernel_params = nn.Parameter(torch.zeros(num_features, 1, kernel_size))
        self._init_gaussian_kernel(init_std)

        if residual_gate:
            # shared gate alpha in (0,1), init ~0.2
            self.gate_param = nn.Parameter(torch.full((1,), -1.386))

    @torch.no_grad()
    def _init_gaussian_kernel(self, std: float):
        """Initialize kernel parameters to symmetric Gaussian."""
        K = self.kernel_size
        coords = torch.arange(K, dtype=torch.float32)
        coords = coords - (K - 1) / 2.0
        g = torch.exp(-(coords**2) / (2 * std * std))
        g = g / g.sum()  # sum=1
        self.kernel_params.copy_(g.log().view(1, 1, K).expand_as(self.kernel_params))

    def forward(
        self,
        x: torch.FloatTensor,
        lengths: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        """Apply depthwise smoothing with masked renormalization (causal or non-causal).
        
        Args:
            x: Input tensor [B, T, F]
            lengths: Sequence lengths [B]
            
        Returns:
            Tuple of (smoothed tensor, unchanged lengths)
        """
        # x: [B, T, F] -> conv wants [B, F, T]
        B, T, num_feats = x.shape
        k = torch.softmax(self.kernel_params, dim=-1)               # [F,1,K]
        xt = x.transpose(1, 2).contiguous()                         # [B,F,T]

        # Build per-sample validity mask m: [B,1,T] (1 for valid frames)
        if lengths is None:
            m = xt.new_ones((B, 1, T))
        else:
            if lengths.device != xt.device:
                lengths = lengths.to(xt.device)
            t = torch.arange(T, device=xt.device).view(1, 1, T)
            m = (t < lengths.view(B, 1, 1)).to(dtype=xt.dtype)      # [B,1,T]

        # Expand mask channel-wise to match depthwise groups
        mF = m.expand(B, num_feats, T)                              # [B,F,T]

        # Apply padding based on causal mode
        if self.causal:
            # Causal padding: pad only on the left (past frames only)
            padding_left = self.kernel_size - 1
            padding_right = 0
            xt_pad = F.pad(xt, (padding_left, padding_right), mode="reflect")  # [B,F,T+p]
            m_pad  = F.pad(mF, (padding_left, padding_right), mode="constant", value=0.0)
        else:
            # Symmetric padding for both x and m (same amount left/right)
            padding = self.kernel_size // 2
            xt_pad = F.pad(xt, (padding, padding), mode="reflect")  # [B,F,T+2p]
            m_pad  = F.pad(mF, (padding, padding), mode="constant", value=0.0)  # mask still needs zero padding

        # Depthwise conv on x*m and on m, using the same per-channel kernel
        num = F.conv1d(xt_pad * m_pad, k, groups=num_feats)         # [B,F,T]
        den = F.conv1d(m_pad,       k, groups=num_feats)            # [B,F,T]

        y = num / (den + self.eps)                                  # renormalize

        # Residual gate (optional): y = (1-a) x + a y
        if self.residual_gate:
            a = torch.sigmoid(self.gate_param)                      # scalar
            y = (1 - a) * xt + a * y

        # Zero out invalid tail explicitly (safety)
        if lengths is not None:
            y = y * mF

        return y.transpose(1, 2).contiguous(), lengths

    def receptive_field(self) -> int:
        """Get the receptive field size."""
        return self.kernel_size

    @torch.no_grad()
    def get_effective_kernels(self) -> torch.Tensor:
        """Get the effective smoothing kernels after softmax normalization.
        
        Returns:
            Normalized kernel weights [F, K]
        """
        return torch.softmax(self.kernel_params, dim=-1).squeeze(1)  # [F,K]

    def export_config(self) -> Dict[str, Any]:
        """Export smoother configuration."""
        return {
            "type": "depthwise",
            "num_features": self.num_features,
            "kernel_size": self.kernel_size,
            "residual_gate": self.residual_gate,
            "causal": self.causal,
            "gate_value": torch.sigmoid(self.gate_param).item() if self.residual_gate else None,
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
        """Apply Gaussian smoothing with masked renormalization.
        
        Args:
            x: Input tensor [B, T, F]
            lengths: Sequence lengths [B]
            
        Returns:
            Tuple of (smoothed tensor, unchanged lengths)
        """
        B, T, num_feats = x.shape
        # kernel 1D, same for all channels
        k1 = self.kernel.view(1, 1, self.kernel_size)               # [1,1,K]
        kF = k1.expand(num_feats, 1, -1)                            # [F,1,K]

        xt = x.transpose(1, 2).contiguous()                         # [B,F,T]

        # Build per-sample validity mask
        if lengths is None:
            m = xt.new_ones((B, 1, T))
        else:
            if lengths.device != xt.device:
                lengths = lengths.to(xt.device)
            t = torch.arange(T, device=xt.device).view(1, 1, T)
            m = (t < lengths.view(B, 1, 1)).to(dtype=xt.dtype)      # [B,1,T]
        mF = m.expand(B, num_feats, T)

        # Symmetric padding for both x and m
        padding = self.kernel_size // 2
        xt_pad = F.pad(xt, (padding, padding), mode="reflect")
        m_pad  = F.pad(mF, (padding, padding), mode="constant", value=0.0)  # mask still needs zero padding

        # Masked renormalization: convolve x*m and m separately, then divide
        num = F.conv1d(xt_pad * m_pad, kF, groups=num_feats)        # [B,F,T]
        den = F.conv1d(m_pad,       kF, groups=num_feats)           # [B,F,T]
        y   = num / (den + 1e-8)

        # safety: zero invalid tail
        if lengths is not None:
            y = y * mF

        return y.transpose(1, 2).contiguous(), lengths
    
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
        
        # Initialize hidden state (ensure dtype compatibility for AMP)
        y = torch.zeros_like(x)
        h = x.new_zeros(batch_size, feat_dim)  # [B, F]
        
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
    'depthwise': DepthwiseSmoother,  # Supports both causal and non-causal modes
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
        smoother_type: Type of smoother ('gaussian', 'depthwise', 'depthwise_causal', 'ema', 'identity')
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
