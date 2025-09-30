"""Window projection layer for gradual dimensionality reduction.

This module implements a window projection layer that gradually reduces
dimensionality from patched input to GRU hidden size using GLU activation.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowProjection(nn.Module):
    """Window projection layer with GLU activation.
    
    Implements gradual dimensionality reduction:
    input_dim → intermediate_dim → GLU → output_dim
    
    This addresses the information bottleneck by providing a smooth
    transition from high-dimensional patched input to GRU hidden size.
    
    Args:
        input_dim: Input dimension (typically 6144 from patching)
        intermediate_dim: Intermediate dimension for GLU (e.g., 1536)
        output_dim: Output dimension for GRU input (e.g., 1024)
        dropout: Dropout probability between layers
        layer_norm: Whether to use layer normalization
        bias: Whether to use bias in linear layers
    """
    
    def __init__(
        self,
        input_dim: int = 6144,
        intermediate_dim: int = 1536,
        output_dim: int = 1024,
        dropout: float = 0.1,
        layer_norm: bool = True,
        bias: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim
        
        # First projection: input_dim → intermediate_dim
        self.input_projection = nn.Linear(input_dim, intermediate_dim, bias=bias)
        
        # GLU layer: intermediate_dim → intermediate_dim (split into gates)
        # GLU needs 2x intermediate_dim to split into (a, b)
        self.glu_projection = nn.Linear(intermediate_dim, intermediate_dim * 2, bias=bias)
        
        # Final projection: intermediate_dim → output_dim
        self.output_projection = nn.Linear(intermediate_dim, output_dim, bias=bias)
        
        # Optional layer normalization
        self.layer_norm = nn.LayerNorm(intermediate_dim) if layer_norm else nn.Identity()
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        # Xavier initialization for linear layers
        for module in [self.input_projection, self.glu_projection, self.output_projection]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through window projection.
        
        Args:
            x: Input tensor [B, T, input_dim]
            lengths: Sequence lengths [B] (unchanged)
            
        Returns:
            Tuple of (projected_x, lengths) where:
            - projected_x: [B, T, output_dim]
            - lengths: [B] (unchanged)
        """
        # First projection: input_dim → intermediate_dim
        x = self.input_projection(x)  # [B, T, intermediate_dim]
        
        # Apply layer norm if enabled
        x = self.layer_norm(x)
        
        # GLU activation: split and gate
        glu_input = self.glu_projection(x)  # [B, T, intermediate_dim * 2]
        
        # Split into two halves for GLU
        a, b = glu_input.chunk(2, dim=-1)  # Each: [B, T, intermediate_dim]
        
        # GLU: a ⊙ σ(b)
        x = a * torch.sigmoid(b)  # [B, T, intermediate_dim]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Final projection: intermediate_dim → output_dim
        x = self.output_projection(x)  # [B, T, output_dim]
        
        return x, lengths
    
    def export_config(self) -> dict:
        """Export configuration for reproducibility."""
        return {
            'type': 'window_projection',
            'input_dim': self.input_dim,
            'intermediate_dim': self.intermediate_dim,
            'output_dim': self.output_dim,
            'dropout': self.dropout.p if hasattr(self.dropout, 'p') else 0.0,
            'layer_norm': isinstance(self.layer_norm, nn.LayerNorm),
            'bias': self.input_projection.bias is not None
        }