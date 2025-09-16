"""Utility functions for neural encoder blocks.

This module provides helper functions for length computation, masking,
and other common operations used across encoder blocks.
"""

from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F
import math


def compute_output_lengths(
    input_lengths: torch.LongTensor,
    ops_config: List[Dict[str, Any]]
) -> torch.LongTensor:
    """Compute output lengths after a sequence of time-reducing operations.
    
    This is a pure function that composes length transformations from
    various operations like patching, convolution, and pooling.
    
    Args:
        input_lengths: Original sequence lengths [B]
        ops_config: List of operations with their configurations
            Each operation dict should have 'type' and relevant params:
            - {'type': 'patch', 'size': N, 'stride': S}
            - {'type': 'conv1d', 'kernel_size': K, 'stride': S, 'padding': P}
            - {'type': 'pool', 'kernel_size': K, 'stride': S}
            
    Returns:
        Output lengths after all operations [B]
        
    Example:
        >>> lengths = torch.tensor([100, 150])
        >>> ops = [
        ...     {'type': 'patch', 'size': 4, 'stride': 2},
        ...     {'type': 'conv1d', 'kernel_size': 3, 'stride': 1, 'padding': 1}
        ... ]
        >>> compute_output_lengths(lengths, ops)
        tensor([49, 74])
    """
    lengths = input_lengths.clone()
    
    for op in ops_config:
        op_type = op['type']
        
        if op_type == 'patch':
            # Patching concatenates 'size' frames with 'stride' step
            # Output length = floor((L - size) / stride) + 1 if L >= size, else 0
            size = op['size']
            stride = op.get('stride', size)  # Default stride = size (non-overlapping)
            if size > 1:
                has_valid_window = lengths >= size
                new_lengths = torch.div(lengths - size, stride, rounding_mode='floor') + 1
                lengths = torch.where(has_valid_window, new_lengths, torch.zeros_like(lengths))
                
        elif op_type == 'conv1d':
            # Standard convolution length formula
            kernel_size = op['kernel_size']
            stride = op.get('stride', 1)
            padding = op.get('padding', 0)
            dilation = op.get('dilation', 1)
            
            effective_kernel = (kernel_size - 1) * dilation + 1
            lengths = torch.floor((lengths + 2 * padding - effective_kernel) / stride).long() + 1
            
        elif op_type == 'pool':
            # Pooling length formula
            kernel_size = op['kernel_size']
            stride = op.get('stride', kernel_size)
            padding = op.get('padding', 0)
            
            lengths = torch.floor((lengths + 2 * padding - kernel_size) / stride).long() + 1
            
        elif op_type == 'downsample':
            # Simple downsampling by factor
            factor = op['factor']
            lengths = torch.ceil(lengths / factor).long()
            
        # Ensure lengths stay positive
        lengths = torch.clamp(lengths, min=1)
    
    return lengths


def mask_logits_(
    logits: torch.FloatTensor,
    lengths: torch.LongTensor,
    mask_value: float = -1e9
) -> None:
    """In-place masking of logits beyond valid lengths.
    
    Sets all logit values beyond each sequence's valid length to a large
    negative value, ensuring they become ~0 after softmax.
    
    Args:
        logits: Logit tensor to mask [B, T, V]
        lengths: Valid lengths for each sequence [B]
        mask_value: Value to set for masked positions (default: -1e9)
        
    Note:
        This function modifies logits in-place for efficiency.
    """
    batch_size, max_len, vocab_size = logits.shape
    
    # Create mask [B, T]
    mask = torch.arange(max_len, device=logits.device).unsqueeze(0)
    mask = mask >= lengths.unsqueeze(1)
    
    # Apply mask to all vocabulary dimensions
    logits.masked_fill_(mask.unsqueeze(-1), mask_value)


def create_padding_mask(
    lengths: torch.LongTensor,
    max_len: Optional[int] = None,
    dtype: torch.dtype = torch.bool
) -> torch.Tensor:
    """Create a padding mask from sequence lengths.
    
    Args:
        lengths: Valid lengths for each sequence [B]
        max_len: Maximum sequence length (inferred if None)
        dtype: Data type for mask (default: bool)
        
    Returns:
        Mask tensor where True indicates valid positions [B, T]
    """
    if max_len is None:
        max_len = lengths.max().item()
    
    mask = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    mask = mask < lengths.unsqueeze(1)
    
    return mask.to(dtype)


def apply_patch_embedding(
    x: torch.FloatTensor,
    patch_size: int,
    patch_stride: int,
    lengths: Optional[torch.LongTensor] = None
) -> Tuple[torch.FloatTensor, Optional[torch.LongTensor]]:
    """Apply patching (frame concatenation) to input sequences.
    
    This operation concatenates consecutive frames to create patches,
    reducing the temporal dimension while increasing the feature dimension.
    
    Args:
        x: Input tensor [B, T, F]
        patch_size: Number of frames to concatenate
        patch_stride: Stride between patches
        lengths: Optional sequence lengths [B]
        
    Returns:
        Tuple of:
        - Patched tensor [B, T_out, F * patch_size]
        - Updated lengths if provided [B]
    """
    if patch_size <= 1:
        return x, lengths
    
    batch_size, seq_len, feat_dim = x.shape
    
    # Use unfold to create patches
    # unfold dimension order: [B, F, T] -> [B, F, num_patches, patch_size]
    x_transposed = x.transpose(1, 2)  # [B, F, T]
    patches = x_transposed.unfold(
        dimension=2,
        size=patch_size,
        step=patch_stride
    )  # [B, F, num_patches, patch_size]
    
    # Reshape to [B, num_patches, F * patch_size]
    num_patches = patches.shape[2]
    patches = patches.permute(0, 2, 1, 3)  # [B, num_patches, F, patch_size]
    patches = patches.reshape(batch_size, num_patches, feat_dim * patch_size)
    
    # Update lengths if provided
    if lengths is not None:
        lengths = compute_output_lengths(
            lengths,
            [{'type': 'patch', 'size': patch_size, 'stride': patch_stride}]
        )
    
    return patches, lengths


def get_activation_fn(name: str) -> torch.nn.Module:
    """Get activation function by name.
    
    Args:
        name: Activation function name
        
    Returns:
        PyTorch activation module
        
    Raises:
        ValueError: If activation name is not recognized
    """
    activations = {
        'relu': torch.nn.ReLU,
        'gelu': torch.nn.GELU,
        'silu': torch.nn.SiLU,
        'softsign': torch.nn.Softsign,
        'tanh': torch.nn.Tanh,
        'sigmoid': torch.nn.Sigmoid,
        'identity': torch.nn.Identity,
    }
    
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
    
    return activations[name]()


def calculate_conv_output_length(
    input_length: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1
) -> int:
    """Calculate output length for 1D convolution.
    
    Args:
        input_length: Input sequence length
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Padding on both sides
        dilation: Dilation factor
        
    Returns:
        Output sequence length
    """
    effective_kernel = (kernel_size - 1) * dilation + 1
    return math.floor((input_length + 2 * padding - effective_kernel) / stride) + 1


def create_causal_mask(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.bool
) -> torch.Tensor:
    """Create a causal mask for autoregressive attention.
    
    Args:
        seq_len: Sequence length
        device: Device to create mask on
        dtype: Data type for mask
        
    Returns:
        Causal mask [seq_len, seq_len] where True indicates valid positions
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=dtype), diagonal=1)
    return ~mask  # Invert so True means "can attend"
