"""Utility functions for neural encoder blocks.

This module provides helper functions for length computation, masking,
and other common operations used across encoder blocks.
"""

from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence




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
        # Patching concatenates 'patch_size' frames with 'patch_stride' step
        # Output length = floor((L - patch_size) / patch_stride) + 1 if L >= patch_size, else 0
        if patch_size > 1:
            has_valid_window = lengths >= patch_size
            new_lengths = torch.div(lengths - patch_size, patch_stride, rounding_mode='floor') + 1
            lengths = torch.where(has_valid_window, new_lengths, torch.zeros_like(lengths))
            # Ensure lengths stay positive
            lengths = torch.clamp(lengths, min=1)
    
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


# RNN Helper Functions
# These are shared between GRU and LSTM implementations

def feat_dim_from_seq(seq, max_len, default):
    """Get feature dimension from sequence, handling both PackedSequence and tensor."""
    if isinstance(seq, PackedSequence):
        # data is [sum(batch_sizes), F]
        return seq.data.size(-1)
    else:
        return seq.size(-1) if seq.dim() == 3 else default


def expand_h0(p, B, device, dtype):
    """Expand learnable h0 parameter to batch size."""
    return p.to(device=device, dtype=dtype).expand(-1, B, -1).contiguous()


def apply_locked_dropout(
    seq: torch.Tensor, 
    dropout_p: float,
    batch_size: int, 
    max_len: int,
    output_size: int
) -> torch.Tensor:
    """Apply locked dropout with consistent masks across time.
    
    Locked dropout uses the same dropout mask across all time steps
    for each sample, which helps preserve temporal dependencies better
    than standard dropout. Masks are regenerated for each forward pass.
    
    Args:
        seq: Input sequence (PackedSequence or tensor)
        dropout_p: Dropout probability
        batch_size: Batch size
        max_len: Maximum sequence length
        output_size: Default output size for feature dimension
        
    Returns:
        Sequence with locked dropout applied
    """
    if dropout_p <= 0:
        return seq
        
    if isinstance(seq, PackedSequence):
        # Unpack for dropout application
        x, lengths = pad_packed_sequence(seq, batch_first=True, total_length=max_len)
        
        # Get feature dimension from actual sequence
        feat_dim = feat_dim_from_seq(seq, max_len, output_size)
        
        # Create mask: [B, 1, F] - broadcasts across time dimension
        # Generate fresh mask for each forward pass
        mask = x.new_empty((batch_size, 1, feat_dim)).bernoulli_(1 - dropout_p).div_(1 - dropout_p)
        
        # Apply mask to all time steps
        x = x * mask
        
        # Repack the sequence
        return pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
    else:
        # Handle regular tensor (not packed)
        B, T, F = seq.shape
        
        # Create mask: [B, 1, F] - broadcasts across time dimension
        # Generate fresh mask for each forward pass
        mask = seq.new_empty((B, 1, F)).bernoulli_(1 - dropout_p).div_(1 - dropout_p)
        
        # Apply mask (broadcasts across time dimension)
        return seq * mask


def apply_output_zoneout(
    seq, zoneout_p, lengths, max_len, bidirectional=False
):
    """Apply output zoneout regularization.
    
    Output zoneout mixes current hidden states with previous time step states,
    providing regularization similar to true zoneout but compatible with cuDNN.
    
    Args:
        seq: Input sequence (PackedSequence or tensor)
        zoneout_p: Zoneout probability
        lengths: Sequence lengths
        max_len: Maximum sequence length
        bidirectional: Whether the sequence is bidirectional
        
    Returns:
        Sequence with zoneout applied
    """
    if zoneout_p <= 0:
        return seq

    if isinstance(seq, PackedSequence):
        x, lens = pad_packed_sequence(seq, batch_first=True, total_length=max_len)
    else:
        x, lens = seq, lengths

    # x: [B, T, F]; build time-shifted prev state (forward dir)
    prev = x.clone()
    prev[:, 1:] = x[:, :-1]
    # for t=0, keep the first state
    
    # bidirectional handling: split, shift independently, then cat
    if bidirectional:
        F = x.size(-1) // 2
        fwd, bwd = x[..., :F], x[..., F:]
        f_prev, b_prev = fwd.clone(), bwd.clone()
        f_prev[:, 1:] = fwd[:, :-1]      # forward uses t-1
        b_prev[:, :-1] = bwd[:, 1:]      # backward uses t+1
        prev = torch.cat([f_prev, b_prev], dim=-1)

    # Build a locked mask: [B, 1, F]
    mask = x.new_empty((x.size(0), 1, x.size(-1))).bernoulli_(1 - zoneout_p)
    # Keep expected value unchanged
    mask = mask / (1 - zoneout_p)

    # Apply: m * prev + (1-m) * x
    y = mask * prev + (1 - mask) * x

    # Zero-out padded timesteps so they don't leak
    if lens is not None:
        # build boolean mask [B, T, 1]
        tt = torch.arange(max_len, device=x.device).unsqueeze(0)
        valid = tt < lens.unsqueeze(1)
        y = y * valid.unsqueeze(-1)

    if isinstance(seq, PackedSequence):
        return pack_padded_sequence(y, lens.cpu(), batch_first=True, enforce_sorted=False)
    else:
        return y


def apply_interlayer_dropout(seq, layer_idx, batch_size, max_len, dropout, dropout_type, num_layers, output_size, dropout_modules, training):
    """Apply inter-layer dropout (locked or standard)."""
    if not training or dropout <= 0 or layer_idx == num_layers - 1:
        return seq
    
    if dropout_type == 'locked':
        return apply_locked_dropout(seq, dropout, batch_size, max_len, output_size)
    else:
        # Standard dropout
        if isinstance(seq, PackedSequence):
            data = dropout_modules[layer_idx](seq.data)
            return PackedSequence(data, seq.batch_sizes, seq.sorted_indices, seq.unsorted_indices)
        else:
            return dropout_modules[layer_idx](seq)


