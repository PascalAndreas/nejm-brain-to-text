"""Utility functions for neural encoder blocks.

This module provides helper functions for length computation, masking,
and other common operations used across encoder blocks.
"""

from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn.functional as F
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


def time_mask_like(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Return a [B,T,1] mask with 1 on valid frames, 0 on padded tail.
    
    Args:
        x: Input tensor [B, T, F]
        lengths: Valid lengths for each sequence [B]
        
    Returns:
        Time mask [B, T, 1] with 1 for valid frames, 0 for padding
    """
    B, T = x.size(0), x.size(1)
    if lengths is None:
        return x.new_ones((B, T, 1))
    if lengths.device != x.device:
        lengths = lengths.to(x.device)
    t = torch.arange(T, device=x.device).unsqueeze(0)         # [1,T]
    m = (t < lengths.unsqueeze(1)).unsqueeze(-1).to(x.dtype)  # [B,T,1]
    return m




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


def force_blank_on_pad_(
    logits: torch.Tensor,
    lengths: torch.LongTensor,
    blank_idx: int
) -> None:
    """Force blank token on padded frames to avoid NaNs in log_softmax.
    
    For t >= length[b], set logits[b, t, blank] = 0 and others = very negative.
    This avoids NaNs in log_softmax and yields near-1.0 blank prob on padded frames.
    
    Args:
        logits: Logit tensor to modify in-place [B, T, V]
        lengths: Valid lengths for each sequence [B]
        blank_idx: Index of the CTC blank token
        
    Note:
        This function modifies logits in-place for efficiency.
    """
    B, T, V = logits.shape
    device = logits.device
    if lengths.device != device:
        lengths = lengths.to(device)
    
    # Build time mask [B, T, 1] - True where padded
    t = torch.arange(T, device=device).view(1, T)
    mask = (t >= lengths.view(B, 1)).unsqueeze(-1)  # [B, T, 1]
    
    if mask.any():
        # Use a large negative value, but not -inf to avoid NaNs
        very_neg = torch.finfo(logits.dtype).min / 2
        
        # Set all classes to very_neg at padded frames
        logits.masked_fill_(mask.expand(-1, -1, V), very_neg)
        
        # Set blank to 0 at padded frames
        logits[..., blank_idx].masked_fill_(mask.squeeze(-1), 0.0)


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
    lengths: Optional[torch.LongTensor] = None,
) -> Tuple[torch.FloatTensor, Optional[torch.LongTensor]]:
    """
    Returns per-sample patched lengths and masks away invalid windows.

    Args:
        x:        [B, T, F] padded batch (zeros beyond each sample's length)
        patch_size:  number of frames per patch (>= 2 in practice here)
        patch_stride: stride between consecutive patches
        lengths:  [B] true frame lengths for each sample (all >= patch_size)

    Returns:
        patches:     [B, K_full, F * patch_size] (K_full based on batch T)
        new_lengths: [B] per-sample valid patch counts
    """
    if patch_size <= 1:
        return x, lengths  # no-op

    B, T, F = x.shape
    device  = x.device

    # [B, F, T] → unfold over time: [B, F, K_full, patch_size]
    xt = x.transpose(1, 2).contiguous()
    patches4 = xt.unfold(dimension=2, size=patch_size, step=patch_stride)
    K_full = patches4.size(2)

    # [B, K_full, F * patch_size]
    patches = patches4.permute(0, 2, 1, 3).reshape(B, K_full, F * patch_size)

    if lengths is None:
        # No masking possible/needed; caller guarantees padding semantics downstream
        return patches, None

    # Ensure lengths on same device
    lengths = lengths.to(device)

    # Valid windows per sample:
    # new_len = floor((L - patch_size) / patch_stride) + 1  (given L >= patch_size)
    new_lengths = torch.div(lengths - patch_size, patch_stride, rounding_mode='floor') + 1

    # Mask out windows beyond each sample's valid count (keep K_full, zero tail)
    if K_full > 0:
        idx = torch.arange(K_full, device=device).view(1, K_full)          # [1, K_full]
        mask = (idx < new_lengths.view(B, 1)).unsqueeze(-1)                 # [B, K_full, 1]
        patches = patches * mask.to(patches.dtype)

    return patches, new_lengths


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
    feat_dim: Optional[int] = None,
    lengths: Optional[torch.LongTensor] = None
) -> torch.Tensor:
    """Apply locked dropout with consistent masks across time.
    
    Locked dropout uses the same dropout mask across all time steps
    for each sample, which helps preserve temporal dependencies better
    than standard dropout. Masks are regenerated for each forward pass.
    
    Optimized to work directly with unpacked tensors for better performance.
    
    Args:
        seq: Input sequence tensor [B, T, F] (should be unpacked)
        dropout_p: Dropout probability
        batch_size: Batch size
        max_len: Maximum sequence length
        feat_dim: Feature dimension (computed from seq if None)
        lengths: Sequence lengths for masking (optional)
        
    Returns:
        Tensor with locked dropout applied
    """
    if dropout_p <= 0:
        return seq
    
    # Work directly with unpacked tensor for efficiency
    if isinstance(seq, PackedSequence):
        raise ValueError("apply_locked_dropout now expects unpacked tensors for better performance")
        
    B, T, F = seq.shape
    feat_dim = F if (feat_dim is None) else feat_dim
    
    # Create mask: [B, 1, F] - broadcasts across time dimension
    # Generate fresh mask for each forward pass
    mask = seq.new_empty((B, 1, feat_dim)).bernoulli_(1 - dropout_p).div_(1 - dropout_p)
    
    # Apply mask (broadcasts across time dimension)
    seq = seq * mask
    
    # Apply length masking to zero out padding
    if lengths is not None:
        lengths_device = lengths.to(seq.device)
        time_mask = torch.arange(T, device=seq.device).unsqueeze(0) < lengths_device.unsqueeze(1)
        seq = seq * time_mask.unsqueeze(-1).float()
    
    return seq


def apply_output_zoneout(
    seq: torch.Tensor, zoneout_p: float, lengths: Optional[torch.LongTensor], 
    max_len: int, bidirectional: bool = False
) -> torch.Tensor:
    """Apply output zoneout regularization.
    
    Output zoneout mixes current hidden states with previous time step states,
    providing regularization similar to true zoneout but compatible with cuDNN.
    
    Optimized to work directly with unpacked tensors for better performance.
    
    Args:
        seq: Input sequence tensor [B, T, F] (should be unpacked)
        zoneout_p: Zoneout probability
        lengths: Sequence lengths for masking (optional)
        max_len: Maximum sequence length
        bidirectional: Whether the sequence is bidirectional
        
    Returns:
        Tensor with zoneout applied
    """
    if zoneout_p <= 0:
        return seq

    # Work directly with unpacked tensor for efficiency
    if isinstance(seq, PackedSequence):
        raise ValueError("apply_output_zoneout now expects unpacked tensors for better performance")

    x = seq

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
    if lengths is not None:
        lengths_device = lengths.to(x.device)
        tt = torch.arange(max_len, device=x.device).unsqueeze(0)
        valid = tt < lengths_device.unsqueeze(1)
        y = y * valid.unsqueeze(-1)

    return y


def apply_interlayer_dropout(seq, layer_idx, batch_size, max_len, dropout, dropout_type, num_layers, dropout_modules, training, lengths=None):
    """Apply inter-layer dropout (locked or standard).
    
    Optimized to work with unpacked tensors for better performance.
    """
    if not training or dropout <= 0 or layer_idx == num_layers - 1:
        return seq
    
    # Work directly with unpacked tensor for efficiency
    if isinstance(seq, PackedSequence):
        raise ValueError("apply_interlayer_dropout now expects unpacked tensors for better performance")
    
    if dropout_type == 'locked':
        feat_dim = seq.size(-1)
        return apply_locked_dropout(seq, dropout, batch_size, max_len, feat_dim, lengths)
    else:
        # Standard dropout - work directly on unpacked tensor
        return dropout_modules[layer_idx](seq)


def flatten_ctc_targets(y: torch.Tensor, y_lens: torch.Tensor) -> torch.Tensor:
    """Flatten padded targets to 1D concatenation for CTCLoss.
    
    CTCLoss expects targets as 1D concatenation of per-example sequences.
    This is a common operation needed across training, calibration, and evaluation.
    
    Args:
        y: Padded targets [B, L_max]
        y_lens: True target lengths [B]
        
    Returns:
        1D concatenated targets with length == sum(y_lens)
    """
    B, L_max = y.size()
    # Ensure y_lens is on the same device as y
    y_lens = y_lens.to(y.device)
    mask = torch.arange(L_max, device=y.device).unsqueeze(0) < y_lens.unsqueeze(1)
    return y[mask]


