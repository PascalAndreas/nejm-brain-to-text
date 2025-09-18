"""Building blocks for neural encoder models.

This module provides reusable components that can be composed to create
various neural encoder architectures.
"""

from .prenet import DayAdapter, PreNet
from .smoothers import (
    SmootherBase,
    GaussianSmoother,
    DepthwiseSmoother,
    EMASmoother,
    IdentitySmoother,
    build_smoother,
    SMOOTHERS
)
from .rnn import GRUBackbone, LSTMBackbone
from .head import ProjectionHead, AuxiliaryHead, MultiHeadProjection, CTCHead
from .utils import (
    mask_logits_,
    force_blank_on_pad_,
    create_padding_mask,
    apply_patch_embedding,
    get_activation_fn,
    flatten_ctc_targets
)

__all__ = [
    # PreNet components
    "DayAdapter", 
    "PreNet",
    
    # Smoothers
    "SmootherBase",
    "GaussianSmoother",
    "DepthwiseSmoother",
    "EMASmoother",
    "IdentitySmoother",
    "build_smoother",
    "SMOOTHERS",
    
    # RNN backbones
    "GRUBackbone",
    "LSTMBackbone",
    
    # Projection heads
    "ProjectionHead",
    "AuxiliaryHead",
    "MultiHeadProjection",
    "CTCHead",
    
    
    # Utilities
    "mask_logits_",
    "force_blank_on_pad_",
    "create_padding_mask",
    "apply_patch_embedding",
    "get_activation_fn",
    "flatten_ctc_targets"
]
