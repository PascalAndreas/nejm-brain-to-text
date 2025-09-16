"""Building blocks for neural encoder models.

This module provides reusable components that can be composed to create
various neural encoder architectures.
"""

from .prenet import DayAdapter, PreNet
from .smoothers import (
    SmootherBase,
    GaussianSmoother,
    DepthwiseCausalSmoother,
    EMASmoother,
    IdentitySmoother,
    build_smoother,
    SMOOTHERS
)
from .rnn import GRUBackbone, LSTMBackbone, VariationalGRU
from .head import ProjectionHead, AuxiliaryHead, MultiHeadProjection, CTCHead
from .calibrator import TemperatureScaling, PlattScaling, EnsembleTemperature
from .utils import (
    compute_output_lengths,
    mask_logits_,
    create_padding_mask,
    apply_patch_embedding,
    get_activation_fn,
    calculate_conv_output_length,
    create_causal_mask
)

__all__ = [
    # PreNet components
    "DayAdapter", 
    "PreNet",
    
    # Smoothers
    "SmootherBase",
    "GaussianSmoother",
    "DepthwiseCausalSmoother",
    "EMASmoother",
    "IdentitySmoother",
    "build_smoother",
    "SMOOTHERS",
    
    # RNN backbones
    "GRUBackbone",
    "LSTMBackbone",
    "VariationalGRU",
    
    # Projection heads
    "ProjectionHead",
    "AuxiliaryHead",
    "MultiHeadProjection",
    "CTCHead",
    
    # Calibration
    "TemperatureScaling",
    "PlattScaling",
    "EnsembleTemperature",
    
    # Utilities
    "compute_output_lengths",
    "mask_logits_",
    "create_padding_mask",
    "apply_patch_embedding",
    "get_activation_fn",
    "calculate_conv_output_length",
    "create_causal_mask"
]
