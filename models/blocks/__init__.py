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
from .calibrator import TemperatureScaling, PlattScaling, EnsembleTemperature
from .utils import (
    mask_logits_,
    create_padding_mask,
    apply_patch_embedding,
    get_activation_fn
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
    
    # Calibration
    "TemperatureScaling",
    "PlattScaling",
    "EnsembleTemperature",
    
    # Utilities
    "mask_logits_",
    "create_padding_mask",
    "apply_patch_embedding",
    "get_activation_fn"
]
