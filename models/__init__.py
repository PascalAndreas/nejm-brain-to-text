"""Neural encoder models for brain-to-text decoding.

This module provides modular neural encoder architectures that convert
neural implant data to phoneme predictions using CTC loss.
"""

from typing import Dict, Optional
from .base import Batch, Emissions, NeuralEncoder
from .gru_ctc import GRUCTC
from .lightning_module import BrainToTextLightningModule, EMA
from .schedulers import (
    build_lr_scheduler,
    build_aux_scheduler,
    AuxiliaryLossScheduler,
    WarmupCosineScheduler
)

# Model registry for easy model selection
ENCODERS: Dict[str, type] = {
    "gru_v1": GRUCTC,
    # Future: "conformer_v1": ConformerCTC,
}


def build_encoder(name: str, config: Dict) -> NeuralEncoder:
    """Factory function to build encoder models.
    
    Args:
        name: Model identifier from the registry
        config: Model configuration dictionary
        
    Returns:
        Instantiated NeuralEncoder model
        
    Raises:
        ValueError: If model name not found in registry
    """
    if name not in ENCODERS:
        raise ValueError(f"Unknown encoder: {name}. Available: {list(ENCODERS.keys())}")
    
    encoder_class = ENCODERS[name]
    return encoder_class(**config)


__all__ = [
    "Batch",
    "Emissions", 
    "NeuralEncoder",
    "GRUCTC",
    "BrainToTextLightningModule",
    "EMA",
    "build_encoder",
    "build_lr_scheduler",
    "build_aux_scheduler",
    "AuxiliaryLossScheduler",
    "WarmupCosineScheduler",
    "ENCODERS"
]
