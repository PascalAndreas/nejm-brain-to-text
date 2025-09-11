"""
Pure-Python decoding package for Brain-to-Text CTC decoding.
Replaces the Kaldi/SRILM stack with TorchAudio CTC decoder + KenLM.

Uses NVIDIA TAO speech-to-text language models for optimal speech recognition performance.
"""

import os
from pathlib import Path
from omegaconf import OmegaConf

# Load default configuration
_config_path = Path(__file__).parent / "config.yaml"
DEFAULT_CONFIG = OmegaConf.load(_config_path)

from .build_tokens import build_tokens
from .build_lexicon import build_lexicon, extract_lm_vocabulary_from_lexicon
from .filter_lm import filter_language_model
from .decode_ctc import CTCDecoder

# Note: tune_decoding_params has dependencies on model_training, 
# so it's not imported here to avoid circular imports

__all__ = [
    'DEFAULT_CONFIG',
    'build_tokens',
    'build_lexicon',
    'extract_lm_vocabulary_from_lexicon', 
    'filter_language_model',
    'CTCDecoder'
]
