"""
Decoder router for CTC decoding.
Routes to appropriate decoder implementation based on type and configuration.
"""

from typing import Dict, Any, Optional
import torch
from pathlib import Path
from omegaconf import OmegaConf


def create_decoder(decoder_type: str, **kwargs) -> Any:
    """
    Factory function to create decoder instances.
    
    Args:
        decoder_type: Type of decoder ('flashlight', 'greedy')
        **kwargs: Additional arguments passed to decoder constructor
        
    Returns:
        Decoder instance
        
    Raises:
        ValueError: If decoder_type is not supported
        ImportError: If required dependencies are not available
    """
    decoder_type = decoder_type.lower()
    
    if decoder_type == 'flashlight':
        try:
            from .flashlight import FlashlightCTCDecoder
            return FlashlightCTCDecoder(**kwargs)
        except ImportError as e:
            if "flashlight" in str(e).lower():
                print("Flashlight not available, falling back to greedy decoder")
                from .greedy import GreedyCTCDecoder
                return GreedyCTCDecoder(**kwargs)
            else:
                raise
    elif decoder_type == 'greedy':
        from .greedy import GreedyCTCDecoder
        return GreedyCTCDecoder(**kwargs)
    else:
        raise ValueError(f"Unsupported decoder type: {decoder_type}")


class Decoder:
    """
    Main decoder router class that handles decoder selection and initialization.
    """
    
    def __init__(self, config_path: Optional[str] = None, **kwargs):
        """
        Initialize decoder router.
        
        Args:
            config_path: Path to configuration file (defaults to config.yaml in same directory)
            **kwargs: Direct decoder parameters or configuration overrides
        """
        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        self.config = OmegaConf.load(config_path)
        
        # Store direct parameters for decoder initialization
        self.direct_params = kwargs
        
        # Determine decoder type
        self.decoder_type = kwargs.get('backend') or self.config.decoder.get('backend', 'auto')
        if self.decoder_type == 'auto':
            self.decoder_type = self._auto_select_decoder()
        
        # Create decoder instance
        self.decoder_instance = self._create_decoder_instance()
    
    def _auto_select_decoder(self) -> str:
        """
        Automatically select the best available decoder.
        
        Returns:
            Selected decoder type
        """
        try:
            from flashlight.lib.text.decoder import LexiconDecoder
            return 'flashlight'
        except ImportError:
            print("Flashlight not available, falling back to greedy decoder")
            return 'greedy'
    
    def _create_decoder_instance(self):
        """Create the appropriate decoder instance based on configuration."""
        # Start with config defaults
        decoder_kwargs = {}
        
        # Common parameters from config
        decoder_kwargs.update({
            'tokens_path': self.config.artifacts.get('tokens_txt'),
            'lexicon_path': self.config.artifacts.get('lexicon_txt'),
            'lm_path': self.config.language_model.get('active_model'),
            'blank_token': self.config.decoder.get('blank_token', 'BLANK'),
            'silence_token': self.config.decoder.get('silence_token', 'SIL'),
            'unk_word': self.config.decoder.get('unk_word', '<unk>'),
        })
        
        # Add decoder-specific parameters from config
        if self.decoder_type == 'flashlight':
            if 'flashlight' in self.config:
                decoder_kwargs.update({
                    'lm_weight': self.config.flashlight.get('lm_weight'),
                    'word_score': self.config.flashlight.get('word_score'),
                    'beam_size': self.config.flashlight.get('beam_size'),
                    'beam_size_token': self.config.flashlight.get('beam_size_token'),
                    'beam_threshold': self.config.flashlight.get('beam_threshold'),
                    'nbest': self.config.flashlight.get('nbest'),
                })
            else:
                # Fallback to decoder section for backward compatibility
                decoder_kwargs.update({
                    'lm_weight': self.config.decoder.get('lm_weight'),
                    'word_score': self.config.decoder.get('word_score'),
                    'beam_size': self.config.decoder.get('beam_size'),
                    'beam_size_token': self.config.decoder.get('beam_size_token'),
                    'beam_threshold': self.config.decoder.get('beam_threshold'),
                    'nbest': self.config.decoder.get('nbest'),
                })
        
        # Override with any direct parameters passed to constructor
        decoder_kwargs.update(self.direct_params)
        
        # Remove 'backend' if it was passed as a direct parameter
        decoder_kwargs.pop('backend', None)
        
        # Filter out None values
        decoder_kwargs = {k: v for k, v in decoder_kwargs.items() if v is not None}
        
        return create_decoder(self.decoder_type, **decoder_kwargs)
    
    def decode(self, logits: torch.Tensor, logit_lengths: Optional[torch.Tensor] = None):
        """
        Decode CTC logits to text with batching handled at this level.
        
        Args:
            logits: CTC logits tensor [batch, time, vocab] or [time, vocab]
            logit_lengths: Lengths of each sequence in batch [batch]
            
        Returns:
            Decoding results from the underlying decoder
        """
        # Handle single sequence case
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)  # Add batch dimension
            single_sequence = True
        else:
            single_sequence = False
        
        batch_size, seq_len, vocab_size = logits.shape
        
        # Also treat batch_size=1 as single sequence for convenience
        if batch_size == 1:
            single_sequence = True
        
        # Default lengths to full sequences if not provided
        if logit_lengths is None:
            logit_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
        
        # Move to CPU for decoding
        logits = logits.cpu()
        logit_lengths = logit_lengths.cpu()
        
        # Call decoder instance with normalized inputs
        results = self.decoder_instance.decode(logits, logit_lengths)
        
        # Return single result if input was single sequence
        if single_sequence and isinstance(results, list):
            return results[0] if results else {
                'words': [], 'tokens': [], 'score': float('-inf'), 
                'nbest': [], 'sentence': ''
            }
        
        return results
    
    @property
    def tokens(self):
        """Access to decoder tokens."""
        return getattr(self.decoder_instance, 'tokens', [])
    
    @property
    def token_to_idx(self):
        """Access to token-to-index mapping."""
        return getattr(self.decoder_instance, 'token_to_idx', {})
    
    @property
    def blank_idx(self):
        """Access to blank token index."""
        return getattr(self.decoder_instance, 'blank_idx', 0)
