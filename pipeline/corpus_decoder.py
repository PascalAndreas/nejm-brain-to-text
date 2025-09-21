"""Corpus-aware decoder that uses different decoding profiles for different corpus types.

This module provides a decoder wrapper that automatically selects the appropriate
decoding hyperparameters based on the corpus type of the input data.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import torch
import logging

# Handle imports for both package and direct execution
try:
    from .bayesian_tuning import DecodingProfile
    from ..decoding.decoder import Decoder
except ImportError:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from pipeline.bayesian_tuning import DecodingProfile
    from decoding.decoder import Decoder


class CorpusAwareDecoder:
    """Decoder that uses corpus-specific hyperparameter profiles.
    
    This decoder wrapper automatically selects the appropriate decoding profile
    based on the corpus type of the input data, falling back to a global profile
    when corpus-specific profiles are not available.
    """
    
    def __init__(
        self,
        profiles: Union[Dict[str, DecodingProfile], str],
        decoder_type: str = 'flashlight',
        fallback_to_global: bool = True
    ):
        """Initialize corpus-aware decoder.
        
        Args:
            profiles: Dictionary of corpus profiles or path to profiles JSON file
            decoder_type: Type of decoder backend ('flashlight' or 'greedy')
            fallback_to_global: Whether to fall back to global profile for unknown corpora
        """
        self.decoder_type = decoder_type
        self.fallback_to_global = fallback_to_global
        
        # Load profiles
        if isinstance(profiles, str):
            self.profiles = self._load_profiles_from_file(profiles)
        else:
            self.profiles = profiles
        
        # Cache decoder instances to avoid repeated initialization
        self._decoder_cache: Dict[str, Decoder] = {}
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initialized CorpusAwareDecoder with {len(self.profiles)} profiles")
        for name, profile in self.profiles.items():
            if profile.performance:
                wer = profile.performance.get('wer', 'N/A')
                self.logger.info(f"  {name}: WER={wer}")
            else:
                self.logger.info(f"  {name}: No performance data")
    
    def _load_profiles_from_file(self, filepath: str) -> Dict[str, DecodingProfile]:
        """Load profiles from JSON file.
        
        Args:
            filepath: Path to profiles JSON file
            
        Returns:
            Dictionary of loaded profiles
        """
        with open(filepath, 'r') as f:
            profiles_dict = json.load(f)
        
        profiles = {}
        for name, profile_data in profiles_dict.items():
            profiles[name] = DecodingProfile(**profile_data)
        
        return profiles
    
    def _get_decoder_for_profile(self, profile: DecodingProfile) -> Decoder:
        """Get or create decoder instance for a profile.
        
        Args:
            profile: Decoding profile
            
        Returns:
            Decoder instance configured with profile parameters
        """
        # Use profile name as cache key
        cache_key = profile.name
        
        if cache_key not in self._decoder_cache:
            # Create new decoder with profile parameters
            decoder_kwargs = profile.to_decoder_kwargs()
            decoder_kwargs['backend'] = self.decoder_type
            
            self._decoder_cache[cache_key] = Decoder(**decoder_kwargs)
            self.logger.debug(f"Created decoder for profile '{profile.name}'")
        
        return self._decoder_cache[cache_key]
    
    def _get_profile_for_corpus(self, corpus: str) -> DecodingProfile:
        """Get the appropriate profile for a corpus.
        
        Args:
            corpus: Corpus name
            
        Returns:
            Best decoding profile for this corpus
            
        Raises:
            ValueError: If no suitable profile is found
        """
        # Try exact match first
        if corpus in self.profiles:
            return self.profiles[corpus]
        
        # Fall back to global profile if enabled
        if self.fallback_to_global and 'global' in self.profiles:
            self.logger.debug(f"Using global profile for corpus '{corpus}'")
            return self.profiles['global']
        
        # No suitable profile found
        available_profiles = list(self.profiles.keys())
        raise ValueError(
            f"No profile found for corpus '{corpus}' and fallback disabled. "
            f"Available profiles: {available_profiles}"
        )
    
    def decode_batch(
        self,
        logits: torch.Tensor,
        logit_lengths: torch.Tensor,
        corpus_info: List[str]
    ) -> List[Dict[str, Any]]:
        """Decode a batch using corpus-specific profiles.
        
        Args:
            logits: CTC logits tensor [batch, time, vocab]
            logit_lengths: Lengths of each sequence [batch]
            corpus_info: List of corpus names for each item in batch
            
        Returns:
            List of decoding results, one per batch item
        """
        batch_size = logits.shape[0]
        
        if len(corpus_info) != batch_size:
            raise ValueError(f"Corpus info length ({len(corpus_info)}) doesn't match batch size ({batch_size})")
        
        # Group batch items by corpus to minimize decoder switching
        corpus_groups: Dict[str, List[int]] = {}
        for i, corpus in enumerate(corpus_info):
            if corpus not in corpus_groups:
                corpus_groups[corpus] = []
            corpus_groups[corpus].append(i)
        
        # Initialize results list
        results = [None] * batch_size
        
        # Process each corpus group
        for corpus, indices in corpus_groups.items():
            try:
                # Get appropriate profile and decoder
                profile = self._get_profile_for_corpus(corpus)
                decoder = self._get_decoder_for_profile(profile)
                
                # Extract logits and lengths for this corpus
                corpus_logits = logits[indices]  # [group_size, time, vocab]
                corpus_lengths = logit_lengths[indices]  # [group_size]
                
                # Decode this group
                corpus_results = decoder.decode(corpus_logits, corpus_lengths)
                
                # Store results in correct positions
                for i, result in enumerate(corpus_results):
                    original_idx = indices[i]
                    results[original_idx] = result
                
                self.logger.debug(f"Decoded {len(indices)} items with {corpus} profile")
                
            except Exception as e:
                self.logger.error(f"Error decoding corpus '{corpus}': {e}")
                
                # Fill with empty results for failed decodes
                for idx in indices:
                    results[idx] = {
                        'words': [],
                        'tokens': [],
                        'score': float('-inf'),
                        'nbest': [],
                        'sentence': '',
                        'error': str(e)
                    }
        
        return results
    
    def decode_single(
        self,
        logits: torch.Tensor,
        logit_lengths: Optional[torch.Tensor] = None,
        corpus: str = 'Unknown'
    ) -> Dict[str, Any]:
        """Decode a single utterance with corpus-specific profile.
        
        Args:
            logits: CTC logits tensor [time, vocab] or [1, time, vocab]
            logit_lengths: Length of sequence (optional)
            corpus: Corpus name for this utterance
            
        Returns:
            Decoding result dictionary
        """
        # Ensure batch dimension
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
        
        if logit_lengths is None:
            logit_lengths = torch.tensor([logits.shape[1]])
        elif logit_lengths.dim() == 0:
            logit_lengths = logit_lengths.unsqueeze(0)
        
        # Use batch decode with single item
        results = self.decode_batch(logits, logit_lengths, [corpus])
        return results[0]
    
    def get_profile_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all loaded profiles.
        
        Returns:
            Dictionary with profile summaries
        """
        summary = {}
        for name, profile in self.profiles.items():
            summary[name] = {
                'hyperparameters': profile.to_decoder_kwargs(),
                'performance': profile.performance
            }
        return summary
    
    def update_profile(self, corpus: str, profile: DecodingProfile):
        """Update or add a profile for a corpus.
        
        Args:
            corpus: Corpus name
            profile: New decoding profile
        """
        self.profiles[corpus] = profile
        
        # Clear cached decoder for this profile
        if profile.name in self._decoder_cache:
            del self._decoder_cache[profile.name]
        
        self.logger.info(f"Updated profile for corpus '{corpus}'")
    
    def save_profiles(self, filepath: str):
        """Save current profiles to file.
        
        Args:
            filepath: Path to save profiles JSON file
        """
        from dataclasses import asdict
        
        profiles_dict = {}
        for name, profile in self.profiles.items():
            profiles_dict[name] = asdict(profile)
        
        with open(filepath, 'w') as f:
            json.dump(profiles_dict, f, indent=2)
        
        self.logger.info(f"Saved {len(profiles_dict)} profiles to {filepath}")


class CorpusAwareEvaluator:
    """Evaluator that can assess corpus-specific decoder performance."""
    
    def __init__(self, corpus_decoder: CorpusAwareDecoder):
        """Initialize evaluator.
        
        Args:
            corpus_decoder: Corpus-aware decoder to evaluate
        """
        self.corpus_decoder = corpus_decoder
        self.logger = logging.getLogger(__name__)
    
    def evaluate_on_emissions(
        self,
        emissions_dict: Dict[str, Dict[str, Any]],
        corpus_breakdown: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate decoder performance on cached emissions.
        
        Args:
            emissions_dict: Dictionary of cached emissions with metadata
            corpus_breakdown: Whether to provide per-corpus breakdown
            
        Returns:
            Dictionary with evaluation results
        """
        from jiwer import wer, cer
        
        # Overall metrics
        total_wer = 0.0
        total_cer = 0.0
        total_words = 0
        total_chars = 0
        successful_decodes = 0
        
        # Per-corpus metrics
        corpus_metrics = {}
        
        for utt_id, emission in emissions_dict.items():
            # Skip if no ground truth
            if 'ground_truth' not in emission.get('meta', {}):
                continue
            
            corpus = emission.get('meta', {}).get('corpus', 'Unknown')
            
            try:
                # Decode with corpus-specific profile
                log_probs = emission['log_probs'].unsqueeze(0)
                out_len = torch.tensor([emission['out_len']])
                
                result = self.corpus_decoder.decode_single(log_probs, out_len, corpus)
                
                pred_sentence = result.get('sentence', '').lower().strip()
                true_sentence = emission['meta']['ground_truth'].lower().strip()
                
                if pred_sentence and true_sentence:
                    # Calculate metrics
                    sample_wer = wer(true_sentence, pred_sentence)
                    sample_cer = cer(true_sentence, pred_sentence)
                    
                    true_words = len(true_sentence.split())
                    true_chars = len(true_sentence)
                    
                    # Update overall metrics
                    total_wer += sample_wer * true_words
                    total_cer += sample_cer * true_chars
                    total_words += true_words
                    total_chars += true_chars
                    successful_decodes += 1
                    
                    # Update corpus-specific metrics
                    if corpus_breakdown:
                        if corpus not in corpus_metrics:
                            corpus_metrics[corpus] = {
                                'total_wer': 0.0, 'total_cer': 0.0,
                                'total_words': 0, 'total_chars': 0,
                                'successful_decodes': 0, 'total_utterances': 0
                            }
                        
                        corpus_metrics[corpus]['total_wer'] += sample_wer * true_words
                        corpus_metrics[corpus]['total_cer'] += sample_cer * true_chars
                        corpus_metrics[corpus]['total_words'] += true_words
                        corpus_metrics[corpus]['total_chars'] += true_chars
                        corpus_metrics[corpus]['successful_decodes'] += 1
                    
            except Exception as e:
                self.logger.warning(f"Failed to decode {utt_id}: {e}")
            
            # Count total utterances per corpus
            if corpus_breakdown:
                if corpus not in corpus_metrics:
                    corpus_metrics[corpus] = {
                        'total_wer': 0.0, 'total_cer': 0.0,
                        'total_words': 0, 'total_chars': 0,
                        'successful_decodes': 0, 'total_utterances': 0
                    }
                corpus_metrics[corpus]['total_utterances'] += 1
        
        # Calculate final metrics
        results = {
            'overall': {
                'wer': total_wer / total_words if total_words > 0 else 1.0,
                'cer': total_cer / total_chars if total_chars > 0 else 1.0,
                'success_rate': successful_decodes / len(emissions_dict),
                'total_utterances': len(emissions_dict),
                'successful_decodes': successful_decodes
            }
        }
        
        # Add corpus-specific results
        if corpus_breakdown:
            for corpus, metrics in corpus_metrics.items():
                if metrics['total_words'] > 0:
                    results[corpus] = {
                        'wer': metrics['total_wer'] / metrics['total_words'],
                        'cer': metrics['total_cer'] / metrics['total_chars'],
                        'success_rate': metrics['successful_decodes'] / metrics['total_utterances'],
                        'total_utterances': metrics['total_utterances'],
                        'successful_decodes': metrics['successful_decodes']
                    }
        
        return results
