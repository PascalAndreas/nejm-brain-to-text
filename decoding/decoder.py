"""
Optimized CTC decoder with separated initialization and parameter updates.
This decoder builds the trie and loads the language model once, then allows
fast parameter updates for hyperparameter tuning.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import os
from pathlib import Path
from omegaconf import OmegaConf

# Handle relative imports for both package and direct execution
from decoding.helpers import load_tokens, load_lexicon_entries
try:
    from flashlight.lib.text.decoder import (
        LexiconDecoder, LexiconDecoderOptions, SmearingMode, CriterionType,
        Trie, ZeroLM, KenLM
    )
    from flashlight.lib.text.dictionary import Dictionary, create_word_dict
    FLASHLIGHT_AVAILABLE = True
except ImportError as e:
    FLASHLIGHT_AVAILABLE = False
    print(f"Warning: flashlight-text not available: {e}")


class OptimizedCTCDecoder:
    """
    Optimized CTC decoder that separates expensive initialization from parameter updates.
    
    This decoder:
    1. Builds trie and loads LM once during initialization
    2. Allows fast parameter updates without rebuilding expensive components
    3. Provides significant speedup for hyperparameter tuning
    """
    
    def __init__(
        self,
        tokens_path: str = 'decoding/artifacts/tokens.txt',
        lexicon_path: str = 'decoding/artifacts/lexicon.csv',
        lm_path: str = 'language_models/4gram-pruned-0_2_7_9-en-lm-set-1.0.bin',
        blank_token: str = "BLANK",
        silence_token: str = "SIL",
        unk_word: str = "<unk>",
        smearing_mode: str = "max",
        log_add: bool = False,
        nbest: int = 50,
        device: Optional[torch.device] = None,
        verbose: bool = False
    ):
        """
        Initialize the decoder with expensive one-time operations.
        
        Args:
            tokens_path: Path to tokens file
            lexicon_path: Path to lexicon file  
            lm_path: Path to language model
            blank_token: CTC blank token
            silence_token: Silence token
            unk_word: Unknown word token
            smearing_mode: Trie smearing mode
            log_add: Use log addition
            nbest: Number of n-best hypotheses
            device: PyTorch device
            verbose: Verbose output
        """
        if not FLASHLIGHT_AVAILABLE:
            raise ImportError("Flashlight is not available. Please install flashlight-text.")
        
        self.verbose = verbose
        self.tokens_path = tokens_path
        self.lexicon_path = lexicon_path
        self.lm_path = lm_path
        self.blank_token = blank_token
        self.silence_token = silence_token
        self.unk_word = unk_word
        self.smearing_mode = smearing_mode
        self.log_add = log_add
        self.nbest = nbest
        self.device = device or torch.device('cpu')
        
        if self.verbose:
            print("🔧 Initializing OptimizedCTCDecoder (one-time setup)...")
        
        # Expensive one-time operations
        self._load_tokens()
        self._load_lexicon()
        self._build_word_dict()
        self._load_language_model()
        self._build_trie()
        
        # Current decoder parameters (will be updated frequently)
        self.current_params = None
        self.decoder_instance = None
        
        if self.verbose:
            print("✅ OptimizedCTCDecoder initialized successfully!")
            print(f"   Tokens: {len(self.tokens)}")
            print(f"   Lexicon entries: {len(self.lexicon_entries)}")
            print(f"   Words in dictionary: {self.word_dict.index_size()}")
    
    def _load_tokens(self):
        """Load tokens and validate special tokens."""
        self.tokens = load_tokens(self.tokens_path)
        self.token_to_idx = {token: idx for idx, token in enumerate(self.tokens)}
        
        # Find special token indices
        self.blank_idx = self.token_to_idx.get(self.blank_token, 0)
        self.silence_idx = self.token_to_idx.get(self.silence_token, -1)
        
        # Validate blank token exists
        if self.blank_token not in self.token_to_idx:
            raise ValueError(f"Blank token '{self.blank_token}' not found in tokens")
        
        # If silence token is specified but not found, set index to -1
        if self.silence_token and self.silence_token not in self.token_to_idx:
            if self.verbose:
                print(f"Warning: Silence token '{self.silence_token}' not found in tokens")
            self.silence_idx = -1
    
    def _load_lexicon(self):
        """Load lexicon entries preserving individual pronunciation probabilities."""
        self.lexicon_entries = load_lexicon_entries(self.lexicon_path, verbose=self.verbose)
        
        # Build flashlight lexicon format: word -> list of phoneme sequences
        self.flashlight_lexicon = {}
        # Build pronunciation-specific probability mapping: (word, phoneme_seq) -> log_prob
        self.pronunciation_probs = {}
        
        for word, phonemes, log_prob in self.lexicon_entries:
            phoneme_tokens = phonemes.split()
            
            if word not in self.flashlight_lexicon:
                self.flashlight_lexicon[word] = []
            self.flashlight_lexicon[word].append(phoneme_tokens)
            
            # Store pronunciation-specific probability
            self.pronunciation_probs[(word, phonemes)] = log_prob
    
    def _build_word_dict(self):
        """Build word dictionary (expensive, done once)."""
        self.word_dict = create_word_dict(self.flashlight_lexicon)
        if self.verbose:
            print(f"   Built word dictionary with {self.word_dict.index_size()} words")
    
    def _load_language_model(self):
        """Load language model (expensive, done once)."""
        if self.lm_path and os.path.exists(self.lm_path):
            try:
                self.lm = KenLM(self.lm_path, self.word_dict)
                if self.verbose:
                    print(f"   Loaded KenLM from: {self.lm_path}")
            except Exception as e:
                if self.verbose:
                    print(f"   Failed to load KenLM: {e}, using ZeroLM")
                self.lm = ZeroLM()
        else:
            if self.verbose:
                print("   No language model specified, using ZeroLM")
            self.lm = ZeroLM()
    
    def _build_trie(self):
        """Build trie from lexicon (expensive, done once)."""
        # Load tokens dictionary
        self.token_dict = Dictionary()
        for token in self.tokens:
            self.token_dict.add_entry(token)
        
        # Build trie from lexicon with pronunciation-specific probabilities
        self.trie = Trie(self.token_dict.index_size(), self.word_dict.index_size())
        
        trie_insertions = 0
        failed_insertions = 0
        
        for word, phonemes, log_prob in self.lexicon_entries:
            word_idx = self.word_dict.get_index(word)
            if word_idx == -1:
                failed_insertions += 1
                continue
            
            phoneme_tokens = phonemes.split()
            
            # Convert phoneme tokens to token indices
            spelling_indices = []
            for token in phoneme_tokens:
                token_idx = self.token_dict.get_index(token)
                if token_idx != -1:  # Valid token
                    spelling_indices.append(token_idx)
                else:
                    if self.verbose:
                        print(f"   Unknown token '{token}' in word '{word}' pronunciation '{phonemes}'")
            
            if spelling_indices:  # Only add if we have valid tokens
                # Use pronunciation-specific log probability as word score
                self.trie.insert(spelling_indices, word_idx, log_prob)
                trie_insertions += 1
        
        # Apply trie smearing for better lexicon lookahead
        smearing_mode_map = {
            'max': SmearingMode.MAX,
            'logadd': SmearingMode.LOGADD,
            'none': SmearingMode.NONE
        }
        smearing_mode = smearing_mode_map.get(self.smearing_mode.lower(), SmearingMode.MAX)
        self.trie.smear(smearing_mode)
        
        if self.verbose:
            print(f"   Built trie with {trie_insertions} insertions, {failed_insertions} failed")
    
    def update_params(
        self,
        lm_weight: float = 2.5,
        word_score: float = -0.2,
        sil_score: float = 0.0,
        beam_size: int = 150,
        beam_size_token: int = 10,
        beam_threshold: float = 20.0
    ):
        """
        Update decoder parameters and rebuild decoder (fast operation).
        
        Args:
            lm_weight: Language model weight
            word_score: Word insertion penalty/bonus
            sil_score: Silence token score
            beam_size: Beam size for decoding
            beam_size_token: Token-level beam size
            beam_threshold: Beam pruning threshold
        """
        new_params = {
            'lm_weight': lm_weight,
            'word_score': word_score,
            'sil_score': sil_score,
            'beam_size': beam_size,
            'beam_size_token': beam_size_token,
            'beam_threshold': beam_threshold
        }
        
        # Only rebuild decoder if parameters changed
        if self.current_params != new_params:
            self.current_params = new_params
            self._build_decoder()
    
    def _build_decoder(self):
        """Build the final Flashlight decoder with current parameters (fast operation)."""
        # Set up decoder options
        decoder_opts = LexiconDecoderOptions(
            beam_size=self.current_params['beam_size'],
            beam_size_token=self.current_params['beam_size_token'],
            beam_threshold=self.current_params['beam_threshold'],
            lm_weight=self.current_params['lm_weight'],
            word_score=self.current_params['word_score'],
            unk_score=-float('inf'),
            sil_score=self.current_params['sil_score'],
            log_add=self.log_add,
            criterion_type=CriterionType.CTC
        )
        
        # Get unk token index
        unk_idx = self.word_dict.get_index(self.unk_word)
        if unk_idx == -1:
            unk_idx = self.word_dict.get_index("<unk>")  # fallback
            if unk_idx == -1:
                unk_idx = 0  # fallback to first word
        
        # Create decoder instance
        self.decoder_instance = LexiconDecoder(
            decoder_opts,
            self.trie,
            self.lm,
            self.silence_idx,
            self.blank_idx,
            unk_idx,
            [],  # transitions (empty for CTC)
            False  # is_token_lm
        )
    
    def decode(
        self, 
        logits: torch.Tensor,
        logit_lengths: Optional[torch.Tensor] = None
    ) -> List[Dict[str, Any]]:
        """
        Decode CTC logits to text using beam search.
        
        Args:
            logits: CTC logits tensor [batch, time, vocab]
            logit_lengths: Lengths of each sequence in batch [batch]
            
        Returns:
            List of decoding results, one per batch item
        """
        if self.decoder_instance is None:
            raise RuntimeError("Decoder not initialized. Call update_params() first.")
        
        # Convert to numpy and ensure correct format
        if isinstance(logits, torch.Tensor):
            logits_np = logits.detach().cpu().numpy()
        else:
            logits_np = logits
        
        # Ensure float32 format
        if logits_np.dtype != np.float32:
            logits_np = logits_np.astype(np.float32)
        
        batch_size, seq_len, vocab_size = logits_np.shape
        results = []
        
        for i in range(batch_size):
            # Get sequence length
            if logit_lengths is not None:
                seq_length = int(logit_lengths[i].item())
            else:
                seq_length = seq_len
            
            # Extract single sequence
            single_logits = logits_np[i, :seq_length, :]  # [time, vocab]
            
            # Decode using Flashlight
            try:
                # Flashlight expects LOG PROBABILITIES in [T, N] format where N is vocab size
                log_probs = single_logits.astype(np.float32)  # [time, vocab]
                
                # Decode with Flashlight - use raw data pointer as integer
                T, N = log_probs.shape
                emissions_ptr = log_probs.ctypes.data  # Raw integer pointer
                decoded_results = self.decoder_instance.decode(emissions_ptr, T, N)
                
                if decoded_results:
                    best_result = decoded_results[0]  # Get best hypothesis
                    
                    # Map word IDs to actual words with bounds checking
                    word_ids = best_result.words
                    token_ids = best_result.tokens
                    
                    words = []
                    for word_id in word_ids:
                        try:
                            if 0 <= word_id < self.word_dict.index_size():
                                word = self.word_dict.get_entry(word_id)
                                words.append(word)
                            else:
                                # Invalid word ID, skip it
                                continue
                        except:
                            # Fallback if word mapping fails, skip invalid entries
                            continue
                    
                    # Map token IDs to phonemes with proper bounds checking
                    tokens = []
                    for token_id in token_ids:
                        if 0 <= token_id < len(self.tokens):
                            tokens.append(self.tokens[token_id])
                    
                    # Create sentence
                    sentence = ' '.join(words).strip()
                    
                    # Get n-best if available
                    nbest_list = []
                    for j, result in enumerate(decoded_results[:min(self.nbest, len(decoded_results))]):
                        result_words = []
                        result_tokens = []
                        
                        # Map words with proper bounds checking
                        for word_id in result.words:
                            try:
                                if 0 <= word_id < self.word_dict.index_size():
                                    word = self.word_dict.get_entry(word_id)
                                    result_words.append(word)
                            except:
                                continue
                        
                        # Map tokens with proper bounds checking
                        for token_id in result.tokens:
                            if 0 <= token_id < len(self.tokens):
                                result_tokens.append(self.tokens[token_id])
                        
                        nbest_sentence = ' '.join(result_words).strip()
                        nbest_list.append({
                            'sentence': nbest_sentence,
                            'score': float(result.score),
                            'words': result_words,
                            'tokens': result_tokens
                        })
                    
                    result_dict = {
                        'words': words,
                        'tokens': tokens,
                        'score': float(best_result.score),
                        'sentence': sentence,
                        'nbest': nbest_list
                    }
                else:
                    # Empty result
                    result_dict = {
                        'words': [],
                        'tokens': [],
                        'score': -float('inf'),
                        'sentence': '',
                        'nbest': []
                    }
                    
            except Exception as e:
                # Fallback for failed decoding
                result_dict = {
                    'words': [],
                    'tokens': [],
                    'score': -float('inf'),
                    'sentence': '',
                    'nbest': [],
                    'error': str(e)
                }
            
            results.append(result_dict)
        
        return results
    
    def get_current_params(self) -> Dict[str, Any]:
        """Get current decoder parameters."""
        return self.current_params.copy() if self.current_params else {}
    
    def __repr__(self) -> str:
        """String representation of decoder."""
        params_str = str(self.current_params) if self.current_params else "Not configured"
        return f"OptimizedCTCDecoder(params={params_str})"


# Factory function for backward compatibility
def create_decoder(
    tokens_path: str = 'decoding/artifacts/tokens.txt',
    lexicon_path: str = 'decoding/artifacts/lexicon.csv',
    lm_path: str = 'language_models/4gram-pruned-0_2_7_9-en-lm-set-1.0.bin',
    **kwargs
) -> OptimizedCTCDecoder:
    """
    Factory function to create an optimized decoder instance.
    
    Args:
        tokens_path: Path to tokens file
        lexicon_path: Path to lexicon file
        lm_path: Path to language model
        **kwargs: Additional arguments for decoder
        
    Returns:
        OptimizedCTCDecoder instance
    """
    return OptimizedCTCDecoder(
        tokens_path=tokens_path,
        lexicon_path=lexicon_path,
        lm_path=lm_path,
        **kwargs
    )