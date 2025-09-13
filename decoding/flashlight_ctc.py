"""
Flashlight CTC decoder implementation.
Implements lexicon-constrained beam search with shallow fusion using Flashlight.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import os
import ctypes
from pathlib import Path
from omegaconf import OmegaConf

# Handle relative imports for both package and direct execution
from decoding.helpers import load_tokens, load_lexicon_dict, load_lexicon_with_probs
try:
    from flashlight.lib.text.decoder import (
        LexiconDecoder, LexiconDecoderOptions, SmearingMode, CriterionType,
        Trie, ZeroLM, KenLM
    )
    from flashlight.lib.text.dictionary import Dictionary, create_word_dict, load_words
    FLASHLIGHT_AVAILABLE = True
except ImportError as e:
    FLASHLIGHT_AVAILABLE = False
    print(f"Warning: flashlight-text not available: {e}")




class FlashlightCTCDecoder:
    """
    Flashlight CTC decoder with KenLM language model integration.
    Supports lexicon-constrained beam search with shallow fusion.
    """
    
    def __init__(
        self,
        tokens_path: Optional[str] = None,
        lexicon_path: Optional[str] = None,
        lm_path: Optional[str] = None,
        lm_weight: Optional[float] = None,
        word_score: Optional[float] = None,
        beam_size: Optional[int] = None,
        beam_size_token: Optional[int] = None,
        beam_threshold: Optional[float] = None,
        nbest: Optional[int] = None,
        blank_token: Optional[str] = None,
        silence_token: Optional[str] = None,
        unk_word: Optional[str] = None,
        sil_score: Optional[float] = None,
        log_add: Optional[bool] = None,
        smearing_mode: Optional[str] = None,
        device: Optional[torch.device] = None,
        verbose: Optional[bool] = False
    ):
        """
        Initialize Flashlight CTC decoder.
        
        Args:
            tokens_path: Path to tokens.txt file (defaults from config.yaml)
            lexicon_path: Path to lexicon.txt file (defaults from config.yaml)
            lm_path: Path to KenLM language model (.arpa or .bin) (defaults from config.yaml)
            lm_weight: Language model weight (lambda) (defaults from config.yaml)
            word_score: Word insertion bonus/penalty (beta) (defaults from config.yaml)
            beam_size: Beam size for decoding (defaults from config.yaml)
            beam_size_token: Token-level beam size (defaults from config.yaml)
            beam_threshold: Beam pruning threshold (defaults from config.yaml)
            nbest: Number of n-best hypotheses to return (defaults from config.yaml)
            blank_token: CTC blank token symbol (defaults from config.yaml)
            silence_token: Silence token symbol (defaults from config.yaml)
            unk_word: Unknown word symbol (defaults from config.yaml)
            device: PyTorch device
        """
        if not FLASHLIGHT_AVAILABLE:
            raise ImportError("Flashlight is not available. Please install flashlight-text.")
        
        # Initialize in deterministic, testable steps
        self.verbose = verbose
        self._load_config(tokens_path, lexicon_path, lm_path, lm_weight, word_score,
                         beam_size, beam_size_token, beam_threshold, nbest,
                         blank_token, silence_token, unk_word, sil_score, log_add, 
                         smearing_mode, device)
        self._load_tokens()
        self._load_lexicon()
        self._load_or_zero_lm()
        self._build_trie()
        self._build_decoder()
        if verbose:
            print(f"Initialized FlashlightCTCDecoder:")
            print(f"  Tokens: {len(self.tokens)} ({self.tokens_path})")
            print(f"  Lexicon: {self.lexicon_path}")
            print(f"  LM: {self.lm_path}")
            print(f"  Blank token: '{self.blank_token}' (idx={self.blank_idx})")
            print(f"  Silence token: '{self.silence_token}' (idx={self.silence_idx})")
            print(f"  Beam size: {self.beam_size}, LM weight: {self.lm_weight}, Word score: {self.word_score}")
    
    def _load_config(self, tokens_path, lexicon_path, lm_path, lm_weight, word_score,
                     beam_size, beam_size_token, beam_threshold, nbest,
                     blank_token, silence_token, unk_word, sil_score, log_add, 
                     smearing_mode, device):
        """Load configuration and set parameters."""
        # Load configuration
        config_path = Path(__file__).parent / "config.yaml"
        config = OmegaConf.load(config_path)
        
        # Use provided values or fall back to config defaults
        self.tokens_path = tokens_path or config.artifacts.tokens_txt
        self.lexicon_path = lexicon_path or config.artifacts.lexicon_txt
        self.lm_path = lm_path or config.language_model.get('active_model', None)
        
        # Flashlight parameters from flashlight section
        self.lm_weight = lm_weight if lm_weight is not None else config.flashlight.lm_weight
        self.word_score = word_score if word_score is not None else config.flashlight.word_score
        self.beam_size = beam_size if beam_size is not None else config.flashlight.beam_size
        self.beam_size_token = beam_size_token if beam_size_token is not None else config.flashlight.beam_size_token
        self.beam_threshold = beam_threshold if beam_threshold is not None else config.flashlight.beam_threshold
        self.nbest = int(nbest) if nbest is not None else int(config.flashlight.nbest)
        self.sil_score = sil_score if sil_score is not None else config.flashlight.get('sil_score', -1.0)
        self.log_add = log_add if log_add is not None else config.flashlight.get('log_add', False)
        self.smearing_mode = smearing_mode if smearing_mode is not None else config.flashlight.get('smearing_mode', 'max')
        
        # Token definitions from decoder section
        self.blank_token = blank_token or config.decoder.blank_token
        self.silence_token = silence_token or config.decoder.silence_token
        self.unk_word = unk_word or config.decoder.unk_word
        self.device = device or torch.device('cpu')
    
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
            print(f"Warning: Silence token '{self.silence_token}' not found in tokens, setting index to -1")
            self.silence_idx = -1
    
    def _load_lexicon(self):
        """Load lexicon with probabilities for both trie building and word dictionary."""
        self.lexicon_dict = load_lexicon_with_probs(self.lexicon_path, verbose=self.verbose)
        
        # Create word-to-logprob mapping for fast trie construction
        self.word_to_logprob = {}
        for phonemes, (words, log_prob) in self.lexicon_dict.items():
            for word in words:
                # Use the highest log prob if word appears multiple times
                if word not in self.word_to_logprob or log_prob > self.word_to_logprob[word]:
                    self.word_to_logprob[word] = log_prob
    
    def _load_or_zero_lm(self):
        """Load language model or create ZeroLM fallback. Build word dictionary."""
        # Convert lexicon_dict to format expected by flashlight
        flashlight_lexicon = {}
        for phonemes, (words, _) in self.lexicon_dict.items():
            for word in words:
                if word not in flashlight_lexicon:
                    flashlight_lexicon[word] = []
                flashlight_lexicon[word].append(phonemes.split())
        
        self.word_dict = create_word_dict(flashlight_lexicon)
        
        # <unk> should already be in the lexicon at the end, no need to add explicitly
        
        # Try to load actual language model
        if self.lm_path and os.path.exists(self.lm_path):
            try:
                from flashlight.lib.text.decoder import KenLM
                if self.verbose:
                    print(f"Loading KenLM from: {self.lm_path}")
                self.lm = KenLM(self.lm_path, self.word_dict)
                if self.verbose:
                    print("Successfully loaded KenLM language model")
            except Exception as e:
                print(f"Failed to load KenLM: {e}")
                print("Falling back to ZeroLM")
                self.lm = ZeroLM()
        else:
            print("No language model path specified, using ZeroLM")
            self.lm = ZeroLM()
    
    def _build_trie(self):
        """Build trie from lexicon with 1-gram log probabilities as word scores."""
        # Load tokens dictionary
        self.token_dict = Dictionary()
        for token in self.tokens:
            self.token_dict.add_entry(token)
        
        # Convert lexicon_dict to format expected by trie building
        flashlight_lexicon = {}
        for phonemes, (words, _) in self.lexicon_dict.items():
            for word in words:
                if word not in flashlight_lexicon:
                    flashlight_lexicon[word] = []
                flashlight_lexicon[word].append(phonemes.split())
        
        # Build trie from lexicon
        self.trie = Trie(self.token_dict.index_size(), self.word_dict.index_size())
        
        trie_insertions = 0
        failed_insertions = 0
        for word, spellings in flashlight_lexicon.items():
            word_idx = self.word_dict.get_index(word)
            if word_idx == -1:
                failed_insertions += 1
                continue
            
            # Get 1-gram log probability for this word using fast lookup
            word_log_prob = self.word_to_logprob.get(word, 0.0)
                
            for spelling in spellings:
                # Convert spelling (list of tokens) to token indices
                spelling_indices = []
                for token in spelling:
                    token_idx = self.token_dict.get_index(token)
                    if token_idx != -1:  # Valid token
                        spelling_indices.append(token_idx)
                    else:
                        print(f"DEBUG: Unknown token '{token}' in word '{word}' spelling {spelling}")
                
                if spelling_indices:  # Only add if we have valid tokens
                    # Use 1-gram log probability as word score
                    self.trie.insert(spelling_indices, word_idx, word_log_prob)
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
            print(f"Built trie with {trie_insertions} insertions, {failed_insertions} failed")
            print(f"Using 1-gram log probabilities as word scores")
    
    def _build_decoder(self):
        """Build the final Flashlight decoder with all components."""
        # Set up decoder options
        decoder_opts = LexiconDecoderOptions(
            beam_size=self.beam_size,
            beam_size_token=self.beam_size_token,
            beam_threshold=self.beam_threshold,
            lm_weight=self.lm_weight,
            word_score=self.word_score,
            unk_score=-float('inf'),
            sil_score=self.sil_score,
            log_add=self.log_add,
            criterion_type=CriterionType.CTC
        )
        
        # Get unk token index
        unk_token_idx = self.word_dict.get_index(self.unk_word)
        if unk_token_idx == -1:
            # Disable unknown words if not found
            unk_token_idx = -1
        
        # Build decoder
        self.decoder = LexiconDecoder(
            decoder_opts,
            self.trie,
            self.lm,
            self.silence_idx,  # sil_token_idx
            self.blank_idx,    # blank_token_idx  
            unk_token_idx,     # unk_token_idx
            [],                # transitions (empty for CTC)
            False              # is_token_lm
        )
        
        if self.decoder is None:
            raise RuntimeError("Failed to build Flashlight decoder")
        if self.verbose:
            print("Built Flashlight CTC decoder with lexicon")
    
    
    def decode(
        self, 
        logits: torch.Tensor,
        logit_lengths: Optional[torch.Tensor] = None
    ) -> List[Dict[str, Any]]:
        """
        Decode CTC logits to text.
        Expects batched inputs [batch, time, vocab] with logit_lengths [batch].
        
        Args:
            logits: CTC logits tensor [batch, time, vocab] (batching handled by base class)
            logit_lengths: Lengths of each sequence in batch [batch]
            
        Returns:
            List of decoding results, one per batch item.
            Each result contains:
                - words: List of decoded words
                - tokens: List of decoded tokens  
                - score: Decoding score
                - nbest: List of n-best hypotheses (if nbest > 1)
                - sentence: Decoded sentence string
        """
        batch_size, seq_len, vocab_size = logits.shape
        
        if not (FLASHLIGHT_AVAILABLE and hasattr(self, 'word_dict')):
            raise RuntimeError("Flashlight decoder not properly initialized")
        
        results = []
        
        try:
            for i in range(batch_size):
                seq_len = logit_lengths[i].item()
                logit_seq = logits[i, :seq_len].cpu().numpy()  # [time, vocab]
                
                # Flashlight expects LOG PROBABILITIES in [T, N] format where N is vocab size
                # Convert raw logits to log probabilities (log-softmax)
                log_probs = torch.log_softmax(torch.from_numpy(logit_seq), dim=-1).float().contiguous().numpy()
                
                # Decode with Flashlight - use raw data pointer as integer
                T, N = log_probs.shape
                emissions_ptr = log_probs.ctypes.data  # Raw integer pointer
                decoder_results = self.decoder.decode(emissions_ptr, T, N)
                
                if decoder_results:
                    # Get best hypothesis
                    best_result = decoder_results[0]
                    
                    # Map word IDs to actual words
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
                    
                    score = best_result.score
                    sentence = ' '.join(words)
                    
                    # Get n-best if available
                    nbest_results = []
                    nbest = int(self.nbest) if self.nbest is not None else 1
                    for j, result in enumerate(decoder_results[:nbest]):
                        result_words = []
                        result_tokens = []
                        
                        # Map words with proper bounds checking
                        for word_id in result.words:
                            try:
                                if 0 <= word_id < self.word_dict.index_size():
                                    word = self.word_dict.get_entry(word_id)
                                    result_words.append(word)
                                else:
                                    # Invalid word ID, skip it
                                    continue
                            except:
                                # Fallback if word mapping fails, skip invalid entries
                                continue
                        
                        # Map tokens with proper bounds checking
                        for token_id in result.tokens:
                            if 0 <= token_id < len(self.tokens):
                                result_tokens.append(self.tokens[token_id])
                        
                        nbest_results.append({
                            'words': result_words,
                            'tokens': result_tokens,
                            'score': result.score,
                            'rank': j,
                            'sentence': ' '.join(result_words)
                        })
                    
                    results.append({
                        'words': words,
                        'tokens': tokens,
                        'score': score,
                        'nbest': nbest_results,
                        'sentence': sentence
                    })
                else:
                    # Empty result
                    results.append({
                        'words': [],
                        'tokens': [],
                        'score': float('-inf'),
                        'nbest': [],
                        'sentence': ''
                    })
            
            return results
            
        except Exception as e:
            print(f"Flashlight decoder failed: {e}")
            raise RuntimeError(f"Flashlight decoding failed: {e}")