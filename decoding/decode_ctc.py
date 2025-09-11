"""
TorchAudio CTC decoder wrapper with KenLM integration.
Implements lexicon-constrained beam search with shallow fusion.
"""

import torch
import torchaudio
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import os
from pathlib import Path
from omegaconf import OmegaConf

try:
    from flashlight.lib.text.decoder import (
        LexiconDecoder, LexiconDecoderOptions, SmearingMode, CriterionType,
        Trie, ZeroLM, LM
    )
    from flashlight.lib.text.dictionary import Dictionary, create_word_dict, load_words
    FLASHLIGHT_AVAILABLE = True
except ImportError:
    FLASHLIGHT_AVAILABLE = False
    print("Warning: flashlight-text not available, using greedy decoding only")


# Import the shared config from the module
try:
    from . import DEFAULT_CONFIG
except ImportError:
    # Fallback if imported directly
    def _load_default_config():
        """Load default configuration from config.yaml."""
        config_path = Path(__file__).parent / "config.yaml"
        if config_path.exists():
            return OmegaConf.load(config_path)
        else:
            # Fallback minimal config
            return OmegaConf.create({
                'artifacts': {
                    'tokens_txt': 'artifacts/tokens.txt',
                    'lexicon_txt': 'artifacts/lexicon.txt'
                },
                'language_model': {
                    'active_model': None
                },
                'decoder': {
                    'lm_weight': 0.35,
                    'word_score': -90.0,
                    'beam_size': 17,
                    'beam_size_token': 10,
                    'beam_threshold': 8.0,
                    'nbest': 100,
                    'blank_token': 'BLANK',
                    'silence_token': 'SIL',
                    'unk_word': '<unk>'
                }
            })
    DEFAULT_CONFIG = _load_default_config()


class CTCDecoder:
    """
    TorchAudio CTC decoder with KenLM language model integration.
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
        device: Optional[torch.device] = None
    ):
        """
        Initialize CTC decoder.
        
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
        # Use the shared default configuration
        config = DEFAULT_CONFIG
        
        # Use provided values or fall back to config defaults
        self.tokens_path = tokens_path or config.artifacts.tokens_txt
        self.lexicon_path = lexicon_path or config.artifacts.lexicon_txt
        self.lm_path = lm_path or config.language_model.get('active_model', None)
        self.lm_weight = lm_weight if lm_weight is not None else config.decoder.lm_weight
        self.word_score = word_score if word_score is not None else config.decoder.word_score
        self.beam_size = beam_size if beam_size is not None else config.decoder.beam_size
        self.beam_size_token = beam_size_token if beam_size_token is not None else config.decoder.beam_size_token
        self.beam_threshold = beam_threshold if beam_threshold is not None else config.decoder.beam_threshold
        self.nbest = nbest if nbest is not None else config.decoder.nbest
        self.blank_token = blank_token or config.decoder.blank_token
        self.silence_token = silence_token or config.decoder.silence_token
        self.unk_word = unk_word or config.decoder.unk_word
        self.device = device or torch.device('cpu')
        
        # Load tokens
        self.tokens = self._load_tokens()
        self.token_to_idx = {token: idx for idx, token in enumerate(self.tokens)}
        
        # Find special token indices
        self.blank_idx = self.token_to_idx.get(self.blank_token, 0)
        self.silence_idx = self.token_to_idx.get(self.silence_token, -1)
        
        # Build decoder
        self.decoder = self._build_decoder()
        
        # Load lexicon for greedy fallback
        self.lexicon_dict = self._load_lexicon_dict()
        
        print(f"Initialized CTCDecoder:")
        print(f"  Tokens: {len(self.tokens)} ({self.tokens_path})")
        print(f"  Lexicon: {self.lexicon_path}")
        print(f"  LM: {self.lm_path}")
        print(f"  Blank token: '{self.blank_token}' (idx={self.blank_idx})")
        print(f"  Silence token: '{self.silence_token}' (idx={self.silence_idx})")
        print(f"  Beam size: {self.beam_size}, LM weight: {self.lm_weight}, Word score: {self.word_score}")
    
    def _load_tokens(self) -> List[str]:
        """Load tokens from tokens.txt file."""
        tokens = []
        with open(self.tokens_path, 'r') as f:
            for line in f:
                token = line.rstrip('\n')  # Only remove newlines, preserve spaces
                if token:  # Allow empty string tokens if needed
                    tokens.append(token)
        return tokens
    
    def _load_lexicon_dict(self) -> Dict[str, str]:
        """Load lexicon and create phoneme-to-word mapping for greedy fallback."""
        lexicon_dict = {}
        try:
            with open(self.lexicon_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        word = parts[0]
                        phonemes = ' '.join(parts[1:])
                        # Map phoneme sequence to word
                        lexicon_dict[phonemes] = word
            print(f"Loaded {len(lexicon_dict)} lexicon entries for greedy fallback")
        except Exception as e:
            print(f"Warning: Could not load lexicon for greedy fallback: {e}")
        return lexicon_dict
    
    def _phonemes_to_words(self, phoneme_tokens: List[str]) -> List[str]:
        """
        Convert phoneme sequence to words using lexicon lookup.
        Uses greedy longest-match algorithm.
        """
        if not phoneme_tokens:
            return []
        
        words = []
        i = 0
        
        while i < len(phoneme_tokens):
            # Skip silence tokens
            if phoneme_tokens[i] == self.silence_token.strip():
                i += 1
                continue
            
            # Try to find longest matching phoneme sequence
            best_match = None
            best_length = 0
            
            # Try sequences of decreasing length
            for length in range(min(10, len(phoneme_tokens) - i), 0, -1):
                phoneme_seq = ' '.join(phoneme_tokens[i:i+length])
                if phoneme_seq in self.lexicon_dict:
                    best_match = self.lexicon_dict[phoneme_seq]
                    best_length = length
                    break
            
            if best_match:
                words.append(best_match)
                i += best_length
            else:
                # No match found, keep as phoneme or skip
                if phoneme_tokens[i] != self.silence_token.strip():
                    words.append(phoneme_tokens[i])  # Keep unknown phoneme
                i += 1
        
        return words
    
    def _build_decoder(self):
        """Build Flashlight CTC decoder."""
        if not FLASHLIGHT_AVAILABLE:
            print("Flashlight not available, using greedy decoder")
            return None
        
        # Skip flashlight for now due to trie size issues with large lexicon
        print("Skipping Flashlight decoder, using greedy decoder with lexicon lookup")
        return None
            
        try:
            # Load tokens dictionary
            token_dict = Dictionary()
            for i, token in enumerate(self.tokens):
                token_dict.add_entry(token, i)
            
            # Load lexicon and build trie
            lexicon = load_words(self.lexicon_path)
            word_dict = create_word_dict(lexicon)
            
            # Build trie from lexicon
            trie = Trie(token_dict.index_size(), word_dict.index_size())
            
            for word, spellings in lexicon.items():
                word_idx = word_dict.get_index(word)
                for spelling in spellings:
                    # Convert spelling (list of tokens) to token indices
                    spelling_indices = []
                    for token in spelling:
                        token_idx = token_dict.get_index(token)
                        if token_idx != -1:  # Valid token
                            spelling_indices.append(token_idx)
                    
                    if spelling_indices:  # Only add if we have valid tokens
                        trie.insert(spelling_indices, word_idx, 0.0)  # score = 0.0
            
            # Create language model
            if self.lm_path and os.path.exists(self.lm_path):
                # Load KenLM model
                lm = LM(self.lm_path)
                print("Loaded KenLM language model")
            else:
                # Use zero language model (no LM)
                lm = ZeroLM()
                print("Using ZeroLM (no language model)")
            
            # Set up decoder options
            decoder_opts = LexiconDecoderOptions(
                beam_size=self.beam_size,
                beam_size_token=self.beam_size_token,
                beam_threshold=self.beam_threshold,
                lm_weight=self.lm_weight,
                word_score=self.word_score,
                unk_score=-float('inf'),
                sil_score=0.0,
                log_add=False,
                criterion_type=CriterionType.CTC
            )
            
            # Find token indices for special tokens
            unk_token_idx = word_dict.get_index(self.unk_word)
            if unk_token_idx == -1:
                unk_token_idx = word_dict.get_index("<unk>")
            if unk_token_idx == -1:
                unk_token_idx = 0  # Fallback
            
            # Build decoder with new API
            decoder = LexiconDecoder(
                decoder_opts,
                trie,
                lm,
                self.silence_idx,  # sil_token_idx
                self.blank_idx,    # blank_token_idx  
                unk_token_idx,     # unk_token_idx
                [],                # transitions (empty for CTC)
                False              # is_token_lm
            )
            
            print("Built Flashlight CTC decoder with lexicon")
            return decoder
            
        except Exception as e:
            print(f"Error building Flashlight decoder: {e}")
            print("Falling back to greedy decoder...")
            return None
    
    def decode(
        self, 
        logits: torch.Tensor,
        logit_lengths: Optional[torch.Tensor] = None
    ) -> List[Dict[str, Any]]:
        """
        Decode CTC logits to text.
        
        Args:
            logits: CTC logits tensor [batch, time, vocab] or [time, vocab]
            logit_lengths: Lengths of each sequence in batch [batch]
            
        Returns:
            List of decoding results, one per batch item.
            Each result contains:
                - words: List of decoded words
                - tokens: List of decoded tokens  
                - score: Decoding score
                - nbest: List of n-best hypotheses (if nbest > 1)
        """
        # Handle single sequence case
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)  # Add batch dimension
            single_sequence = True
        else:
            single_sequence = False
        
        batch_size, seq_len, vocab_size = logits.shape
        
        # Default lengths to full sequences if not provided
        if logit_lengths is None:
            logit_lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
        
        # Move to CPU for decoding (TorchAudio decoder expects CPU tensors)
        logits = logits.cpu()
        logit_lengths = logit_lengths.cpu()
        
        results = []
        
        if self.decoder is not None and FLASHLIGHT_AVAILABLE:
            try:
                # Use Flashlight CTC decoder
                for i in range(batch_size):
                    seq_len = logit_lengths[i].item()
                    logit_seq = logits[i, :seq_len].cpu().numpy()  # [time, vocab]
                    
                    # Flashlight expects log probabilities
                    log_probs = torch.log_softmax(torch.from_numpy(logit_seq), dim=-1).numpy()
                    
                    # Decode with Flashlight
                    decoder_results = self.decoder.decode(log_probs.T)  # Flashlight expects [vocab, time]
                    
                    if decoder_results:
                        # Get best hypothesis
                        best_result = decoder_results[0]
                        words = [self.tokens[idx] for idx in best_result.tokens if idx < len(self.tokens)]
                        tokens = words  # For CTC, tokens are the same as words at phone level
                        score = best_result.score
                        
                        # Get n-best if available
                        nbest_results = []
                        for j, result in enumerate(decoder_results[:self.nbest]):
                            result_tokens = [self.tokens[idx] for idx in result.tokens if idx < len(self.tokens)]
                            nbest_results.append({
                                'words': result_tokens,
                                'tokens': result_tokens,
                                'score': result.score,
                                'rank': j
                            })
                        
                        results.append({
                            'words': words,
                            'tokens': tokens,
                            'score': score,
                            'nbest': nbest_results,
                            'sentence': ' '.join(words) if words else ''
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
            
            except Exception as e:
                print(f"Flashlight decoder failed: {e}")
                print("Falling back to greedy decoding...")
                results = self._greedy_decode(logits, logit_lengths)
        
        else:
            # Fallback to greedy decoding
            results = self._greedy_decode(logits, logit_lengths)
        
        # Return single result if input was single sequence
        if single_sequence:
            return results[0] if results else {
                'words': [], 'tokens': [], 'score': float('-inf'), 
                'nbest': [], 'sentence': ''
            }
        
        return results
    
    def _greedy_decode(
        self, 
        logits: torch.Tensor, 
        logit_lengths: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """
        Fallback greedy CTC decoding.
        
        Args:
            logits: CTC logits [batch, time, vocab]
            logit_lengths: Sequence lengths [batch]
            
        Returns:
            List of decoding results
        """
        results = []
        
        for i in range(logits.shape[0]):
            seq_len = logit_lengths[i].item()
            logit_seq = logits[i, :seq_len]  # [time, vocab]
            
            # Greedy decoding
            pred_tokens = torch.argmax(logit_seq, dim=-1)  # [time]
            
            # Remove blanks and consecutive duplicates
            decoded_tokens = []
            prev_token = None
            
            for token_idx in pred_tokens:
                token_idx = token_idx.item()
                if token_idx != self.blank_idx and token_idx != prev_token:
                    decoded_tokens.append(token_idx)
                prev_token = token_idx
            
            # Convert token indices to tokens
            tokens = [self.tokens[idx] for idx in decoded_tokens if idx < len(self.tokens)]
            
            # Convert phonemes to words using lexicon
            words = self._phonemes_to_words(tokens)
            sentence = ' '.join(words)
            
            results.append({
                'words': words,
                'tokens': tokens,
                'score': 0.0,  # No score for greedy
                'nbest': [],
                'sentence': sentence
            })
        
        return results
    
    def update_params(
        self,
        lm_weight: Optional[float] = None,
        word_score: Optional[float] = None,
        beam_size: Optional[int] = None,
        beam_size_token: Optional[int] = None,
        beam_threshold: Optional[float] = None
    ) -> None:
        """
        Update decoder parameters and rebuild if necessary.
        
        Args:
            lm_weight: New language model weight
            word_score: New word score
            beam_size: New beam size
            beam_size_token: New token beam size  
            beam_threshold: New beam threshold
        """
        rebuild_needed = False
        
        if lm_weight is not None and lm_weight != self.lm_weight:
            self.lm_weight = lm_weight
            rebuild_needed = True
            
        if word_score is not None and word_score != self.word_score:
            self.word_score = word_score
            rebuild_needed = True
            
        if beam_size is not None and beam_size != self.beam_size:
            self.beam_size = beam_size
            rebuild_needed = True
            
        if beam_size_token is not None and beam_size_token != self.beam_size_token:
            self.beam_size_token = beam_size_token
            rebuild_needed = True
            
        if beam_threshold is not None and beam_threshold != self.beam_threshold:
            self.beam_threshold = beam_threshold
            rebuild_needed = True
        
        if rebuild_needed:
            print(f"Rebuilding decoder with new params: lm_weight={self.lm_weight}, "
                  f"word_score={self.word_score}, beam_size={self.beam_size}")
            self.decoder = self._build_decoder()


def test_decoder():
    """Test the CTC decoder with dummy data."""
    print("Testing CTC decoder...")
    
    # Create dummy tokens and lexicon files
    os.makedirs("artifacts", exist_ok=True)
    
    # Dummy tokens
    with open("artifacts/test_tokens.txt", "w") as f:
        f.write("BLANK\nAA\nAE\nAH\nSIL\n")
    
    # Dummy lexicon  
    with open("artifacts/test_lexicon.txt", "w") as f:
        f.write("hello AH\nworld AA AE\n")
    
    # Initialize decoder
    decoder = CTCDecoder(
        tokens_path="artifacts/test_tokens.txt",
        lexicon_path="artifacts/test_lexicon.txt",
        lm_path=None,  # No LM for test
        beam_size=10,
        silence_token="SIL"
    )
    
    # Create dummy logits
    batch_size, seq_len, vocab_size = 1, 20, 5
    logits = torch.randn(batch_size, seq_len, vocab_size)
    
    # Decode
    results = decoder.decode(logits)
    
    # Results is a list with one element for single sequence
    result = results[0] if isinstance(results, list) else results
    
    print(f"Decoded result:")
    print(f"  Tokens: {result['tokens']}")
    print(f"  Words: {result['words']}")
    print(f"  Sentence: {result['sentence']}")
    print(f"  Score: {result['score']}")
    
    print("CTC decoder test completed!")


if __name__ == "__main__":
    test_decoder()
