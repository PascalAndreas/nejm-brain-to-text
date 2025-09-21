"""
Greedy CTC decoder implementation.
Simple greedy decoding without beam search or language model.
"""

import torch
from typing import List, Dict, Optional, Any
from pathlib import Path

# Handle relative imports for both package and direct execution
from decoding.helpers import load_tokens, load_lexicon_dict, phonemes_to_words


class GreedyCTCDecoder:
    """
    Simple greedy CTC decoder.
    Performs greedy decoding and converts phonemes to words using lexicon lookup.
    """
    
    def __init__(
        self,
        tokens_path: str,
        lexicon_path: str,
        blank_token: str = "BLANK",
        silence_token: str = "SIL",
        unk_word: str = "<unk>",
        device: Optional[torch.device] = None,
        **kwargs  # Accept but ignore other parameters for compatibility
    ):
        """
        Initialize greedy CTC decoder.
        
        Args:
            tokens_path: Path to tokens.txt file
            lexicon_path: Path to lexicon.txt file
            blank_token: CTC blank token symbol
            silence_token: Silence token symbol
            unk_word: Unknown word symbol
            device: PyTorch device (ignored, always uses CPU for decoding)
            **kwargs: Other parameters (ignored for compatibility)
        """
        self.tokens_path = tokens_path
        self.lexicon_path = lexicon_path
        self.blank_token = blank_token
        self.silence_token = silence_token
        self.unk_word = unk_word
        self.device = device or torch.device('cpu')
        
        # Load tokens
        self.tokens = load_tokens(self.tokens_path)
        self.token_to_idx = {token: idx for idx, token in enumerate(self.tokens)}
        
        # Find special token indices
        self.blank_idx = self.token_to_idx.get(self.blank_token, 0)
        self.silence_idx = self.token_to_idx.get(self.silence_token, -1)
        
        # Load lexicon for phoneme-to-word conversion
        self.lexicon_dict = load_lexicon_dict(self.lexicon_path)
        
        print(f"Initialized GreedyCTCDecoder:")
        print(f"  Tokens: {len(self.tokens)} ({self.tokens_path})")
        print(f"  Lexicon: {self.lexicon_path}")
        print(f"  Blank token: '{self.blank_token}' (idx={self.blank_idx})")
        print(f"  Silence token: '{self.silence_token}' (idx={self.silence_idx})")
    
    
    def decode(
        self, 
        logits: torch.Tensor,
        logit_lengths: Optional[torch.Tensor] = None
    ) -> List[Dict[str, Any]]:
        """
        Decode CTC logits to text using greedy decoding.
        Expects batched inputs [batch, time, vocab] with logit_lengths [batch].
        
        Args:
            logits: CTC logits tensor [batch, time, vocab] (batching handled by base class)
            logit_lengths: Lengths of each sequence in batch [batch]
            
        Returns:
            List of decoding results, one per batch item.
            Each result contains:
                - words: List of decoded words
                - tokens: List of decoded tokens  
                - score: Decoding score (0.0 for greedy)
                - nbest: List of n-best hypotheses (empty for greedy)
                - sentence: Decoded sentence string
        """
        batch_size, seq_len, vocab_size = logits.shape
        results = []
        
        for i in range(batch_size):
            seq_len = logit_lengths[i].item()
            logit_seq = logits[i, :seq_len]  # [time, vocab]
            
            # Greedy decoding - argmax works on both logits and log probabilities
            pred_tokens = torch.argmax(logit_seq, dim=-1)  # [time]
            
            # Remove blanks and consecutive duplicates
            decoded_tokens = []
            prev_token = None
            
            for token_idx in pred_tokens:
                token_idx = token_idx.item()
                if token_idx != self.blank_idx and token_idx != prev_token:
                    decoded_tokens.append(token_idx)
                prev_token = token_idx
            
            # Convert token indices to tokens with proper bounds checking
            tokens = [self.tokens[idx] for idx in decoded_tokens if 0 <= idx < len(self.tokens)]
            
            # Convert phonemes to words using lexicon
            words = phonemes_to_words(tokens, self.lexicon_dict, self.silence_token)
            sentence = ' '.join(words)
            
            results.append({
                'words': words,
                'tokens': tokens,
                'score': 0.0,  # No score for greedy
                'nbest': [],   # No n-best for greedy
                'sentence': sentence
            })
        
        return results
