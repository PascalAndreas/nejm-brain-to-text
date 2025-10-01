"""
Shared helper functions for CTC decoders.
Contains common functionality used across different decoder implementations.
"""

from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import csv
import os


def load_tokens(tokens_path: str) -> List[str]:
    """Load tokens from tokens.txt file."""
    tokens = []
    with open(tokens_path, 'r') as f:
        for line in f:
            token = line.rstrip('\n')  # Only remove newlines, preserve spaces
            if token:  # Allow empty string tokens if needed
                tokens.append(token)
    return tokens


def load_lexicon_entries(lexicon_path: str, verbose: Optional[bool] = False) -> List[Tuple[str, str, float]]:
    """
    Load lexicon entries with individual probabilities preserved.
    Returns list of (word, phoneme_sequence, log_prob) tuples.
    """
    entries = []
    
    try:
        with open(lexicon_path, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            
            for row in reader:
                if len(row) >= 3:
                    word = row[0]
                    phonemes = row[1]
                    log_prob = float(row[2])
                    entries.append((word, phonemes, log_prob))
        
        if verbose:
            print(f"Loaded {len(entries)} lexicon entries with individual probabilities")
        
    except Exception as e:
        print(f"Warning: Could not load lexicon: {e}")
        entries = []
    
    return entries


def load_lexicon_with_probs(lexicon_path: str, verbose: Optional[bool] = False) -> Dict[str, Tuple[List[str], float]]:
    """
    Load lexicon with 1-gram log probabilities from CSV format.
    Returns dict mapping phoneme sequences to (word_list, log_prob) tuples.
    
    DEPRECATED: This function collapses probabilities incorrectly. Use load_lexicon_entries() instead.
    Kept for backward compatibility with greedy decoder.
    """
    entries = load_lexicon_entries(lexicon_path, verbose=False)
    lexicon_dict = {}
    
    phoneme_to_entries = defaultdict(list)
    for word, phonemes, log_prob in entries:
        phoneme_to_entries[phonemes].append((word, log_prob))
    
    # For each phoneme sequence, collect words and use the best log prob
    for phonemes, word_prob_pairs in phoneme_to_entries.items():
        words = [entry[0] for entry in word_prob_pairs]
        # Use the highest log probability among all words for this phoneme sequence
        best_log_prob = max(entry[1] for entry in word_prob_pairs)
        lexicon_dict[phonemes] = (words, best_log_prob)
    
    if verbose:
        total_entries = sum(len(words) for words, _ in lexicon_dict.values())
        print(f"Loaded {len(lexicon_dict)} unique pronunciations with {total_entries} word entries")
    
    return lexicon_dict


def load_lexicon_dict(lexicon_path: str) -> Dict[str, List[str]]:
    """
    Load lexicon and create phoneme-to-word mapping (backward compatible).
    This function maintains the original interface for compatibility with greedy.py.
    """
    lexicon_with_probs = load_lexicon_with_probs(lexicon_path)
    
    # Convert to old format (drop probabilities)
    lexicon_dict = {}
    for phonemes, (words, _) in lexicon_with_probs.items():
        lexicon_dict[phonemes] = words
    
    return lexicon_dict


def select_best_word(word_candidates: List[str]) -> str:
    """
    Select the best word from candidates for a phoneme sequence.
    Since lexicon is pre-filtered by build_lexicon.py, just take the first one.
    """
    if not word_candidates:
        return ""
    # Since build_lexicon.py already handles word selection/filtering, just take first
    return word_candidates[0]


def phonemes_to_words(phoneme_tokens: List[str], lexicon_dict: Dict[str, List[str]], silence_token: str = "SIL") -> List[str]:
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
        if phoneme_tokens[i] == silence_token.strip():
            i += 1
            continue
        
        # Try to find longest matching phoneme sequence
        best_candidates = None
        best_length = 0
        
        # Try sequences of decreasing length
        for length in range(len(phoneme_tokens) - i, 0, -1):
            phoneme_seq = ' '.join(phoneme_tokens[i:i+length])
            if phoneme_seq in lexicon_dict:
                best_candidates = lexicon_dict[phoneme_seq]
                best_length = length
                break
        
        if best_candidates:
            # Select best word from candidates
            best_word = select_best_word(best_candidates)
            words.append(best_word)
            i += best_length
        else:
            # No match found, keep as phoneme or skip
            if phoneme_tokens[i] != silence_token.strip():
                words.append(phoneme_tokens[i])  # Keep unknown phoneme
            i += 1
    
    return words

LOGIT_TO_PHONEME = [
    'BLANK',
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B',  'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH',
    'SIL',
]

PHONE_MAPPING = {
    # Vowels - remove stress digits and map allophones
    'AA0': 'AA', 'AA1': 'AA', 'AA2': 'AA',
    'AE0': 'AE', 'AE1': 'AE', 'AE2': 'AE', 
    'AH0': 'AH', 'AH1': 'AH', 'AH2': 'AH',
    'AO0': 'AO', 'AO1': 'AO', 'AO2': 'AO',
    'AW0': 'AW', 'AW1': 'AW', 'AW2': 'AW',
    'AY0': 'AY', 'AY1': 'AY', 'AY2': 'AY',
    'EH0': 'EH', 'EH1': 'EH', 'EH2': 'EH',
    'ER0': 'ER', 'ER1': 'ER', 'ER2': 'ER',
    'EY0': 'EY', 'EY1': 'EY', 'EY2': 'EY',
    'IH0': 'IH', 'IH1': 'IH', 'IH2': 'IH',
    'IY0': 'IY', 'IY1': 'IY', 'IY2': 'IY',
    'OW0': 'OW', 'OW1': 'OW', 'OW2': 'OW',
    'OY0': 'OY', 'OY1': 'OY', 'OY2': 'OY',
    'UH0': 'UH', 'UH1': 'UH', 'UH2': 'UH',
    'UW0': 'UW', 'UW1': 'UW', 'UW2': 'UW',
    
    # Allophone mappings
    'AX': 'AH',   # schwa -> AH
    'IX': 'IH',   # barred i -> IH  
    'AXR': 'ER',  # schwa+r -> ER
    'UX': 'UW',   # high back -> UW
    
    # Consonants
    'B': 'B', 'CH': 'CH', 'D': 'D', 'DH': 'DH', 'F': 'F', 'G': 'G',
    'HH': 'HH', 'JH': 'JH', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N',
    'NG': 'NG', 'P': 'P', 'R': 'R', 'S': 'S', 'SH': 'SH', 'T': 'T',
    'TH': 'TH', 'V': 'V', 'W': 'W', 'Y': 'Y', 'Z': 'Z', 'ZH': 'ZH',
    
    # Flap mapping
    'DX': 'T',    # flap -> T
    
    # Syllabic consonants
    'EN': 'N', 'NX': 'N',  # syllabic n -> N
    'EM': 'M',             # syllabic m -> M
    'EL': 'L',             # syllabic l -> L
    
    # Alternative consonant forms
    'H': 'HH',    # alternative H -> HH
}