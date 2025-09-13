"""
Build lexicon.txt file from CMUdict + G2P fallback with phone mapping.
Maps words to phone sequences for lexicon-constrained CTC decoding.
"""

import os
import re
import sys
import warnings
from pathlib import Path
from typing import Set, Dict, List, Tuple
from collections import defaultdict
import urllib.request
import csv

# Suppress G2P numpy warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="g2p_en")

from g2p_en import G2p

from decoding.helpers import LOGIT_TO_PHONEME, PHONE_MAPPING

try:
    import kenlm
    KENLM_AVAILABLE = True
except ImportError:
    KENLM_AVAILABLE = False
    print("Warning: kenlm not available. 1-gram probabilities will not be computed.")


def download_cmudict(cache_path: str = "artifacts/cmudict.dict") -> str:
    """Download CMUdict if not already cached."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    if os.path.exists(cache_path):
        print(f"Using cached CMUdict at: {cache_path}")
        return cache_path
    
    print("Downloading CMUdict...")
    url = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
    urllib.request.urlretrieve(url, cache_path)
    print(f"Downloaded CMUdict to: {cache_path}")
    return cache_path


def load_cmudict(cmudict_path: str) -> Dict[str, List[List[str]]]:
    """Load CMUdict and return word -> pronunciations mapping."""
    pronunciations = defaultdict(list)
    
    with open(cmudict_path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';;;'):
                continue
            
            # Parse: WORD  PHONE1 PHONE2 ...
            # or:    WORD(2)  PHONE1 PHONE2 ... (variant pronunciations)
            parts = line.split()
            if len(parts) < 2:
                continue
            
            word = parts[0]
            # Remove variant indicators like (2), (3), etc.
            word = re.sub(r'\(\d+\)$', '', word).upper()
            
            phones = parts[1:]
            pronunciations[word].append(phones)
    
    print(f"Loaded CMUdict with {len(pronunciations)} words")
    return dict(pronunciations)


def map_phones_to_logit_set(phones: List[str]) -> List[str]:
    """Map CMUdict phones to our LOGIT_TO_PHONEME set."""
    mapped_phones = []
    
    # Create target phone set (exclude BLANK and silence token)
    target_phones = set(LOGIT_TO_PHONEME[1:40])  # Exclude BLANK (0) and ' | ' (40)
    
    for phone in phones:
        # Remove stress digits
        phone_clean = re.sub(r'[0-9]', '', phone)
        
        # Map to our phone set
        if phone_clean in PHONE_MAPPING:
            mapped_phone = PHONE_MAPPING[phone_clean]
            # Only add if it's in our target phone set
            if mapped_phone in target_phones:
                mapped_phones.append(mapped_phone)
        elif phone_clean in target_phones:
            mapped_phones.append(phone_clean)
    
    return mapped_phones


def load_kenlm_model(lm_path: str):
    """Load KenLM model for computing 1-gram probabilities."""
    if not KENLM_AVAILABLE:
        print("Warning: KenLM not available, cannot compute 1-gram probabilities")
        return None
    
    if not lm_path or not os.path.exists(lm_path):
        print(f"Warning: Language model not found at {lm_path}")
        return None
    
    try:
        print(f"Loading KenLM model from: {lm_path}")
        model = kenlm.Model(lm_path)
        print(f"Successfully loaded KenLM model (order: {model.order})")
        return model
    except Exception as e:
        print(f"Failed to load KenLM model: {e}")
        return None


def get_word_1gram_logprob(model, word: str) -> float:
    """Get 1-gram log probability for a word from KenLM model."""
    if model is None:
        return 0.0  # Default score if no model
    
    try:
        # Query 1-gram probability (no context)
        # Use bos=False, eos=False to get just the word probability
        log_prob = model.score(word.lower(), bos=False, eos=False)
        return log_prob
    except Exception as e:
        print(f"Warning: Failed to get log probability for word '{word}': {e}")
        return 0.0  # Default score on failure


def extract_lm_vocabulary_from_lexicon(lexicon_file_path: str, top_n: int = 50000) -> List[str]:
    """
    Extract top-N vocabulary from NVIDIA TAO lexicon file (already frequency-ordered).
    
    Args:
        lexicon_file_path: Path to NVIDIA TAO lexicon.txt file  
        top_n: Number of top words to extract
        
    Returns:
        List of vocabulary words in frequency order
    """
    vocab = []
    
    if not os.path.exists(lexicon_file_path):
        print(f"Warning: Lexicon file not found: {lexicon_file_path}")
        return vocab
    
    print(f"Loading vocabulary from TAO lexicon: {lexicon_file_path}")
    
    with open(lexicon_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if len(vocab) >= top_n:
                break
                
            parts = line.strip().split('\t')
            if len(parts) >= 1:
                word = parts[0].strip()
                # Only include alphabetic words (no punctuation, numbers, etc.)
                if word and word.replace("'", "").replace("-", "").isalpha():
                    vocab.append(word.upper())
    
    print(f"Extracted {len(vocab)} words from TAO lexicon")
    return vocab


def build_lexicon(
    output_path: str,
    vocab_boost_size: int = 50000,
    cmudict_path: str = None,
    lm_vocab_file: str = None,
    lm_path: str = None
) -> Tuple[int, int, float]:
    """
    Build lexicon.txt file from CMUdict + G2P fallback using LM vocabulary.
    
    Args:
        output_path: Path to write lexicon.txt (or lexicon.csv if including probabilities)
        vocab_boost_size: Number of high-frequency words to add from LM vocab
        cmudict_path: Path to CMUdict file (will download if None)
        lm_vocab_file: Path to LM vocabulary file (e.g., NVIDIA TAO vocab)
        lm_path: Path to KenLM model for computing 1-gram probabilities
        
    Returns:
        Tuple of (total_words, total_pronunciations, oov_rate)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load KenLM model for 1-gram probabilities (required for CSV format)
    kenlm_model = load_kenlm_model(lm_path) if lm_path else None
    if not kenlm_model:
        raise ValueError("KenLM model is required for lexicon building. Please provide lm_path.")
    
    # Download/load CMUdict
    if cmudict_path is None:
        cmudict_path = download_cmudict()
    cmudict = load_cmudict(cmudict_path)
    
    # Get vocabulary from NVIDIA TAO lexicon (frequency-ordered)
    tao_lexicon_file = lm_vocab_file.replace('en_4.0_dict_vocab.txt', 'lexicon.txt') if lm_vocab_file else None
    if tao_lexicon_file and os.path.exists(tao_lexicon_file):
        vocab_list = extract_lm_vocabulary_from_lexicon(tao_lexicon_file, vocab_boost_size)
        print(f"Using TAO lexicon vocabulary: {len(vocab_list)} words")
    else:
        print("No TAO lexicon file found, using CMUdict only")
        vocab_list = []
    
    # Initialize G2P for OOV words
    g2p = G2p()
    
    # Build lexicon with 1-gram probabilities
    lexicon = {}  # word -> (pronunciations, log_prob)
    oov_count = 0
    total_words = 0
    words_added = 0
    
    print(f"Building lexicon with up to {vocab_boost_size} words...")
    
    # Process TAO vocabulary words first (already in frequency order)
    for word in vocab_list:
        if words_added >= vocab_boost_size:
            break
            
        word_clean = word.upper().strip()
        if not word_clean or not word_clean.replace("'", "").replace("-", "").isalpha():
            continue
            
        total_words += 1
        pronunciations = []
        
        if word_clean in cmudict:
            # Use CMUdict pronunciations
            for pron in cmudict[word_clean]:
                mapped_phones = map_phones_to_logit_set(pron)
                if mapped_phones:  # Only add if mapping was successful
                    pronunciations.append(mapped_phones)
        
        if not pronunciations:
            # Fall back to G2P
            oov_count += 1
            try:
                g2p_phones = g2p(word_clean.lower())
                # Remove spaces and non-alphabetic characters
                g2p_phones = [p for p in g2p_phones if p.isalpha()]
                mapped_phones = map_phones_to_logit_set(g2p_phones)
                if mapped_phones:
                    pronunciations.append(mapped_phones)
            except Exception as e:
                print(f"G2P failed for word '{word_clean}': {e}")
                continue
        
        if pronunciations:
            # Get 1-gram log probability
            log_prob = get_word_1gram_logprob(kenlm_model, word_clean)
            lexicon[word_clean] = (pronunciations, log_prob)
            words_added += 1
    
    print(f"Successfully added {words_added} words to lexicon")
    
    # Sort words by 1-gram log probability (highest first)
    sorted_words = sorted(lexicon.keys(), 
                        key=lambda w: lexicon[w][1], reverse=True)
    print(f"Sorted {len(sorted_words)} words by 1-gram log probability")
    
    # Write CSV lexicon file with log probabilities
    total_pronunciations = 0
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['word', 'phonemes', 'log_prob'])  # Header
        
        for word in sorted_words:
            pronunciations, log_prob = lexicon[word]
            for pron in pronunciations:
                writer.writerow([word.lower(), ' '.join(pron), log_prob])
                total_pronunciations += 1
        
        # Add <unk> token at the end with a low probability
        writer.writerow(['<unk>', 'AH N K', -10.0])
        total_pronunciations += 1
    
    # Calculate statistics
    oov_rate = oov_count / total_words if total_words > 0 else 0
    
    print(f"Built lexicon at: {output_path}")
    print(f"Total words: {len(lexicon)}")
    print(f"Total pronunciations: {total_pronunciations}")
    print(f"OOV rate: {oov_rate:.1%} ({oov_count}/{total_words})")
    
    # Validate that all phones are in LOGIT_TO_PHONEME
    all_phones_in_lexicon = set()
    with open(output_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:
                phones = row[1].split()
                all_phones_in_lexicon.update(phones)
    
    invalid_phones = all_phones_in_lexicon - set(LOGIT_TO_PHONEME[1:40])  # Exclude BLANK (0) and 'SIL' (40)
    if invalid_phones:
        print(f"WARNING: Invalid phones in lexicon: {invalid_phones}")
    else:
        print("✓ All phones in lexicon are valid")
    
    return len(lexicon), total_pronunciations, oov_rate


if __name__ == "__main__":
    import yaml
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        vocab_size = config.get('lexicon', {}).get('lm_vocab_size', 50000)
        lm_vocab_file = config.get('language_model', {}).get('nvidia_tao', {}).get('lexicon_file')
        lm_path = config.get('language_model', {}).get('active_model')
        output_path = config.get('artifacts', {}).get('lexicon_txt')
        
        # Always use CSV format
        output_path = output_path.replace('.txt', '.csv')
    else:
        vocab_size = 50000
        lm_vocab_file = None
        lm_path = None
        output_path = 'artifacts/lexicon.csv'
    
    # Build lexicon with configuration settings
    build_lexicon(
        output_path, 
        vocab_boost_size=vocab_size,
        lm_vocab_file=lm_vocab_file,
        lm_path=lm_path
    )
