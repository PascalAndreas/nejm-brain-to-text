"""
Build lexicon.txt file from CMUdict + G2P fallback with phone mapping.
Maps words to phone sequences for lexicon-constrained CTC decoding.
"""

import os
import re
import sys
from pathlib import Path
from typing import Set, Dict, List, Tuple
from collections import defaultdict
import urllib.request
from g2p_en import G2p

# Add parent directory to path to import nejm_b2txt_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nejm_b2txt_utils.general_utils import remove_punctuation

# Import the correct phoneme mapping from evaluate_model_helpers.py
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model_training'))
from evaluate_model_helpers import LOGIT_TO_PHONEME


# Phone mapping from CMUdict ARPAbet to our 39-phone set
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


def extract_lm_vocabulary(vocab_file_path: str, top_n: int = 50000) -> Set[str]:
    """
    Extract top-N vocabulary from language model vocabulary file.
    
    Args:
        vocab_file_path: Path to vocabulary file (e.g., NVIDIA TAO en_4.0_dict_vocab.txt)
        top_n: Number of top words to extract
        
    Returns:
        Set of vocabulary words
    """
    vocab = set()
    
    if not os.path.exists(vocab_file_path):
        print(f"Warning: Vocabulary file not found: {vocab_file_path}")
        return vocab
    
    print(f"Loading vocabulary from: {vocab_file_path}")
    
    with open(vocab_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= top_n:
                break
                
            word = line.strip().split()[0] if line.strip() else ""
            if word and word.replace("'", "").replace("-", "").isalpha():
                vocab.add(word.upper())
    
    print(f"Extracted {len(vocab)} words from LM vocabulary")
    return vocab


def build_lexicon(
    output_path: str,
    vocab_boost_size: int = 50000,
    cmudict_path: str = None,
    lm_vocab_file: str = None
) -> Tuple[int, int, float]:
    """
    Build lexicon.txt file from CMUdict + G2P fallback using LM vocabulary.
    
    Args:
        output_path: Path to write lexicon.txt
        vocab_boost_size: Number of high-frequency words to add from LM vocab
        cmudict_path: Path to CMUdict file (will download if None)
        lm_vocab_file: Path to LM vocabulary file (e.g., NVIDIA TAO vocab)
        
    Returns:
        Tuple of (total_words, total_pronunciations, oov_rate)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Download/load CMUdict
    if cmudict_path is None:
        cmudict_path = download_cmudict()
    cmudict = load_cmudict(cmudict_path)
    
    # Get vocabulary from language model
    if lm_vocab_file and os.path.exists(lm_vocab_file):
        vocab = extract_lm_vocabulary(lm_vocab_file, vocab_boost_size)
        print(f"Using LM vocabulary: {len(vocab)} words")
    else:
        print("No LM vocabulary file provided, using CMUdict only")
        vocab = set()
    
    # Initialize G2P for OOV words
    g2p = G2p()
    
    # Build lexicon
    lexicon = {}
    oov_count = 0
    total_words = 0
    
    # Process all words in vocabulary
    all_words = set(cmudict.keys()) | vocab
    
    for word in all_words:
        word_clean = word.upper().strip()
        if not word_clean or not word_clean.replace("'", "").isalpha():
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
            lexicon[word_clean] = pronunciations
    
    # Write lexicon file
    total_pronunciations = 0
    with open(output_path, 'w') as f:
        for word, pronunciations in sorted(lexicon.items()):
            for pron in pronunciations:
                f.write(f"{word.lower()} {' '.join(pron)}\n")
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
        for line in f:
            parts = line.strip().split()
            if len(parts) > 1:
                phones = parts[1:]
                all_phones_in_lexicon.update(phones)
    
    invalid_phones = all_phones_in_lexicon - set(LOGIT_TO_PHONEME[1:40])  # Exclude BLANK (0) and ' | ' (40)
    if invalid_phones:
        print(f"WARNING: Invalid phones in lexicon: {invalid_phones}")
    else:
        print("✓ All phones in lexicon are valid")
    
    return len(lexicon), total_pronunciations, oov_rate


if __name__ == "__main__":
    # Build lexicon with default settings
    output_path = "artifacts/lexicon.txt"
    build_lexicon(output_path)
