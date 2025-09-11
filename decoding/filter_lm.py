"""
Filter KenLM language model by lexicon vocabulary to reduce memory usage.
"""

import os
import tempfile
import subprocess
from pathlib import Path
from typing import Set


def load_lexicon_vocabulary(lexicon_path: str) -> Set[str]:
    """Load vocabulary from lexicon.txt file."""
    vocab = set()
    
    with open(lexicon_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 1:
                word = parts[0].lower()
                vocab.add(word)
    
    print(f"Loaded {len(vocab)} words from lexicon")
    return vocab


def filter_arpa_lm(
    input_lm_path: str,
    output_lm_path: str, 
    vocab: Set[str],
    keep_unk: bool = True
) -> None:
    """
    Filter ARPA language model by vocabulary.
    
    Args:
        input_lm_path: Path to input ARPA LM file
        output_lm_path: Path to write filtered ARPA LM
        vocab: Set of words to keep
        keep_unk: Whether to keep <unk> token
    """
    os.makedirs(os.path.dirname(output_lm_path), exist_ok=True)
    
    print(f"Filtering ARPA LM: {input_lm_path} -> {output_lm_path}")
    
    # Add special tokens to vocabulary
    special_tokens = {'<s>', '</s>'}
    if keep_unk:
        special_tokens.add('<unk>')
    
    vocab_with_special = vocab | special_tokens
    
    with open(input_lm_path, 'r') as infile, open(output_lm_path, 'w') as outfile:
        in_ngram_section = False
        current_section = None
        
        for line in infile:
            line = line.strip()
            
            # Copy header information
            if line.startswith('\\data\\') or line.startswith('ngram'):
                outfile.write(line + '\n')
                continue
            
            # Track n-gram sections
            if line.startswith('\\') and line.endswith('-grams:'):
                current_section = line
                in_ngram_section = True
                outfile.write(line + '\n')
                continue
            
            if line.startswith('\\end\\'):
                outfile.write(line + '\n')
                break
            
            if in_ngram_section and line:
                # Parse n-gram line: prob word1 word2 ... [backoff]
                parts = line.split('\t')
                if len(parts) >= 2:
                    prob = parts[0]
                    ngram_part = parts[1]
                    backoff = parts[2] if len(parts) > 2 else None
                    
                    # Extract words from n-gram
                    words = ngram_part.split()
                    
                    # Keep n-gram if all words are in vocabulary
                    keep_ngram = all(word in vocab_with_special for word in words)
                    
                    if keep_ngram:
                        if backoff:
                            outfile.write(f"{prob}\t{ngram_part}\t{backoff}\n")
                        else:
                            outfile.write(f"{prob}\t{ngram_part}\n")
            elif not line:
                # Empty line marks end of section
                in_ngram_section = False
                outfile.write('\n')
    
    print(f"Filtered ARPA LM written to: {output_lm_path}")


def convert_arpa_to_binary(arpa_path: str, binary_path: str) -> None:
    """
    Convert ARPA LM to binary format using KenLM's build_binary.
    
    Args:
        arpa_path: Path to ARPA LM file
        binary_path: Path to write binary LM file
    """
    try:
        # Try to use KenLM's build_binary tool
        cmd = ['build_binary', arpa_path, binary_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Converted to binary LM: {binary_path}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: Could not convert to binary format: {e}")
        print("You may need to install KenLM tools or use the ARPA format directly")


def filter_language_model(
    input_lm_path: str,
    lexicon_path: str,
    output_dir: str = "artifacts",
    convert_to_binary: bool = True
) -> str:
    """
    Filter language model by lexicon vocabulary.
    
    Args:
        input_lm_path: Path to input language model (ARPA format)
        lexicon_path: Path to lexicon.txt file
        output_dir: Directory to write filtered LM
        convert_to_binary: Whether to convert to binary format
        
    Returns:
        Path to filtered language model file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load vocabulary from lexicon
    vocab = load_lexicon_vocabulary(lexicon_path)
    
    # Determine output paths
    base_name = Path(input_lm_path).stem
    filtered_arpa_path = os.path.join(output_dir, f"{base_name}_filtered.arpa")
    filtered_binary_path = os.path.join(output_dir, f"{base_name}_filtered.bin")
    
    # Filter ARPA LM
    filter_arpa_lm(input_lm_path, filtered_arpa_path, vocab)
    
    # Convert to binary if requested
    if convert_to_binary:
        convert_arpa_to_binary(filtered_arpa_path, filtered_binary_path)
        return filtered_binary_path
    else:
        return filtered_arpa_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter KenLM language model by lexicon vocabulary")
    parser.add_argument("input_lm", help="Path to input ARPA language model")
    parser.add_argument("lexicon", help="Path to lexicon.txt file")
    parser.add_argument("--output-dir", default="artifacts", help="Output directory")
    parser.add_argument("--no-binary", action="store_true", help="Don't convert to binary format")
    
    args = parser.parse_args()
    
    filtered_lm_path = filter_language_model(
        args.input_lm,
        args.lexicon,
        args.output_dir,
        convert_to_binary=not args.no_binary
    )
    
    print(f"Filtered language model: {filtered_lm_path}")
