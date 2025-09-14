"""
Build tokens.txt file from LOGIT_TO_PHONEME for CTC decoding.
This file maps token indices to phonemes and must match the model output exactly.
"""

import os
import sys
from pathlib import Path

# Use the exact same mapping as evaluate_model_helpers.py for consistency
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


def build_tokens(output_path: str) -> None:
    """
    Build tokens.txt file from LOGIT_TO_PHONEME.
    
    Args:
        output_path: Path to write tokens.txt file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for token in LOGIT_TO_PHONEME:
            f.write(f'{token}\n')
    
    print(f"Built tokens.txt with {len(LOGIT_TO_PHONEME)} tokens at: {output_path}")
    
    # Verify the file
    with open(output_path, 'r') as f:
        tokens = [line.strip() for line in f]
    
    print(f"Tokens: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
    print(f"Total tokens: {len(tokens)}")


if __name__ == "__main__":
    # Resolve path relative to project root
    project_root = os.path.dirname(os.path.dirname(__file__))  # Go up one level from /decoding
    output_path = os.path.join(project_root, "decoding/artifacts/tokens.txt")
    build_tokens(output_path)
