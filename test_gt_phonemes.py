#!/usr/bin/env python3
"""
Test CTC decoding pipeline with ground truth phoneme sequences.
This should achieve near-perfect accuracy since we're feeding the decoder
the exact phonemes that should be decoded.

Usage:
    python test_gt_phonemes.py --num_samples 6 --perfection 0.9 --time_expansion 1.5
"""

import os
import sys
import torch
import numpy as np
import h5py
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from jiwer import cer, wer

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decoding.decode_ctc import CTCDecoder
from decoding import DEFAULT_CONFIG
from model_training.evaluate_model_helpers import LOGIT_TO_PHONEME


def generate_logits(
    gt_phoneme_indices: np.ndarray, 
    num_phonemes: int,
    perfection: float = 0.95,
    time_expansion: float = 2.0,
    noise_level: float = 0.1,
    blank_probability: float = 0.3
) -> torch.Tensor:
    """
    Generate realistic CTC logits with controllable perfection.
    
    Args:
        gt_phoneme_indices: Ground truth phoneme sequence
        num_phonemes: Total number of phoneme classes
        perfection: How perfect the logits are (0.0 = random, 1.0 = perfect)
        time_expansion: How much longer the time sequence should be vs phonemes
        noise_level: Amount of random noise to add
        blank_probability: Probability of inserting blanks between phonemes
    
    Returns:
        Logits tensor [1, time, vocab] with realistic CTC patterns
    """
    # Calculate realistic time steps based on phoneme sequence and expansion
    base_time_steps = len(gt_phoneme_indices)
    num_time_steps = int(base_time_steps * time_expansion)
    num_time_steps = max(num_time_steps, base_time_steps + 5)  # Minimum expansion
    
    # Initialize with low baseline probabilities
    baseline_logit = -8.0 + np.random.normal(0, noise_level, (num_time_steps, num_phonemes))
    logits = baseline_logit.astype(np.float32)
    
    blank_idx = 0  # BLANK is at index 0
    
    # Create a realistic CTC alignment pattern
    alignment = []
    
    for i, ph_idx in enumerate(gt_phoneme_indices):
        # Add optional blank before phoneme
        if i > 0 and np.random.random() < blank_probability:
            alignment.append(blank_idx)
        
        # Add the phoneme (possibly repeated)
        phoneme_repeats = np.random.poisson(2) + 1  # 1-4 repeats typically
        for _ in range(phoneme_repeats):
            alignment.append(ph_idx)
    
    # Add final blank with some probability
    if np.random.random() < blank_probability:
        alignment.append(blank_idx)
    
    # Stretch or compress alignment to fit time steps
    if len(alignment) > num_time_steps:
        # Compress: sample without replacement
        indices = np.linspace(0, len(alignment) - 1, num_time_steps, dtype=int)
        alignment = [alignment[i] for i in indices]
    elif len(alignment) < num_time_steps:
        # Expand: interpolate with repetition
        expanded_alignment = []
        for i in range(num_time_steps):
            idx = int(i * len(alignment) / num_time_steps)
            expanded_alignment.append(alignment[min(idx, len(alignment) - 1)])
        alignment = expanded_alignment
    
    # Apply perfection to the aligned logits
    for t, target_phoneme in enumerate(alignment):
        if target_phoneme < num_phonemes:
            # Perfect component: high probability for correct phoneme
            perfect_logit = 12.0
            
            # Imperfect component: add confusion
            confusion_strength = (1.0 - perfection) * 6.0
            
            # Set target phoneme probability
            logits[t, target_phoneme] = perfect_logit * perfection + np.random.normal(0, noise_level)
            
            # Add confusion to similar phonemes (simulate acoustic similarity)
            if confusion_strength > 0:
                # Add some probability to random other phonemes
                confusion_phonemes = np.random.choice(
                    num_phonemes, 
                    size=min(5, num_phonemes - 1), 
                    replace=False
                )
                for conf_ph in confusion_phonemes:
                    if conf_ph != target_phoneme:
                        logits[t, conf_ph] += np.random.exponential(confusion_strength)
    
    # Add temporal smoothing (acoustic models have temporal dependencies)
    if num_time_steps > 1:
        for t in range(1, num_time_steps):
            # Slight correlation with previous time step
            smoothing_factor = 0.1
            logits[t] = (1 - smoothing_factor) * logits[t] + smoothing_factor * logits[t-1]
    
    # Add global noise
    if noise_level > 0:
        logits += np.random.normal(0, noise_level, logits.shape)
    
    return torch.from_numpy(logits).unsqueeze(0)  # Add batch dimension


def extract_ground_truth_text(transcription, sentence_label):
    """Extract ground truth text from the dataset."""
    if isinstance(sentence_label, (bytes, np.bytes_)):
        sentence_label = sentence_label.decode('utf-8')
    elif isinstance(sentence_label, str):
        pass  # Already string
    else:
        # It's an array of character codes
        sentence_label = ''.join([chr(int(code)) for code in sentence_label if int(code) != 0])
    
    return sentence_label.strip().lower()


def test_ctc_pipeline_with_ground_truth(
    num_samples: int = 6,
    perfection: float = 0.95,
    time_expansion: float = 2.0,
    noise_level: float = 0.1,
    blank_probability: float = 0.3
):
    """Test the CTC decoding pipeline using ground truth phoneme sequences."""
    print("🧪 Testing CTC Pipeline with Ground Truth Phonemes")
    print("=" * 60)
    print(f"🎯 Parameters:")
    print(f"   Samples: {num_samples}")
    print(f"   Perfection: {perfection:.2f}")
    print(f"   Time expansion: {time_expansion:.1f}x")
    print(f"   Noise level: {noise_level:.2f}")
    print(f"   Blank probability: {blank_probability:.2f}")
    
    # Load configuration
    config = DEFAULT_CONFIG
    print(f"\n📁 Using config from: {config}")
    
    # Initialize decoder
    print("\n🔧 Initializing CTC decoder...")
    decoder = CTCDecoder()  # Will use config defaults
    
    print(f"✅ Decoder initialized:")
    print(f"   Tokens: {len(decoder.tokens)}")
    print(f"   Blank idx: {decoder.blank_idx}")
    print(f"   Silence idx: {decoder.silence_idx}")
    
    # Load samples from the dataset
    data_dir = Path("data/t15_copyTask_neuralData/hdf5_data_final")
    sessions = ["t15.2023.08.11", "t15.2023.08.13"]
    
    print(f"\n📊 Loading test data from: {data_dir}")
    
    sample_data = []
    total_samples = 0
    
    for session in sessions:
        session_path = data_dir / session / "data_train.hdf5"
        if not session_path.exists():
            print(f"⚠️ Session file not found: {session_path}")
            continue
            
        print(f"📂 Loading from: {session}")
        
        with h5py.File(session_path, 'r') as f:
            # Get trials from this session
            trial_keys = [key for key in f.keys() if key.startswith('trial_')]
            
            for trial_key in trial_keys:
                trial = f[trial_key]
                
                sample_data.append({
                    'session': session,
                    'trial': trial_key,
                    'seq_class_ids': trial['seq_class_ids'][:],
                    'transcription': trial['transcription'][:] if 'transcription' in trial else None,
                    'sentence_label': trial.attrs.get('sentence_label', ''),
                    'n_time_steps': trial.attrs.get('n_time_steps', 100)
                })
                total_samples += 1
                
                if total_samples >= num_samples:
                    break
        
        if total_samples >= num_samples:
            break
    
    print(f"📈 Loaded {len(sample_data)} samples for testing")
    
    # Test each sample
    print(f"\n🔍 Testing samples...")
    
    results = []
    for i, sample in enumerate(sample_data):
        print(f"\n--- Sample {i+1}/{len(sample_data)} ---")
        print(f"Session: {sample['session']}, Trial: {sample['trial']}")
        
        # Extract ground truth and remove padding (assuming 0 is padding)
        gt_phoneme_indices = sample['seq_class_ids']
        # Remove padding (typically 0 values at the end)
        gt_phoneme_indices = gt_phoneme_indices[gt_phoneme_indices != 0]
        gt_phonemes = [LOGIT_TO_PHONEME[idx] for idx in gt_phoneme_indices if idx < len(LOGIT_TO_PHONEME)]
        gt_text = extract_ground_truth_text(sample['transcription'], sample['sentence_label'])
        
        print(f"GT phonemes ({len(gt_phonemes)} total): {' '.join(gt_phonemes[:20])}{'...' if len(gt_phonemes) > 20 else ''}")
        print(f"GT text: '{gt_text}'")
        
        # Generate logits with specified perfection
        dummy_logits = generate_logits(
            gt_phoneme_indices,
            len(LOGIT_TO_PHONEME),
            perfection=perfection,
            time_expansion=time_expansion,
            noise_level=noise_level,
            blank_probability=blank_probability
        )
        
        print(f"Logits shape: {dummy_logits.shape}")
        print(f"Expected time expansion: {len(gt_phoneme_indices)} phonemes * {time_expansion} = ~{int(len(gt_phoneme_indices) * time_expansion)} time steps")
        
        # Print some sample logits values
        print(f"Sample logits (first 5 time steps, first 10 phonemes):")
        logits_sample = dummy_logits[0, :5, :10].numpy()
        for t in range(min(5, dummy_logits.shape[1])):
            values_str = ' '.join([f"{val:6.2f}" for val in logits_sample[t]])
            print(f"  t={t:2d}: [{values_str}]")
        print(f"  Max logit in sequence: {dummy_logits.max().item():.2f}, Min: {dummy_logits.min().item():.2f}")
        
        # Decode
        try:
            decoded_results = decoder.decode(dummy_logits)
            if decoded_results and len(decoded_results) > 0:
                result = decoded_results[0] if isinstance(decoded_results, list) else decoded_results
                predicted_text = result.get('sentence', result.get('text', 'N/A'))
                predicted_tokens = result.get('tokens', [])
                
                print(f"Predicted: '{predicted_text}'")
                print(f"Tokens: {' '.join(predicted_tokens[:10])}{'...' if len(predicted_tokens) > 10 else ''}")
                
                # Calculate metrics if we have ground truth text
                if gt_text and predicted_text != 'N/A':
                    try:
                        char_error_rate = cer(gt_text, predicted_text)
                        word_error_rate = wer(gt_text, predicted_text)
                        print(f"CER: {char_error_rate:.3f}, WER: {word_error_rate:.3f}")
                        
                        results.append({
                            'sample': i + 1,
                            'gt_text': gt_text,
                            'pred_text': predicted_text,
                            'gt_phonemes': len(gt_phonemes),
                            'pred_tokens': len(predicted_tokens),
                            'cer': char_error_rate,
                            'wer': word_error_rate
                        })
                    except Exception as e:
                        print(f"⚠️ Error calculating metrics: {e}")
                        results.append({
                            'sample': i + 1,
                            'gt_text': gt_text,
                            'pred_text': predicted_text,
                            'gt_phonemes': len(gt_phonemes),
                            'pred_tokens': len(predicted_tokens),
                            'cer': float('inf'),
                            'wer': float('inf')
                        })
                else:
                    print("⚠️ No ground truth text available for metrics")
                    results.append({
                        'sample': i + 1,
                        'gt_text': gt_text,
                        'pred_text': predicted_text,
                        'gt_phonemes': len(gt_phonemes),
                        'pred_tokens': len(predicted_tokens),
                        'cer': float('inf'),
                        'wer': float('inf')
                    })
            else:
                print("❌ No decoding results")
                results.append({
                    'sample': i + 1,
                    'gt_text': gt_text,
                    'pred_text': 'N/A',
                    'gt_phonemes': len(gt_phonemes),
                    'pred_tokens': 0,
                    'cer': float('inf'),
                    'wer': float('inf')
                })
        except Exception as e:
            print(f"❌ Decoding failed: {e}")
            results.append({
                'sample': i + 1,
                'gt_text': gt_text,
                'pred_text': 'ERROR',
                'gt_phonemes': len(gt_phonemes),
                'pred_tokens': 0,
                'cer': float('inf'),
                'wer': float('inf')
            })
    
    # Summary
    print(f"\n📊 SUMMARY")
    print("=" * 60)
    
    valid_results = [r for r in results if r['cer'] != float('inf')]
    if valid_results:
        avg_cer = np.mean([r['cer'] for r in valid_results])
        avg_wer = np.mean([r['wer'] for r in valid_results])
        print(f"Average CER: {avg_cer:.3f}")
        print(f"Average WER: {avg_wer:.3f}")
        print(f"Valid samples: {len(valid_results)}/{len(results)}")
        
        if avg_wer < 0.1:  # Less than 10% error
            print("🎉 SUCCESS: Pipeline working correctly with ground truth phonemes!")
        else:
            print("⚠️ WARNING: Higher than expected error rate")
    else:
        print("❌ FAILED: No valid results obtained")
    
    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test CTC decoding pipeline with ground truth phonemes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--num_samples', 
        type=int, 
        default=6,
        help='Number of samples to test'
    )
    
    parser.add_argument(
        '--perfection', 
        type=float, 
        default=0.95,
        help='How perfect the generated logits should be (0.0-1.0)'
    )
    
    parser.add_argument(
        '--time_expansion', 
        type=float, 
        default=2.0,
        help='How much longer the time sequence should be vs phonemes'
    )
    
    parser.add_argument(
        '--noise_level', 
        type=float, 
        default=0.1,
        help='Amount of random noise to add to logits'
    )
    
    parser.add_argument(
        '--blank_probability', 
        type=float, 
        default=0.3,
        help='Probability of inserting blanks between phonemes'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    test_ctc_pipeline_with_ground_truth(
        num_samples=args.num_samples,
        perfection=args.perfection,
        time_expansion=args.time_expansion,
        noise_level=args.noise_level,
        blank_probability=args.blank_probability
    )
