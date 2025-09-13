#!/usr/bin/env python3
"""
Test script to load the RNN model checkpoint and examine phoneme outputs.
Tests the full pipeline: neural data -> RNN logits -> CTC decoder -> text.

Usage:
    python test_rnn_model_phonemes.py [--num_samples N] [--decode_logits]

Options:
    --decode_logits: Also test CTC decoding of model logits (requires decoder setup)
"""

import os
import sys
import torch
import numpy as np
import h5py
import argparse
from omegaconf import OmegaConf
from tqdm import tqdm
from pathlib import Path
from jiwer import cer, wer

# Add project root to path
sys.path.append('/Users/pascalandreas/Documents/repositories/nejm-brain-to-text')

from model_training.rnn_model import GRUDecoder
from model_training.data_augmentations import gauss_smooth
from decoding import Decoder
from decoding import DEFAULT_CONFIG
from model_training.evaluate_model_helpers import LOGIT_TO_PHONEME
from nejm_b2txt_utils.general_utils import remove_punctuation

def runSingleDecodingStep(x, input_layer, model, model_args, device):
    """Single decoding step function - smooths data and puts it through the model."""
    # Use autocast for efficiency
    with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=model_args['use_amp'], dtype=torch.bfloat16):
        
        x = gauss_smooth(
            inputs=x, 
            device=device,
            smooth_kernel_std=model_args['dataset']['data_transforms']['smooth_kernel_std'],
            smooth_kernel_size=model_args['dataset']['data_transforms']['smooth_kernel_size'],
            padding='valid',
        )

        with torch.no_grad():
            logits, _ = model(
                x=x,
                day_idx=torch.tensor([input_layer], device=device),
                states=None,  # no initial states
                return_state=True,
            )

    # convert logits from bfloat16 to float32
    logits = logits.float().cpu().numpy()

    return logits

def load_phoneme_mapping(tokens_path):
    """Load phoneme tokens and create index to phoneme mapping."""
    with open(tokens_path, 'r') as f:
        phonemes = [line.strip() for line in f.readlines()]

    # Create mapping from index to phoneme
    idx_to_phoneme = {i: phoneme for i, phoneme in enumerate(phonemes)}
    return idx_to_phoneme


def extract_ground_truth_text(transcription, sentence_label):
    """Extract ground truth text from the dataset."""
    if isinstance(sentence_label, (bytes, np.bytes_)):
        sentence_label = sentence_label.decode('utf-8')
    elif isinstance(sentence_label, str):
        pass  # Already string
    else:
        # It's an array of character codes
        sentence_label = ''.join([chr(int(code)) for code in sentence_label if int(code) != 0])

    # Remove punctuation and normalize
    text = remove_punctuation(sentence_label.strip())
    return text


def decode_logits_and_evaluate(logits, gt_text, decoder):
    """Decode logits using CTC decoder and evaluate against ground truth."""
    try:
        decoded_results = decoder.decode(logits)
        if decoded_results:
            # Handle different possible return formats
            if isinstance(decoded_results, list) and len(decoded_results) > 0:
                result = decoded_results[0]
            else:
                result = decoded_results

            # Try different possible field names for the predicted text
            predicted_text = None
            for field in ['sentence', 'text', 'prediction']:
                if hasattr(result, field):
                    predicted_text = getattr(result, field)
                    break
                elif isinstance(result, dict) and field in result:
                    predicted_text = result[field]
                    break

            if predicted_text is None:
                predicted_text = str(result)

            if gt_text and predicted_text and predicted_text != 'N/A':
                try:
                    char_error_rate = cer(gt_text, predicted_text)
                    word_error_rate = wer(gt_text, predicted_text)
                    return {
                        'predicted_text': predicted_text,
                        'cer': char_error_rate,
                        'wer': word_error_rate,
                        'success': True
                    }
                except Exception as e:
                    return {
                        'predicted_text': predicted_text,
                        'cer': float('inf'),
                        'wer': float('inf'),
                        'success': False,
                        'error': f'Metrics calculation failed: {str(e)}'
                    }
            else:
                return {
                    'predicted_text': predicted_text if predicted_text else 'N/A',
                    'cer': float('inf'),
                    'wer': float('inf'),
                    'success': False,
                    'error': 'No ground truth text or no predicted text'
                }
        else:
            return {
                'predicted_text': 'N/A',
                'cer': float('inf'),
                'wer': float('inf'),
                'success': False,
                'error': 'No decoding results'
            }
    except Exception as e:
        return {
            'predicted_text': 'ERROR',
            'cer': float('inf'),
            'wer': float('inf'),
            'success': False,
            'error': str(e)
        }

def load_model_and_config(model_path):
    """Load the RNN model and its configuration."""
    # Load model config
    config_path = os.path.join(model_path, 'checkpoint/args.yaml')
    model_args = OmegaConf.load(config_path)
    
    # Initialize model
    model = GRUDecoder(
        neural_dim=model_args['model']['n_input_features'],
        n_units=model_args['model']['n_units'], 
        n_days=len(model_args['dataset']['sessions']),
        n_classes=model_args['dataset']['n_classes'],
        rnn_dropout=model_args['model']['rnn_dropout'],
        input_dropout=model_args['model']['input_network']['input_layer_dropout'],
        n_layers=model_args['model']['n_layers'],
        patch_size=model_args['model']['patch_size'],
        patch_stride=model_args['model']['patch_stride'],
    )
    
    # Load model weights
    checkpoint_path = os.path.join(model_path, 'checkpoint/best_checkpoint')
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
    
    # Clean up keys (remove module. and _orig_mod. prefixes)
    state_dict = {}
    for key, value in checkpoint['model_state_dict'].items():
        clean_key = key.replace("module.", "").replace("_orig_mod.", "")
        state_dict[clean_key] = value
    
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"Model parameters:")
    print(f"  - Neural dimensions: {model_args['model']['n_input_features']}")
    print(f"  - RNN units: {model_args['model']['n_units']}")
    print(f"  - RNN layers: {model_args['model']['n_layers']}")
    print(f"  - Number of classes: {model_args['dataset']['n_classes']}")
    print(f"  - Number of days: {len(model_args['dataset']['sessions'])}")
    
    return model, model_args

def load_sample_data(data_dir, session_name, max_samples=20):
    """Load sample neural data from the first available session."""
    session_path = os.path.join(data_dir, session_name)
    
    # Try different data file types
    for data_type in ['data_test.hdf5', 'data_val.hdf5', 'data_train.hdf5']:
        data_file = os.path.join(session_path, data_type)
        if os.path.exists(data_file):
            print(f"Loading data from: {data_file}")
            break
    else:
        raise FileNotFoundError(f"No data files found in {session_path}")
    
    samples = []
    with h5py.File(data_file, 'r') as f:
        trial_keys = [k for k in f.keys() if k.startswith('trial_')]
        trial_keys = sorted(trial_keys)[:max_samples]  # Get first 20 trials
        
        for trial_key in trial_keys:
            trial_group = f[trial_key]
            
            # Extract neural features
            neural_features = trial_group['input_features'][:]
            
            # Extract metadata
            block_num = trial_group.attrs.get('block_num', -1)
            trial_num = trial_group.attrs.get('trial_num', -1)
            n_time_steps = trial_group.attrs.get('n_time_steps', neural_features.shape[0])
            
            # Extract ground truth phoneme sequence if available
            gt_phonemes = None
            if 'seq_class_ids' in trial_group:
                gt_phonemes = trial_group['seq_class_ids'][:]
                seq_len = trial_group.attrs.get('seq_len', len(gt_phonemes))
                gt_phonemes = gt_phonemes[:seq_len]  # Trim to actual length

            # Extract transcription and sentence label for ground truth text
            transcription = None
            sentence_label = trial_group.attrs.get('sentence_label', '')
            if 'transcription' in trial_group:
                transcription = trial_group['transcription'][:]

            samples.append({
                'trial_key': trial_key,
                'neural_features': neural_features,
                'block_num': block_num,
                'trial_num': trial_num,
                'n_time_steps': n_time_steps,
                'gt_phonemes': gt_phonemes,
                'transcription': transcription,
                'sentence_label': sentence_label
            })
    
    print(f"Loaded {len(samples)} samples from {data_file}")
    return samples

def predict_phonemes(model, model_args, neural_data, day_idx, device):
    """Run model inference on neural data to get phoneme predictions."""
    # Add batch dimension
    neural_input = np.expand_dims(neural_data, axis=0)

    # Convert to torch tensor
    neural_input = torch.tensor(neural_input, device=device, dtype=torch.float32)

    # Run decoding step
    logits = runSingleDecodingStep(neural_input, day_idx, model, model_args, device)

    # Convert logits to torch tensor for decoder compatibility
    logits_tensor = torch.from_numpy(logits)

    # Ensure correct shape for decoder (batch_size, seq_len, vocab_size)
    if len(logits_tensor.shape) == 2:  # (seq_len, vocab_size)
        logits_tensor = logits_tensor.unsqueeze(0)  # Add batch dimension -> (1, seq_len, vocab_size)
    elif len(logits_tensor.shape) == 3:  # Already (1, seq_len, vocab_size) or (batch, seq_len, vocab_size)
        pass  # Shape is already correct
    elif len(logits_tensor.shape) == 4:  # (1, 1, seq_len, vocab_size) - remove extra batch dim
        logits_tensor = logits_tensor.squeeze(0)

    # Get argmax predictions
    pred_seq = np.argmax(logits[0], axis=-1)  # Remove batch dimension

    return pred_seq, logits[0], logits_tensor

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test RNN model phoneme predictions')
    parser.add_argument('--num_samples', '-n', type=int, default=20,
                        help='Number of samples to process (default: 20)')
    parser.add_argument('--decode_logits', action='store_true',
                        help='Also test CTC decoding of model logits and calculate WER/CER')
    args = parser.parse_args()

    # Paths
    model_path = '/Users/pascalandreas/Documents/repositories/nejm-brain-to-text/data/t15_pretrained_rnn_baseline/t15_pretrained_rnn_baseline'
    data_dir = '/Users/pascalandreas/Documents/repositories/nejm-brain-to-text/data/t15_copyTask_neuralData/hdf5_data_final'
    tokens_path = '/Users/pascalandreas/Documents/repositories/nejm-brain-to-text/artifacts/tokens.txt'
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load phoneme mapping
    idx_to_phoneme = load_phoneme_mapping(tokens_path)
    print(f"Loaded {len(idx_to_phoneme)} phonemes")
    
    # Load model
    model, model_args = load_model_and_config(model_path)
    model.to(device)

    # Initialize decoder if requested
    decoder = None
    if args.decode_logits:
        print("Initializing CTC decoder...")
        decoder = Decoder()
        print(f"Decoder initialized with {len(decoder.tokens)} tokens")

    # Get first available session
    sessions = model_args['dataset']['sessions']
    first_session = sessions[0]
    print(f"Using session: {first_session}")

    # Load sample data
    samples = load_sample_data(data_dir, first_session, max_samples=args.num_samples)
    
    print(f"\n{'='*80}")
    print(f"PHONEME PREDICTIONS FOR FIRST {args.num_samples} SAMPLES")
    print(f"{'='*80}")
    
    # Initialize results tracking if decoding
    results = [] if args.decode_logits else None

    # Process each sample
    for i, sample in enumerate(samples):
        print(f"\nSample {i+1}/{len(samples)}:")
        print(f"Trial: {sample['trial_key']}, Block: {sample['block_num']}, Trial num: {sample['trial_num']}")
        print(f"Neural data shape: {sample['neural_features'].shape}")
        print(f"Time steps: {sample['n_time_steps']}")

        # Get predictions
        day_idx = 0  # First session corresponds to day index 0
        pred_seq, logits, logits_tensor = predict_phonemes(
            model, model_args, sample['neural_features'], day_idx, device
        )

        # Convert predictions to phonemes
        pred_phonemes = [idx_to_phoneme[idx] for idx in pred_seq]

        # Remove consecutive duplicates and blanks for cleaner output
        cleaned_pred = []
        for j, phoneme in enumerate(pred_phonemes):
            if phoneme != 'BLANK' and (j == 0 or phoneme != pred_phonemes[j-1]):
                cleaned_pred.append(phoneme)

        print(f"Raw predictions (first 50): {pred_seq[:50]}")
        print(f"Predicted phonemes (cleaned): {' '.join(cleaned_pred[:20])}")  # Show first 20 clean phonemes

        # Show ground truth if available
        if sample['gt_phonemes'] is not None:
            gt_phonemes = [idx_to_phoneme[idx] for idx in sample['gt_phonemes']]
            print(f"Ground truth phonemes: {' '.join(gt_phonemes)}")

        # Extract ground truth text
        gt_text = extract_ground_truth_text(sample['transcription'], sample['sentence_label'])
        if gt_text:
            print(f"Ground truth text: '{gt_text}'")

        # Decode logits and evaluate if requested
        if args.decode_logits and decoder is not None:
            print("\n--- CTC Decoding ---")
            decode_result = decode_logits_and_evaluate(logits_tensor, gt_text, decoder)
            print(f"Predicted text: '{decode_result['predicted_text']}'")
            if decode_result['success']:
                print(f"CER: {decode_result['cer']:.3f}, WER: {decode_result['wer']:.3f}")
            else:
                print(f"Decoding failed: {decode_result.get('error', 'Unknown error')}")
            results.append(decode_result)

        print(f"Max logit value: {np.max(logits):.3f}, Min: {np.min(logits):.3f}")
        print("-" * 60)

    # Print summary if decoding was performed
    if args.decode_logits and results:
        valid_results = [r for r in results if r['success']]
        if valid_results:
            avg_cer = np.mean([r['cer'] for r in valid_results])
            avg_wer = np.mean([r['wer'] for r in valid_results])

            print(f"\n{'='*80}")
            print("DECODING SUMMARY")
            print(f"{'='*80}")
            print(f"Average CER: {avg_cer:.3f}")
            print(f"Average WER: {avg_wer:.3f}")
            print(f"Valid samples: {len(valid_results)}/{len(results)}")

            if avg_wer < 0.5:  # Less than 50% error
                print("🎉 DECODING PERFORMANCE: Good results!")
            elif avg_wer < 0.8:
                print("⚠️ DECODING PERFORMANCE: Moderate results")
            else:
                print("❌ DECODING PERFORMANCE: Poor results - high error rate")
        else:
            print(f"\n❌ DECODING SUMMARY: No valid results obtained from {len(results)} samples")

if __name__ == "__main__":
    main()
