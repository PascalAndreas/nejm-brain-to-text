#!/usr/bin/env python3
"""
Test script to load the RNN model checkpoint and examine phoneme outputs for the first 20 samples.
"""

import os
import sys
import torch
import numpy as np
import h5py
from omegaconf import OmegaConf
from tqdm import tqdm

# Add project root to path
sys.path.append('/Users/pascalandreas/Documents/repositories/nejm-brain-to-text')

from model_training.rnn_model import GRUDecoder
from model_training.data_augmentations import gauss_smooth

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
            
            samples.append({
                'trial_key': trial_key,
                'neural_features': neural_features,
                'block_num': block_num,
                'trial_num': trial_num,
                'n_time_steps': n_time_steps,
                'gt_phonemes': gt_phonemes
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
    
    # Get argmax predictions
    pred_seq = np.argmax(logits[0], axis=-1)  # Remove batch dimension
    
    return pred_seq, logits[0]

def main():
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
    
    # Get first available session
    sessions = model_args['dataset']['sessions']
    first_session = sessions[0]
    print(f"Using session: {first_session}")
    
    # Load sample data
    samples = load_sample_data(data_dir, first_session, max_samples=20)
    
    print(f"\n{'='*80}")
    print("PHONEME PREDICTIONS FOR FIRST 20 SAMPLES")
    print(f"{'='*80}")
    
    # Process each sample
    for i, sample in enumerate(samples):
        print(f"\nSample {i+1}/20:")
        print(f"Trial: {sample['trial_key']}, Block: {sample['block_num']}, Trial num: {sample['trial_num']}")
        print(f"Neural data shape: {sample['neural_features'].shape}")
        print(f"Time steps: {sample['n_time_steps']}")
        
        # Get predictions
        day_idx = 0  # First session corresponds to day index 0
        pred_seq, logits = predict_phonemes(
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
        
        print(f"Max logit value: {np.max(logits):.3f}, Min: {np.min(logits):.3f}")
        print("-" * 60)

if __name__ == "__main__":
    main()
