"""
Pytest configuration and fixtures for decoder tests.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model_training.rnn_model import GRUDecoder
from model_training.dataset import BrainToTextDataset
from decoding.decoder import Decoder


@pytest.fixture
def sample_logits():
    """Generate sample logits for testing."""
    # Create realistic-looking logits: batch_size=2, seq_len=100, vocab_size=41
    batch_size, seq_len, vocab_size = 2, 100, 41
    
    # Generate logits with some structure (not completely random)
    logits = torch.randn(batch_size, seq_len, vocab_size) * 2.0
    
    # Make blank token (index 0) more likely at some positions
    blank_positions = torch.randint(0, seq_len, (batch_size, seq_len // 4))
    for b in range(batch_size):
        logits[b, blank_positions[b], 0] += 3.0
    
    # Add some realistic sequence lengths
    logit_lengths = torch.tensor([seq_len - 10, seq_len - 5])
    
    return logits, logit_lengths


@pytest.fixture
def rnn_model_checkpoint():
    """Path to the RNN model checkpoint."""
    checkpoint_path = project_root / "data" / "t15_pretrained_rnn_baseline" / "t15_pretrained_rnn_baseline" / "checkpoint" / "best_checkpoint"
    if not checkpoint_path.exists():
        pytest.skip(f"RNN checkpoint not found at {checkpoint_path}")
    return checkpoint_path


@pytest.fixture
def rnn_model_args():
    """Load RNN model arguments from checkpoint."""
    args_path = project_root / "data" / "t15_pretrained_rnn_baseline" / "t15_pretrained_rnn_baseline" / "checkpoint" / "args.yaml"
    if not args_path.exists():
        pytest.skip(f"RNN args not found at {args_path}")
    
    from omegaconf import OmegaConf
    args = OmegaConf.load(args_path)
    return args


@pytest.fixture
def loaded_rnn_model(rnn_model_checkpoint, rnn_model_args):
    """Load the pretrained RNN model using the same approach as test_rnn_model_phonemes.py."""
    try:
        # Create model with saved args (using correct field names)
        model = GRUDecoder(
            neural_dim=rnn_model_args['model']['n_input_features'],
            n_units=rnn_model_args['model']['n_units'], 
            n_days=len(rnn_model_args['dataset']['sessions']),
            n_classes=rnn_model_args['dataset']['n_classes'],
            rnn_dropout=rnn_model_args['model']['rnn_dropout'],
            input_dropout=rnn_model_args['model']['input_network']['input_layer_dropout'],
            n_layers=rnn_model_args['model']['n_layers'],
            patch_size=rnn_model_args['model']['patch_size'],
            patch_stride=rnn_model_args['model']['patch_stride'],
        )
        
        # Load checkpoint and clean up keys
        checkpoint = torch.load(rnn_model_checkpoint, weights_only=False, map_location='cpu')
        
        # Clean up keys (remove module. and _orig_mod. prefixes)
        state_dict = {}
        for key, value in checkpoint['model_state_dict'].items():
            clean_key = key.replace("module.", "").replace("_orig_mod.", "")
            state_dict[clean_key] = value
        
        model.load_state_dict(state_dict)
        model.eval()
        
        return model, rnn_model_args
    except Exception as e:
        pytest.skip(f"Failed to load RNN model: {e}")


@pytest.fixture
def sample_neural_data(rnn_model_args):
    """Generate sample neural data for RNN model with correct dimensions."""
    # Use actual model dimensions
    neural_dim = rnn_model_args['model']['n_input_features']
    batch_size, seq_len = 2, 100
    day_indices = torch.tensor([0, 0])  # Use same day for both samples
    
    # Create more structured synthetic data that's more likely to produce meaningful outputs
    neural_data = torch.randn(batch_size, seq_len, neural_dim) * 0.5
    # Add some structure to make it more realistic (sinusoidal patterns)
    time_steps = torch.arange(seq_len).float().unsqueeze(0).unsqueeze(-1)
    neural_data[:, :, :10] += torch.sin(time_steps * 0.1)
    neural_data[:, :, 10:20] += torch.cos(time_steps * 0.05)
    
    return neural_data, day_indices


@pytest.fixture
def real_neural_data_sample():
    """Load a small sample of real neural data for testing."""
    data_dir = project_root / "data" / "t15_copyTask_neuralData" / "hdf5_data_final"
    
    # Find first available session
    session_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('t15.')]
    if not session_dirs:
        pytest.skip("No neural data sessions found")
    
    session_dir = session_dirs[0]  # Use first session
    
    # Try to find a data file
    for data_file in ['data_test.hdf5', 'data_val.hdf5', 'data_train.hdf5']:
        data_path = session_dir / data_file
        if data_path.exists():
            break
    else:
        pytest.skip(f"No data files found in {session_dir}")
    
    try:
        import h5py
        with h5py.File(data_path, 'r') as f:
            # Get first trial
            trial_keys = [k for k in f.keys() if k.startswith('trial_')]
            if not trial_keys:
                pytest.skip("No trials found in data file")
            
            trial_key = trial_keys[0]
            trial_group = f[trial_key]
            
            # Load neural features
            neural_features = trial_group['input_features'][:]
            
            # Load ground truth if available
            gt_phonemes = None
            if 'seq_class_ids' in trial_group:
                gt_phonemes = trial_group['seq_class_ids'][:]
                seq_len = trial_group.attrs.get('seq_len', len(gt_phonemes))
                gt_phonemes = gt_phonemes[:seq_len]
            
            # Load text labels
            sentence_label = trial_group.attrs.get('sentence_label', '')
            transcription = None
            if 'transcription' in trial_group:
                transcription = trial_group['transcription'][:]
            
            return {
                'neural_features': neural_features,
                'gt_phonemes': gt_phonemes,
                'sentence_label': sentence_label,
                'transcription': transcription,
                'trial_key': trial_key
            }
    except Exception as e:
        pytest.skip(f"Failed to load real neural data: {e}")


@pytest.fixture
def greedy_decoder():
    """Create a greedy decoder for testing."""
    return Decoder(backend='greedy')


@pytest.fixture
def flashlight_decoder():
    """Create a flashlight decoder for testing (if available)."""
    try:
        return Decoder(backend='flashlight')
    except ImportError:
        pytest.skip("Flashlight decoder not available")
