"""
Generate submission.csv for test data using trained CTC model and optimized decoding parameters.
Updated to use the new decoder architecture from the decoding module.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
import argparse
import logging
from pathlib import Path
import time
import yaml

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model_training.rnn_model import GRUDecoder
from model_training.evaluate_model_helpers import load_h5py_file
from model_training.data_augmentations import gauss_smooth
from decoding.decoder import Decoder
from nejm_b2txt_utils.general_utils import remove_punctuation

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def setup_logging(output_dir: str):
    """Set up logging for the submission script."""
    os.makedirs(output_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, 'submission_generation.log'))
        ]
    )
    return logging.getLogger(__name__)


def load_model_and_data(model_path: str, data_dir: str, eval_type: str, device: torch.device):
    """Load the trained model and test/validation data."""
    logger = logging.getLogger(__name__)
    
    # Load model config
    model_args = OmegaConf.load(os.path.join(model_path, 'checkpoint/args.yaml'))
    
    # Load CSV metadata
    csv_path = os.path.join(os.path.dirname(data_dir), 't15_copyTaskData_description.csv')
    if not os.path.exists(csv_path):
        # Try alternative location
        csv_path = os.path.join(data_dir, '..', 't15_copyTaskData_description.csv')
    b2txt_csv_df = pd.read_csv(csv_path)
    
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
    checkpoint = torch.load(os.path.join(model_path, 'checkpoint/best_checkpoint'), weights_only=False)
    # Clean up keys (remove module. and _orig_mod. prefixes)
    state_dict = {}
    for key, value in checkpoint['model_state_dict'].items():
        clean_key = key.replace("module.", "").replace("_orig_mod.", "")
        state_dict[clean_key] = value
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {model_path}")
    
    # Load evaluation data
    eval_data = {}
    total_trials = 0
    
    for session in model_args['dataset']['sessions']:
        session_dir = os.path.join(data_dir, session)
        eval_file = os.path.join(session_dir, f'data_{eval_type}.hdf5')
        
        if os.path.exists(eval_file):
            data = load_h5py_file(eval_file, b2txt_csv_df)
            eval_data[session] = data
            total_trials += len(data["neural_features"])
            logger.info(f'Loaded {len(data["neural_features"])} {eval_type} trials for {session}')
    
    logger.info(f'Total {eval_type} trials: {total_trials}')
    
    return model, model_args, eval_data


def run_single_decoding_step(x, input_layer, model, model_args, device):
    """Single decoding step function - smooths data and puts it through the model."""
    # Use autocast for efficiency
    with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", 
                       enabled=model_args['use_amp'], dtype=torch.bfloat16):
        
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


def get_model_logits(model, model_args, eval_data, device):
    """Get logits from the model for all evaluation data."""
    logger = logging.getLogger(__name__)
    
    logger.info("Generating logits for evaluation data...")
    
    with torch.no_grad():
        for session, data in tqdm(eval_data.items(), desc="Sessions"):
            data['logits'] = []
            input_layer = model_args['dataset']['sessions'].index(session)
            
            for trial in tqdm(range(len(data['neural_features'])), desc=f"{session} trials", leave=False):
                # Get neural input
                neural_input = data['neural_features'][trial]
                neural_input = np.expand_dims(neural_input, axis=0)
                neural_input = torch.tensor(neural_input, device=device, dtype=torch.float32)
                
                # Run model
                logits = run_single_decoding_step(neural_input, input_layer, model, model_args, device)
                data['logits'].append(logits)
    
    logger.info("Logits generation completed")


def decode_all_trials(eval_data, decoder_config, submission_config):
    """Decode all trials with the specified decoder configuration."""
    logger = logging.getLogger(__name__)
    
    # Initialize decoder using new architecture
    decoder = Decoder(
        backend=decoder_config.get('backend', 'flashlight'),
        **{k: v for k, v in decoder_config.items() if k != 'backend'}
    )
    
    logger.info(f"Initialized {decoder.decoder_type} decoder")
    
    # Collect results in chronological order
    submission_results = []
    
    # Process sessions in order they appear in the config
    for session in eval_data.keys():
        data = eval_data[session]
        
        for trial in tqdm(range(len(data['logits'])), desc=f"Decoding {session}"):
            # Get logits
            logits = torch.tensor(data['logits'][trial][0], dtype=torch.float32)
            
            # Decode using new decoder architecture
            result = decoder.decode(logits)
            
            # Handle both single result and batch result formats
            if isinstance(result, list):
                result = result[0] if result else {'sentence': ''}
            
            pred_sentence = result.get('sentence', '') if isinstance(result, dict) else str(result)
            
            # Apply submission text normalization
            if submission_config.get('lowercase', True):
                pred_sentence = pred_sentence.lower()
            
            if submission_config.get('strip_punct', True):
                pred_sentence = remove_punctuation(pred_sentence)
            
            if submission_config.get('collapse_space', True):
                pred_sentence = ' '.join(pred_sentence.split())
            
            pred_sentence = pred_sentence.strip()
            
            # Store result with metadata for ordering
            submission_results.append({
                'session': session,
                'block_num': data['block_num'][trial],
                'trial_num': data['trial_num'][trial],
                'predicted_sentence': pred_sentence
            })
    
    logger.info(f"Decoded {len(submission_results)} trials")
    return submission_results


def create_submission_csv(submission_results, output_path: str, eval_type: str):
    """Create submission.csv in the required format."""
    logger = logging.getLogger(__name__)
    
    # Sort results to ensure chronological order
    # Sort by session order, then block, then trial
    submission_results.sort(key=lambda x: (x['session'], x['block_num'], x['trial_num']))
    
    # Create submission DataFrame
    submission_data = {
        'id': list(range(len(submission_results))),
        'text': [result['predicted_sentence'] for result in submission_results]
    }
    
    submission_df = pd.DataFrame(submission_data)
    
    # Save to CSV
    submission_df.to_csv(output_path, index=False)
    
    logger.info(f"Submission CSV saved to: {output_path}")
    logger.info(f"Total submissions: {len(submission_df)}")
    
    # Log some examples
    logger.info("Sample predictions:")
    for i in range(min(5, len(submission_df))):
        logger.info(f"  ID {i}: '{submission_df.iloc[i]['text']}'")
    
    return submission_df


def generate_submission(
    model_path: str,
    data_dir: str,
    output_dir: str,
    eval_type: str = 'test',
    config_path: str = None,
    decoder_backend: str = 'flashlight',
    device: str = 'cuda'
):
    """Generate submission file for test data."""
    logger = setup_logging(output_dir)
    
    # Set up device
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    # Load configuration
    if config_path and os.path.exists(config_path):
        config = OmegaConf.load(config_path)
    else:
        # Try to load config from decoding module
        decoding_config_path = project_root / "decoding" / "config.yaml"
        if decoding_config_path.exists():
            config = OmegaConf.load(decoding_config_path)
        else:
            # Fallback config
            config = OmegaConf.create({
                'decoding': {
                    'backend': decoder_backend,
                    'lm_weight': 1.0,
                    'word_score': -0.5,
                    'beam_size': 500,
                    'beam_size_token': 100,
                    'beam_threshold': 15.0,
                    'nbest': 1,
                    'blank_token': 'BLANK',
                    'silence_token': 'SIL',
                    'unk_word': '<unk>'
                },
                'submission': {
                    'lowercase': True,
                    'strip_punct': True,
                    'collapse_space': True
                }
            })
    
    # Extract decoder and submission configs
    if 'decoding' in config:
        decoder_config = dict(config.decoding)
    elif 'flashlight' in config:
        # Handle old config format
        decoder_config = dict(config.flashlight)
        decoder_config['backend'] = 'flashlight'
    else:
        decoder_config = {'backend': decoder_backend}
    
    submission_config = config.get('submission', {
        'lowercase': True,
        'strip_punct': True,
        'collapse_space': True
    })
    
    logger.info(f"Decoder config: {decoder_config}")
    logger.info(f"Submission config: {submission_config}")
    
    # Load model and data
    model, model_args, eval_data = load_model_and_data(model_path, data_dir, eval_type, device)
    
    # Get logits
    start_time = time.time()
    get_model_logits(model, model_args, eval_data, device)
    logits_time = time.time() - start_time
    logger.info(f"Logits generation took {logits_time:.2f} seconds")
    
    # Decode all trials
    start_time = time.time()
    submission_results = decode_all_trials(eval_data, decoder_config, submission_config)
    decode_time = time.time() - start_time
    logger.info(f"Decoding took {decode_time:.2f} seconds")
    
    # Create submission CSV
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"submission_{eval_type}_{decoder_config.get('backend', 'unknown')}_{timestamp}.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    submission_df = create_submission_csv(submission_results, output_path, eval_type)
    
    # Log summary statistics
    logger.info(f"Total processing time: {logits_time + decode_time:.2f} seconds")
    logger.info(f"Average time per trial: {(logits_time + decode_time) / len(submission_results):.3f} seconds")
    
    # Save processing log
    log_data = {
        'model_path': model_path,
        'data_dir': data_dir,
        'eval_type': eval_type,
        'total_trials': len(submission_results),
        'logits_time': logits_time,
        'decode_time': decode_time,
        'total_time': logits_time + decode_time,
        'decoder_config': decoder_config,
        'submission_config': submission_config,
        'output_file': output_path
    }
    
    log_path = os.path.join(output_dir, f"submission_log_{timestamp}.yaml")
    with open(log_path, 'w') as f:
        yaml.dump(log_data, f, default_flow_style=False)
    
    logger.info(f"Processing log saved to: {log_path}")
    
    return submission_df, output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate submission CSV for test data")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model directory')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='submission_output',
                        help='Output directory for submission files')
    parser.add_argument('--eval_type', type=str, default='test', choices=['val', 'test'],
                        help='Evaluation type: val or test')
    parser.add_argument('--config_path', type=str, default=None,
                        help='Path to config file with decoding parameters')
    parser.add_argument('--decoder_backend', type=str, default='flashlight',
                        choices=['greedy', 'flashlight'], help='Decoder backend to use')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Generate submission
    submission_df, output_path = generate_submission(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        eval_type=args.eval_type,
        config_path=args.config_path,
        decoder_backend=args.decoder_backend,
        device=args.device
    )
    
    print(f"\nSubmission generation completed!")
    print(f"Output file: {output_path}")
    print(f"Total submissions: {len(submission_df)}")
    print(f"Sample predictions:")
    for i in range(min(3, len(submission_df))):
        print(f"  ID {i}: '{submission_df.iloc[i]['text']}'")
