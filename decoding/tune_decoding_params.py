"""
Hyperparameter tuning grid for CTC decoding parameters (lm_weight, word_score).
Runs validation decoding with different parameter combinations to find optimal settings.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import OmegaConf
from itertools import product
import argparse
import logging
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_training.rnn_model import GRUDecoder
from model_training.evaluate_model_helpers import load_h5py_file, runSingleDecodingStep
from model_training.data_augmentations import gauss_smooth  # Fix import issue
from . import Decoder
from nejm_b2txt_utils.general_utils import calculate_aggregate_error_rate, remove_punctuation

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def setup_logging():
    """Set up logging for the tuning script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('tune_decoding_params.log')
        ]
    )
    return logging.getLogger(__name__)


def load_model_and_data(model_path: str, data_dir: str, device: torch.device):
    """Load the trained model and validation data."""
    logger = logging.getLogger(__name__)
    
    # Load model config
    model_args = OmegaConf.load(os.path.join(model_path, 'checkpoint/args.yaml'))
    
    # Load CSV metadata
    csv_path = os.path.join('data', 't15_copyTaskData_description.csv')
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
    checkpoint = torch.load(os.path.join(model_path, 'checkpoint/best_checkpoint'), 
                          weights_only=False, map_location=device)
    # Clean up keys (remove module. and _orig_mod. prefixes)
    state_dict = {}
    for key, value in checkpoint['model_state_dict'].items():
        clean_key = key.replace("module.", "").replace("_orig_mod.", "")
        state_dict[clean_key] = value
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {model_path}")
    
    # Load validation data
    val_data = {}
    total_trials = 0
    
    for session in model_args['dataset']['sessions']:
        session_dir = os.path.join(data_dir, session)
        val_file = os.path.join(session_dir, 'data_val.hdf5')
        
        if os.path.exists(val_file):
            data = load_h5py_file(val_file, b2txt_csv_df)
            val_data[session] = data
            total_trials += len(data["neural_features"])
            logger.info(f'Loaded {len(data["neural_features"])} val trials for {session}')
    
    logger.info(f'Total validation trials: {total_trials}')
    
    return model, model_args, val_data


def get_model_logits(model, model_args, val_data, device):
    """Get logits from the model for all validation data."""
    logger = logging.getLogger(__name__)
    
    logger.info("Generating logits for validation data...")
    
    with torch.no_grad():
        for session, data in tqdm(val_data.items(), desc="Sessions"):
            data['logits'] = []
            input_layer = model_args['dataset']['sessions'].index(session)
            
            for trial in tqdm(range(len(data['neural_features'])), desc=f"{session} trials", leave=False):
                # Get neural input
                neural_input = data['neural_features'][trial]
                neural_input = np.expand_dims(neural_input, axis=0)
                neural_input = torch.tensor(neural_input, device=device, dtype=torch.float32)
                
                # Run model
                logits = runSingleDecodingStep(neural_input, input_layer, model, model_args, device)
                data['logits'].append(logits)
    
    logger.info("Logits generation completed")


def decode_with_params(val_data, lm_weight, word_score, config):
    """Decode validation data with specific parameters."""
    logger = logging.getLogger(__name__)
    
    # Initialize decoder with current parameters
    decoder = Decoder(
        tokens_path=config.artifacts.tokens_txt,
        lexicon_path=config.artifacts.lexicon_txt,
        lm_path=config.language_model.active_model,
        lm_weight=lm_weight,
        word_score=word_score,
        beam_size=config.decoder.get('beam_size', 17),
        beam_size_token=config.decoder.get('beam_size_token', 10),
        beam_threshold=config.decoder.get('beam_threshold', 8.0),
        nbest=config.decoder.get('nbest', 100),
        blank_token=config.decoder.get('blank_token', '<blk>'),
        silence_token=config.decoder.get('silence_token', 'SIL'),
        unk_word=config.decoder.get('unk_word', '<unk>')
    )
    
    # Collect predictions and ground truth
    predictions = []
    ground_truth = []
    
    for session, data in val_data.items():
        for trial in range(len(data['logits'])):
            # Get logits
            logits = torch.tensor(data['logits'][trial][0], dtype=torch.float32)
            
            # Decode
            result = decoder.decode(logits)
            pred_sentence = result['sentence'] if isinstance(result, dict) else result.get('sentence', '')
            
            # Clean prediction
            pred_sentence = remove_punctuation(pred_sentence).strip()
            
            # Get ground truth
            true_sentence = remove_punctuation(data['sentence_label'][trial]).strip()
            
            predictions.append(pred_sentence)
            ground_truth.append(true_sentence)
    
    return predictions, ground_truth


def calculate_metrics(predictions, ground_truth):
    """Calculate WER and CER metrics."""
    import jiwer
    
    # Calculate WER using jiwer
    wer = jiwer.wer(ground_truth, predictions)
    
    # Calculate CER using jiwer  
    cer = jiwer.cer(ground_truth, predictions)
    
    return {
        'WER': wer,
        'CER': cer,
        'num_samples': len(predictions)
    }


def run_parameter_grid(
    model_path: str,
    data_dir: str,
    output_dir: str,
    lm_weights: list,
    word_scores: list,
    beam_sizes: list = None,
    device: str = 'cuda'
):
    """Run hyperparameter grid search."""
    logger = setup_logging()
    
    # Set up device
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and data
    model, model_args, val_data = load_model_and_data(model_path, data_dir, device)
    
    # Get logits (only once)
    get_model_logits(model, model_args, val_data, device)
    
    # Load decoder config from decoding/config.yaml
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    if os.path.exists(config_path):
        config = OmegaConf.load(config_path)
        decoder_config = config.decoder
    else:
        # Fallback decoder config
        decoder_config = {
            'lm_weight': 0.35,
            'word_score': -90.0,
            'beam_size': 17,
            'beam_size_token': 10,
            'beam_threshold': 8.0,
            'nbest': 100,
            'blank_token': '<blk>',
            'silence_token': 'SIL',
            'unk_word': '<unk>'
        }
        # Fallback paths
        config = OmegaConf.create({
            'artifacts': {
                'tokens_txt': 'artifacts/tokens.txt',
                'lexicon_txt': 'artifacts/lexicon.txt'
            },
            'language_model': {
                'active_model': 'models/language_models/nvidia_tao_v4.1/4gram-pruned-0_2_7_9-en-lm-set-1.0.bin'
            },
            'decoder': decoder_config
        })
    
    # Initialize W&B if available
    wandb_run = None
    if WANDB_AVAILABLE:
        try:
            wandb_run = wandb.init(
                project="nejm-brain-to-text-tuning",
                name="decoding_param_grid_search",
                config={
                    'lm_weights': lm_weights,
                    'word_scores': word_scores,
                    'beam_sizes': beam_sizes or [config.decoder.get('beam_size', 17)],
                    'model_path': model_path,
                    'decoder_config': OmegaConf.to_yaml(config.decoder)
                }
            )
            logger.info(f"Initialized W&B run: {wandb_run.name}")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
    
    # Run grid search
    results = []
    best_wer = float('inf')
    best_params = None
    
    # Use beam_sizes if provided, otherwise just use the default
    beam_sizes = beam_sizes or [config.decoder.get('beam_size', 17)]
    
    total_combinations = len(lm_weights) * len(word_scores) * len(beam_sizes)
    logger.info(f"Running grid search over {total_combinations} parameter combinations")
    
    with tqdm(total=total_combinations, desc="Parameter combinations") as pbar:
        for lm_weight, word_score, beam_size in product(lm_weights, word_scores, beam_sizes):
            logger.info(f"Testing: lm_weight={lm_weight}, word_score={word_score}, beam_size={beam_size}")
            
            # Update decoder config for this iteration
            current_config = OmegaConf.create(config)
            current_config.decoder.beam_size = beam_size
            
            try:
                # Decode with current parameters
                predictions, ground_truth = decode_with_params(
                    val_data, lm_weight, word_score, current_config
                )
                
                # Calculate metrics
                metrics = calculate_metrics(predictions, ground_truth)
                
                # Store results
                result = {
                    'lm_weight': lm_weight,
                    'word_score': word_score,
                    'beam_size': beam_size,
                    'WER': metrics['WER'],
                    'CER': metrics['CER'],
                    'num_samples': metrics['num_samples']
                }
                results.append(result)
                
                # Log to W&B
                if wandb_run:
                    wandb_run.log(result)
                
                # Track best result
                if metrics['WER'] < best_wer:
                    best_wer = metrics['WER']
                    best_params = result.copy()
                    logger.info(f"New best WER: {best_wer:.4f} with params: {best_params}")
                
                logger.info(f"WER: {metrics['WER']:.4f}, CER: {metrics['CER']:.4f}")
                
            except Exception as e:
                logger.error(f"Error with params lm_weight={lm_weight}, word_score={word_score}, beam_size={beam_size}: {e}")
                
            pbar.update(1)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, 'decoding_param_grid_results.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved to: {results_path}")
    
    # Log best parameters
    if best_params:
        logger.info(f"Best parameters: {best_params}")
        
        # Save best parameters to YAML
        best_params_path = os.path.join(output_dir, 'best_decoding_params.yaml')
        with open(best_params_path, 'w') as f:
            import yaml
            yaml.dump(best_params, f, default_flow_style=False)
        logger.info(f"Best parameters saved to: {best_params_path}")
        
        if wandb_run:
            wandb_run.log({
                'best_WER': best_params['WER'],
                'best_CER': best_params['CER'],
                'best_lm_weight': best_params['lm_weight'],
                'best_word_score': best_params['word_score'],
                'best_beam_size': best_params['beam_size']
            })
    
    if wandb_run:
        wandb_run.finish()
    
    return results_df, best_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune CTC decoding parameters")
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model directory')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data directory containing validation data')
    parser.add_argument('--output_dir', type=str, default='tuning_results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to use')
    
    # Parameter ranges
    parser.add_argument('--lm_weights', nargs='+', type=float,
                        default=[0.1, 0.2, 0.35, 0.5, 1.0, 2.0, 3.0],
                        help='LM weights to test')
    parser.add_argument('--word_scores', nargs='+', type=float,
                        default=[-100.0, -90.0, -50.0, -10.0, -1.0, -0.5, 0.0, 0.5],
                        help='Word scores to test')
    parser.add_argument('--beam_sizes', nargs='+', type=int,
                        default=None, help='Beam sizes to test (optional)')
    
    args = parser.parse_args()
    
    # Run grid search
    results_df, best_params = run_parameter_grid(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        lm_weights=args.lm_weights,
        word_scores=args.word_scores,
        beam_sizes=args.beam_sizes,
        device=args.device
    )
    
    print(f"\nGrid search completed!")
    print(f"Best parameters: {best_params}")
    print(f"Results saved to: {args.output_dir}")
