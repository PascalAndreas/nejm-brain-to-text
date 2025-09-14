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
from model_training.data_augmentations import gauss_smooth
from decoding.decoder import Decoder
from dataset import BrainToTextDataset, collate_fn
from torch.utils.data import DataLoader
from nejm_b2txt_utils.general_utils import remove_punctuation
from jiwer import wer, cer

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
    """Load the trained model and test/validation data using the new dataset class."""
    logger = logging.getLogger(__name__)
    
    # Load model config
    model_args = OmegaConf.load(os.path.join(model_path, 'checkpoint/args.yaml'))
    
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
    
    # Load evaluation data using new dataset class
    # data_dir points to hdf5_data_final, but dataset expects the parent directory
    dataset_root = os.path.dirname(data_dir) if data_dir.endswith('hdf5_data_final') else data_dir
    dataset = BrainToTextDataset(
        data_root=dataset_root,
        split=eval_type,
        random_seed=42  # For reproducible results
    )
    
    logger.info(f'Loaded {len(dataset)} {eval_type} trials using new dataset class')
    
    return model, model_args, dataset


def process_and_decode_streaming(model, model_args, dataset, device, submission_config, batch_size=32):
    """Stream processing: fetch batch -> RNN -> decode -> yield results."""
    logger = logging.getLogger(__name__)
    
    logger.info("Starting streaming processing: batch -> RNN -> decode -> results")
    
    # Initialize decoder once
    decoder = Decoder()
    logger.info(f"Initialized {decoder.decoder_type} decoder with optimized parameters")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Keep original order for submission
        collate_fn=collate_fn,
        num_workers=0
    )
    
    total_processed = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing & decoding batches"):
            # Step 1: RNN processing
            input_features = batch['input_features'].to(device)
            day_indices = batch['day_indices'].to(device)
            
            with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", 
                               enabled=model_args['use_amp'], dtype=torch.bfloat16):
                
                # Apply smoothing
                smoothed_input = gauss_smooth(
                    inputs=input_features, 
                    device=device,
                    smooth_kernel_std=model_args['dataset']['data_transforms']['smooth_kernel_std'],
                    smooth_kernel_size=model_args['dataset']['data_transforms']['smooth_kernel_size'],
                    padding='valid',
                )
                
                # Run model
                logits, _ = model(
                    x=smoothed_input,
                    day_idx=day_indices,
                    states=None,
                    return_state=True,
                )
            
            # Convert to float32 and move to CPU
            logits = logits.float().cpu()
            
            # Step 2: Decode each trial in the batch
            for i in range(len(batch['sessions'])):
                # Calculate the correct logit length using the same formula as training
                # IMPORTANT: In training, adjusted_lens uses the ORIGINAL n_time_steps, not smoothed length!
                # The training code applies smoothing but doesn't update n_time_steps for the length calculation
                original_length = batch['n_time_steps'][i].item()
                
                # Use the exact same formula as training: adjusted_lens = ((n_time_steps - patch_size) / patch_stride + 1)
                # This uses ORIGINAL length, not smoothed length!
                patch_size = model_args['model']['patch_size']
                patch_stride = model_args['model']['patch_stride']
                adjusted_lens = int((original_length - patch_size) / patch_stride + 1)
                
                # Ensure we don't exceed the actual logit tensor size
                trim_length = min(adjusted_lens, logits.shape[1])
                
                # Extract logits for this trial (trim to correct length - NO PADDING!)
                trial_logits = logits[i:i+1, :trim_length]  # Keep batch dimension
                logit_lengths = torch.tensor([trim_length])
                
                # Decode
                result = decoder.decode(trial_logits, logit_lengths)
                
                # Handle result format
                if isinstance(result, list):
                    result = result[0] if result else {'sentence': ''}
                
                pred_sentence = result.get('sentence', '') if isinstance(result, dict) else str(result)
                
                # Apply text normalization
                if submission_config.get('lowercase', True):
                    pred_sentence = pred_sentence.lower()
                
                if submission_config.get('strip_punct', True):
                    pred_sentence = remove_punctuation(pred_sentence)
                
                if submission_config.get('collapse_space', True):
                    pred_sentence = ' '.join(pred_sentence.split())
                
                pred_sentence = pred_sentence.strip()
                
                # Prepare result
                result_data = {
                    'session': batch['sessions'][i],
                    'block_num': batch['block_nums'][i].item(),
                    'trial_num': batch['trial_nums'][i].item(),
                    'predicted_sentence': pred_sentence,
                    'corpus': batch['corpora'][i]
                }
                
                # Add ground truth if available (validation mode)
                if 'sentence_labels' in batch:
                    result_data['ground_truth'] = batch['sentence_labels'][i]
                
                # Yield result immediately
                yield result_data
                
                total_processed += 1
    
    logger.info(f"Completed streaming processing of {total_processed} trials")


def calculate_and_log_wer(submission_results, logger, submission_config):
    """Calculate and log WER for validation results."""
    total_wer = 0.0
    total_cer = 0.0
    total_words = 0
    total_chars = 0
    individual_wers = []
    individual_cers = []
    
    logger.info("Calculating WER for validation results...")
    
    for i, result in enumerate(submission_results):
        pred_sentence = result['predicted_sentence']
        true_sentence = result['ground_truth']
        
        # Apply same text normalization to ground truth
        if submission_config.get('lowercase', True):
            true_sentence = true_sentence.lower()
        
        if submission_config.get('strip_punct', True):
            true_sentence = remove_punctuation(true_sentence)
        
        if submission_config.get('collapse_space', True):
            true_sentence = ' '.join(true_sentence.split())
        
        true_sentence = true_sentence.strip()
        
        # Calculate WER and CER for this sample
        try:
            sample_wer = wer(true_sentence, pred_sentence)
            sample_cer = cer(true_sentence, pred_sentence)
            
            # Count words and characters
            true_words = len(true_sentence.split())
            true_chars = len(true_sentence)
            
            # Accumulate for aggregate metrics
            total_wer += sample_wer * true_words
            total_cer += sample_cer * true_chars
            total_words += true_words
            total_chars += true_chars
            
            individual_wers.append(sample_wer)
            individual_cers.append(sample_cer)
            
            # Log first few examples
            if i < 5:
                logger.info(f"Sample {i}:")
                logger.info(f"  True: '{true_sentence}'")
                logger.info(f"  Pred: '{pred_sentence}'")
                logger.info(f"  WER: {sample_wer:.3f}, CER: {sample_cer:.3f}")
                
        except Exception as e:
            logger.warning(f"Error calculating metrics for sample {i}: {e}")
            individual_wers.append(1.0)
            individual_cers.append(1.0)
    
    # Calculate aggregate metrics
    if total_words > 0 and total_chars > 0:
        aggregate_wer = total_wer / total_words
        aggregate_cer = total_cer / total_chars
        mean_wer = np.mean(individual_wers)
        mean_cer = np.mean(individual_cers)
        
        logger.info(f"\n=== WER RESULTS ===")
        logger.info(f"Aggregate WER: {aggregate_wer:.4f} ({aggregate_wer*100:.2f}%)")
        logger.info(f"Aggregate CER: {aggregate_cer:.4f} ({aggregate_cer*100:.2f}%)")
        logger.info(f"Mean WER: {mean_wer:.4f} ({mean_wer*100:.2f}%)")
        logger.info(f"Mean CER: {mean_cer:.4f} ({mean_cer*100:.2f}%)")
        logger.info(f"Total samples: {len(submission_results)}")
        logger.info(f"Total words: {total_words}")
        logger.info(f"Total characters: {total_chars}")
    else:
        logger.warning("Could not calculate aggregate WER - no valid samples")


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
    
    # Load submission config from decoding config
    decoding_config_path = project_root / "decoding" / "config.yaml"
    if decoding_config_path.exists():
        config = OmegaConf.load(decoding_config_path)
        submission_config = config.get('submission', {
            'lowercase': True,
            'strip_punct': True,
            'collapse_space': True
        })
    else:
        submission_config = {
            'lowercase': True,
            'strip_punct': True,
            'collapse_space': True
        }
    
    logger.info(f"Submission config: {submission_config}")
    
    # Load model and data
    model, model_args, dataset = load_model_and_data(model_path, data_dir, eval_type, device)
    
    # Stream processing: batch -> RNN -> decode -> results
    start_time = time.time()
    submission_results = list(process_and_decode_streaming(
        model, model_args, dataset, device, submission_config
    ))
    total_time = time.time() - start_time
    logger.info(f"Total streaming processing took {total_time:.2f} seconds")
    
    # Create submission CSV
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"submission_{eval_type}_flashlight_{timestamp}.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    submission_df = create_submission_csv(submission_results, output_path, eval_type)
    
    # Calculate WER if validation mode (ground truth available)
    if eval_type == 'val' and submission_results and 'ground_truth' in submission_results[0]:
        calculate_and_log_wer(submission_results, logger, submission_config)
    
    # Log summary statistics
    logger.info(f"Average time per trial: {total_time / len(submission_results):.3f} seconds")
    
    # Save processing log
    log_data = {
        'model_path': model_path,
        'data_dir': data_dir,
        'eval_type': eval_type,
        'total_trials': len(submission_results),
        'total_time': total_time,
        'processing_type': 'streaming',
        'submission_config': dict(submission_config),  # Convert to plain dict
        'output_file': output_path
    }
    
    log_path = os.path.join(output_dir, f"submission_log_{timestamp}.yaml")
    try:
        with open(log_path, 'w') as f:
            yaml.dump(log_data, f, default_flow_style=False)
        logger.info(f"Processing log saved to: {log_path}")
    except Exception as e:
        logger.warning(f"Could not save YAML log: {e}")
        # Save as JSON instead
        import json
        log_path_json = log_path.replace('.yaml', '.json')
        with open(log_path_json, 'w') as f:
            json.dump(log_data, f, indent=2)
        logger.info(f"Processing log saved as JSON to: {log_path_json}")
    
    return submission_df, output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate submission CSV for test data")
    parser.add_argument('--model_path', type=str, 
                        default='data/t15_pretrained_rnn_baseline/t15_pretrained_rnn_baseline',
                        help='Path to trained model directory')
    parser.add_argument('--data_dir', type=str, 
                        default='data/t15_copyTask_neuralData/hdf5_data_final',
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='submission_output',
                        help='Output directory for submission files')
    parser.add_argument('--eval_type', type=str, default='test', choices=['val', 'test'],
                        help='Evaluation type: val or test')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    # Generate submission
    submission_df, output_path = generate_submission(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        eval_type=args.eval_type,
        device=args.device
    )
    
    print(f"\nSubmission generation completed!")
    print(f"Output file: {output_path}")
    print(f"Total submissions: {len(submission_df)}")
    print(f"Sample predictions:")
    for i in range(min(3, len(submission_df))):
        print(f"  ID {i}: '{submission_df.iloc[i]['text']}'")
