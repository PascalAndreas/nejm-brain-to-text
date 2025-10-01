#!/usr/bin/env python3
"""
Generate submission.csv for test data using cached emissions and optimized decoder parameters.

This script uses the new optimized approach:
1. Uses cached emissions (no model inference needed)
2. Uses optimized decoder hyperparameters from tuning results
3. Generates proper CSV format for submission
"""

import os
import sys
import json
import string
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any
from tqdm import tqdm
import argparse
import logging
import time
from omegaconf import OmegaConf

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.emit import EmissionCache
from decoding.decoder import OptimizedCTCDecoder
from jiwer import wer, cer


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


def strip_punctuation_and_normalize(text: str) -> str:
    """Strip punctuation and normalize text for evaluation."""
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    text = text.lower()
    return text.strip()


def load_hyperparameters(tuning_results_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load hyperparameters from tuning results or use defaults."""
    if tuning_results_dir and Path(tuning_results_dir).exists():
        best_profile_path = Path(tuning_results_dir) / "best_profile.json"
        if best_profile_path.exists():
            with open(best_profile_path, 'r') as f:
                profile = json.load(f)
            
            params = {
                'lm_weight': profile['lm_weight'],
                'word_score': profile['word_score'],
                'sil_score': profile['sil_score'],
                'beam_size': profile['beam_size'],
                'beam_size_token': profile['beam_size_token'],
                'beam_threshold': profile['beam_threshold']
            }
            print(f"Loaded optimized parameters from {best_profile_path}")
            print(f"  WER achieved: {profile['performance']['wer']:.4f}")
            return params
    
    # Default parameters (fallback)
    print("Using default parameters (no tuning results found)")
    return {
        'lm_weight': 3.75,
        'word_score': -1.0,
        'sil_score': 0.0,
        'beam_size': 150,
        'beam_size_token': 25,
        'beam_threshold': 24.0
    }


def process_emissions_to_predictions(
    emissions_dict: Dict[str, Any], 
    decoder: OptimizedCTCDecoder,
    split: str = 'test'
) -> List[Dict[str, Any]]:
    """Process cached emissions to generate predictions."""
    logger = logging.getLogger(__name__)
    
    utterance_ids = list(emissions_dict.keys())
    logger.info(f"Processing {len(utterance_ids)} {split} utterances...")
    
    results = []
    successful_decodes = 0
    failed_decodes = 0
    
    for utt_id in tqdm(utterance_ids, desc=f"Decoding {split} utterances"):
        emission = emissions_dict[utt_id]
        
        try:
            # Prepare inputs
            log_probs = emission['log_probs'].unsqueeze(0)
            out_len = torch.tensor([emission['out_len']])
            
            # Decode
            result = decoder.decode(log_probs, out_len)
            
            # Extract prediction
            if isinstance(result, list):
                result = result[0] if result else {'sentence': ''}
            pred_sentence = result.get('sentence', '') if isinstance(result, dict) else str(result)
            
            # Apply text normalization
            pred_sentence = strip_punctuation_and_normalize(pred_sentence)
            
            # Get metadata
            meta = emission.get('meta', {})
            
            result_data = {
                'utterance_id': utt_id,
                'predicted_sentence': pred_sentence,
                'session': meta.get('session', ''),
                'block_num': meta.get('block_num', 0),
                'trial_num': meta.get('trial_num', 0),
                'corpus': meta.get('corpus', '')
            }
            
            # Add ground truth if available (for validation)
            if 'ground_truth' in meta:
                result_data['ground_truth'] = meta['ground_truth']
            
            results.append(result_data)
            successful_decodes += 1
            
        except Exception as e:
            logger.warning(f"Failed to decode utterance {utt_id}: {e}")
            # Still add a result with empty prediction to maintain order
            meta = emission.get('meta', {})
            result_data = {
                'utterance_id': utt_id,
                'predicted_sentence': '',
                'session': meta.get('session', ''),
                'block_num': meta.get('block_num', 0),
                'trial_num': meta.get('trial_num', 0),
                'corpus': meta.get('corpus', '')
            }
            if 'ground_truth' in meta:
                result_data['ground_truth'] = meta['ground_truth']
            results.append(result_data)
            failed_decodes += 1
    
    logger.info(f"Decoding complete: {successful_decodes} successful, {failed_decodes} failed")
    return results


def calculate_validation_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate WER/CER metrics for validation results."""
    logger = logging.getLogger(__name__)
    
    if not results or 'ground_truth' not in results[0]:
        logger.info("No ground truth available - skipping metrics calculation")
        return {}
    
    logger.info("Calculating validation metrics...")
    
    wer_scores = []
    cer_scores = []
    
    for result in results:
        pred_sentence = result['predicted_sentence']
        true_sentence = strip_punctuation_and_normalize(result['ground_truth'])
        
        if pred_sentence and true_sentence:
            try:
                sample_wer = wer(true_sentence, pred_sentence)
                sample_cer = cer(true_sentence, pred_sentence)
                wer_scores.append(sample_wer)
                cer_scores.append(sample_cer)
            except Exception as e:
                logger.warning(f"Error calculating metrics: {e}")
                wer_scores.append(1.0)
                cer_scores.append(1.0)
    
    if wer_scores:
        metrics = {
            'mean_wer': float(sum(wer_scores) / len(wer_scores)),
            'mean_cer': float(sum(cer_scores) / len(cer_scores)),
            'total_samples': len(wer_scores),
            'perfect_samples': sum(1 for w in wer_scores if w == 0.0)
        }
        
        logger.info(f"Validation metrics:")
        logger.info(f"  Mean WER: {metrics['mean_wer']:.4f} ({metrics['mean_wer']*100:.2f}%)")
        logger.info(f"  Mean CER: {metrics['mean_cer']:.4f} ({metrics['mean_cer']*100:.2f}%)")
        logger.info(f"  Perfect samples: {metrics['perfect_samples']}/{metrics['total_samples']} ({metrics['perfect_samples']/metrics['total_samples']*100:.1f}%)")
        
        return metrics
    
    return {}


def create_submission_csv(results: List[Dict[str, Any]], output_path: str) -> pd.DataFrame:
    """Create submission.csv in the required format."""
    logger = logging.getLogger(__name__)
    
    # Sort results to ensure proper order
    # Sort by session, then block, then trial
    results.sort(key=lambda x: (x.get('session', ''), x.get('block_num', 0), x.get('trial_num', 0)))
    
    # Create submission DataFrame with required format
    submission_data = {
        'id': list(range(len(results))),
        'text': [result['predicted_sentence'] for result in results]
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
    cache_dir: str = 'cache/emissions',
    split: str = 'test',
    output_dir: str = 'submission_output',
    tuning_results_dir: Optional[str] = None,
    decoder_params: Optional[Dict[str, Any]] = None
) -> tuple[pd.DataFrame, str]:
    """Generate submission file using cached emissions and optimized decoder."""
    logger = setup_logging(output_dir)
    
    logger.info("🚀 Starting optimized submission generation")
    logger.info(f"Split: {split}")
    logger.info(f"Cache dir: {cache_dir}")
    logger.info(f"Output dir: {output_dir}")
    
    # Load cached emissions
    logger.info("Loading cached emissions...")
    emission_cache = EmissionCache(cache_dir=cache_dir)
    
    try:
        emissions_dict = emission_cache.load_emissions(split)
        logger.info(f"Loaded {len(emissions_dict)} {split} emissions")
    except Exception as e:
        logger.error(f"Failed to load cached emissions: {e}")
        logger.error(f"Make sure emissions are cached for split '{split}' in directory '{cache_dir}'")
        raise
    
    # Load hyperparameters
    if decoder_params:
        params = decoder_params
        logger.info("Using provided decoder parameters")
    else:
        params = load_hyperparameters(tuning_results_dir)
    
    logger.info(f"Decoder parameters: {params}")
    
    # Initialize decoder
    logger.info("Initializing optimized decoder...")
    decoder = OptimizedCTCDecoder(verbose=False)
    decoder.update_params(**params)
    
    # Process emissions to generate predictions
    start_time = time.time()
    results = process_emissions_to_predictions(emissions_dict, decoder, split)
    processing_time = time.time() - start_time
    
    logger.info(f"Processing completed in {processing_time:.2f} seconds")
    logger.info(f"Average time per utterance: {processing_time/len(results):.3f} seconds")
    
    # Calculate metrics if validation data
    metrics = {}
    if split == 'val':
        metrics = calculate_validation_metrics(results)
    
    # Create submission CSV
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"submission_{split}_optimized_{timestamp}.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    submission_df = create_submission_csv(results, output_path)
    
    # Save processing log
    log_data = {
        'split': split,
        'cache_dir': cache_dir,
        'tuning_results_dir': tuning_results_dir,
        'decoder_params': params,
        'total_utterances': len(results),
        'processing_time': processing_time,
        'output_file': output_path,
        'metrics': metrics,
        'timestamp': timestamp
    }
    
    log_path = os.path.join(output_dir, f"submission_log_{timestamp}.json")
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    logger.info(f"Processing log saved to: {log_path}")
    
    logger.info("✅ Submission generation completed successfully!")
    return submission_df, output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate submission CSV using cached emissions and optimized decoder"
    )
    
    # Required arguments
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['test', 'val'],
        help='Data split to process (test or val)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='cache/emissions',
        help='Directory containing cached emissions'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='submission_output',
        help='Output directory for submission files'
    )
    
    parser.add_argument(
        '--tuning_results_dir',
        type=str,
        help='Directory containing tuning results with best_profile.json'
    )
    
    # Manual parameter specification
    parser.add_argument('--lm_weight', type=float, help='Language model weight')
    parser.add_argument('--word_score', type=float, help='Word insertion score')
    parser.add_argument('--sil_score', type=float, help='Silence score')
    parser.add_argument('--beam_size', type=int, help='Beam size')
    parser.add_argument('--beam_size_token', type=int, help='Token beam size')
    parser.add_argument('--beam_threshold', type=float, help='Beam threshold')
    
    args = parser.parse_args()
    
    # Validate cache directory
    cache_path = Path(args.cache_dir)
    if not cache_path.exists():
        print(f"Error: Cache directory not found: {args.cache_dir}")
        sys.exit(1)
    
    # Check if manual parameters are provided
    manual_params = None
    if any([args.lm_weight, args.word_score, args.sil_score, 
            args.beam_size, args.beam_size_token, args.beam_threshold]):
        manual_params = {}
        if args.lm_weight is not None:
            manual_params['lm_weight'] = args.lm_weight
        if args.word_score is not None:
            manual_params['word_score'] = args.word_score
        if args.sil_score is not None:
            manual_params['sil_score'] = args.sil_score
        if args.beam_size is not None:
            manual_params['beam_size'] = args.beam_size
        if args.beam_size_token is not None:
            manual_params['beam_size_token'] = args.beam_size_token
        if args.beam_threshold is not None:
            manual_params['beam_threshold'] = args.beam_threshold
    
    print("🚀 Starting optimized submission generation")
    print(f"Split: {args.split}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.tuning_results_dir:
        print(f"Tuning results: {args.tuning_results_dir}")
    if manual_params:
        print(f"Manual parameters: {manual_params}")
    print()
    
    # Generate submission
    try:
        submission_df, output_path = generate_submission(
            cache_dir=args.cache_dir,
            split=args.split,
            output_dir=args.output_dir,
            tuning_results_dir=args.tuning_results_dir,
            decoder_params=manual_params
        )
        
        print(f"\n✅ Submission generation completed!")
        print(f"Output file: {output_path}")
        print(f"Total submissions: {len(submission_df)}")
        print(f"Sample predictions:")
        for i in range(min(3, len(submission_df))):
            print(f"  ID {i}: '{submission_df.iloc[i]['text']}'")
        
    except Exception as e:
        print(f"\n❌ Submission generation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
