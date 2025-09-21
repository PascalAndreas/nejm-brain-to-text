#!/usr/bin/env python3
"""
Command-line interface for running Bayesian hyperparameter optimization.

This script provides an easy way to run the complete Bayesian optimization
pipeline for decoder hyperparameters with corpus-specific profiles.
"""

import argparse
import sys
from pathlib import Path
from omegaconf import OmegaConf
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.bayesian_tuning import run_full_optimization
from pipeline.corpus_decoder import CorpusAwareDecoder, CorpusAwareEvaluator
from pipeline.emit import EmissionCache


def main():
    parser = argparse.ArgumentParser(
        description="Run Bayesian optimization for decoder hyperparameters"
    )
    
    # Required arguments
    parser.add_argument(
        '--model_checkpoint', 
        type=str, 
        required=True,
        help='Path to trained model checkpoint'
    )
    
    # Optional arguments
    parser.add_argument(
        '--config', 
        type=str, 
        default='pipeline/config.yaml',
        help='Path to pipeline configuration file'
    )
    
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='tuning_results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--n_trials', 
        type=int, 
        default=100,
        help='Number of Bayesian optimization trials'
    )
    
    parser.add_argument(
        '--n_startup_trials', 
        type=int, 
        default=10,
        help='Number of random startup trials'
    )
    
    parser.add_argument(
        '--device', 
        type=str, 
        default='auto',
        choices=['auto', 'cuda', 'cpu', 'mps'],
        help='Device for model inference'
    )
    
    parser.add_argument(
        '--cache_format', 
        type=str, 
        default='npz',
        choices=['npz', 'pt', 'hdf5'],
        help='Format for emission caching'
    )
    
    parser.add_argument(
        '--skip_caching', 
        action='store_true',
        help='Skip emission caching (assumes cache already exists)'
    )
    
    parser.add_argument(
        '--evaluate_only', 
        type=str,
        help='Skip optimization and evaluate existing profiles from this file'
    )
    
    parser.add_argument(
        '--min_corpus_utterances', 
        type=int, 
        default=50,
        help='Minimum utterances required for corpus-specific optimization'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Auto-detect device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Prepare configurations
    data_config = {
        'data_root': config.dataset.data_root,
        'batch_size': config.dataset.batch_size,
        'num_workers': config.dataset.num_workers,
        'splits': ['val'],  # Only need validation for tuning
        'use_amp': config.training.precision != 32
    }
    
    cache_config = {
        'cache_dir': config.emission_cache.cache_dir,
        'format': args.cache_format,
        'compress': config.emission_cache.compress
    }
    
    tuning_config = {
        'n_trials': args.n_trials,
        'n_startup_trials': args.n_startup_trials,
        'output_dir': args.output_dir,
        'decoder_type': config.decoding.impl.replace('_ctc', ''),  # flashlight_ctc -> flashlight
        'min_utterances': args.min_corpus_utterances
    }
    
    if args.evaluate_only:
        # Evaluation mode: load existing profiles and evaluate
        print(f"Evaluation mode: loading profiles from {args.evaluate_only}")
        
        # Load profiles
        corpus_decoder = CorpusAwareDecoder(args.evaluate_only)
        evaluator = CorpusAwareEvaluator(corpus_decoder)
        
        # Load cached emissions
        cache_manager = EmissionCache(**cache_config)
        emissions_dict = cache_manager.load_emissions('val')
        
        # Evaluate
        print("Evaluating corpus-aware decoder...")
        results = evaluator.evaluate_on_emissions(emissions_dict, corpus_breakdown=True)
        
        # Print results
        print("\n=== EVALUATION RESULTS ===")
        for corpus, metrics in results.items():
            print(f"{corpus}:")
            print(f"  WER: {metrics['wer']:.4f} ({metrics['wer']*100:.2f}%)")
            print(f"  CER: {metrics['cer']:.4f} ({metrics['cer']*100:.2f}%)")
            print(f"  Success rate: {metrics['success_rate']:.3f}")
            print(f"  Utterances: {metrics['successful_decodes']}/{metrics['total_utterances']}")
        
    else:
        # Optimization mode: run full pipeline
        print("Starting Bayesian optimization pipeline...")
        
        if args.skip_caching:
            print("Skipping emission caching (using existing cache)")
            # Just run tuning
            from pipeline.bayesian_tuning import BayesianDecoderTuner
            from pipeline.emit import EmissionCache
            
            cache_manager = EmissionCache(**cache_config)
            tuner = BayesianDecoderTuner(cache_manager, **tuning_config)
            
            # Run optimization
            global_profile = tuner.optimize_global_profile()
            corpus_profiles = tuner.optimize_corpus_specific_profiles(
                base_profile=global_profile,
                min_utterances=args.min_corpus_utterances
            )
            
            # Save results
            profiles_file = tuner.save_profiles()
            
        else:
            # Run full pipeline including caching
            profiles_dict, profiles_file = run_full_optimization(
                model_checkpoint=args.model_checkpoint,
                data_config=data_config,
                cache_config=cache_config,
                tuning_config=tuning_config,
                device=device,
                model_config=config.model
            )
        
        print(f"\nOptimization completed!")
        print(f"Profiles saved to: {profiles_file}")
        
        # Quick evaluation
        print("\nRunning quick evaluation...")
        corpus_decoder = CorpusAwareDecoder(profiles_file)
        evaluator = CorpusAwareEvaluator(corpus_decoder)
        
        from pipeline.emit import EmissionCache
        cache_manager = EmissionCache(**cache_config)
        emissions_dict = cache_manager.load_emissions('val')
        
        results = evaluator.evaluate_on_emissions(emissions_dict, corpus_breakdown=True)
        
        print("\n=== FINAL RESULTS ===")
        for corpus, metrics in results.items():
            print(f"{corpus}: WER={metrics['wer']:.4f}, CER={metrics['cer']:.4f}")


if __name__ == "__main__":
    main()
