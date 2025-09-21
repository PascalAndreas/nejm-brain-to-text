#!/usr/bin/env python3
"""
Example usage of the Bayesian optimization and corpus-aware decoding system.

This script demonstrates how to:
1. Cache model emissions
2. Run Bayesian optimization for decoder hyperparameters
3. Use corpus-specific decoding profiles
4. Evaluate performance
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipeline.emit import EmissionCache, cache_model_emissions
from pipeline.bayesian_tuning import BayesianDecoderTuner, run_full_optimization
from pipeline.corpus_decoder import CorpusAwareDecoder, CorpusAwareEvaluator


def example_full_pipeline():
    """Example of running the complete optimization pipeline."""
    
    # Configuration (adjust paths as needed)
    model_checkpoint = "models/checkpoints/best_model.ckpt"
    
    data_config = {
        'data_root': 'data/t15_copyTask_neuralData',
        'batch_size': 16,
        'num_workers': 4,
        'splits': ['val'],
        'use_amp': True
    }
    
    cache_config = {
        'cache_dir': 'cache/emissions',
        'format': 'npz',
        'compress': True
    }
    
    tuning_config = {
        'n_trials': 50,  # Reduced for example
        'n_startup_trials': 10,
        'output_dir': 'tuning_results',
        'decoder_type': 'flashlight'
    }
    
    print("Running full Bayesian optimization pipeline...")
    
    # Run complete pipeline
    profiles_dict, profiles_file = run_full_optimization(
        model_checkpoint=model_checkpoint,
        data_config=data_config,
        cache_config=cache_config,
        tuning_config=tuning_config,
        device='cuda'
    )
    
    print(f"Optimization completed! Profiles saved to: {profiles_file}")
    
    # Use the optimized profiles
    corpus_decoder = CorpusAwareDecoder(profiles_file)
    
    # Print profile summary
    summary = corpus_decoder.get_profile_summary()
    for corpus, info in summary.items():
        print(f"\n{corpus} profile:")
        for param, value in info['hyperparameters'].items():
            print(f"  {param}: {value}")
        if info['performance']:
            print(f"  WER: {info['performance']['wer']:.4f}")


def example_step_by_step():
    """Example of running optimization step by step."""
    
    print("Step-by-step Bayesian optimization example...")
    
    # Step 1: Cache emissions (if not already done)
    print("Step 1: Caching emissions...")
    cache_manager = EmissionCache(cache_dir='cache/emissions', format='npz')
    
    # Assume emissions are already cached, otherwise use:
    # cache_model_emissions(model_checkpoint, data_config, cache_config)
    
    # Step 2: Initialize tuner
    print("Step 2: Initializing tuner...")
    tuner = BayesianDecoderTuner(
        cache_manager=cache_manager,
        split_name='val',
        decoder_type='flashlight',
        n_trials=30,  # Reduced for example
        output_dir='tuning_results'
    )
    
    # Step 3: Optimize global profile
    print("Step 3: Optimizing global profile...")
    global_profile = tuner.optimize_global_profile()
    print(f"Global profile WER: {global_profile.performance['wer']:.4f}")
    
    # Step 4: Optimize corpus-specific profiles
    print("Step 4: Optimizing corpus-specific profiles...")
    corpus_profiles = tuner.optimize_corpus_specific_profiles(
        base_profile=global_profile,
        min_utterances=20  # Lower threshold for example
    )
    
    # Step 5: Save and use profiles
    print("Step 5: Saving profiles...")
    profiles_file = tuner.save_profiles()
    
    # Load and use the profiles
    corpus_decoder = CorpusAwareDecoder(profiles_file)
    
    # Example decoding with corpus information
    import torch
    
    # Simulate some log probabilities
    log_probs = torch.randn(1, 100, 41)  # [batch=1, time=100, vocab=41]
    lengths = torch.tensor([100])
    corpus = "50-Word"
    
    result = corpus_decoder.decode_single(log_probs, lengths, corpus)
    print(f"Decoded with {corpus} profile: '{result['sentence']}'")


def example_evaluation():
    """Example of evaluating corpus-aware decoder."""
    
    print("Evaluation example...")
    
    # Load existing profiles
    profiles_file = "tuning_results/profiles_val.json"
    
    try:
        corpus_decoder = CorpusAwareDecoder(profiles_file)
        evaluator = CorpusAwareEvaluator(corpus_decoder)
        
        # Load cached emissions
        cache_manager = EmissionCache(cache_dir='cache/emissions')
        emissions_dict = cache_manager.load_emissions('val')
        
        # Evaluate performance
        results = evaluator.evaluate_on_emissions(emissions_dict, corpus_breakdown=True)
        
        print("\nEvaluation Results:")
        for corpus, metrics in results.items():
            print(f"{corpus}:")
            print(f"  WER: {metrics['wer']:.4f} ({metrics['wer']*100:.2f}%)")
            print(f"  CER: {metrics['cer']:.4f} ({metrics['cer']*100:.2f}%)")
            print(f"  Success: {metrics['successful_decodes']}/{metrics['total_utterances']}")
    
    except FileNotFoundError:
        print(f"Profiles file not found: {profiles_file}")
        print("Run optimization first to generate profiles.")


def example_manual_profiles():
    """Example of creating and using manual profiles."""
    
    print("Manual profiles example...")
    
    from pipeline.bayesian_tuning import DecodingProfile
    
    # Create manual profiles
    global_profile = DecodingProfile(
        name='global',
        lm_weight=2.5,
        word_score=-0.2,
        sil_score=0.0,
        beam_size=100,
        beam_size_token=10,
        beam_threshold=20.0
    )
    
    switchboard_profile = DecodingProfile(
        name='Switchboard',
        lm_weight=3.0,  # Higher LM weight for conversational speech
        word_score=-0.1,
        sil_score=-0.5,
        beam_size=150,
        beam_size_token=15,
        beam_threshold=15.0
    )
    
    profiles = {
        'global': global_profile,
        'Switchboard': switchboard_profile
    }
    
    # Use profiles
    corpus_decoder = CorpusAwareDecoder(profiles)
    
    # Example usage
    import torch
    log_probs = torch.randn(1, 50, 41)
    lengths = torch.tensor([50])
    
    # Decode with Switchboard profile
    result = corpus_decoder.decode_single(log_probs, lengths, 'Switchboard')
    print(f"Switchboard result: '{result['sentence']}'")
    
    # Decode with unknown corpus (falls back to global)
    result = corpus_decoder.decode_single(log_probs, lengths, 'Unknown')
    print(f"Unknown corpus result: '{result['sentence']}'")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Bayesian optimization examples")
    parser.add_argument(
        '--example', 
        choices=['full', 'step_by_step', 'evaluation', 'manual'],
        default='manual',
        help='Which example to run'
    )
    
    args = parser.parse_args()
    
    if args.example == 'full':
        example_full_pipeline()
    elif args.example == 'step_by_step':
        example_step_by_step()
    elif args.example == 'evaluation':
        example_evaluation()
    elif args.example == 'manual':
        example_manual_profiles()
