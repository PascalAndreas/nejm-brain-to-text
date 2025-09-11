#!/usr/bin/env python3
"""
Hyperparameter grid search for CTC decoding pipeline.
Tests different combinations of lm_weight, word_score, and beam_size
to find optimal parameters for ground truth phoneme decoding.
"""

import os
import sys
import itertools
import time
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test_gt_phonemes import test_ctc_pipeline_with_ground_truth


def run_grid_search(
    lm_weights=[0.0, 0.1, 0.2],  # Start from 0.0 which worked
    word_scores=[-1.0, 0.0, 1.0],  # Around neutral which worked
    beam_sizes=[50, 100, 150],  # Large beams which worked
    num_samples=8,  # Reasonable sample size
    perfection=0.90,
    time_expansion=2.0,
    perfect_one_hot=False,
    output_file="hyperparameter_results.csv"
):
    """
    Run grid search over hyperparameters for CTC decoding.
    
    Args:
        lm_weights: List of language model weights to test
        word_scores: List of word scores to test
        beam_sizes: List of beam sizes to test
        num_samples: Number of samples to test per configuration
        perfection: Perfection level for generated logits
        time_expansion: Time expansion factor for logits
        perfect_one_hot: If True, generate perfect one-hot logits
        output_file: CSV file to save results
    
    Returns:
        DataFrame with results
    """
    print("🔍 Starting CTC Decoding Hyperparameter Grid Search")
    print("=" * 60)
    print(f"Parameters to search:")
    print(f"  LM weights: {lm_weights}")
    print(f"  Word scores: {word_scores}")
    print(f"  Beam sizes: {beam_sizes}")
    print(f"  Samples per config: {num_samples}")
    print(f"  Perfection: {perfection}")
    print(f"  Time expansion: {time_expansion}")
    print(f"  Perfect one-hot: {perfect_one_hot}")
    
    # Generate all combinations
    combinations = list(itertools.product(lm_weights, word_scores, beam_sizes))
    total_combinations = len(combinations)
    print(f"\n🎯 Total combinations to test: {total_combinations}")
    
    results = []
    start_time = time.time()
    
    for i, (lm_weight, word_score, beam_size) in enumerate(combinations):
        print(f"\n--- Configuration {i+1}/{total_combinations} ---")
        print(f"LM weight: {lm_weight}, Word score: {word_score}, Beam size: {beam_size}")
        
        config_start_time = time.time()
        
        try:
            # Run test with current configuration
            avg_wer = test_ctc_pipeline_with_ground_truth(
                num_samples=num_samples,
                perfection=perfection,
                time_expansion=time_expansion,
                lm_weight=lm_weight,
                word_score=word_score,
                beam_size=beam_size,
                perfect_one_hot=perfect_one_hot,
                verbose=False  # Suppress detailed output for grid search
            )
            
            config_time = time.time() - config_start_time
            
            result = {
                'lm_weight': lm_weight,
                'word_score': word_score,
                'beam_size': beam_size,
                'avg_wer': avg_wer,
                'time_seconds': config_time,
                'status': 'success' if avg_wer != float('inf') else 'failed'
            }
            
            results.append(result)
            
            print(f"✅ Average WER: {avg_wer:.3f} (took {config_time:.1f}s)")
            
        except Exception as e:
            config_time = time.time() - config_start_time
            print(f"❌ Configuration failed: {e}")
            
            result = {
                'lm_weight': lm_weight,
                'word_score': word_score,
                'beam_size': beam_size,
                'avg_wer': float('inf'),
                'time_seconds': config_time,
                'status': 'error'
            }
            results.append(result)
        
        # Estimate remaining time
        elapsed_time = time.time() - start_time
        avg_time_per_config = elapsed_time / (i + 1)
        remaining_configs = total_combinations - (i + 1)
        estimated_remaining_time = avg_time_per_config * remaining_configs
        
        print(f"Progress: {i+1}/{total_combinations} ({100*(i+1)/total_combinations:.1f}%)")
        print(f"Estimated remaining time: {estimated_remaining_time/60:.1f} minutes")
    
    # Convert to DataFrame and analyze results
    df = pd.DataFrame(results)
    
    # Save results
    df.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
    
    # Analysis
    print(f"\n📊 GRID SEARCH RESULTS")
    print("=" * 60)
    
    successful_results = df[df['status'] == 'success']
    if len(successful_results) > 0:
        # Best configuration
        best_result = successful_results.loc[successful_results['avg_wer'].idxmin()]
        print(f"🏆 Best configuration:")
        print(f"  LM weight: {best_result['lm_weight']}")
        print(f"  Word score: {best_result['word_score']}")
        print(f"  Beam size: {best_result['beam_size']}")
        print(f"  Average WER: {best_result['avg_wer']:.3f}")
        
        # Top 5 configurations
        print(f"\n🥇 Top 10 configurations:")
        top_10 = successful_results.nsmallest(10, 'avg_wer')
        for idx, row in top_10.iterrows():
            print(f"  {row.name+1}. LM:{row['lm_weight']:5.2f} Word:{row['word_score']:6.1f} Beam:{row['beam_size']:2.0f} -> WER:{row['avg_wer']:.3f}")
        
        # Statistics
        print(f"\n📈 Statistics:")
        print(f"  Successful configurations: {len(successful_results)}/{len(df)}")
        print(f"  Best WER: {successful_results['avg_wer'].min():.3f}")
        print(f"  Worst WER: {successful_results['avg_wer'].max():.3f}")
        print(f"  Mean WER: {successful_results['avg_wer'].mean():.3f}")
        print(f"  Std WER: {successful_results['avg_wer'].std():.3f}")
        
        # Parameter analysis
        print(f"\n🔍 Parameter Analysis:")
        for param in ['lm_weight', 'word_score', 'beam_size']:
            param_performance = successful_results.groupby(param)['avg_wer'].mean().sort_values()
            print(f"  Best {param}: {param_performance.index[0]} (avg WER: {param_performance.iloc[0]:.3f})")
    
    else:
        print("❌ No successful configurations found!")
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Total search time: {total_time/60:.1f} minutes")
    
    return df


def main():
    """Main function with example grid search configurations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run hyperparameter grid search for CTC decoding")
    parser.add_argument('--samples', type=int, default=20, help='Number of samples per configuration')
    parser.add_argument('--output', type=str, default='hyperparameter_results.csv', help='Output CSV file')
    parser.add_argument('--perfect_one_hot', action='store_true', help='Generate perfect one-hot logits with no noise')
    args = parser.parse_args()
    
    # Run the grid search
    results_df = run_grid_search(
        num_samples=args.samples,
        perfect_one_hot=args.perfect_one_hot,
        output_file=args.output
    )
    
    print(f"\n🎉 Grid search completed! Results saved to {args.output}")


if __name__ == "__main__":
    main()
