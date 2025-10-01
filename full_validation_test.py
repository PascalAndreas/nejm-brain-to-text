#!/usr/bin/env python3
"""
Test decoder performance on the FULL validation set to get accurate metrics.
"""

import torch
import numpy as np
from pipeline.emit import EmissionCache
from decoding.decoder import OptimizedCTCDecoder
from jiwer import wer, cer
import string
import matplotlib.pyplot as plt
from tqdm import tqdm

def strip_punctuation_and_normalize(text: str) -> str:
    """Strip punctuation and normalize text for evaluation."""
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    text = text.lower()
    return text.strip()

def test_full_validation():
    """Test decoder on the complete validation set."""
    
    # Load emissions
    print("Loading cached emissions...")
    emission_cache = EmissionCache(cache_dir='cache/emissions')
    emissions_dict = emission_cache.load_emissions('val')
    utterance_ids = list(emissions_dict.keys())
    
    print(f"Loaded {len(emissions_dict)} validation emissions")
    print("Testing on ALL validation utterances (no sampling)")
    
    # Initialize decoder
    print("Initializing decoder...")
    decoder = OptimizedCTCDecoder(verbose=False)
    
    # Use the best parameters we found from epoch 13
    print("Using best parameters from epoch 13 optimization:")
    best_params = {
        'lm_weight': 4.2,
        'word_score': -1.8,
        'sil_score': -0.2,
        'beam_size': 150,
        'beam_size_token': 17,
        'beam_threshold': 30
    }
    print(f"  {best_params}")
    
    decoder.update_params(**best_params)
    
    print(f"\nTesting decoder on ALL {len(utterance_ids)} validation utterances...")
    
    wer_scores = []
    cer_scores = []
    successful_decodes = 0
    failed_decodes = 0
    
    # Process all utterances with progress bar
    for i, utt_id in enumerate(tqdm(utterance_ids, desc="Processing")):
        emission = emissions_dict[utt_id]
        
        # Skip if no ground truth
        if 'ground_truth' not in emission.get('meta', {}):
            failed_decodes += 1
            continue
        
        try:
            # Get ground truth
            true_sentence = emission['meta']['ground_truth']
            
            # Prepare inputs
            log_probs = emission['log_probs'].unsqueeze(0)
            out_len = torch.tensor([emission['out_len']])
            
            # Decode
            result = decoder.decode(log_probs, out_len)
            
            # Extract prediction
            if isinstance(result, list):
                result = result[0] if result else {'sentence': ''}
            pred_sentence = result.get('sentence', '') if isinstance(result, dict) else str(result)
            
            # Normalize both
            pred_normalized = strip_punctuation_and_normalize(pred_sentence)
            true_normalized = strip_punctuation_and_normalize(true_sentence)
            
            if pred_normalized and true_normalized:
                # Calculate metrics
                sample_wer = wer(true_normalized, pred_normalized)
                sample_cer = cer(true_normalized, pred_normalized)
                
                wer_scores.append(sample_wer)
                cer_scores.append(sample_cer)
                successful_decodes += 1
                
                # Print some examples of worst cases (but not too many)
                if sample_wer > 0.9 and len([w for w in wer_scores if w > 0.9]) <= 5:
                    print(f"\nWORST CASE #{len([w for w in wer_scores if w > 0.9])} (WER={sample_wer:.3f}):")
                    print(f"  True: '{true_normalized}'")
                    print(f"  Pred: '{pred_normalized}'")
            else:
                failed_decodes += 1
            
        except Exception as e:
            failed_decodes += 1
    
    print(f"\nCompleted processing {len(utterance_ids)} samples")
    
    # Calculate comprehensive statistics
    if wer_scores:
        avg_wer = np.mean(wer_scores)
        median_wer = np.median(wer_scores)
        std_wer = np.std(wer_scores)
        min_wer = np.min(wer_scores)
        max_wer = np.max(wer_scores)
        
        avg_cer = np.mean(cer_scores)
        median_cer = np.median(cer_scores)
        
        # Percentiles
        p25_wer = np.percentile(wer_scores, 25)
        p75_wer = np.percentile(wer_scores, 75)
        p90_wer = np.percentile(wer_scores, 90)
        p95_wer = np.percentile(wer_scores, 95)
        
        print(f"\n" + "="*50)
        print(f"FULL VALIDATION SET PERFORMANCE")
        print(f"="*50)
        print(f"Total utterances: {len(utterance_ids)}")
        print(f"Successful decodes: {successful_decodes}")
        print(f"Failed decodes: {failed_decodes}")
        print(f"Success rate: {successful_decodes/(successful_decodes+failed_decodes):.3f}")
        
        print(f"\nWER Statistics:")
        print(f"  Mean: {avg_wer:.4f} ({avg_wer*100:.2f}%)")
        print(f"  Median: {median_wer:.4f} ({median_wer*100:.2f}%)")
        print(f"  Std: {std_wer:.4f}")
        print(f"  Min: {min_wer:.4f} ({min_wer*100:.2f}%)")
        print(f"  Max: {max_wer:.4f} ({max_wer*100:.2f}%)")
        print(f"  25th percentile: {p25_wer:.4f} ({p25_wer*100:.2f}%)")
        print(f"  75th percentile: {p75_wer:.4f} ({p75_wer*100:.2f}%)")
        print(f"  90th percentile: {p90_wer:.4f} ({p90_wer*100:.2f}%)")
        print(f"  95th percentile: {p95_wer:.4f} ({p95_wer*100:.2f}%)")
        
        print(f"\nCER Statistics:")
        print(f"  Mean: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
        print(f"  Median: {median_cer:.4f} ({median_cer*100:.2f}%)")
        
        # Performance distribution
        perfect_count = sum(1 for w in wer_scores if w == 0.0)
        excellent_count = sum(1 for w in wer_scores if 0.0 < w <= 0.1)
        good_count = sum(1 for w in wer_scores if 0.1 < w <= 0.2)
        ok_count = sum(1 for w in wer_scores if 0.2 < w <= 0.5)
        bad_count = sum(1 for w in wer_scores if 0.5 < w <= 0.8)
        terrible_count = sum(1 for w in wer_scores if w > 0.8)
        
        print(f"\nPerformance Distribution:")
        print(f"  Perfect (0% WER): {perfect_count} ({perfect_count/len(wer_scores)*100:.1f}%)")
        print(f"  Excellent (0-10% WER): {excellent_count} ({excellent_count/len(wer_scores)*100:.1f}%)")
        print(f"  Good (10-20% WER): {good_count} ({good_count/len(wer_scores)*100:.1f}%)")
        print(f"  OK (20-50% WER): {ok_count} ({ok_count/len(wer_scores)*100:.1f}%)")
        print(f"  Bad (50-80% WER): {bad_count} ({bad_count/len(wer_scores)*100:.1f}%)")
        print(f"  Terrible (>80% WER): {terrible_count} ({terrible_count/len(wer_scores)*100:.1f}%)")
        
        # Comparison to targets
        print(f"\nComparison to Targets:")
        print(f"  Current performance: {avg_wer:.4f} ({avg_wer*100:.2f}% WER)")
        print(f"  Your baseline: 0.13 (13% WER)")
        print(f"  Top 10 leaderboard: 0.07 (7% WER)")
        print(f"  Gap to baseline: {((avg_wer - 0.13) / 0.13 * 100):+.1f}%")
        print(f"  Gap to top 10: {((avg_wer - 0.07) / 0.07 * 100):+.1f}%")
        
        # Create comprehensive histogram
        plt.figure(figsize=(15, 10))
        
        # Main histogram
        plt.subplot(2, 2, 1)
        plt.hist(wer_scores, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(avg_wer, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_wer:.3f}')
        plt.axvline(median_wer, color='green', linestyle='--', linewidth=2, label=f'Median: {median_wer:.3f}')
        plt.axvline(0.13, color='orange', linestyle=':', linewidth=2, label='Your baseline: 0.13')
        plt.axvline(0.07, color='purple', linestyle=':', linewidth=2, label='Top 10: 0.07')
        plt.xlabel('WER')
        plt.ylabel('Count')
        plt.title(f'WER Distribution (Full Validation Set, n={len(wer_scores)})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # CER histogram
        plt.subplot(2, 2, 2)
        plt.hist(cer_scores, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(avg_cer, color='red', linestyle='--', linewidth=2, label=f'Mean: {avg_cer:.3f}')
        plt.axvline(median_cer, color='green', linestyle='--', linewidth=2, label=f'Median: {median_cer:.3f}')
        plt.xlabel('CER')
        plt.ylabel('Count')
        plt.title('CER Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Cumulative distribution
        plt.subplot(2, 2, 3)
        sorted_wer = np.sort(wer_scores)
        cumulative = np.arange(1, len(sorted_wer) + 1) / len(sorted_wer)
        plt.plot(sorted_wer, cumulative, linewidth=2)
        plt.axvline(0.13, color='orange', linestyle=':', linewidth=2, label='Your baseline: 0.13')
        plt.axvline(0.07, color='purple', linestyle=':', linewidth=2, label='Top 10: 0.07')
        plt.xlabel('WER')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative WER Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Box plot by performance categories
        plt.subplot(2, 2, 4)
        categories = []
        for w in wer_scores:
            if w == 0.0:
                categories.append('Perfect')
            elif w <= 0.1:
                categories.append('Excellent')
            elif w <= 0.2:
                categories.append('Good')
            elif w <= 0.5:
                categories.append('OK')
            elif w <= 0.8:
                categories.append('Bad')
            else:
                categories.append('Terrible')
        
        import pandas as pd
        df = pd.DataFrame({'WER': wer_scores, 'Category': categories})
        category_order = ['Perfect', 'Excellent', 'Good', 'OK', 'Bad', 'Terrible']
        df['Category'] = pd.Categorical(df['Category'], categories=category_order, ordered=True)
        df = df.sort_values('Category')
        
        box_data = [df[df['Category'] == cat]['WER'].values for cat in category_order if cat in df['Category'].values]
        box_labels = [cat for cat in category_order if cat in df['Category'].values]
        
        plt.boxplot(box_data, labels=box_labels)
        plt.ylabel('WER')
        plt.title('WER Distribution by Performance Category')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = 'full_validation_performance2.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"\nSaved comprehensive performance analysis to: {plot_file}")
        
    else:
        print(f"\nNo successful decodes!")

if __name__ == "__main__":
    test_full_validation()
