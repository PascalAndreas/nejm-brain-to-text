#!/usr/bin/env python3
"""
Adaptable grid search for decoder hyperparameters.
Supports different decoder types and generates visual heatmaps for parameter combinations.

Usage:
    python grid_search.py --config grid_search_config.yaml
    python grid_search.py --decoder flashlight --params lm_weight:0.5,1.0,1.5 word_score:-1.0,-0.5,0.0
"""

import os
import sys
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Tuple
import csv
import torch
import h5py
from tqdm import tqdm
from jiwer import wer, cer
from omegaconf import OmegaConf
import random

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from decoding.decoder import Decoder
from model_training.rnn_model import GRUDecoder
from model_training.data_augmentations import gauss_smooth
from nejm_b2txt_utils.general_utils import remove_punctuation


class GridSearchRunner:
    """
    Adaptable grid search runner for decoder hyperparameters.
    """
    
    def __init__(self, decoder_type: str, param_grid: Dict[str, List], 
                 num_samples: int = 50, output_dir: str = "grid_search_results"):
        """
        Initialize grid search runner.
        
        Args:
            decoder_type: Type of decoder ('greedy', 'flashlight')
            param_grid: Dictionary of parameter names to lists of values to try
            num_samples: Number of samples to evaluate on
            output_dir: Directory to save results
        """
        self.decoder_type = decoder_type
        self.param_grid = param_grid
        self.num_samples = num_samples
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load RNN model for generating logits
        self.rnn_model, self.rnn_args = self._load_rnn_model()
        
        # Load test data
        self.test_samples = self._load_test_samples()
        
        print(f"Initialized GridSearchRunner:")
        print(f"  Decoder type: {decoder_type}")
        print(f"  Parameter grid: {param_grid}")
        print(f"  Test samples: {len(self.test_samples)}")
        print(f"  Output directory: {output_dir}")
    
    def _load_rnn_model(self):
        """Load the pretrained RNN model."""
        model_path = project_root / "data" / "t15_pretrained_rnn_baseline" / "t15_pretrained_rnn_baseline"
        config_path = model_path / "checkpoint" / "args.yaml"
        checkpoint_path = model_path / "checkpoint" / "best_checkpoint"
        
        # Load model config
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
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
        
        # Clean up keys
        state_dict = {}
        for key, value in checkpoint['model_state_dict'].items():
            clean_key = key.replace("module.", "").replace("_orig_mod.", "")
            state_dict[clean_key] = value
        
        model.load_state_dict(state_dict)
        model.eval()
        
        print(f"Loaded RNN model from {model_path}")
        return model, model_args
    
    def _load_test_samples(self):
        """Load test samples from neural data across multiple days."""
        # Create mapping from session names to day indices
        sessions = self.rnn_args['dataset']['sessions']
        session_to_day_idx = {session: idx for idx, session in enumerate(sessions)}
        
        # Get available data directories
        data_base_dir = project_root / "data" / "t15_copyTask_neuralData" / "hdf5_data_final"
        available_sessions = []
        
        for session_dir in data_base_dir.iterdir():
            if session_dir.is_dir() and session_dir.name in session_to_day_idx:
                train_file = session_dir / "data_val.hdf5"
                if train_file.exists():
                    available_sessions.append((session_dir.name, train_file, session_to_day_idx[session_dir.name]))
        
        available_sessions.sort(key=lambda x: x[2])  # Sort by day index
        print(f"Found {len(available_sessions)} available sessions with training data")
        
        # First, collect all available trials with their metadata
        all_trials = []
        for session_name, data_file, day_idx in available_sessions:
            with h5py.File(data_file, 'r') as f:
                trial_keys = [k for k in f.keys() if k.startswith('trial_')]
                
                for trial_key in trial_keys:
                    trial_group = f[trial_key]
                    
                    # Extract ground truth text to filter valid trials
                    sentence_label = trial_group.attrs.get('sentence_label', '')
                    transcription = None
                    if 'transcription' in trial_group:
                        transcription = trial_group['transcription'][:]
                    
                    gt_text = self._extract_ground_truth_text(transcription, sentence_label)
                    
                    if gt_text:  # Only include trials with ground truth
                        all_trials.append({
                            'session_name': session_name,
                            'data_file': data_file,
                            'day_idx': day_idx,
                            'trial_key': trial_key,
                            'gt_text': gt_text
                        })
        
        print(f"Found {len(all_trials)} total valid trials across {len(available_sessions)} sessions")
        
        # Randomly sample the requested number of trials
        if len(all_trials) > self.num_samples:
            selected_trials = random.sample(all_trials, self.num_samples)
        else:
            selected_trials = all_trials
            print(f"Warning: Only {len(all_trials)} trials available, using all of them")
        
        # Now load the actual data for selected trials
        samples = []
        session_counts = {}
        
        for trial_info in selected_trials:
            with h5py.File(trial_info['data_file'], 'r') as f:
                trial_group = f[trial_info['trial_key']]
                neural_features = trial_group['input_features'][:]
                
                samples.append({
                    'trial_key': trial_info['trial_key'],
                    'neural_features': neural_features,
                    'gt_text': trial_info['gt_text'],
                    'day_idx': trial_info['day_idx'],
                    'session_name': trial_info['session_name']
                })
                
                # Count samples per session for reporting
                session_name = trial_info['session_name']
                session_counts[session_name] = session_counts.get(session_name, 0) + 1
        
        # Report the distribution
        for session_name, count in sorted(session_counts.items()):
            day_idx = next(s['day_idx'] for s in samples if s['session_name'] == session_name)
            print(f"Loaded {count} samples from {session_name} (day_idx={day_idx})")
        
        print(f"Loaded {len(samples)} total test samples with ground truth from {len(set(s['day_idx'] for s in samples))} days")
        return samples
    
    def _extract_ground_truth_text(self, transcription, sentence_label):
        """Extract ground truth text from the dataset."""
        if isinstance(sentence_label, (bytes, np.bytes_)):
            sentence_label = sentence_label.decode('utf-8')
        elif isinstance(sentence_label, str):
            pass  # Already string
        else:
            # It's an array of character codes
            sentence_label = ''.join([chr(int(code)) for code in sentence_label if int(code) != 0])
        
        # Remove punctuation and normalize
        text = remove_punctuation(sentence_label.strip())
        return text if text else None
    
    def _generate_logits(self, neural_features, day_idx):
        """Generate logits from neural features using RNN model."""
        device = torch.device('cpu')
        self.rnn_model.to(device)
        
        # Add batch dimension
        neural_input = np.expand_dims(neural_features, axis=0)
        neural_input = torch.tensor(neural_input, device=device, dtype=torch.float32)
        
        with torch.autocast(device_type="cpu", enabled=self.rnn_args['use_amp'], dtype=torch.bfloat16):
            # Apply smoothing
            smoothed_data = gauss_smooth(
                inputs=neural_input, 
                device=device,
                smooth_kernel_std=self.rnn_args['dataset']['data_transforms']['smooth_kernel_std'],
                smooth_kernel_size=self.rnn_args['dataset']['data_transforms']['smooth_kernel_size'],
                padding='valid',
            )
            
            with torch.no_grad():
                logits, _ = self.rnn_model(
                    x=smoothed_data,
                    day_idx=torch.tensor([day_idx], device=device),
                    states=None,
                    return_state=True,
                )
        
        # Convert to float32 and return
        return logits.float()
    
    def _evaluate_decoder(self, decoder, sample):
        """Evaluate decoder on a single sample."""
        try:
            # Generate logits using the correct day index
            logits = self._generate_logits(sample['neural_features'], sample['day_idx'])
            logit_lengths = torch.tensor([logits.shape[1]])
            
            # Decode
            result = decoder.decode(logits, logit_lengths)
            
            if result and 'sentence' in result:
                predicted_text = result['sentence']
                gt_text = sample['gt_text']
                
                if predicted_text and gt_text:
                    word_error_rate = wer(gt_text, predicted_text)
                    char_error_rate = cer(gt_text, predicted_text)
                    
                    return {
                        'wer': word_error_rate,
                        'cer': char_error_rate,
                        'predicted': predicted_text,
                        'ground_truth': gt_text,
                        'success': True
                    }
                
            return {
                'wer': 1.0,
                'cer': 1.0,
                'predicted': '',
                'ground_truth': sample['gt_text'],
                'success': False
            }
            
        except Exception as e:
            print(f"Error evaluating sample: {e}")
            return {
                'wer': 1.0,
                'cer': 1.0,
                'predicted': 'ERROR',
                'ground_truth': sample['gt_text'],
                'success': False,
                'error': str(e)
            }
    
    def run_grid_search(self):
        """Run the grid search and generate results."""
        # Generate all parameter combinations
        param_names = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        print(f"\n🔍 Starting grid search with {len(param_combinations)} parameter combinations")
        print(f"📊 Testing on {len(self.test_samples)} samples per combination")
        print(f"⏱️  Estimated time: ~{len(param_combinations) * len(self.test_samples) * 0.5 / 60:.1f} minutes")
        
        results = []
        best_wer_so_far = float('inf')
        best_params_so_far = None
        
        # Progress bar for parameter combinations
        pbar = tqdm(param_combinations, desc="🔍 Grid Search Progress", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc}')
        
        for i, param_combo in enumerate(pbar):
            # Create parameter dictionary
            params = dict(zip(param_names, param_combo))
            
            # Update progress bar description
            param_str = ', '.join([f"{k}={v}" for k, v in params.items()])
            pbar.set_description(f"🔍 Testing: {param_str}")
            
            try:
                # Create decoder with these parameters
                decoder = Decoder(backend=self.decoder_type, **params)
                
                # Evaluate on all test samples with inner progress bar
                sample_results = []
                sample_pbar = tqdm(self.test_samples, desc="📝 Samples", leave=False,
                                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
                
                for sample in sample_pbar:
                    sample_result = self._evaluate_decoder(decoder, sample)
                    sample_results.append(sample_result)
                
                sample_pbar.close()
                
                # Calculate average metrics
                successful_results = [r for r in sample_results if r['success']]
                if successful_results:
                    avg_wer = np.mean([r['wer'] for r in successful_results])
                    avg_cer = np.mean([r['cer'] for r in successful_results])
                    success_rate = len(successful_results) / len(sample_results)
                else:
                    avg_wer = 1.0
                    avg_cer = 1.0
                    success_rate = 0.0
                
                result = {
                    'params': params,
                    'avg_wer': avg_wer,
                    'avg_cer': avg_cer,
                    'success_rate': success_rate,
                    'num_samples': len(sample_results),
                    'num_successful': len(successful_results)
                }
                
                results.append(result)
                
                # Track best result so far
                if avg_wer < best_wer_so_far:
                    best_wer_so_far = avg_wer
                    best_params_so_far = params.copy()
                
                # Update progress with current best
                status_msg = f"✅ WER: {avg_wer:.3f} | Best so far: {best_wer_so_far:.3f}"
                pbar.set_postfix_str(status_msg)
                
                print(f"\n📊 Combo {i+1}/{len(param_combinations)}: {param_str}")
                print(f"   Results: WER={avg_wer:.3f}, CER={avg_cer:.3f}, Success={success_rate:.1%}")
                if avg_wer == best_wer_so_far:
                    print(f"   🏆 NEW BEST RESULT!")
                
            except Exception as e:
                print(f"\n❌ Failed to evaluate combo {params}: {e}")
                result = {
                    'params': params,
                    'avg_wer': 1.0,
                    'avg_cer': 1.0,
                    'success_rate': 0.0,
                    'num_samples': len(self.test_samples),
                    'num_successful': 0,
                    'error': str(e)
                }
                results.append(result)
                pbar.set_postfix_str(f"❌ Error: {str(e)[:30]}...")
        
        pbar.close()
        print(f"\n🎉 Grid search completed!")
        print(f"🏆 Best result: WER={best_wer_so_far:.3f} with {best_params_so_far}")
        
        # Save results
        self._save_results(results)
        
        # Generate visualizations
        self._generate_visualizations(results)
        
        return results
    
    def _save_results(self, results):
        """Save results to CSV file."""
        results_file = self.output_dir / f"{self.decoder_type}_grid_search_results.csv"
        
        if not results:
            print("No results to save")
            return
        
        # Dynamically determine fieldnames based on actual parameters used
        param_names = list(results[0]['params'].keys())
        metric_names = ['avg_wer', 'avg_cer', 'success_rate', 'num_samples', 'num_successful']
        fieldnames = param_names + metric_names
        
        with open(results_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                # Flatten the nested params dict and convert numpy scalars to Python floats
                row = {}
                row.update(result['params'])
                row['avg_wer'] = float(result['avg_wer'])
                row['avg_cer'] = float(result['avg_cer'])
                row['success_rate'] = float(result['success_rate'])
                row['num_samples'] = result['num_samples']
                row['num_successful'] = result['num_successful']
                writer.writerow(row)
        
        print(f"Results saved to {results_file}")
    
    def _generate_visualizations(self, results):
        """Generate heatmap visualizations for parameter combinations."""
        if not results:
            print("No results available for visualization")
            return
            
        param_names = list(self.param_grid.keys())
        print(f"Generating visualizations for parameters: {param_names}")
        
        if len(param_names) < 2:
            print("Need at least 2 parameters for heatmap visualization")
            return
        
        # Generate heatmaps for all pairs of parameters
        for i in range(len(param_names)):
            for j in range(i + 1, len(param_names)):
                param1, param2 = param_names[i], param_names[j]
                print(f"Creating heatmaps for {param1} vs {param2}")
                try:
                    self._create_heatmap(results, param1, param2, 'avg_wer', 'WER')
                    self._create_heatmap(results, param1, param2, 'avg_cer', 'CER')
                except Exception as e:
                    print(f"Error creating heatmap for {param1} vs {param2}: {e}")
    
    def _create_heatmap(self, results, param1, param2, metric, metric_name):
        """Create a heatmap for two parameters and a metric."""
        try:
            # Get unique values for each parameter
            param1_values = sorted(list(set([r['params'][param1] for r in results])))
            param2_values = sorted(list(set([r['params'][param2] for r in results])))
            
            print(f"  {param1} values: {param1_values}")
            print(f"  {param2} values: {param2_values}")
            
            # Create matrix for heatmap
            matrix = np.full((len(param2_values), len(param1_values)), np.nan)
            
            for result in results:
                p1_val = result['params'][param1]
                p2_val = result['params'][param2]
                
                try:
                    i = param1_values.index(p1_val)
                    j = param2_values.index(p2_val)
                    matrix[j, i] = result[metric]
                except (ValueError, KeyError) as e:
                    print(f"  Warning: Could not place result {p1_val}, {p2_val}: {e}")
                    continue
            
            # Check if we have any valid data
            if np.all(np.isnan(matrix)):
                print(f"  Warning: No valid data for heatmap {param1} vs {param2} for {metric}")
                return
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            
            # Use a colormap where lower values (better performance) are darker
            cmap = 'RdYlBu_r' if metric in ['avg_wer', 'avg_cer'] else 'viridis'
            
            sns.heatmap(matrix, 
                       xticklabels=[f"{param1}={v}" for v in param1_values],
                       yticklabels=[f"{param2}={v}" for v in param2_values],
                       annot=True, 
                       fmt='.3f',
                       cmap=cmap,
                       cbar_kws={'label': metric_name})
            
            plt.title(f'{self.decoder_type.title()} Decoder: {metric_name} vs {param1} and {param2}')
            plt.xlabel(param1)
            plt.ylabel(param2)
            plt.tight_layout()
            
            # Save plot
            plot_file = self.output_dir / f"{self.decoder_type}_{param1}_{param2}_{metric}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✅ Saved heatmap: {plot_file}")
            
        except Exception as e:
            print(f"  ❌ Error creating heatmap: {e}")
            import traceback
            traceback.print_exc()


def parse_param_string(param_string):
    """Parse parameter string like 'lm_weight:0.5,1.0,1.5 word_score:-1.0,-0.5,0.0'"""
    param_grid = {}
    
    for param_spec in param_string.split():
        if ':' not in param_spec:
            continue
        
        param_name, values_str = param_spec.split(':', 1)
        values = []
        
        for value_str in values_str.split(','):
            value_str = value_str.strip()
            try:
                # Try to parse as float
                value = float(value_str)
                values.append(value)
            except ValueError:
                # Keep as string
                values.append(value_str)
        
        param_grid[param_name] = values
    
    return param_grid


def main():
    parser = argparse.ArgumentParser(description='Run grid search for decoder parameters')
    parser.add_argument('--config', type=str, default='decoding/config.yaml',
                       help='Path to main config file (default: decoding/config.yaml)')
    parser.add_argument('--num_samples', type=int, default=None, 
                       help='Number of test samples to evaluate on (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results (overrides config)')
    
    args = parser.parse_args()
    
    # Load main configuration
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return
    
    config = OmegaConf.load(config_path)
    
    # Extract grid search parameters from tuning section
    tuning_config = config.get('tuning', {})
    if not tuning_config:
        print("No 'tuning' section found in config.yaml")
        return
    
    # Get decoder type
    decoder_type = config.get('decoder', {}).get('backend', 'flashlight')
    
    # Get parameter grid for this decoder type
    decoder_params = tuning_config.get(decoder_type, {})
    if not decoder_params:
        print(f"No parameter grid found for decoder type '{decoder_type}' in tuning config")
        print(f"Available decoder types: {list(tuning_config.keys())}")
        return
    
    # Convert OmegaConf to regular dict for parameter grid
    param_grid = dict(decoder_params)
    num_samples = args.num_samples or tuning_config.get('num_samples', 50)
    output_dir = args.output_dir or tuning_config.get('output_dir', 'grid_search_results')
    primary_metric = tuning_config.get('primary_metric', 'WER')
    
    print(f"Starting grid search:")
    print(f"  Decoder: {decoder_type}")
    print(f"  Parameters: {param_grid}")
    print(f"  Samples: {num_samples}")
    print(f"  Output: {output_dir}")
    
    # Run grid search
    runner = GridSearchRunner(decoder_type, param_grid, num_samples, output_dir)
    results = runner.run_grid_search()
    
    # Print best results
    best_result = min(results, key=lambda x: x['avg_wer'])
    print(f"\nBest result:")
    print(f"  Parameters: {best_result['params']}")
    print(f"  WER: {best_result['avg_wer']:.3f}")
    print(f"  CER: {best_result['avg_cer']:.3f}")
    print(f"  Success rate: {best_result['success_rate']:.3f}")


if __name__ == "__main__":
    main()
