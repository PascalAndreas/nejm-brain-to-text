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
import yaml
import torch
import h5py
from tqdm import tqdm
from jiwer import wer, cer
from omegaconf import OmegaConf

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
        """Load test samples from neural data."""
        data_dir = project_root / "data" / "t15_copyTask_neuralData" / "hdf5_data_final"
        
        # Get first available session
        sessions = self.rnn_args['dataset']['sessions']
        first_session = sessions[0]
        session_path = data_dir / first_session
        
        # Try different data file types
        for data_type in ['data_test.hdf5', 'data_val.hdf5', 'data_train.hdf5']:
            data_file = session_path / data_type
            if data_file.exists():
                print(f"Loading test data from: {data_file}")
                break
        else:
            raise FileNotFoundError(f"No data files found in {session_path}")
        
        samples = []
        with h5py.File(data_file, 'r') as f:
            trial_keys = [k for k in f.keys() if k.startswith('trial_')]
            trial_keys = sorted(trial_keys)[:self.num_samples]
            
            for trial_key in trial_keys:
                trial_group = f[trial_key]
                
                # Extract neural features
                neural_features = trial_group['input_features'][:]
                
                # Extract ground truth text
                sentence_label = trial_group.attrs.get('sentence_label', '')
                transcription = None
                if 'transcription' in trial_group:
                    transcription = trial_group['transcription'][:]
                
                # Extract ground truth text
                gt_text = self._extract_ground_truth_text(transcription, sentence_label)
                
                if gt_text:  # Only include samples with ground truth
                    samples.append({
                        'trial_key': trial_key,
                        'neural_features': neural_features,
                        'gt_text': gt_text
                    })
        
        print(f"Loaded {len(samples)} test samples with ground truth")
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
    
    def _generate_logits(self, neural_features):
        """Generate logits from neural features using RNN model."""
        device = torch.device('cpu')
        self.rnn_model.to(device)
        
        # Add batch dimension
        neural_input = np.expand_dims(neural_features, axis=0)
        neural_input = torch.tensor(neural_input, device=device, dtype=torch.float32)
        
        day_idx = 0  # First session corresponds to day index 0
        
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
            # Generate logits
            logits = self._generate_logits(sample['neural_features'])
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
        
        print(f"Running grid search with {len(param_combinations)} parameter combinations...")
        
        results = []
        
        for i, param_combo in enumerate(tqdm(param_combinations, desc="Grid search")):
            # Create parameter dictionary
            params = dict(zip(param_names, param_combo))
            
            try:
                # Create decoder with these parameters
                decoder = Decoder(backend=self.decoder_type, **params)
                
                # Evaluate on all test samples
                sample_results = []
                for sample in self.test_samples:
                    sample_result = self._evaluate_decoder(decoder, sample)
                    sample_results.append(sample_result)
                
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
                
                print(f"Combo {i+1}/{len(param_combinations)}: {params} -> WER: {avg_wer:.3f}, CER: {avg_cer:.3f}")
                
            except Exception as e:
                print(f"Failed to evaluate combo {params}: {e}")
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
        
        # Save results
        self._save_results(results)
        
        # Generate visualizations
        self._generate_visualizations(results)
        
        return results
    
    def _save_results(self, results):
        """Save results to YAML file."""
        results_file = self.output_dir / f"{self.decoder_type}_grid_search_results.yaml"
        
        with open(results_file, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
        
        print(f"Results saved to {results_file}")
    
    def _generate_visualizations(self, results):
        """Generate heatmap visualizations for parameter combinations."""
        param_names = list(self.param_grid.keys())
        
        if len(param_names) < 2:
            print("Need at least 2 parameters for heatmap visualization")
            return
        
        # Generate heatmaps for all pairs of parameters
        for i in range(len(param_names)):
            for j in range(i + 1, len(param_names)):
                param1, param2 = param_names[i], param_names[j]
                self._create_heatmap(results, param1, param2, 'avg_wer', 'WER')
                self._create_heatmap(results, param1, param2, 'avg_cer', 'CER')
    
    def _create_heatmap(self, results, param1, param2, metric, metric_name):
        """Create a heatmap for two parameters and a metric."""
        # Get unique values for each parameter
        param1_values = sorted(list(set([r['params'][param1] for r in results])))
        param2_values = sorted(list(set([r['params'][param2] for r in results])))
        
        # Create matrix for heatmap
        matrix = np.full((len(param2_values), len(param1_values)), np.nan)
        
        for result in results:
            p1_val = result['params'][param1]
            p2_val = result['params'][param2]
            
            try:
                i = param1_values.index(p1_val)
                j = param2_values.index(p2_val)
                matrix[j, i] = result[metric]
            except (ValueError, KeyError):
                continue
        
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
        
        print(f"Saved heatmap: {plot_file}")


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
    parser = argparse.ArgumentParser(description='Run adaptable grid search for decoder parameters')
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    parser.add_argument('--decoder', type=str, choices=['greedy', 'flashlight'], 
                       default='flashlight', help='Decoder type')
    parser.add_argument('--params', type=str, 
                       help='Parameter grid as string: "param1:val1,val2 param2:val3,val4"')
    parser.add_argument('--num_samples', type=int, default=50, 
                       help='Number of test samples to evaluate on')
    parser.add_argument('--output_dir', type=str, default='grid_search_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        decoder_type = config.get('decoder_type', 'flashlight')
        param_grid = config.get('param_grid', {})
        num_samples = config.get('num_samples', 50)
        output_dir = config.get('output_dir', 'grid_search_results')
    else:
        # Use command line arguments
        decoder_type = args.decoder
        
        if args.params:
            param_grid = parse_param_string(args.params)
        else:
            # Default parameter grids
            if decoder_type == 'flashlight':
                param_grid = {
                    'lm_weight': [0.5, 1.0, 1.5, 2.0],
                    'word_score': [-1.0, -0.5, 0.0, 0.5],
                    'beam_size': [100, 500, 1000]
                }
            else:
                print("No parameters to search for greedy decoder")
                return
        
        num_samples = args.num_samples
        output_dir = args.output_dir
    
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
