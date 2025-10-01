"""
Fast Bayesian hyperparameter optimization for decoder parameters.

This module implements a simple, fast Bayesian optimization approach that:
1. Uses a small fixed budget of utterances for all trials
2. Leverages the optimized decoder with fast parameter updates
3. Provides real-time progress visualization
4. Focuses on speed over complex racing logic
"""

import os
import json
import pickle
import string
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import torch
from tqdm import tqdm
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from jiwer import wer, cer

# Handle imports for both package and direct execution
try:
    from .emit import EmissionCache
    from ..dataset import BrainToTextDataset
except ImportError:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from pipeline.emit import EmissionCache
    from dataset import BrainToTextDataset


@dataclass
class DecodingProfile:
    """Simplified decoding hyperparameter profile."""
    name: str
    lm_weight: float
    word_score: float
    sil_score: float
    beam_size: int = 150  # Fixed
    beam_size_token: int = 10  # Tune this
    beam_threshold: float = 20.0
    performance: Optional[Dict[str, float]] = None


def strip_punctuation_and_normalize(text: str) -> str:
    """Strip punctuation and normalize text for evaluation."""
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    text = text.lower()
    return text.strip()


class FastBayesianTuner:
    """Fast Bayesian optimization focusing on challenging utterances only."""
    
    def __init__(
        self,
        emission_cache: EmissionCache,
        decoder_type: str = 'flashlight',
        output_dir: str = 'tuning_results_fast',
        n_trials: int = 100,
        n_startup_trials: int = 15,
        random_seed: int = 42
    ):
        self.emission_cache = emission_cache
        self.decoder_type = decoder_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials
        self.random_seed = random_seed
        
        # Set up logging (suppress verbose output)
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Load emissions and shuffle for unbiased sampling
        print("Loading cached emissions...")
        self.emissions_dict = self.emission_cache.load_emissions('val')
        self.utterance_ids = list(self.emissions_dict.keys())
        np.random.seed(self.random_seed)
        np.random.shuffle(self.utterance_ids)  # Shuffle for unbiased sampling
        
        print(f"Loaded {len(self.emissions_dict)} validation emissions")
        
        # Initialize optimized decoder once (expensive operation)
        print("🔧 Initializing optimized decoder (one-time setup)...")
        from decoding.decoder import OptimizedCTCDecoder
        self.decoder = OptimizedCTCDecoder(verbose=False)
        
        # Set good starting parameters
        self.good_starting_params = {
            'lm_weight': 3.75,
            'word_score': -1.0,
            'sil_score': 0.0,
            'beam_size': 150,
            'beam_size_token': 25,
            'beam_threshold': 24.0
        }
        print("✅ Decoder initialized!")
        
        # Filter out easy samples by evaluating all with good starting params
        print("\n🔍 Filtering out easy samples with good starting parameters...")
        self.filtered_utterance_ids = self._filter_challenging_utterances()
        
        print(f"Fast tuning configuration:")
        print(f"  Original validation set: {len(self.utterance_ids)} utterances")
        print(f"  Challenging utterances: {len(self.filtered_utterance_ids)} utterances")
        print(f"  Filtered out {len(self.utterance_ids) - len(self.filtered_utterance_ids)} easy samples (0.0 WER)")
        print(f"  Focus factor: {len(self.filtered_utterance_ids) / len(self.utterance_ids):.2%} of original set")
        
        # Storage for optimization results
        self.best_profile = None
        self.study = None
        self.trial_history = []
    
    def _filter_challenging_utterances(self) -> List[str]:
        """Filter out utterances that achieve 0.0 WER with good starting parameters."""
        # Set decoder to good starting parameters
        self._update_decoder_params(self.good_starting_params)
        
        challenging_utterances = []
        perfect_count = 0
        
        print(f"Evaluating all {len(self.utterance_ids)} utterances with starting parameters...")
        
        for utt_id in tqdm(self.utterance_ids, desc="Filtering easy samples"):
            emission = self.emissions_dict[utt_id]
            
            # Skip if no ground truth
            if 'ground_truth' not in emission.get('meta', {}):
                continue
            
            try:
                # Prepare inputs
                log_probs = emission['log_probs'].unsqueeze(0)
                out_len = torch.tensor([emission['out_len']])
                
                # Decode
                result = self.decoder.decode(log_probs, out_len)
                
                # Extract prediction
                if isinstance(result, list):
                    result = result[0] if result else {'sentence': ''}
                pred_sentence = result.get('sentence', '') if isinstance(result, dict) else str(result)
                
                # Get ground truth
                true_sentence = emission['meta']['ground_truth']
                
                # Normalize both
                pred_normalized = strip_punctuation_and_normalize(pred_sentence)
                true_normalized = strip_punctuation_and_normalize(true_sentence)
                
                if pred_normalized and true_normalized:
                    # Calculate WER for this utterance
                    sample_wer = wer(true_normalized, pred_normalized)
                    
                    # Only keep utterances that don't achieve perfect score
                    if sample_wer > 0.0:
                        challenging_utterances.append(utt_id)
                    else:
                        perfect_count += 1
                else:
                    # Keep utterances with decoding issues as they're challenging
                    challenging_utterances.append(utt_id)
                
            except Exception as e:
                # Keep utterances that fail to decode as they're challenging
                challenging_utterances.append(utt_id)
        
        print(f"✅ Filtering complete: {perfect_count} perfect samples removed, {len(challenging_utterances)} challenging samples remain")
        return challenging_utterances
    
    def _update_decoder_params(self, params: Dict[str, Any]):
        """Update decoder parameters (fast operation)."""
        self.decoder.update_params(
            lm_weight=params['lm_weight'],
            word_score=params['word_score'],
            sil_score=params['sil_score'],
            beam_size=150,  # Fixed
            beam_size_token=params['beam_size_token'],
            beam_threshold=params['beam_threshold']
        )
    
    def _evaluate_params(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate parameters on challenging utterances only."""
        # Update decoder parameters (fast operation)
        self._update_decoder_params(params)
        
        # Use all filtered challenging utterances
        utterance_batch = self.filtered_utterance_ids
        
        # Evaluate on batch
        total_wer = 0.0
        total_cer = 0.0
        total_words = 0
        total_chars = 0
        successful_decodes = 0
        
        for utt_id in utterance_batch:
            emission = self.emissions_dict[utt_id]
            
            # Skip if no ground truth
            if 'ground_truth' not in emission.get('meta', {}):
                continue
            
            try:
                # Prepare inputs
                log_probs = emission['log_probs'].unsqueeze(0)
                out_len = torch.tensor([emission['out_len']])
                
                # Decode
                result = self.decoder.decode(log_probs, out_len)
                
                # Extract prediction
                if isinstance(result, list):
                    result = result[0] if result else {'sentence': ''}
                pred_sentence = result.get('sentence', '') if isinstance(result, dict) else str(result)
                
                # Get ground truth
                true_sentence = emission['meta']['ground_truth']
                
                # Normalize both
                pred_normalized = strip_punctuation_and_normalize(pred_sentence)
                true_normalized = strip_punctuation_and_normalize(true_sentence)
                
                if pred_normalized and true_normalized:
                    # Calculate metrics
                    sample_wer = wer(true_normalized, pred_normalized)
                    sample_cer = cer(true_normalized, pred_normalized)
                    
                    # Accumulate
                    true_words = len(true_normalized.split())
                    true_chars = len(true_normalized)
                    
                    total_wer += sample_wer * true_words
                    total_cer += sample_cer * true_chars
                    total_words += true_words
                    total_chars += true_chars
                    successful_decodes += 1
                
            except Exception as e:
                # Skip failed decodes
                continue
        
        # Calculate final metrics
        if total_words > 0 and successful_decodes > 0:
            final_wer = total_wer / total_words
            final_cer = total_cer / total_chars
            success_rate = successful_decodes / len(utterance_batch)
        else:
            final_wer = 1.0
            final_cer = 1.0
            success_rate = 0.0
        
        return {
            'wer': final_wer,
            'cer': final_cer,
            'success_rate': success_rate,
            'utterances_evaluated': len(utterance_batch)
        }
    
    def _create_objective(self):
        """Create objective function for Optuna optimization."""
        def objective(trial):
            # Define hyperparameter search space centered around good starting values
            params = {
                'lm_weight': trial.suggest_float('lm_weight', 3.5, 4.5),  # Centered around 3.0
                'word_score': trial.suggest_float('word_score', -1.8, -0.8),  # Centered around -0.8
                'sil_score': trial.suggest_float('sil_score', -0.5, 0.5),  # Centered around 0.0
                'beam_size_token': trial.suggest_int('beam_size_token', 10, 35),  # Centered around 25
                'beam_threshold': trial.suggest_float('beam_threshold', 20, 40),  # Centered around 24.0
            }
            
            # Evaluate parameters
            metrics = self._evaluate_params(params)
            
            # Store additional metrics for analysis
            trial.set_user_attr('cer', metrics['cer'])
            trial.set_user_attr('success_rate', metrics['success_rate'])
            trial.set_user_attr('utterances_evaluated', metrics['utterances_evaluated'])
            
            # Store trial history for visualization
            self.trial_history.append({
                'trial_number': trial.number,
                'wer': metrics['wer'],
                'cer': metrics['cer'],
                'success_rate': metrics['success_rate'],
                'utterances_evaluated': metrics['utterances_evaluated'],
                **params
            })
            
            # Return WER as the objective to minimize
            return metrics['wer']
        
        return objective
    
    def optimize(self) -> DecodingProfile:
        """Run fast Bayesian optimization."""
        print("\n⚡ Starting Fast Bayesian Optimization")
        print(f"Target trials: {self.n_trials}")
        print(f"Evaluating on: {len(self.filtered_utterance_ids)} challenging utterances per trial")
        
        # Create Optuna study
        self.study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_seed, n_startup_trials=self.n_startup_trials),
            pruner=MedianPruner(n_startup_trials=self.n_startup_trials),
            study_name="fast_decoder_tuning"
        )
        
        # Enqueue good starting parameters as the first trial
        print(f"🎯 Starting with good parameters: {self.good_starting_params}")
        self.study.enqueue_trial(self.good_starting_params)
        
        # Create main progress bar for trials
        trial_progress = tqdm(
            total=self.n_trials,
            desc="Optimizing",
            unit="trial",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} trials [{elapsed}<{remaining}] Best WER: {postfix}'
        )
        
        # Custom callback to update progress
        def progress_callback(study, trial):
            best_wer = study.best_value if study.best_value is not None else 1.0
            trial_progress.set_postfix_str(f"{best_wer:.4f}")
            trial_progress.update(1)
        
        # Optimize with progress tracking
        objective = self._create_objective()
        self.study.optimize(
            objective, 
            n_trials=self.n_trials,
            callbacks=[progress_callback]
        )
        
        trial_progress.close()
        
        # Get best parameters
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        print(f"\n✅ Optimization completed!")
        print(f"Best WER: {best_value:.4f}")
        print(f"Best parameters: {best_params}")
        
        # Create profile
        self.best_profile = DecodingProfile(
            name='fast_optimized',
            lm_weight=best_params['lm_weight'],
            word_score=best_params['word_score'],
            sil_score=best_params['sil_score'],
            beam_size=150,  # Fixed
            beam_size_token=best_params['beam_size_token'],
            beam_threshold=best_params['beam_threshold'],
            performance={
                'wer': best_value,
                'cer': self.study.best_trial.user_attrs['cer'],
                'success_rate': self.study.best_trial.user_attrs['success_rate'],
                'utterances_evaluated': self.study.best_trial.user_attrs['utterances_evaluated']
            }
        )
        
        # Save results and create visualizations
        self._save_results()
        self._create_visualizations()
        
        return self.best_profile
    
    def _save_results(self):
        """Save optimization results."""
        # Save study
        study_file = self.output_dir / "fast_study.pkl"
        with open(study_file, 'wb') as f:
            pickle.dump(self.study, f)
        
        # Save profile
        profile_file = self.output_dir / "best_profile.json"
        with open(profile_file, 'w') as f:
            json.dump(asdict(self.best_profile), f, indent=2)
        
        # Save trial history
        history_file = self.output_dir / "trial_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.trial_history, f, indent=2)
        
        print(f"Results saved to {self.output_dir}")
    
    def _create_visualizations(self):
        """Create visualizations of the optimization."""
        if not self.study or not self.trial_history:
            return
        
        print("Creating optimization visualizations...")
        
        # Convert trial history to DataFrame
        df = pd.DataFrame(self.trial_history)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Optimization progress
        ax1 = plt.subplot(2, 3, 1)
        plt.plot(df['trial_number'], df['wer'], 'o-', alpha=0.7, markersize=4)
        running_best = df['wer'].cummin()
        plt.plot(df['trial_number'], running_best, 'r-', linewidth=2, label=f'Best: {running_best.iloc[-1]:.4f}')
        plt.xlabel('Trial Number')
        plt.ylabel('WER')
        plt.title('Fast Optimization Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2-5. Parameter vs WER scatter plots
        param_names = ['lm_weight', 'word_score', 'sil_score', 'beam_size_token']
        
        for i, param in enumerate(param_names):
            ax = plt.subplot(2, 3, i + 2)
            scatter = plt.scatter(df[param], df['wer'], c=df['trial_number'], cmap='viridis', alpha=0.7)
            plt.xlabel(param.replace('_', ' ').title())
            plt.ylabel('WER')
            plt.title(f'{param.replace("_", " ").title()} vs WER')
            plt.colorbar(scatter, label='Trial Number')
            plt.grid(True, alpha=0.3)
        
        # 6. WER vs CER correlation
        ax6 = plt.subplot(2, 3, 6)
        plt.scatter(df['wer'], df['cer'], c=df['trial_number'], cmap='viridis', alpha=0.7)
        plt.xlabel('WER')
        plt.ylabel('CER')
        plt.title('WER vs CER Correlation')
        plt.colorbar(label='Trial Number')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = self.output_dir / "fast_optimization_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print summary
        total_evaluations = df['utterances_evaluated'].sum()
        avg_evaluations = df['utterances_evaluated'].mean()
        
        print(f"\n⚡ Fast Optimization Summary:")
        print(f"Total evaluations: {total_evaluations:,}")
        print(f"Challenging utterances per trial: {avg_evaluations:.0f}")
        print(f"Best trial WER: {df['wer'].min():.4f}")
        print(f"Improvement over first trial: {((df['wer'].iloc[0] - df['wer'].min()) / df['wer'].iloc[0] * 100):.1f}%")
        
        print(f"Visualizations saved to {self.output_dir}")


def run_fast_optimization(
    model_checkpoint: str,
    config_path: str,
    output_dir: str = 'tuning_results_fast',
    n_trials: int = 100,
    device: str = 'auto',
    use_cached_emissions: bool = False,
    cache_dir: str = 'cache/emissions'
) -> DecodingProfile:
    """Run the complete fast optimization pipeline."""
    from omegaconf import OmegaConf
    
    # Load configuration
    config = OmegaConf.load(config_path)
    
    # Auto-detect device
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Set up emission cache
    emission_cache = EmissionCache(cache_dir=cache_dir)
    cache_file = emission_cache._get_cache_path('val')
    
    # Handle emission generation/loading logic
    if use_cached_emissions:
        if cache_file.exists():
            print("Using existing cached emissions...")
        else:
            print(f"❌ Error: --use_cached_emissions specified but no cache found at {cache_file}")
            print("Either generate emissions first or remove --use_cached_emissions flag")
            raise FileNotFoundError(f"No cached emissions found at {cache_file}")
    else:
        if cache_file.exists():
            print(f"⚠️  Found existing cached emissions at {cache_file}")
            print("Generating fresh emissions with the specified model...")
        else:
            print("No cached emissions found. Generating fresh emissions...")
        
        # Use the working emission caching function from the old implementation
        from .emit import cache_model_emissions
        
        data_config = {
            'data_root': config.dataset.data_root,
            'batch_size': 16,
            'num_workers': 4,
            'splits': ['val'],
            'corpus_filter': getattr(config.dataset, 'corpus_filter', None),
            'bad_trials_dict': getattr(config.dataset, 'bad_trials_dict', None),
            'random_seed': getattr(config.dataset, 'random_seed', 42)
        }
        
        cache_config = {
            'cache_dir': cache_dir,
            'format': 'npz',
            'compress': True
        }
        
        print(f"Generating emissions with model: {model_checkpoint}")
        cache_model_emissions(
            model_checkpoint=model_checkpoint,
            data_config=data_config,
            cache_config=cache_config,
            device=device,
            model_config=config.model
        )
        print("✅ Fresh emissions generated and cached!")
    
    # Run fast optimization
    tuner = FastBayesianTuner(
        emission_cache=emission_cache,
        decoder_type='flashlight',
        output_dir=output_dir,
        n_trials=n_trials
    )
    
    best_profile = tuner.optimize()
    
    print(f"\n🎉 Fast Optimization completed!")
    print(f"Best WER: {best_profile.performance['wer']:.4f}")
    print(f"Best CER: {best_profile.performance['cer']:.4f}")
    print(f"Challenging utterances per trial: {best_profile.performance['utterances_evaluated']}")
    print(f"Best parameters:")
    print(f"  lm_weight: {best_profile.lm_weight:.3f}")
    print(f"  word_score: {best_profile.word_score:.3f}")
    print(f"  sil_score: {best_profile.sil_score:.3f}")
    print(f"  beam_size: {best_profile.beam_size} (fixed)")
    print(f"  beam_size_token: {best_profile.beam_size_token}")
    print(f"  beam_threshold: {best_profile.beam_threshold:.1f}")
    print(f"\nResults and visualizations saved to: {output_dir}")
    
    return best_profile
