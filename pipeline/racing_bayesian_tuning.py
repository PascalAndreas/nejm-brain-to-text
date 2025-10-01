"""
Racing-based Bayesian hyperparameter optimization for decoder parameters.

This module implements an intelligent racing approach that:
1. Uses successive halving over utterances (not epochs)
2. Dynamically selects informative trials
3. Adaptively allocates utterance budgets
4. Provides real-time progress visualization
5. Implements early stopping with statistical bounds
"""

import os
import json
import pickle
import string
import math
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
import time

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


@dataclass
class RacingTrial:
    """Trial in the racing system with adaptive evaluation."""
    trial_id: int
    params: Dict[str, Any]
    utterances_evaluated: int = 0
    total_wer: float = 0.0
    total_cer: float = 0.0
    total_words: int = 0
    total_chars: int = 0
    successful_decodes: int = 0
    is_alive: bool = True
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    
    @property
    def current_wer(self) -> float:
        """Current WER estimate."""
        if self.total_words > 0:
            return self.total_wer / self.total_words
        return 1.0
    
    @property
    def current_cer(self) -> float:
        """Current CER estimate."""
        if self.total_chars > 0:
            return self.total_cer / self.total_chars
        return 1.0
    
    @property
    def success_rate(self) -> float:
        """Success rate of decoding."""
        if self.utterances_evaluated > 0:
            return self.successful_decodes / self.utterances_evaluated
        return 0.0


def strip_punctuation_and_normalize(text: str) -> str:
    """Strip punctuation and normalize text for evaluation."""
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    text = text.lower()
    return text.strip()


def hoeffding_bound(n: int, confidence: float = 0.95) -> float:
    """Calculate Hoeffding bound for early stopping."""
    if n <= 0:
        return 1.0
    return math.sqrt(-math.log(1 - confidence) / (2 * n))


class RacingBayesianTuner:
    """Racing-based Bayesian optimization with dynamic trial selection."""
    
    def __init__(
        self,
        emission_cache: EmissionCache,
        decoder_type: str = 'flashlight',
        output_dir: str = 'tuning_results_racing',
        n_trials: int = 100,
        n_startup_trials: int = 15,
        initial_budget: int = 64,      # Start with 64 utterances
        max_budget: int = 512,         # Max 512 utterances per trial
        survival_rate: float = 0.5,    # Keep top 50% of trials
        confidence_level: float = 0.95, # For statistical bounds
        early_stop_margin: float = 0.05, # Stop if worse than best by 5%
        random_seed: int = 42
    ):
        self.emission_cache = emission_cache
        self.decoder_type = decoder_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials
        self.initial_budget = initial_budget
        self.max_budget = max_budget
        self.survival_rate = survival_rate
        self.confidence_level = confidence_level
        self.early_stop_margin = early_stop_margin
        self.random_seed = random_seed
        
        # Set up logging (suppress verbose output)
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Load emissions and shuffle for racing
        print("Loading cached emissions...")
        self.emissions_dict = self.emission_cache.load_emissions('val')
        self.utterance_ids = list(self.emissions_dict.keys())
        np.random.seed(self.random_seed)
        np.random.shuffle(self.utterance_ids)  # Shuffle for unbiased racing
        
        print(f"Loaded {len(self.emissions_dict)} validation emissions")
        print(f"Racing configuration:")
        print(f"  Initial budget: {self.initial_budget} utterances")
        print(f"  Max budget: {self.max_budget} utterances")
        print(f"  Survival rate: {self.survival_rate*100:.0f}%")
        print(f"  Early stop margin: {self.early_stop_margin*100:.1f}%")
        
        # Initialize optimized decoder once (expensive operation)
        print("🔧 Initializing optimized decoder (one-time setup)...")
        from decoding.decoder import OptimizedCTCDecoder
        self.decoder = OptimizedCTCDecoder(verbose=False)
        
        # Set good starting parameters
        self.good_starting_params = {
            'lm_weight': 2.5,
            'word_score': -0.7,
            'sil_score': 0.0,
            'beam_size': 150,
            'beam_size_token': 25,
            'beam_threshold': 24.0
        }
        print("✅ Decoder initialized! Racing will be much faster now.")
        
        # Storage for optimization results
        self.best_profile = None
        self.study = None
        self.racing_trials: List[RacingTrial] = []
        self.trial_history = []
        
        # Progress tracking
        self.current_round = 0
        self.total_rounds = 0
    
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
    
    def _evaluate_trial_on_utterances(
        self, 
        trial: RacingTrial, 
        utterance_batch: List[str],
        progress_bar: Optional[tqdm] = None
    ) -> None:
        """Evaluate a trial on a batch of utterances."""
        # Update decoder parameters (fast operation)
        self._update_decoder_params(trial.params)
        
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
                    
                    trial.total_wer += sample_wer * true_words
                    trial.total_cer += sample_cer * true_chars
                    trial.total_words += true_words
                    trial.total_chars += true_chars
                    trial.successful_decodes += 1
                
                trial.utterances_evaluated += 1
                
            except Exception as e:
                # Failed decode
                trial.utterances_evaluated += 1
                continue
            
            # Update progress bar
            if progress_bar:
                progress_bar.update(1)
    
    def _calculate_confidence_interval(self, trial: RacingTrial) -> Tuple[float, float]:
        """Calculate confidence interval for trial's WER using Hoeffding bound."""
        if trial.utterances_evaluated == 0:
            return (0.0, 1.0)
        
        current_wer = trial.current_wer
        bound = hoeffding_bound(trial.utterances_evaluated, self.confidence_level)
        
        lower = max(0.0, current_wer - bound)
        upper = min(1.0, current_wer + bound)
        
        return (lower, upper)
    
    def _should_eliminate_trial(self, trial: RacingTrial, best_wer: float) -> bool:
        """Determine if a trial should be eliminated based on statistical bounds."""
        if trial.utterances_evaluated < 10:  # Need minimum samples
            return False
        
        # Calculate confidence interval
        trial.confidence_interval = self._calculate_confidence_interval(trial)
        lower_bound, upper_bound = trial.confidence_interval
        
        # Eliminate if lower bound is worse than best + margin
        return lower_bound > best_wer + self.early_stop_margin
    
    def _run_racing_round(self, trials: List[RacingTrial], budget_per_trial: int) -> List[RacingTrial]:
        """Run one round of racing evaluation."""
        if not trials:
            return []
        
        # Calculate total evaluations needed
        total_evaluations = len(trials) * budget_per_trial
        
        # Create progress bar for this round
        desc = f"Round {self.current_round+1}/{self.total_rounds} (Budget: {budget_per_trial})"
        progress_bar = tqdm(
            total=total_evaluations,
            desc=desc,
            unit="eval",
            leave=False,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        # Evaluate each trial on its budget
        for trial in trials:
            if not trial.is_alive:
                continue
            
            # Get utterances for this trial
            start_idx = trial.utterances_evaluated
            end_idx = min(start_idx + budget_per_trial, len(self.utterance_ids))
            
            if start_idx >= end_idx:
                continue
            
            utterance_batch = self.utterance_ids[start_idx:end_idx]
            self._evaluate_trial_on_utterances(trial, utterance_batch, progress_bar)
        
        progress_bar.close()
        
        # Find best WER for elimination decisions
        alive_trials = [t for t in trials if t.is_alive and t.utterances_evaluated > 0]
        if not alive_trials:
            return trials
        
        best_wer = min(t.current_wer for t in alive_trials)
        
        # Apply early stopping based on statistical bounds
        for trial in alive_trials:
            if self._should_eliminate_trial(trial, best_wer):
                trial.is_alive = False
        
        # Keep only top fraction of trials (survival_rate)
        alive_trials = [t for t in trials if t.is_alive]
        if len(alive_trials) > 1:
            alive_trials.sort(key=lambda t: t.current_wer)
            n_survivors = max(1, int(len(alive_trials) * self.survival_rate))
            
            # Mark eliminated trials as dead
            for trial in alive_trials[n_survivors:]:
                trial.is_alive = False
        
        return trials
    
    def _create_objective(self):
        """Create objective function for Optuna optimization with racing."""
        def objective(trial):
            # Define hyperparameter search space centered around good starting values
            params = {
                'lm_weight': trial.suggest_float('lm_weight', 1.5, 4.0),  # Centered around 2.5
                'word_score': trial.suggest_float('word_score', -1.2, -0.2),  # Centered around -0.7
                'sil_score': trial.suggest_float('sil_score', -0.5, 0.5),  # Centered around 0.0
                'beam_size_token': trial.suggest_int('beam_size_token', 15, 35),  # Centered around 25
                'beam_threshold': trial.suggest_float('beam_threshold', 16.0, 32.0),  # Centered around 24.0
            }
            
            # Create racing trial
            racing_trial = RacingTrial(
                trial_id=trial.number,
                params=params
            )
            
            # Add to racing trials list
            self.racing_trials.append(racing_trial)
            
            # Run racing evaluation
            current_budget = self.initial_budget
            
            while (racing_trial.is_alive and 
                   racing_trial.utterances_evaluated < self.max_budget and
                   current_budget <= self.max_budget):
                
                # Evaluate on current budget
                start_idx = racing_trial.utterances_evaluated
                end_idx = min(start_idx + current_budget, len(self.utterance_ids))
                
                if start_idx >= end_idx:
                    break
                
                utterance_batch = self.utterance_ids[start_idx:end_idx]
                self._evaluate_trial_on_utterances(racing_trial, utterance_batch)
                
                # Check if we should stop early
                if len(self.racing_trials) > 1:
                    alive_trials = [t for t in self.racing_trials if t.is_alive and t.utterances_evaluated > 0]
                    if alive_trials:
                        best_wer = min(t.current_wer for t in alive_trials)
                        if self._should_eliminate_trial(racing_trial, best_wer):
                            racing_trial.is_alive = False
                            break
                
                # Double budget for next round
                current_budget = min(current_budget * 2, self.max_budget)
            
            # Store trial results
            final_wer = racing_trial.current_wer
            final_cer = racing_trial.current_cer
            
            # Store additional metrics for analysis
            trial.set_user_attr('cer', final_cer)
            trial.set_user_attr('success_rate', racing_trial.success_rate)
            trial.set_user_attr('utterances_evaluated', racing_trial.utterances_evaluated)
            trial.set_user_attr('confidence_interval', racing_trial.confidence_interval)
            
            # Store trial history for visualization
            self.trial_history.append({
                'trial_number': trial.number,
                'wer': final_wer,
                'cer': final_cer,
                'success_rate': racing_trial.success_rate,
                'utterances_evaluated': racing_trial.utterances_evaluated,
                'confidence_interval': racing_trial.confidence_interval,
                **params
            })
            
            return final_wer
        
        return objective
    
    def optimize(self) -> DecodingProfile:
        """Run racing-based Bayesian optimization."""
        print("\n🏁 Starting Racing-based Bayesian Optimization")
        print(f"Target trials: {self.n_trials}")
        
        # Create Optuna study
        self.study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_seed, n_startup_trials=self.n_startup_trials),
            pruner=MedianPruner(n_startup_trials=self.n_startup_trials),
            study_name="racing_decoder_tuning"
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
            name='racing_optimized',
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
        study_file = self.output_dir / "racing_study.pkl"
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
        """Create comprehensive visualizations of the racing optimization."""
        if not self.study or not self.trial_history:
            return
        
        print("Creating racing visualizations...")
        
        # Convert trial history to DataFrame
        df = pd.DataFrame(self.trial_history)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Racing progress over trials
        ax1 = plt.subplot(3, 3, 1)
        plt.plot(df['trial_number'], df['wer'], 'o-', alpha=0.7, markersize=4)
        running_best = df['wer'].cummin()
        plt.plot(df['trial_number'], running_best, 'r-', linewidth=2, label=f'Best: {running_best.iloc[-1]:.4f}')
        plt.xlabel('Trial Number')
        plt.ylabel('WER')
        plt.title('Racing Optimization Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 2. Utterances evaluated vs WER (efficiency analysis)
        ax2 = plt.subplot(3, 3, 2)
        scatter = plt.scatter(df['utterances_evaluated'], df['wer'], 
                            c=df['trial_number'], cmap='viridis', alpha=0.7)
        plt.xlabel('Utterances Evaluated')
        plt.ylabel('WER')
        plt.title('Evaluation Efficiency')
        plt.colorbar(scatter, label='Trial Number')
        plt.grid(True, alpha=0.3)
        
        # 3. Parameter vs WER scatter plots
        param_names = ['lm_weight', 'word_score', 'sil_score', 'beam_size_token', 'beam_threshold']
        
        for i, param in enumerate(param_names[:4]):
            ax = plt.subplot(3, 3, i + 3)
            scatter = plt.scatter(df[param], df['wer'], c=df['utterances_evaluated'], 
                                cmap='plasma', alpha=0.7)
            plt.xlabel(param.replace('_', ' ').title())
            plt.ylabel('WER')
            plt.title(f'{param.replace("_", " ").title()} vs WER')
            plt.colorbar(scatter, label='Utterances Evaluated')
            plt.grid(True, alpha=0.3)
        
        # 7. Confidence intervals visualization
        ax7 = plt.subplot(3, 3, 7)
        # Extract confidence intervals
        ci_lower = [ci[0] if isinstance(ci, (list, tuple)) and len(ci) == 2 else 0 for ci in df['confidence_interval']]
        ci_upper = [ci[1] if isinstance(ci, (list, tuple)) and len(ci) == 2 else 1 for ci in df['confidence_interval']]
        
        plt.scatter(df['trial_number'], df['wer'], alpha=0.7, label='WER')
        plt.fill_between(df['trial_number'], ci_lower, ci_upper, alpha=0.3, label='95% CI')
        plt.xlabel('Trial Number')
        plt.ylabel('WER')
        plt.title('WER with Confidence Intervals')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 8. Success rate vs WER
        ax8 = plt.subplot(3, 3, 8)
        plt.scatter(df['success_rate'], df['wer'], c=df['utterances_evaluated'], 
                   cmap='viridis', alpha=0.7)
        plt.xlabel('Success Rate')
        plt.ylabel('WER')
        plt.title('Success Rate vs WER')
        plt.colorbar(label='Utterances Evaluated')
        plt.grid(True, alpha=0.3)
        
        # 9. Racing efficiency histogram
        ax9 = plt.subplot(3, 3, 9)
        plt.hist(df['utterances_evaluated'], bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(df['utterances_evaluated'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {df["utterances_evaluated"].mean():.0f}')
        plt.xlabel('Utterances Evaluated')
        plt.ylabel('Count')
        plt.title('Racing Efficiency Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = self.output_dir / "racing_optimization_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create racing-specific analysis
        self._create_racing_analysis()
        
        print(f"Visualizations saved to {self.output_dir}")
    
    def _create_racing_analysis(self):
        """Create racing-specific analysis plots."""
        df = pd.DataFrame(self.trial_history)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Racing efficiency vs performance
        ax1 = axes[0, 0]
        efficiency = df['wer'] / (df['utterances_evaluated'] / self.max_budget)
        ax1.scatter(df['utterances_evaluated'], efficiency, alpha=0.7, c=df['wer'], cmap='RdYlBu_r')
        ax1.set_xlabel('Utterances Evaluated')
        ax1.set_ylabel('WER / Budget Fraction')
        ax1.set_title('Racing Efficiency Analysis')
        plt.colorbar(ax1.collections[0], ax=ax1, label='WER')
        ax1.grid(True, alpha=0.3)
        
        # Early stopping effectiveness
        ax2 = axes[0, 1]
        budget_bins = pd.cut(df['utterances_evaluated'], bins=5, labels=['Very Early', 'Early', 'Medium', 'Late', 'Full'])
        df['budget_bin'] = budget_bins
        df.boxplot(column='wer', by='budget_bin', ax=ax2)
        ax2.set_title('Early Stopping Effectiveness')
        ax2.set_xlabel('Evaluation Budget')
        ax2.set_ylabel('WER')
        
        # Best trials analysis
        ax3 = axes[1, 0]
        best_trials = df.nsmallest(10, 'wer')
        ax3.hist(best_trials['utterances_evaluated'], bins=10, alpha=0.7, edgecolor='black')
        ax3.axvline(best_trials['utterances_evaluated'].mean(), color='red', linestyle='--',
                   label=f'Mean: {best_trials["utterances_evaluated"].mean():.0f}')
        ax3.set_xlabel('Utterances Evaluated')
        ax3.set_ylabel('Count')
        ax3.set_title('Budget Usage (Top 10 Trials)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Racing convergence
        ax4 = axes[1, 1]
        # Calculate cumulative best WER
        cumulative_best = df['wer'].cummin()
        ax4.plot(df['trial_number'], cumulative_best, 'b-', linewidth=2, label='Best WER')
        
        # Add total evaluations line
        cumulative_evals = df['utterances_evaluated'].cumsum()
        ax4_twin = ax4.twinx()
        ax4_twin.plot(df['trial_number'], cumulative_evals, 'g--', alpha=0.7, label='Total Evaluations')
        
        ax4.set_xlabel('Trial Number')
        ax4.set_ylabel('Best WER', color='b')
        ax4_twin.set_ylabel('Cumulative Evaluations', color='g')
        ax4.set_title('Racing Convergence')
        ax4.grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        
        plt.tight_layout()
        
        # Save racing analysis
        racing_plot_file = self.output_dir / "racing_analysis.png"
        plt.savefig(racing_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print racing summary
        total_evaluations = df['utterances_evaluated'].sum()
        avg_evaluations = df['utterances_evaluated'].mean()
        efficiency_gain = (self.n_trials * self.max_budget) / total_evaluations
        
        print(f"\n🏁 Racing Analysis Summary:")
        print(f"Total evaluations: {total_evaluations:,} (vs {self.n_trials * self.max_budget:,} full)")
        print(f"Average per trial: {avg_evaluations:.0f} (vs {self.max_budget} full)")
        print(f"Efficiency gain: {efficiency_gain:.1f}x speedup")
        print(f"Best trial used: {self.study.best_trial.user_attrs['utterances_evaluated']} utterances")


def run_racing_optimization(
    model_checkpoint: str,
    config_path: str,
    output_dir: str = 'tuning_results_racing',
    n_trials: int = 100,
    device: str = 'auto'
) -> DecodingProfile:
    """Run the complete racing optimization pipeline."""
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
    cache_config = {
        'cache_dir': config.emission_cache.cache_dir,
        'format': 'npz',
        'compress': True
    }
    
    emission_cache = EmissionCache(**cache_config)
    
    # Check if we need to generate emissions
    cache_file = emission_cache._get_cache_path('val')
    if not cache_file.exists():
        print("Generating cached emissions...")
        
        # Load model and dataset
        from models import build_encoder
        model = build_encoder(config.model)
        checkpoint = torch.load(model_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        
        # Load dataset
        dataset_config = {
            'data_root': config.dataset.data_root,
            'batch_size': 16,
            'num_workers': 4,
            'splits': ['val']
        }
        dataset = BrainToTextDataset(**dataset_config)
        val_loader = dataset.val_dataloader()
        
        # Cache emissions
        emission_cache.cache_emissions(model, val_loader, 'val', device=device)
    
    # Run racing optimization
    tuner = RacingBayesianTuner(
        emission_cache=emission_cache,
        decoder_type='flashlight',
        output_dir=output_dir,
        n_trials=n_trials,
        initial_budget=64,    # Start small
        max_budget=512,       # Reasonable max
        survival_rate=0.5,    # Keep top 50%
        early_stop_margin=0.05  # 5% margin for early stopping
    )
    
    best_profile = tuner.optimize()
    
    print(f"\n🎉 Racing Optimization completed!")
    print(f"Best WER: {best_profile.performance['wer']:.4f}")
    print(f"Best CER: {best_profile.performance['cer']:.4f}")
    print(f"Utterances used: {best_profile.performance['utterances_evaluated']}")
    print(f"Best parameters:")
    print(f"  lm_weight: {best_profile.lm_weight:.3f}")
    print(f"  word_score: {best_profile.word_score:.3f}")
    print(f"  sil_score: {best_profile.sil_score:.3f}")
    print(f"  beam_size: {best_profile.beam_size} (fixed)")
    print(f"  beam_size_token: {best_profile.beam_size_token}")
    print(f"  beam_threshold: {best_profile.beam_threshold:.1f}")
    print(f"\nResults and visualizations saved to: {output_dir}")
    
    return best_profile
