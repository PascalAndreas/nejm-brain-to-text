"""Bayesian hyperparameter optimization for decoder parameters.

This module implements efficient Bayesian optimization of decoder hyperparameters
using cached emissions and corpus-specific evaluation.
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
import numpy as np
import torch
from tqdm import tqdm
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import logging

# Handle imports for both package and direct execution
try:
    from .emit import EmissionCache
    from ..decoding.decoder import Decoder
    from ..dataset import BrainToTextDataset
except ImportError:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from pipeline.emit import EmissionCache
    from decoding.decoder import Decoder
    from dataset import BrainToTextDataset
from jiwer import wer, cer


@dataclass
class DecodingProfile:
    """Decoding hyperparameter profile for a specific corpus or global use.
    
    Attributes:
        name: Profile name (e.g., 'global', '50-Word', 'Switchboard')
        lm_weight: Language model weight
        word_score: Word insertion penalty/bonus
        sil_score: Silence token score
        beam_size: Beam search width
        beam_size_token: Token-level beam size
        beam_threshold: Beam pruning threshold
        performance: Performance metrics on validation set
    """
    name: str
    lm_weight: float
    word_score: float
    sil_score: float
    beam_size: int
    beam_size_token: int
    beam_threshold: float
    performance: Optional[Dict[str, float]] = None
    
    def to_decoder_kwargs(self) -> Dict[str, Any]:
        """Convert profile to decoder initialization arguments."""
        return {
            'lm_weight': self.lm_weight,
            'word_score': self.word_score,
            'sil_score': self.sil_score,
            'beam_size': self.beam_size,
            'beam_size_token': self.beam_size_token,
            'beam_threshold': self.beam_threshold,
        }


class BayesianDecoderTuner:
    """Bayesian optimization for decoder hyperparameters with corpus-specific profiles.
    
    This class performs efficient hyperparameter tuning by:
    1. Using cached emissions to avoid repeated model inference
    2. Supporting corpus-specific optimization
    3. Using Bayesian optimization (Optuna) for efficient search
    """
    
    def __init__(
        self,
        cache_manager: EmissionCache,
        split_name: str = 'val',
        decoder_type: str = 'flashlight',
        output_dir: str = 'tuning_results',
        n_trials: int = 100,
        n_startup_trials: int = 10,
        random_seed: int = 42
    ):
        """Initialize Bayesian decoder tuner.
        
        Args:
            cache_manager: Emission cache manager
            split_name: Dataset split to use for tuning
            decoder_type: Type of decoder ('flashlight' or 'greedy')
            output_dir: Directory to save results
            n_trials: Number of optimization trials
            n_startup_trials: Number of random trials before Bayesian optimization
            random_seed: Random seed for reproducibility
        """
        self.cache_manager = cache_manager
        self.split_name = split_name
        self.decoder_type = decoder_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.n_trials = n_trials
        self.n_startup_trials = n_startup_trials
        self.random_seed = random_seed
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Load cached emissions
        self.emissions_dict = self.cache_manager.load_emissions(split_name)
        self.logger.info(f"Loaded {len(self.emissions_dict)} cached emissions for {split_name}")
        
        # Group emissions by corpus
        self.corpus_groups = self._group_emissions_by_corpus()
        self.logger.info(f"Found {len(self.corpus_groups)} corpus types: {list(self.corpus_groups.keys())}")
        
        # Store best profiles
        self.best_profiles: Dict[str, DecodingProfile] = {}
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the tuner."""
        logger = logging.getLogger(f"bayesian_tuner_{self.split_name}")
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.output_dir / f"tuning_{self.split_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _group_emissions_by_corpus(self) -> Dict[str, List[str]]:
        """Group emission utterance IDs by corpus type."""
        corpus_groups = {}
        
        for utt_id, emission in self.emissions_dict.items():
            corpus = emission.get('meta', {}).get('corpus', 'Unknown')
            if corpus not in corpus_groups:
                corpus_groups[corpus] = []
            corpus_groups[corpus].append(utt_id)
        
        # Sort by corpus size for better logging
        corpus_groups = dict(sorted(corpus_groups.items(), key=lambda x: len(x[1]), reverse=True))
        
        for corpus, utt_ids in corpus_groups.items():
            self.logger.info(f"  {corpus}: {len(utt_ids)} utterances")
        
        return corpus_groups
    
    def _evaluate_hyperparameters(
        self,
        params: Dict[str, Any],
        corpus_filter: Optional[str] = None
    ) -> Dict[str, float]:
        """Evaluate decoder hyperparameters on cached emissions.
        
        Args:
            params: Hyperparameter dictionary
            corpus_filter: If specified, evaluate only on this corpus
            
        Returns:
            Dictionary with performance metrics
        """
        # Create decoder with these parameters
        decoder = Decoder(backend=self.decoder_type, **params)
        
        # Select utterances to evaluate
        if corpus_filter:
            if corpus_filter not in self.corpus_groups:
                raise ValueError(f"Corpus '{corpus_filter}' not found in data")
            utt_ids = self.corpus_groups[corpus_filter]
        else:
            utt_ids = list(self.emissions_dict.keys())
        
        # Evaluate on selected utterances
        total_wer = 0.0
        total_cer = 0.0
        total_words = 0
        total_chars = 0
        successful_decodes = 0
        
        for utt_id in utt_ids:
            emission = self.emissions_dict[utt_id]
            
            # Skip if no ground truth available
            if 'ground_truth' not in emission.get('meta', {}):
                continue
            
            try:
                # Prepare inputs for decoder
                log_probs = emission['log_probs'].unsqueeze(0)  # Add batch dimension
                out_len = torch.tensor([emission['out_len']])
                
                # Decode
                result = decoder.decode(log_probs, out_len)
                
                # Handle result format
                if isinstance(result, list):
                    result = result[0] if result else {'sentence': ''}
                
                pred_sentence = result.get('sentence', '') if isinstance(result, dict) else str(result)
                true_sentence = emission['meta']['ground_truth']
                
                # Normalize text (proper submission normalization)
                from nejm_b2txt_utils.general_utils import remove_punctuation
                pred_sentence = remove_punctuation(pred_sentence)
                true_sentence = remove_punctuation(true_sentence)
                
                if pred_sentence and true_sentence:
                    # Calculate metrics
                    sample_wer = wer(true_sentence, pred_sentence)
                    sample_cer = cer(true_sentence, pred_sentence)
                    
                    # Accumulate
                    true_words = len(true_sentence.split())
                    true_chars = len(true_sentence)
                    
                    total_wer += sample_wer * true_words
                    total_cer += sample_cer * true_chars
                    total_words += true_words
                    total_chars += true_chars
                    successful_decodes += 1
                
            except Exception as e:
                # Skip failed decodes
                continue
        
        # Calculate aggregate metrics
        if total_words > 0 and successful_decodes > 0:
            avg_wer = total_wer / total_words
            avg_cer = total_cer / total_chars
            success_rate = successful_decodes / len(utt_ids)
        else:
            avg_wer = 1.0
            avg_cer = 1.0
            success_rate = 0.0
        
        return {
            'wer': avg_wer,
            'cer': avg_cer,
            'success_rate': success_rate,
            'num_utterances': len(utt_ids),
            'successful_decodes': successful_decodes
        }
    
    def _create_objective(self, corpus_filter: Optional[str] = None):
        """Create objective function for Optuna optimization.
        
        Args:
            corpus_filter: If specified, optimize only on this corpus
            
        Returns:
            Objective function for Optuna
        """
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'lm_weight': trial.suggest_float('lm_weight', 0.5, 5.0),
                'word_score': trial.suggest_float('word_score', -2.0, 1.0),
                'sil_score': trial.suggest_float('sil_score', -2.0, 2.0),
                'beam_size': trial.suggest_int('beam_size', 20, 200),
                'beam_size_token': trial.suggest_int('beam_size_token', 5, 20),
                'beam_threshold': trial.suggest_float('beam_threshold', 5.0, 50.0),
            }
            
            # Evaluate hyperparameters
            metrics = self._evaluate_hyperparameters(params, corpus_filter)
            
            # Log trial results
            corpus_str = f" ({corpus_filter})" if corpus_filter else ""
            self.logger.info(
                f"Trial {trial.number}{corpus_str}: WER={metrics['wer']:.4f}, "
                f"CER={metrics['cer']:.4f}, Success={metrics['success_rate']:.3f}"
            )
            
            # Store additional metrics for analysis
            trial.set_user_attr('cer', metrics['cer'])
            trial.set_user_attr('success_rate', metrics['success_rate'])
            trial.set_user_attr('num_utterances', metrics['num_utterances'])
            trial.set_user_attr('successful_decodes', metrics['successful_decodes'])
            
            # Return WER as the objective to minimize
            return metrics['wer']
        
        return objective
    
    def optimize_global_profile(self) -> DecodingProfile:
        """Optimize hyperparameters on the full validation set.
        
        Returns:
            Best global decoding profile
        """
        self.logger.info("Starting global hyperparameter optimization")
        
        # Create Optuna study
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_seed, n_startup_trials=self.n_startup_trials),
            pruner=MedianPruner(n_startup_trials=self.n_startup_trials),
            study_name=f"global_{self.split_name}"
        )
        
        # Optimize
        objective = self._create_objective(corpus_filter=None)
        study.optimize(objective, n_trials=self.n_trials)
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        self.logger.info(f"Global optimization completed. Best WER: {best_value:.4f}")
        self.logger.info(f"Best parameters: {best_params}")
        
        # Create profile
        profile = DecodingProfile(
            name='global',
            lm_weight=best_params['lm_weight'],
            word_score=best_params['word_score'],
            sil_score=best_params['sil_score'],
            beam_size=best_params['beam_size'],
            beam_size_token=best_params['beam_size_token'],
            beam_threshold=best_params['beam_threshold'],
            performance={
                'wer': best_value,
                'cer': study.best_trial.user_attrs['cer'],
                'success_rate': study.best_trial.user_attrs['success_rate']
            }
        )
        
        self.best_profiles['global'] = profile
        
        # Save study results
        study_file = self.output_dir / f"study_global_{self.split_name}.pkl"
        with open(study_file, 'wb') as f:
            pickle.dump(study, f)
        
        return profile
    
    def optimize_corpus_specific_profiles(
        self,
        base_profile: Optional[DecodingProfile] = None,
        min_utterances: int = 50
    ) -> Dict[str, DecodingProfile]:
        """Optimize corpus-specific hyperparameters.
        
        Args:
            base_profile: Base profile to initialize search (uses global if None)
            min_utterances: Minimum utterances required for corpus-specific optimization
            
        Returns:
            Dictionary mapping corpus names to their best profiles
        """
        if base_profile is None:
            base_profile = self.best_profiles.get('global')
            if base_profile is None:
                raise ValueError("No base profile provided and no global profile available")
        
        self.logger.info("Starting corpus-specific hyperparameter optimization")
        
        corpus_profiles = {}
        
        for corpus, utt_ids in self.corpus_groups.items():
            if len(utt_ids) < min_utterances:
                self.logger.info(f"Skipping {corpus} (only {len(utt_ids)} utterances, need {min_utterances})")
                # Use global profile for small corpora
                corpus_profiles[corpus] = DecodingProfile(
                    name=corpus,
                    lm_weight=base_profile.lm_weight,
                    word_score=base_profile.word_score,
                    sil_score=base_profile.sil_score,
                    beam_size=base_profile.beam_size,
                    beam_size_token=base_profile.beam_size_token,
                    beam_threshold=base_profile.beam_threshold,
                    performance=None  # Not optimized
                )
                continue
            
            self.logger.info(f"Optimizing for {corpus} ({len(utt_ids)} utterances)")
            
            # Create study for this corpus
            study = optuna.create_study(
                direction='minimize',
                sampler=TPESampler(seed=self.random_seed, n_startup_trials=self.n_startup_trials),
                pruner=MedianPruner(n_startup_trials=self.n_startup_trials),
                study_name=f"{corpus}_{self.split_name}"
            )
            
            # Optimize with fewer trials for corpus-specific tuning
            corpus_trials = max(20, self.n_trials // 3)  # Use 1/3 of global trials, minimum 20
            objective = self._create_objective(corpus_filter=corpus)
            study.optimize(objective, n_trials=corpus_trials)
            
            # Get best parameters
            best_params = study.best_params
            best_value = study.best_value
            
            self.logger.info(f"{corpus} optimization completed. Best WER: {best_value:.4f}")
            
            # Create profile
            profile = DecodingProfile(
                name=corpus,
                lm_weight=best_params['lm_weight'],
                word_score=best_params['word_score'],
                sil_score=best_params['sil_score'],
                beam_size=best_params['beam_size'],
                beam_size_token=best_params['beam_size_token'],
                beam_threshold=best_params['beam_threshold'],
                performance={
                    'wer': best_value,
                    'cer': study.best_trial.user_attrs['cer'],
                    'success_rate': study.best_trial.user_attrs['success_rate']
                }
            )
            
            corpus_profiles[corpus] = profile
            
            # Save study results
            study_file = self.output_dir / f"study_{corpus}_{self.split_name}.pkl"
            with open(study_file, 'wb') as f:
                pickle.dump(study, f)
        
        self.best_profiles.update(corpus_profiles)
        return corpus_profiles
    
    def save_profiles(self, filename: Optional[str] = None) -> str:
        """Save all optimized profiles to disk.
        
        Args:
            filename: Optional filename (defaults to profiles_{split_name}.json)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"profiles_{self.split_name}.json"
        
        filepath = self.output_dir / filename
        
        # Convert profiles to serializable format
        profiles_dict = {}
        for name, profile in self.best_profiles.items():
            profiles_dict[name] = asdict(profile)
        
        with open(filepath, 'w') as f:
            json.dump(profiles_dict, f, indent=2)
        
        self.logger.info(f"Saved {len(profiles_dict)} profiles to {filepath}")
        return str(filepath)
    
    def load_profiles(self, filepath: str) -> Dict[str, DecodingProfile]:
        """Load profiles from disk.
        
        Args:
            filepath: Path to profiles JSON file
            
        Returns:
            Dictionary of loaded profiles
        """
        with open(filepath, 'r') as f:
            profiles_dict = json.load(f)
        
        profiles = {}
        for name, profile_data in profiles_dict.items():
            profiles[name] = DecodingProfile(**profile_data)
        
        self.best_profiles.update(profiles)
        self.logger.info(f"Loaded {len(profiles)} profiles from {filepath}")
        return profiles
    
    def get_profile_for_corpus(self, corpus: str) -> DecodingProfile:
        """Get the best profile for a specific corpus.
        
        Args:
            corpus: Corpus name
            
        Returns:
            Best decoding profile for this corpus (falls back to global)
        """
        if corpus in self.best_profiles:
            return self.best_profiles[corpus]
        elif 'global' in self.best_profiles:
            self.logger.warning(f"No specific profile for {corpus}, using global profile")
            return self.best_profiles['global']
        else:
            raise ValueError(f"No profile available for {corpus} and no global profile")


def run_full_optimization(
    model_checkpoint: str,
    data_config: Dict[str, Any],
    cache_config: Optional[Dict[str, Any]] = None,
    tuning_config: Optional[Dict[str, Any]] = None,
    device: str = 'cuda',
    model_config: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, DecodingProfile], str]:
    """Run complete Bayesian optimization pipeline.
    
    Args:
        model_checkpoint: Path to trained model checkpoint
        data_config: Data loading configuration
        cache_config: Emission caching configuration
        tuning_config: Bayesian tuning configuration
        device: Device for model inference
        
    Returns:
        Tuple of (profiles_dict, profiles_file_path)
    """
    from .emit import cache_model_emissions
    
    # Set default configurations
    cache_config = cache_config or {'format': 'npz', 'compress': True}
    tuning_config = tuning_config or {'n_trials': 100, 'n_startup_trials': 10}
    
    # Step 1: Cache emissions if not already cached
    print("Step 1: Caching model emissions...")
    cache_files = cache_model_emissions(
        model_checkpoint=model_checkpoint,
        data_config=data_config,
        cache_config=cache_config,
        device=device,
        model_config=model_config
    )
    
    # Step 2: Initialize tuner
    print("Step 2: Initializing Bayesian tuner...")
    cache_manager = EmissionCache(**cache_config)
    
    # Separate min_utterances from tuner config
    min_utterances = tuning_config.pop('min_utterances', 50)
    
    tuner = BayesianDecoderTuner(
        cache_manager=cache_manager,
        split_name='val',
        **tuning_config
    )
    
    # Step 3: Optimize global profile
    print("Step 3: Optimizing global profile...")
    global_profile = tuner.optimize_global_profile()
    
    # Step 4: Optimize corpus-specific profiles
    print("Step 4: Optimizing corpus-specific profiles...")
    corpus_profiles = tuner.optimize_corpus_specific_profiles(
        base_profile=global_profile, 
        min_utterances=min_utterances
    )
    
    # Step 5: Save results
    print("Step 5: Saving results...")
    profiles_file = tuner.save_profiles()
    
    # Print summary
    print("\n=== OPTIMIZATION RESULTS ===")
    print(f"Global profile WER: {global_profile.performance['wer']:.4f}")
    for corpus, profile in corpus_profiles.items():
        if profile.performance:
            print(f"{corpus} profile WER: {profile.performance['wer']:.4f}")
        else:
            print(f"{corpus} profile: Using global parameters (insufficient data)")
    
    return tuner.best_profiles, profiles_file
