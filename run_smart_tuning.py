#!/usr/bin/env python3
"""
Runner script for smart Bayesian hyperparameter optimization.

Usage:
    python run_smart_tuning.py --model models/checkpoints/A1/last.ckpt --config pipeline/configA1.yaml
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pipeline.fast_bayesian_tuning import run_fast_optimization


def main():
    parser = argparse.ArgumentParser(
        description="Run smart Bayesian optimization for decoder hyperparameters"
    )
    
    # Required arguments
    parser.add_argument(
        '--model', 
        type=str, 
        required=True,
        help='Path to trained model checkpoint'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        required=True,
        help='Path to pipeline configuration file'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default='tuning_results_fast',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--n_trials', 
        type=int, 
        default=100,
        help='Number of Bayesian optimization trials'
    )
    
    parser.add_argument(
        '--device', 
        type=str, 
        default='auto',
        choices=['auto', 'cuda', 'cpu', 'mps'],
        help='Device for model inference'
    )
    
    parser.add_argument(
        '--use_cached_emissions',
        action='store_true',
        help='Use existing cached emissions instead of generating new ones'
    )
    
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='cache/emissions',
        help='Directory for emission cache'
    )
    
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.model).exists():
        print(f"Error: Model checkpoint not found: {args.model}")
        sys.exit(1)
    
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    print("⚡ Starting Fast Bayesian Hyperparameter Optimization")
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    print(f"Output: {args.output_dir}")
    print(f"Trials: {args.n_trials}")
    print(f"Device: {args.device}")
    print(f"Cache dir: {args.cache_dir}")
    print(f"Use cached emissions: {args.use_cached_emissions}")
    print()
    
    # Run optimization
    try:
        best_profile = run_fast_optimization(
            model_checkpoint=args.model,
            config_path=args.config,
            output_dir=args.output_dir,
            n_trials=args.n_trials,
            device=args.device,
            use_cached_emissions=args.use_cached_emissions,
            cache_dir=args.cache_dir
        )
        
        print("\n✅ Optimization completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
