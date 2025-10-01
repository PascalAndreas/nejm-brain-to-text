#!/usr/bin/env python3
"""
Generate test emissions and then create submission using optimized decoder parameters.

This script:
1. Generates cached emissions for the test split
2. Uses optimized decoder parameters from tuning results
3. Generates the final submission CSV
"""

import sys
import argparse
from pathlib import Path
from omegaconf import OmegaConf
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pipeline.emit import cache_model_emissions
from pipeline.generate_submission_optimized import generate_submission


def generate_test_emissions(
    model_checkpoint: str,
    config_path: str,
    cache_dir: str = 'cache/emissions',
    device: str = 'auto'
):
    """Generate cached emissions for test split."""
    print("🔧 Generating test emissions...")
    
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
    
    # Set up data configuration for test split
    data_config = {
        'data_root': config.dataset.data_root,
        'batch_size': 16,
        'num_workers': 4,
        'splits': ['test'],  # Only generate test emissions
        'corpus_filter': getattr(config.dataset, 'corpus_filter', None),
        'bad_trials_dict': getattr(config.dataset, 'bad_trials_dict', None),
        'random_seed': getattr(config.dataset, 'random_seed', 42),
        'use_amp': getattr(config, 'use_amp', True)
    }
    
    cache_config = {
        'cache_dir': cache_dir,
        'format': 'npz',
        'compress': True
    }
    
    print(f"Model checkpoint: {model_checkpoint}")
    print(f"Data root: {data_config['data_root']}")
    print(f"Cache directory: {cache_dir}")
    
    # Generate emissions
    cache_files = cache_model_emissions(
        model_checkpoint=model_checkpoint,
        data_config=data_config,
        cache_config=cache_config,
        device=device,
        model_config=config.model
    )
    
    print("✅ Test emissions generated successfully!")
    return cache_files.get('test')


def main():
    parser = argparse.ArgumentParser(
        description="Generate test emissions and create submission with optimized parameters"
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
    
    parser.add_argument(
        '--tuning_results',
        type=str,
        required=True,
        help='Directory containing tuning results (with best_profile.json)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='cache/emissions',
        help='Directory for emission cache'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='submission_output',
        help='Output directory for submission files'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu', 'mps'],
        help='Device for model inference'
    )
    
    parser.add_argument(
        '--skip_emission_generation',
        action='store_true',
        help='Skip emission generation (use existing test emissions)'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.model).exists():
        print(f"Error: Model checkpoint not found: {args.model}")
        sys.exit(1)
    
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    if not Path(args.tuning_results).exists():
        print(f"Error: Tuning results directory not found: {args.tuning_results}")
        sys.exit(1)
    
    best_profile_path = Path(args.tuning_results) / "best_profile.json"
    if not best_profile_path.exists():
        print(f"Error: best_profile.json not found in {args.tuning_results}")
        sys.exit(1)
    
    print("🚀 Starting test emission generation and submission creation")
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    print(f"Tuning results: {args.tuning_results}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print()
    
    try:
        # Step 1: Generate test emissions (if not skipped)
        if not args.skip_emission_generation:
            cache_file = generate_test_emissions(
                model_checkpoint=args.model,
                config_path=args.config,
                cache_dir=args.cache_dir,
                device=args.device
            )
            print(f"Test emissions cached to: {cache_file}")
        else:
            print("⏭️  Skipping emission generation (using existing test emissions)")
        
        print()
        
        # Step 2: Generate submission using optimized parameters
        print("📝 Generating submission with optimized parameters...")
        submission_df, output_path = generate_submission(
            cache_dir=args.cache_dir,
            split='test',
            output_dir=args.output_dir,
            tuning_results_dir=args.tuning_results
        )
        
        print(f"\n🎉 Success! Complete pipeline finished:")
        print(f"✅ Test emissions generated")
        print(f"✅ Submission created: {output_path}")
        print(f"📊 Total entries: {len(submission_df)}")
        print(f"🔍 Sample predictions:")
        for i in range(min(3, len(submission_df))):
            print(f"    ID {i}: '{submission_df.iloc[i]['text']}'")
        
        print(f"\n🎯 Ready for submission!")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
