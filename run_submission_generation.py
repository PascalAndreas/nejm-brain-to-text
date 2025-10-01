#!/usr/bin/env python3
"""
Convenient wrapper script for generating submissions with optimized parameters.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pipeline.generate_submission_optimized import generate_submission


def main():
    parser = argparse.ArgumentParser(
        description="Generate submission using cached emissions and optimized decoder parameters"
    )
    
    # Required arguments
    parser.add_argument(
        '--cache_dir',
        type=str,
        default='cache/emissions',
        help='Directory containing cached emissions'
    )
    
    parser.add_argument(
        '--tuning_results',
        type=str,
        required=True,
        help='Directory containing tuning results (with best_profile.json)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        choices=['test', 'val'],
        help='Data split to process'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='submission_output',
        help='Output directory for submission files'
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not Path(args.cache_dir).exists():
        print(f"Error: Cache directory not found: {args.cache_dir}")
        sys.exit(1)
    
    if not Path(args.tuning_results).exists():
        print(f"Error: Tuning results directory not found: {args.tuning_results}")
        sys.exit(1)
    
    best_profile_path = Path(args.tuning_results) / "best_profile.json"
    if not best_profile_path.exists():
        print(f"Error: best_profile.json not found in {args.tuning_results}")
        sys.exit(1)
    
    print("🚀 Generating submission with optimized parameters")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Tuning results: {args.tuning_results}")
    print(f"Split: {args.split}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Generate submission
    try:
        submission_df, output_path = generate_submission(
            cache_dir=args.cache_dir,
            split=args.split,
            output_dir=args.output_dir,
            tuning_results_dir=args.tuning_results
        )
        
        print(f"\n✅ Success! Submission generated:")
        print(f"File: {output_path}")
        print(f"Entries: {len(submission_df)}")
        
    except Exception as e:
        print(f"\n❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
