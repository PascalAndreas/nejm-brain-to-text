#!/usr/bin/env python3
"""
Helper script to add corpus information to HDF5 files.

This script reads the corpus mapping from the CSV file and adds the corpus
information as an attribute to each trial in the HDF5 files.
"""

import os
from pathlib import Path
import h5py
import pandas as pd
from tqdm import tqdm


def add_corpus_to_hdf5_files(data_root: str, dry_run: bool = False):
    """
    Add corpus information to all HDF5 files.
    
    Args:
        data_root: Root directory containing the data (e.g., 'data/t15_copyTask_neuralData')
        dry_run: If True, only print what would be done without modifying files
    """
    data_root = Path(data_root)
    
    # Load corpus mapping
    csv_path = data_root.parent / 't15_copyTaskData_description.csv'
    df = pd.read_csv(csv_path)
    df['session'] = 't15.' + df['Date'].str.replace('-', '.')
    
    # Create a mapping dictionary for quick lookup
    corpus_map = {}
    for _, row in df.iterrows():
        key = (row['session'], row['Block number'])
        corpus_map[key] = row['Corpus']
    
    # Process all HDF5 files
    hdf5_root = data_root / 'hdf5_data_final'
    
    files_to_process = []
    for session_dir in sorted(hdf5_root.iterdir()):
        if not session_dir.is_dir() or not session_dir.name.startswith('t15'):
            continue
        
        for hdf5_file in session_dir.glob('data_*.hdf5'):
            files_to_process.append(hdf5_file)
    
    print(f"Found {len(files_to_process)} HDF5 files to process")
    
    if dry_run:
        print("DRY RUN - No files will be modified")
    
    # Process each file
    for hdf5_file in tqdm(files_to_process, desc="Processing files"):
        session_name = hdf5_file.parent.name
        
        if dry_run:
            # Just check what would be done
            with h5py.File(hdf5_file, 'r') as f:
                trials_needing_corpus = []
                for trial_key in f.keys():
                    if not trial_key.startswith('trial_'):
                        continue
                    
                    trial_data = f[trial_key]
                    if 'corpus' not in trial_data.attrs:
                        block_num = trial_data.attrs['block_num']
                        key = (session_name, block_num)
                        corpus = corpus_map.get(key, 'Unknown')
                        trials_needing_corpus.append((trial_key, corpus))
                
                if trials_needing_corpus:
                    print(f"\n{hdf5_file.name}: Would add corpus to {len(trials_needing_corpus)} trials")
                    # Show first few as examples
                    for trial_key, corpus in trials_needing_corpus[:3]:
                        print(f"  {trial_key}: {corpus}")
                    if len(trials_needing_corpus) > 3:
                        print(f"  ... and {len(trials_needing_corpus) - 3} more")
        else:
            # Actually modify the file
            trials_updated = 0
            with h5py.File(hdf5_file, 'r+') as f:
                for trial_key in f.keys():
                    if not trial_key.startswith('trial_'):
                        continue
                    
                    trial_data = f[trial_key]
                    
                    # Check if corpus already exists
                    if 'corpus' in trial_data.attrs:
                        continue
                    
                    # Get corpus from mapping
                    block_num = trial_data.attrs['block_num']
                    key = (session_name, block_num)
                    corpus = corpus_map.get(key, 'Unknown')
                    
                    # Add corpus attribute
                    trial_data.attrs['corpus'] = corpus
                    trials_updated += 1
            
            if trials_updated > 0:
                tqdm.write(f"Updated {trials_updated} trials in {hdf5_file.name}")


def verify_corpus_addition(data_root: str):
    """
    Verify that corpus information has been added to all trials.
    
    Args:
        data_root: Root directory containing the data
    """
    data_root = Path(data_root)
    hdf5_root = data_root / 'hdf5_data_final'
    
    total_trials = 0
    trials_with_corpus = 0
    trials_without_corpus = 0
    
    for session_dir in sorted(hdf5_root.iterdir()):
        if not session_dir.is_dir() or not session_dir.name.startswith('t15'):
            continue
        
        for hdf5_file in session_dir.glob('data_*.hdf5'):
            with h5py.File(hdf5_file, 'r') as f:
                for trial_key in f.keys():
                    if not trial_key.startswith('trial_'):
                        continue
                    
                    total_trials += 1
                    trial_data = f[trial_key]
                    
                    if 'corpus' in trial_data.attrs:
                        trials_with_corpus += 1
                    else:
                        trials_without_corpus += 1
    
    print(f"\nCorpus Verification Summary:")
    print(f"Total trials: {total_trials}")
    print(f"Trials with corpus: {trials_with_corpus}")
    print(f"Trials without corpus: {trials_without_corpus}")
    
    if trials_without_corpus == 0:
        print("✓ All trials have corpus information!")
    else:
        print(f"⚠ {trials_without_corpus} trials are missing corpus information")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add corpus information to HDF5 files")
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/t15_copyTask_neuralData",
        help="Root directory containing the data"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show what would be done without modifying files"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify that corpus information has been added"
    )
    
    args = parser.parse_args()
    
    if args.verify:
        verify_corpus_addition(args.data_root)
    else:
        add_corpus_to_hdf5_files(args.data_root, dry_run=args.dry_run)
