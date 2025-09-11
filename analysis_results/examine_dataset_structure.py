#!/usr/bin/env python3
"""
Script to examine the structure of validation and test datasets
to understand what ground truth data is available.
"""

import h5py
import pandas as pd
import os

def examine_hdf5_structure(file_path, max_trials=3):
    """Examine the structure of an HDF5 file"""
    print(f"\n=== Examining: {file_path} ===")
    
    if not os.path.exists(file_path):
        print(f"File does not exist: {file_path}")
        return
    
    try:
        with h5py.File(file_path, 'r') as f:
            keys = list(f.keys())
            print(f"Number of trials: {len(keys)}")
            print(f"First few trial keys: {keys[:5]}")
            
            # Examine first few trials in detail
            for i, key in enumerate(keys[:max_trials]):
                print(f"\n--- Trial {key} ---")
                trial_group = f[key]
                
                # List all datasets and attributes
                print("Datasets:")
                for dataset_name in trial_group.keys():
                    dataset = trial_group[dataset_name]
                    print(f"  {dataset_name}: shape={dataset.shape}, dtype={dataset.dtype}")
                    
                    # Show first few values for small datasets
                    if dataset.size < 20:
                        print(f"    Values: {dataset[:]}")
                    elif dataset_name == 'transcription':
                        # Convert transcription to string
                        trans_bytes = dataset[:]
                        trans_str = ''.join([chr(b) for b in trans_bytes if b != 0])
                        print(f"    Text: '{trans_str}'")
                    elif dataset_name == 'seq_class_ids':
                        print(f"    First 10 phoneme IDs: {dataset[:10]}")
                
                print("Attributes:")
                for attr_name in trial_group.attrs.keys():
                    attr_value = trial_group.attrs[attr_name]
                    print(f"  {attr_name}: {attr_value}")
                    
    except Exception as e:
        print(f"Error reading file: {e}")

def main():
    # Load CSV for context
    csv_path = '../data/t15_copyTaskData_description.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print("Dataset description CSV loaded successfully")
        print(f"Columns: {list(df.columns)}")
        print(f"Number of entries: {len(df)}")
        
        # Show split distribution
        if 'Split' in df.columns:
            print("\nSplit distribution:")
            print(df['Split'].value_counts())
    else:
        print("CSV file not found")
    
    # Pick a few representative sessions to examine
    base_dir = '../data/t15_copyTask_neuralData/hdf5_data_final'
    
    # Find sessions with different file types
    sessions_to_check = []
    
    # Look for sessions with val, test, and train files
    for session in ['t15.2023.08.13', 't15.2023.08.18', 't15.2023.10.20']:
        session_dir = os.path.join(base_dir, session)
        if os.path.exists(session_dir):
            sessions_to_check.append(session)
    
    for session in sessions_to_check:
        session_dir = os.path.join(base_dir, session)
        
        print(f"\n\n{'='*60}")
        print(f"SESSION: {session}")
        print(f"{'='*60}")
        
        # Check each split type
        for split in ['train', 'val', 'test']:
            file_path = os.path.join(session_dir, f'data_{split}.hdf5')
            examine_hdf5_structure(file_path, max_trials=2)
    
    # Also check what the CSV says about these sessions
    if os.path.exists(csv_path):
        print(f"\n\n{'='*60}")
        print("CSV INFO FOR EXAMINED SESSIONS")
        print(f"{'='*60}")
        
        df = pd.read_csv(csv_path)
        for session in sessions_to_check:
            # Extract date from session name
            parts = session.split('.')
            if len(parts) >= 4:
                year, month, day = parts[1], parts[2], parts[3]
                date_str = f'{year}-{month}-{day}'
                
                session_rows = df[df['Date'] == date_str]
                if not session_rows.empty:
                    print(f"\n{session} ({date_str}):")
                    print(session_rows[['Block number', 'Number of sentences', 'Corpus', 'Split']].to_string(index=False))

if __name__ == "__main__":
    main()
