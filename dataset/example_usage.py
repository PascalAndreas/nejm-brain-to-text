#!/usr/bin/env python3
"""
Example usage of the BrainToTextDataset class.
"""

from torch.utils.data import DataLoader
from dataset import BrainToTextDataset, collate_fn


def main():
    # Example 1: Load training data with all corpora
    train_dataset = BrainToTextDataset(
        data_root='data/t15_copyTask_neuralData',
        split='train',
        random_seed=42
    )
    print(f"Training dataset size: {len(train_dataset)} trials")
    
    # Example 2: Load validation data filtered by corpus
    val_dataset = BrainToTextDataset(
        data_root='data/t15_copyTask_neuralData',
        split='val',
        corpus_filter=['50-Word', 'Switchboard'],
        random_seed=42
    )
    print(f"Validation dataset size (50-Word + Switchboard): {len(val_dataset)} trials")
    
    # Example 3: Load test data with bad trials excluded
    bad_trials = {
        't15.2023.08.13': {
            '1': [0, 1, 2],  # Exclude trials 0, 1, 2 from block 1
            '2': [5]         # Exclude trial 5 from block 2
        }
    }
    test_dataset = BrainToTextDataset(
        data_root='data/t15_copyTask_neuralData',
        split='test',
        bad_trials_dict=bad_trials,
        random_seed=42
    )
    print(f"Test dataset size (with exclusions): {len(test_dataset)} trials")
    
    # Example 4: Create a DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0  # Set to > 0 for parallel data loading
    )
    
    # Example 5: Iterate through a batch
    print("\nExample batch from training data:")
    for batch_idx, batch in enumerate(train_loader):
        if batch_idx == 0:  # Just show first batch
            print(f"  input_features shape: {batch['input_features'].shape}")
            print(f"  seq_class_ids shape: {batch['seq_class_ids'].shape}")
            print(f"  n_time_steps: {batch['n_time_steps'][:5]}...")
            print(f"  phone_seq_lens: {batch['phone_seq_lens'][:5]}...")
            print(f"  day_indices: {batch['day_indices'][:5]}...")
            print(f"  sessions (first 3): {batch['sessions'][:3]}")
            print(f"  corpora (first 3): {batch['corpora'][:3]}")
            break
    
    # Example 6: Access a single trial
    print("\nExample single trial:")
    trial = train_dataset[0]
    print(f"  input_features shape: {trial['input_features'].shape}")
    print(f"  seq_class_ids shape: {trial['seq_class_ids'].shape}")
    print(f"  session: {trial['session']}")
    print(f"  corpus: {trial['corpus']}")
    print(f"  sentence: {trial['sentence_label']}")
    print(f"  block_num: {trial['block_num']}, trial_num: {trial['trial_num']}")
    
    # Example 7: Check session configuration
    print(f"\nDataset configuration:")
    print(f"  Total sessions: {len(train_dataset.config.sessions)}")
    print(f"  First 5 sessions: {train_dataset.config.sessions[:5]}")
    print(f"  Available corpus types: {train_dataset.config.corpus_types}")


if __name__ == "__main__":
    main()
