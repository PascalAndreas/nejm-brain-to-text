import os
from pathlib import Path
from typing import Dict, List, Optional, Literal, Tuple
import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf


class BrainToTextDataset(Dataset):
    """
    Dataset for brain-to-text data.
    
    This dataset handles train/val/test splits that are already present in the data structure,
    and supports filtering by corpus type.
    """
    
    def __init__(
        self,
        data_root: str,
        split: Literal['train', 'val', 'test'] = 'train',
        corpus_filter: Optional[List[str]] = None,
        bad_trials_dict: Optional[Dict] = None,
        random_seed: int = -1
    ):
        """
        Initialize the BrainToText dataset.
        
        Args:
            data_root: Root directory containing the data (e.g., 'data/t15_copyTask_neuralData')
            split: Which split to use ('train', 'val', or 'test')
            corpus_filter: Optional list of corpus types to include 
                          (e.g., ['50-Word', 'Switchboard']). If None, includes all.
            bad_trials_dict: Dictionary of trials to exclude from the dataset. Formatted as:
                {
                    'session_name_1': {block_num_1: [trial_nums], block_num_2: [trial_nums], ...},
                    'session_name_2': {block_num_1: [trial_nums], block_num_2: [trial_nums], ...},
                }
            random_seed: Random seed for reproducibility. -1 means no seed.
        """
        super().__init__()
        
        # Set random seed if specified
        if random_seed != -1:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
        
        self.data_root = Path(data_root)
        self.split = split
        self.corpus_filter = corpus_filter
        self.bad_trials_dict = bad_trials_dict or {}
        
        # Load configuration
        self.config = self._load_config()
        
        # Load corpus mapping from CSV
        self.corpus_df = self._load_corpus_mapping()
        
        # Build index of all valid trials
        self.trial_index = self._build_trial_index()
        
    def _load_config(self):
        """Load dataset configuration."""
        config_path = Path(__file__).parent / "config.yaml"
        return OmegaConf.load(config_path)
    
    def _load_corpus_mapping(self) -> pd.DataFrame:
        """Load the corpus mapping from the CSV file."""
        csv_path = self.data_root.parent / 't15_copyTaskData_description.csv'
        df = pd.read_csv(csv_path)
        # Convert date to session format (e.g., '2023-08-11' -> 't15.2023.08.11')
        df['session'] = 't15.' + df['Date'].str.replace('-', '.')
        return df
    
    def _get_day_index_for_session(self, session: str) -> int:
        """Get the day index for a session, consistent with RNN model indexing."""
        try:
            return self.config.sessions.index(session)
        except ValueError:
            # Session not found in config sessions
            print(f"Warning: Session {session} not found in config sessions, using -1")
            return -1
    
    def _get_corpus_for_trial(self, session: str, block_num: int) -> Optional[str]:
        """Get the corpus type for a given session and block number."""
        mask = (self.corpus_df['session'] == session) & (self.corpus_df['Block number'] == block_num)
        matches = self.corpus_df[mask]
        if len(matches) > 0:
            return matches.iloc[0]['Corpus']
        return None
    
    def _is_bad_trial(self, session: str, block_num: int, trial_num: int) -> bool:
        """Check if a trial is in the bad trials list."""
        if session not in self.bad_trials_dict:
            return False
        if str(block_num) not in self.bad_trials_dict[session]:
            return False
        return trial_num in self.bad_trials_dict[session][str(block_num)]
    
    def _build_trial_index(self) -> List[Tuple[Path, str]]:
        """
        Build an index of all valid trials for this split.
        
        Returns:
            List of tuples (file_path, trial_key)
        """
        trial_index = []
        hdf5_root = self.data_root / 'hdf5_data_final'
        
        # Iterate through all session directories
        for session_dir in sorted(hdf5_root.iterdir()):
            if not session_dir.is_dir() or not session_dir.name.startswith('t15'):
                continue
            
            session_name = session_dir.name
            
            # Look for the appropriate split file
            split_file = session_dir / f'data_{self.split}.hdf5'
            if not split_file.exists():
                continue
            
            # Open the file and index all valid trials
            with h5py.File(split_file, 'r') as f:
                for trial_key in f.keys():
                    if not trial_key.startswith('trial_'):
                        continue
                    
                    trial_data = f[trial_key]
                    block_num = trial_data.attrs['block_num']
                    trial_num = trial_data.attrs['trial_num']
                    
                    # Check if this is a bad trial
                    if self._is_bad_trial(session_name, block_num, trial_num):
                        continue
                    
                    # Check corpus filter
                    if self.corpus_filter is not None:
                        # First check if corpus is in HDF5, otherwise use CSV mapping
                        corpus = trial_data.attrs.get('corpus', None)
                        if corpus is None:
                            corpus = self._get_corpus_for_trial(session_name, block_num)
                        if corpus not in self.corpus_filter:
                            continue
                    
                    trial_index.append((split_file, trial_key))
        
        return trial_index
    
    def __len__(self) -> int:
        """Return the number of trials in this dataset."""
        return len(self.trial_index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single trial from the dataset.
        
        Returns:
            Dictionary containing:
                - input_features: Neural data tensor [time_steps, features]
                - seq_class_ids: Phoneme label sequence [phoneme_seq_len]
                - transcription: Character-level transcription [char_seq_len]
                - n_time_steps: Number of time steps in the trial
                - phone_seq_len: Number of phonemes in the label
                - day_index: Day index (session index)
                - block_num: Block number
                - trial_num: Trial number within block
                - session: Session name (e.g., 't15.2023.08.11')
                - corpus: Corpus type (e.g., '50-Word', 'Switchboard')
                - sentence_label: Original sentence text
        """
        file_path, trial_key = self.trial_index[idx]
        
        with h5py.File(file_path, 'r') as f:
            trial_data = f[trial_key]
            
            # Extract all the data
            input_features = torch.from_numpy(trial_data['input_features'][:])
            
            # Get attributes
            n_time_steps = trial_data.attrs['n_time_steps']
            block_num = trial_data.attrs['block_num']
            trial_num = trial_data.attrs['trial_num']
            session = trial_data.attrs['session']
            
            # Get corpus type - first check if it's in the HDF5 file, otherwise use CSV mapping
            corpus = trial_data.attrs.get('corpus', None)
            if corpus is None:
                corpus = self._get_corpus_for_trial(session, block_num)
            
            # Map session to day index (for compatibility with day-specific layers)
            day_index = self._get_day_index_for_session(session)
            
            # Build return dictionary with required fields
            result = {
                'input_features': input_features,
                'n_time_steps': n_time_steps,
                'day_index': day_index,
                'block_num': block_num,
                'trial_num': trial_num,
                'session': session,
                'corpus': corpus if corpus else 'Unknown',
            }
            
            # Add optional fields only if they exist (not for test data)
            if 'seq_class_ids' in trial_data:
                result['seq_class_ids'] = torch.from_numpy(trial_data['seq_class_ids'][:])
                result['phone_seq_len'] = trial_data.attrs.get('seq_len', 0)
            
            if 'transcription' in trial_data:
                result['transcription'] = torch.from_numpy(trial_data['transcription'][:])
            
            if 'sentence_label' in trial_data.attrs:
                result['sentence_label'] = trial_data.attrs['sentence_label']
            
            return result


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to handle batching of variable-length sequences.
    Handles optional fields that may not be present in test data.
    
    Args:
        batch: List of dictionaries from __getitem__
        
    Returns:
        Batched dictionary with padded sequences
    """
    # Initialize collated dict with required fields
    collated = {
        'input_features': [],
        'n_time_steps': [],
        'day_indices': [],
        'block_nums': [],
        'trial_nums': [],
        'sessions': [],
        'corpora': []
    }
    
    # Check which optional fields are present
    has_seq_class_ids = 'seq_class_ids' in batch[0]
    has_transcription = 'transcription' in batch[0]
    has_sentence_label = 'sentence_label' in batch[0]
    has_phone_seq_len = 'phone_seq_len' in batch[0]
    
    if has_seq_class_ids:
        collated['seq_class_ids'] = []
    if has_transcription:
        collated['transcriptions'] = []
    if has_sentence_label:
        collated['sentence_labels'] = []
    if has_phone_seq_len:
        collated['phone_seq_lens'] = []
    
    # Collect data from batch
    for item in batch:
        collated['input_features'].append(item['input_features'])
        collated['n_time_steps'].append(item['n_time_steps'])
        collated['day_indices'].append(item['day_index'])
        collated['block_nums'].append(item['block_num'])
        collated['trial_nums'].append(item['trial_num'])
        collated['sessions'].append(item['session'])
        collated['corpora'].append(item['corpus'])
        
        if has_seq_class_ids:
            collated['seq_class_ids'].append(item['seq_class_ids'])
        if has_transcription:
            collated['transcriptions'].append(item['transcription'])
        if has_sentence_label:
            collated['sentence_labels'].append(item['sentence_label'])
        if has_phone_seq_len:
            collated['phone_seq_lens'].append(item['phone_seq_len'])
    
    # Pad sequences
    collated['input_features'] = pad_sequence(
        collated['input_features'], 
        batch_first=True, 
        padding_value=0
    )
    
    if has_seq_class_ids:
        collated['seq_class_ids'] = pad_sequence(
            collated['seq_class_ids'], 
            batch_first=True, 
            padding_value=0
        )
    
    if has_transcription:
        collated['transcriptions'] = torch.stack(collated['transcriptions'])
    
    # Convert to tensors
    collated['n_time_steps'] = torch.tensor(collated['n_time_steps'])
    collated['day_indices'] = torch.tensor(collated['day_indices'])
    collated['block_nums'] = torch.tensor(collated['block_nums'])
    collated['trial_nums'] = torch.tensor(collated['trial_nums'])
    
    if has_phone_seq_len:
        collated['phone_seq_lens'] = torch.tensor(collated['phone_seq_lens'])
    
    return collated
