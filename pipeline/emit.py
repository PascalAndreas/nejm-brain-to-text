"""Emission caching for efficient decoder tuning.

This module provides functionality to cache model emissions (log probabilities)
to disk, enabling efficient hyperparameter tuning of the decoder without
re-running the neural encoder.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import torch
import numpy as np
from tqdm import tqdm
import pickle
import h5py

from ..models.base import Batch, Emissions, NeuralEncoder
from ..models import build_encoder


class EmissionCache:
    """Manager for caching and loading neural encoder emissions.
    
    This class handles:
    - Forward pass through encoder and emission caching
    - Loading cached emissions for decoder tuning
    - Support for multiple file formats (NPZ, PT, HDF5)
    
    Args:
        cache_dir: Directory to store cached emissions
        format: File format ('npz', 'pt', or 'hdf5')
        compress: Whether to compress cached files
    """
    
    def __init__(
        self,
        cache_dir: str = 'cache/emissions',
        format: str = 'npz',
        compress: bool = True
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.format = format.lower()
        self.compress = compress
        
        if self.format not in ['npz', 'pt', 'hdf5']:
            raise ValueError(f"Unknown format: {format}. Use 'npz', 'pt', or 'hdf5'")
    
    def cache_emissions(
        self,
        model: NeuralEncoder,
        dataloader: torch.utils.data.DataLoader,
        split_name: str = 'val',
        device: str = 'cuda',
        use_amp: bool = True,
        batch_limit: Optional[int] = None
    ) -> str:
        """Generate and cache emissions for a dataset split.
        
        Args:
            model: Neural encoder model
            dataloader: Data loader for the split
            split_name: Name of the split (for filename)
            device: Device to run model on
            use_amp: Whether to use automatic mixed precision
            batch_limit: Limit number of batches (for debugging)
            
        Returns:
            Path to the cache file
        """
        model = model.to(device)
        model.eval()
        
        cache_file = self._get_cache_path(split_name)
        emissions_dict = {}
        
        with torch.no_grad():
            autocast_context = torch.cuda.amp.autocast if use_amp and device == 'cuda' else torch.no_grad
            
            with autocast_context():
                for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=f"Caching {split_name}")):
                    if batch_limit and batch_idx >= batch_limit:
                        break
                    
                    # Convert to standard batch format
                    batch = Batch.from_dataset_batch(batch_data)
                    batch = batch.to(device)
                    
                    # Forward pass
                    emissions = model(batch)
                    
                    # Store emissions for each utterance
                    for i in range(batch.batch_size):
                        utt_id = batch.utt_id[i] if batch.utt_id else f"{split_name}_{batch_idx}_{i}"
                        
                        # Extract individual emission
                        log_probs = emissions.log_probs[i, :emissions.out_lens[i]].cpu()
                        
                        emissions_dict[utt_id] = {
                            'log_probs': log_probs.numpy() if self.format == 'npz' else log_probs,
                            'out_len': emissions.out_lens[i].item(),
                            'meta': {
                                'day_id': batch.day_id[i].item() if batch.day_id is not None else None,
                                'session': batch.meta['sessions'][i] if batch.meta else None,
                                'block': batch.meta['block_nums'][i].item() if batch.meta else None,
                                'trial': batch.meta['trial_nums'][i].item() if batch.meta else None,
                            }
                        }
                        
                        # Add auxiliary outputs if present
                        if emissions.aux:
                            if 'aux_log_probs' in emissions.aux:
                                aux_log_probs = emissions.aux['aux_log_probs'][i, :emissions.out_lens[i]].cpu()
                                emissions_dict[utt_id]['aux_log_probs'] = (
                                    aux_log_probs.numpy() if self.format == 'npz' else aux_log_probs
                                )
        
        # Save to disk
        self._save_emissions(emissions_dict, cache_file)
        print(f"Cached {len(emissions_dict)} emissions to {cache_file}")
        
        return str(cache_file)
    
    def load_emissions(
        self,
        split_name: str,
        utterance_ids: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Load cached emissions from disk.
        
        Args:
            split_name: Name of the split
            utterance_ids: Optional list of specific utterance IDs to load
            
        Returns:
            Dictionary mapping utterance IDs to emissions
        """
        cache_file = self._get_cache_path(split_name)
        
        if not cache_file.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_file}")
        
        emissions_dict = self._load_emissions(cache_file)
        
        # Filter by utterance IDs if specified
        if utterance_ids is not None:
            emissions_dict = {
                utt_id: emissions
                for utt_id, emissions in emissions_dict.items()
                if utt_id in utterance_ids
            }
        
        print(f"Loaded {len(emissions_dict)} emissions from {cache_file}")
        return emissions_dict
    
    def _get_cache_path(self, split_name: str) -> Path:
        """Get cache file path for a split.
        
        Args:
            split_name: Name of the split
            
        Returns:
            Path to cache file
        """
        extension = {
            'npz': '.npz',
            'pt': '.pt',
            'hdf5': '.h5'
        }[self.format]
        
        return self.cache_dir / f"{split_name}_emissions{extension}"
    
    def _save_emissions(self, emissions_dict: Dict, cache_file: Path):
        """Save emissions to disk.
        
        Args:
            emissions_dict: Dictionary of emissions
            cache_file: Path to save to
        """
        if self.format == 'npz':
            # Convert to flat dictionary for NPZ
            flat_dict = {}
            metadata = {}
            
            for utt_id, emission in emissions_dict.items():
                flat_dict[f"{utt_id}_log_probs"] = emission['log_probs']
                flat_dict[f"{utt_id}_out_len"] = emission['out_len']
                
                if 'aux_log_probs' in emission:
                    flat_dict[f"{utt_id}_aux_log_probs"] = emission['aux_log_probs']
                
                metadata[utt_id] = emission.get('meta', {})
            
            # Save NPZ
            if self.compress:
                np.savez_compressed(cache_file, **flat_dict)
            else:
                np.savez(cache_file, **flat_dict)
            
            # Save metadata separately
            metadata_file = cache_file.with_suffix('.meta.pkl')
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
        
        elif self.format == 'pt':
            # Save as PyTorch file
            torch.save(emissions_dict, cache_file)
        
        elif self.format == 'hdf5':
            # Save as HDF5
            with h5py.File(cache_file, 'w') as f:
                for utt_id, emission in emissions_dict.items():
                    grp = f.create_group(utt_id)
                    grp.create_dataset('log_probs', data=emission['log_probs'])
                    grp.attrs['out_len'] = emission['out_len']
                    
                    if 'aux_log_probs' in emission:
                        grp.create_dataset('aux_log_probs', data=emission['aux_log_probs'])
                    
                    # Store metadata as attributes
                    if 'meta' in emission:
                        for key, value in emission['meta'].items():
                            if value is not None:
                                grp.attrs[key] = value
    
    def _load_emissions(self, cache_file: Path) -> Dict:
        """Load emissions from disk.
        
        Args:
            cache_file: Path to load from
            
        Returns:
            Dictionary of emissions
        """
        if self.format == 'npz':
            # Load NPZ file
            data = np.load(cache_file, allow_pickle=True)
            
            # Load metadata
            metadata_file = cache_file.with_suffix('.meta.pkl')
            if metadata_file.exists():
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
            else:
                metadata = {}
            
            # Reconstruct emissions dictionary
            emissions_dict = {}
            utt_ids = set()
            
            for key in data.files:
                if key.endswith('_log_probs'):
                    utt_id = key[:-10]  # Remove '_log_probs'
                    utt_ids.add(utt_id)
            
            for utt_id in utt_ids:
                emissions_dict[utt_id] = {
                    'log_probs': torch.from_numpy(data[f"{utt_id}_log_probs"]),
                    'out_len': int(data[f"{utt_id}_out_len"]),
                    'meta': metadata.get(utt_id, {})
                }
                
                if f"{utt_id}_aux_log_probs" in data:
                    emissions_dict[utt_id]['aux_log_probs'] = torch.from_numpy(
                        data[f"{utt_id}_aux_log_probs"]
                    )
            
            return emissions_dict
        
        elif self.format == 'pt':
            # Load PyTorch file
            return torch.load(cache_file)
        
        elif self.format == 'hdf5':
            # Load HDF5 file
            emissions_dict = {}
            
            with h5py.File(cache_file, 'r') as f:
                for utt_id in f.keys():
                    grp = f[utt_id]
                    
                    emission = {
                        'log_probs': torch.from_numpy(grp['log_probs'][:]),
                        'out_len': grp.attrs['out_len'],
                        'meta': {}
                    }
                    
                    if 'aux_log_probs' in grp:
                        emission['aux_log_probs'] = torch.from_numpy(grp['aux_log_probs'][:])
                    
                    # Load metadata from attributes
                    for key in ['day_id', 'session', 'block', 'trial']:
                        if key in grp.attrs:
                            emission['meta'][key] = grp.attrs[key]
                    
                    emissions_dict[utt_id] = emission
            
            return emissions_dict


def cache_model_emissions(
    model_checkpoint: str,
    data_config: Dict[str, Any],
    cache_config: Optional[Dict[str, Any]] = None,
    device: str = 'cuda'
) -> Dict[str, str]:
    """High-level function to cache emissions for all splits.
    
    Args:
        model_checkpoint: Path to model checkpoint
        data_config: Configuration for data loading
        cache_config: Configuration for caching
        device: Device to run model on
        
    Returns:
        Dictionary mapping split names to cache file paths
    """
    from torch.utils.data import DataLoader
    from ..dataset import BrainToTextDataset, collate_fn
    
    # Load model
    checkpoint = torch.load(model_checkpoint, map_location=device)
    
    if 'model_config' in checkpoint:
        model = build_encoder(
            checkpoint['model_config']['name'],
            checkpoint['model_config']['params']
        )
    else:
        # Try to load from state dict
        from ..models.gru_ctc import GRUCTC
        model = GRUCTC()
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create cache manager
    cache_config = cache_config or {}
    cache_manager = EmissionCache(**cache_config)
    
    # Process each split
    cache_files = {}
    
    for split in ['train', 'val', 'test']:
        if split not in data_config.get('splits', ['val']):
            continue
        
        # Create dataset and dataloader
        dataset = BrainToTextDataset(
            data_root=data_config['data_root'],
            split=split,
            corpus_filter=data_config.get('corpus_filter'),
            bad_trials_dict=data_config.get('bad_trials_dict')
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=data_config.get('batch_size', 32),
            shuffle=False,
            num_workers=data_config.get('num_workers', 4),
            collate_fn=collate_fn,
            pin_memory=(device == 'cuda')
        )
        
        # Cache emissions
        cache_file = cache_manager.cache_emissions(
            model=model,
            dataloader=dataloader,
            split_name=split,
            device=device,
            use_amp=data_config.get('use_amp', True)
        )
        
        cache_files[split] = cache_file
    
    return cache_files
