#!/usr/bin/env python3
"""
Compare emissions between legacy and new models.
Analyzes entropy, blank probability, and PER to understand performance differences.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from model_training.rnn_model import GRUDecoder
from model_training.data_augmentations import gauss_smooth
from models.lightning_module import BrainToTextLightningModule, EMA
from dataset import BrainToTextDataset, collate_fn
from models.base import Batch
from torch.utils.data import DataLoader


def load_legacy_model(model_path: str, device: str = 'cpu'):
    """Load legacy model using generate_submission.py approach."""
    print(f"Loading legacy model from: {model_path}")
    
    # Load model config
    model_args = OmegaConf.load(os.path.join(model_path, 'checkpoint/args.yaml'))
    
    # Initialize model
    model = GRUDecoder(
        neural_dim=model_args['model']['n_input_features'],
        n_units=model_args['model']['n_units'], 
        n_days=len(model_args['dataset']['sessions']),
        n_classes=model_args['dataset']['n_classes'],
        rnn_dropout=model_args['model']['rnn_dropout'],
        input_dropout=model_args['model']['input_network']['input_layer_dropout'],
        n_layers=model_args['model']['n_layers'],
        patch_size=model_args['model']['patch_size'],
        patch_stride=model_args['model']['patch_stride'],
    )
    
    # Load model weights
    checkpoint = torch.load(os.path.join(model_path, 'checkpoint/best_checkpoint'), 
                           weights_only=False, map_location=device)
    
    # Clean up keys (remove module. and _orig_mod. prefixes)
    state_dict = {}
    for key, value in checkpoint['model_state_dict'].items():
        clean_key = key.replace("module.", "").replace("_orig_mod.", "")
        state_dict[clean_key] = value
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    print("✅ Legacy model loaded successfully")
    return model, model_args


def load_new_model_with_ema(checkpoint_path: str, device: str = 'cpu'):
    """Load new model with EMA weights applied."""
    print(f"Loading new model from: {checkpoint_path}")
    
    # Load the Lightning model
    model = BrainToTextLightningModule.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()
    
    # Apply EMA weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'ema_state' in checkpoint:
        ema = EMA(model.model)
        ema.load_state_dict(checkpoint['ema_state'])
        ema.apply_shadow()
        print("✅ New model loaded with EMA weights applied")
    else:
        print("⚠️  No EMA weights found, using raw weights")
    
    return model


def get_legacy_emissions(model, model_args, batch_data, device):
    """Get emissions from legacy model (returns logits, not log_probs)."""
    
    # Prepare data like in generate_submission.py
    neural_data = batch_data['input_features'].to(device)
    day_indices = batch_data['day_indices'].to(device)
    n_time_steps = batch_data['n_time_steps'].to(device)
    
    # Map day indices to legacy model range (0-44)
    day_indices = torch.clamp(day_indices - 1, 0, 44)
    
    # Apply smoothing (as done in legacy training/inference)
    neural_data = gauss_smooth(neural_data, device)
    
    # Forward pass
    logits, _ = model(
        x=neural_data,
        day_idx=day_indices,
        states=None,
        return_state=True,
    )
    
    # Calculate output lengths using same formula as training
    patch_size = model_args['model']['patch_size']
    patch_stride = model_args['model']['patch_stride']
    adjusted_lens = ((n_time_steps - patch_size) / patch_stride + 1).to(torch.int32)
    
    return logits, adjusted_lens


def get_new_emissions(model, batch_data, device):
    """Get emissions from new model (returns log_probs)."""
    
    # Convert to Batch object
    batch = Batch.from_dataset_batch(batch_data)
    batch = batch.to(device)
    
    # Forward pass
    emissions = model(batch)
    
    return emissions.log_probs, emissions.out_lens


def calculate_phoneme_error_rate(predictions, targets, blank_idx=0):
    """Calculate phoneme error rate with CTC decoding."""
    
    # Convert log probabilities to predicted phonemes
    pred_phonemes = torch.argmax(predictions, dim=-1)  # [time]
    
    # Remove blanks and consecutive duplicates (basic CTC alignment)
    def ctc_decode(sequence, blank_idx):
        result = []
        prev = None
        for token in sequence:
            token = token.item()
            if token != blank_idx and token != prev:
                result.append(token)
            prev = token
        return result
    
    pred_sequence = ctc_decode(pred_phonemes, blank_idx)
    target_sequence = [t.item() for t in targets if t.item() != blank_idx and t.item() != -1]  # Remove padding/blanks
    
    # Calculate edit distance (simple implementation)
    if len(target_sequence) == 0:
        return 1.0 if len(pred_sequence) > 0 else 0.0
    
    # Simple character-level error rate
    correct = 0
    min_len = min(len(pred_sequence), len(target_sequence))
    
    for i in range(min_len):
        if pred_sequence[i] == target_sequence[i]:
            correct += 1
    
    # Account for length differences
    total_errors = abs(len(pred_sequence) - len(target_sequence)) + (min_len - correct)
    per = total_errors / len(target_sequence)
    
    return per


def analyze_emission_statistics(emissions, lengths, targets, model_name, is_logits=False):
    """Analyze emission statistics including entropy, blank probability, and PER."""
    
    print(f"\n📊 {model_name} Emission Analysis")
    print("=" * 50)
    
    # Convert logits to log_probs if needed
    if is_logits:
        log_probs = torch.log_softmax(emissions, dim=-1)
        print("  (Converted logits to log_probs)")
    else:
        log_probs = emissions
    
    probs = torch.exp(log_probs)
    
    # Collect statistics
    entropies = []
    blank_probs = []
    max_probs = []
    blank_ratios = []  # Fraction of timesteps that are blank predictions
    pers = []  # Phoneme error rates
    
    batch_size = emissions.shape[0]
    
    for i in range(batch_size):
        # Use actual sequence length
        seq_len = lengths[i].item()
        sample_log_probs = log_probs[i, :seq_len]  # [T, C]
        sample_probs = probs[i, :seq_len]  # [T, C]
        
        # Entropy per timestep
        sample_entropy = -torch.sum(sample_probs * sample_log_probs, dim=-1)  # [T]
        entropies.extend(sample_entropy.cpu().numpy())
        
        # Blank probability (class 0)
        sample_blank_probs = sample_probs[:, 0]  # [T]
        blank_probs.extend(sample_blank_probs.cpu().numpy())
        
        # Maximum probability per timestep
        sample_max_probs = torch.max(sample_probs, dim=-1)[0]  # [T]
        max_probs.extend(sample_max_probs.cpu().numpy())
        
        # Blank ratio (fraction of timesteps predicted as blank)
        predicted_classes = torch.argmax(sample_log_probs, dim=-1)  # [T]
        blank_ratio = (predicted_classes == 0).float().mean().item()
        blank_ratios.append(blank_ratio)
        
        # Calculate PER if targets available
        if targets is not None and i < len(targets):
            sample_targets = targets[i]
            if sample_targets is not None:
                try:
                    per = calculate_phoneme_error_rate(sample_log_probs, sample_targets)
                    pers.append(per)
                except Exception as e:
                    # Skip if PER calculation fails
                    continue
    
    # Convert to numpy arrays
    entropies = np.array(entropies)
    blank_probs = np.array(blank_probs)
    max_probs = np.array(max_probs)
    blank_ratios = np.array(blank_ratios)
    pers = np.array(pers) if pers else np.array([])
    
    # Print statistics
    print(f"  Total timesteps analyzed: {len(entropies)}")
    print(f"  Total sequences analyzed: {batch_size}")
    print(f"  Average entropy: {entropies.mean():.4f} ± {entropies.std():.4f}")
    print(f"  Average blank probability: {blank_probs.mean():.4f} ± {blank_probs.std():.4f}")
    print(f"  Average max probability: {max_probs.mean():.4f} ± {max_probs.std():.4f}")
    print(f"  Average blank ratio per sequence: {blank_ratios.mean():.4f} ± {blank_ratios.std():.4f}")
    
    if len(pers) > 0:
        print(f"  Average PER: {pers.mean():.4f} ({pers.mean()*100:.2f}%)")
        print(f"  PER std: {pers.std():.4f}")
    else:
        print(f"  PER: Could not calculate (no valid targets)")
    
    # Diagnosis
    if entropies.mean() < 0.5:
        print("  🚨 Very low entropy - likely overfit!")
    elif entropies.mean() < 1.0:
        print("  ⚠️  Low entropy - possibly overfit")
    else:
        print("  ✅ Entropy looks reasonable")
    
    if max_probs.mean() > 0.9:
        print("  🚨 Very high confidence - likely overfit!")
    elif max_probs.mean() > 0.8:
        print("  ⚠️  High confidence - possibly overfit")
    else:
        print("  ✅ Confidence looks reasonable")
    
    if blank_ratios.mean() > 0.8:
        print("  ⚠️  Very high blank ratio - model may be too sparse")
    elif blank_ratios.mean() < 0.5:
        print("  ⚠️  Low blank ratio - model may be too dense")
    else:
        print("  ✅ Blank ratio looks reasonable")
    
    if len(pers) > 0:
        if pers.mean() > 0.4:
            print("  🚨 Very high PER - poor phoneme recognition!")
        elif pers.mean() > 0.25:
            print("  ⚠️  High PER - suboptimal phoneme recognition")
        else:
            print("  ✅ PER looks reasonable")
    
    return {
        'entropies': entropies,
        'blank_probs': blank_probs,
        'max_probs': max_probs,
        'blank_ratios': blank_ratios,
        'pers': pers
    }


def compare_models():
    """Compare legacy and new model emissions."""
    
    print("🔬 COMPARING LEGACY vs NEW MODEL EMISSIONS")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load models
    legacy_model, legacy_args = load_legacy_model(
        'data/t15_pretrained_rnn_baseline/t15_pretrained_rnn_baseline', 
        device
    )
    new_model = load_new_model_with_ema('models/checkpoints/trained.ckpt', device)
    
    # Load validation data
    print("\nLoading validation dataset...")
    with open('pipeline/config.yaml', 'r') as f:
        config = OmegaConf.load(f)
    
    val_dataset = BrainToTextDataset(
        data_root=config.dataset.data_root,
        split='val',
        corpus_filter=None,
        bad_trials_dict=None
    )
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
    
    print(f"Validation dataset: {len(val_dataset)} samples")
    
    # Collect emissions from both models
    legacy_stats_all = {'entropies': [], 'blank_probs': [], 'max_probs': [], 'blank_ratios': [], 'pers': []}
    new_stats_all = {'entropies': [], 'blank_probs': [], 'max_probs': [], 'blank_ratios': [], 'pers': []}
    
    max_batches = 25  # Process more batches for better statistics
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(val_loader, desc="Comparing models")):
            if batch_idx >= max_batches:
                break
            
            try:
                # Get targets for PER calculation
                targets = batch_data.get('seq_class_ids', None)
                
                # Get legacy emissions (logits)
                legacy_logits, legacy_lens = get_legacy_emissions(
                    legacy_model, legacy_args, batch_data, device
                )
                
                # Get new emissions (log_probs)
                new_log_probs, new_lens = get_new_emissions(
                    new_model, batch_data, device
                )
                
                # Analyze both (only print detailed stats for first batch)
                print_detailed = (batch_idx == 0)
                
                legacy_stats = analyze_emission_statistics(
                    legacy_logits, legacy_lens, targets, 
                    "Legacy Model" if print_detailed else "", 
                    is_logits=True
                )
                new_stats = analyze_emission_statistics(
                    new_log_probs, new_lens, targets, 
                    "New Model" if print_detailed else "", 
                    is_logits=False
                )
                
                # Accumulate stats
                for key in legacy_stats_all:
                    if len(legacy_stats[key]) > 0:
                        legacy_stats_all[key].extend(legacy_stats[key])
                    if len(new_stats[key]) > 0:
                        new_stats_all[key].extend(new_stats[key])
                
            except Exception as e:
                print(f"  Error processing batch {batch_idx}: {e}")
                continue
    
    # Overall comparison
    print(f"\n🔍 OVERALL COMPARISON ({max_batches} batches)")
    print("=" * 60)
    
    metrics = [
        ('entropies', 'Entropy'),
        ('blank_probs', 'Blank Probability'), 
        ('max_probs', 'Max Probability'),
        ('blank_ratios', 'Blank Ratio'),
        ('pers', 'Phoneme Error Rate (PER)')
    ]
    
    for key, description in metrics:
        legacy_vals = np.array(legacy_stats_all[key])
        new_vals = np.array(new_stats_all[key])
        
        if len(legacy_vals) == 0 or len(new_vals) == 0:
            print(f"\n{description}: No data available")
            continue
            
        print(f"\n{description}:")
        print(f"  Legacy: {legacy_vals.mean():.4f} ± {legacy_vals.std():.4f}")
        print(f"  New:    {new_vals.mean():.4f} ± {new_vals.std():.4f}")
        
        diff = new_vals.mean() - legacy_vals.mean()
        
        if key == 'entropies':
            if diff < -0.5:
                print(f"  🚨 New model much lower entropy ({diff:.3f}) - likely overfit!")
            elif diff < -0.2:
                print(f"  ⚠️  New model lower entropy ({diff:.3f}) - possibly overfit")
            elif diff > 0.2:
                print(f"  ✅ New model higher entropy ({diff:.3f}) - better calibration!")
            else:
                print(f"  ✅ Entropy difference acceptable ({diff:.3f})")
        elif key == 'max_probs':
            if diff > 0.1:
                print(f"  🚨 New model much more confident ({diff:.3f}) - likely overfit!")
            elif diff > 0.05:
                print(f"  ⚠️  New model more confident ({diff:.3f}) - possibly overfit")
            elif diff < -0.05:
                print(f"  ✅ New model less confident ({diff:.3f}) - better calibration!")
            else:
                print(f"  ✅ Confidence difference acceptable ({diff:.3f})")
        elif key == 'blank_ratios':
            if abs(diff) > 0.2:
                print(f"  ⚠️  Large difference in blank ratios ({diff:.3f})")
            else:
                print(f"  ✅ Blank ratio difference acceptable ({diff:.3f})")
        elif key == 'pers':
            if diff > 0.1:
                print(f"  🚨 New model much higher PER ({diff:.3f}) - worse phoneme recognition!")
            elif diff > 0.05:
                print(f"  ⚠️  New model higher PER ({diff:.3f}) - somewhat worse")
            elif diff < -0.05:
                print(f"  ✅ New model lower PER ({diff:.3f}) - better phoneme recognition!")
            else:
                print(f"  ✅ PER difference acceptable ({diff:.3f})")
    
    # Final diagnosis
    print(f"\n🏥 COMPREHENSIVE DIAGNOSIS")
    print("=" * 30)
    
    new_entropy = np.array(new_stats_all['entropies']).mean()
    new_confidence = np.array(new_stats_all['max_probs']).mean()
    new_per = np.array(new_stats_all['pers']).mean() if len(new_stats_all['pers']) > 0 else 0
    
    legacy_entropy = np.array(legacy_stats_all['entropies']).mean()
    legacy_confidence = np.array(legacy_stats_all['max_probs']).mean()
    legacy_per = np.array(legacy_stats_all['pers']).mean() if len(legacy_stats_all['pers']) > 0 else 0
    
    print(f"Legacy Model:")
    print(f"  Entropy: {legacy_entropy:.4f}, Confidence: {legacy_confidence:.4f}, PER: {legacy_per:.4f}")
    
    print(f"New Model:")
    print(f"  Entropy: {new_entropy:.4f}, Confidence: {new_confidence:.4f}, PER: {new_per:.4f}")
    
    # Overall assessment
    if new_per > legacy_per + 0.1:
        print("\n🚨 NEW MODEL HAS SIGNIFICANTLY WORSE PHONEME RECOGNITION")
        print("   This explains the poor WER despite better calibration properties")
    elif abs(new_per - legacy_per) < 0.05:
        print("\n✅ MODELS HAVE SIMILAR PHONEME RECOGNITION")
        print("   The WER gap is likely due to decoder/hyperparameter issues")
    else:
        print(f"\n📊 PER difference: {new_per - legacy_per:.3f}")
    
    return legacy_stats_all, new_stats_all


if __name__ == "__main__":
    legacy_stats, new_stats = compare_models()
