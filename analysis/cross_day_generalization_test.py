#!/usr/bin/env python3
"""
Cross-day generalization test for the RNN model.
Tests how much performance drops when using wrong day-specific transformations.
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import h5py
from tqdm import tqdm
import editdistance

# Add model_training to path to import modules
import sys
sys.path.append('../model_training')
from rnn_model import GRUDecoder
from evaluate_model_helpers import load_h5py_file, runSingleDecodingStep, LOGIT_TO_PHONEME
from data_augmentations import gauss_smooth

def load_pretrained_model():
    """Load the pretrained RNN model"""
    model_path = '../data/t15_pretrained_rnn_baseline/t15_pretrained_rnn_baseline'
    
    # Load model args
    model_args = OmegaConf.load(os.path.join(model_path, 'checkpoint/args.yaml'))
    
    # Set up device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Define model
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
                          map_location=device, weights_only=False)
    
    # Clean up state dict keys
    state_dict = checkpoint['model_state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "").replace("_orig_mod.", "")
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    
    return model, model_args, device

def compute_phoneme_error_rate(pred_phonemes, true_phonemes):
    """Compute phoneme error rate using edit distance"""
    if len(true_phonemes) == 0:
        return 1.0 if len(pred_phonemes) > 0 else 0.0
    
    # Convert to strings for edit distance computation
    pred_str = ' '.join(pred_phonemes)
    true_str = ' '.join(true_phonemes)
    
    edit_dist = editdistance.eval(pred_str, true_str)
    per = edit_dist / len(true_str.split())
    return per

def compute_ctc_loss(logits, targets, input_lengths, target_lengths):
    """Compute CTC loss for a batch"""
    ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    # Convert to tensors if needed
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits, dtype=torch.float32)
    if not isinstance(targets, torch.Tensor):
        targets = torch.tensor(targets, dtype=torch.int32)
    if not isinstance(input_lengths, torch.Tensor):
        input_lengths = torch.tensor(input_lengths, dtype=torch.int32)
    if not isinstance(target_lengths, torch.Tensor):
        target_lengths = torch.tensor(target_lengths, dtype=torch.int32)
    
    # CTC expects log probabilities in format [T, N, C]
    log_probs = torch.log_softmax(logits, dim=-1)
    log_probs = log_probs.permute(1, 0, 2)  # [T, N, C]
    
    loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
    return loss.item()

def load_validation_data(model_args, csv_path):
    """Load validation data from all available sessions"""
    data_dir = '../data/t15_copyTask_neuralData/hdf5_data_final'
    b2txt_csv_df = pd.read_csv(csv_path)
    
    val_data = {}
    sessions = model_args['dataset']['sessions']
    
    print("Loading validation data from all sessions...")
    for i, session in enumerate(sessions):
        session_path = os.path.join(data_dir, session)
        val_file = os.path.join(session_path, 'data_val.hdf5')
        
        if os.path.exists(val_file):
            print(f"Loading validation data from {session}")
            data = load_h5py_file(val_file, b2txt_csv_df)
            if len(data['neural_features']) > 0:
                val_data[session] = {
                    'data': data,
                    'session_idx': i
                }
                print(f"  Loaded {len(data['neural_features'])} validation trials")
    
    print(f"Total sessions with validation data: {len(val_data)}")
    return val_data

def run_cross_day_test(model, model_args, device, val_data):
    """Run the cross-day generalization test"""
    sessions = list(val_data.keys())
    n_sessions = len(sessions)
    all_day_indices = list(range(len(model_args['dataset']['sessions'])))
    
    print(f"\nRunning cross-day generalization test...")
    print(f"Testing {n_sessions} validation sessions against {len(all_day_indices)} day transformations")
    
    # Results matrix: [val_session][day_transform] = {'per': float, 'ctc_loss': float}
    results = {}
    
    for val_session in sessions:
        print(f"\n=== Testing validation data from {val_session} ===")
        
        val_session_data = val_data[val_session]['data']
        correct_day_idx = val_data[val_session]['session_idx']
        
        results[val_session] = {}
        
        # Test this validation data against all possible day transformations
        for day_idx in tqdm(all_day_indices, desc=f"Day transforms for {val_session}"):
            per_scores = []
            ctc_losses = []
            
            # Process each trial in this validation session
            for trial_idx in range(len(val_session_data['neural_features'])):
                # Get neural input and ground truth
                neural_input = val_session_data['neural_features'][trial_idx]
                true_seq_ids = val_session_data['seq_class_ids'][trial_idx]
                seq_len = val_session_data['seq_len'][trial_idx]
                
                # Get true phoneme sequence
                true_seq_ids = true_seq_ids[:seq_len]  # Trim to actual length
                true_phonemes = [LOGIT_TO_PHONEME[p] for p in true_seq_ids]
                
                # Prepare neural input
                neural_input = np.expand_dims(neural_input, axis=0)  # Add batch dim
                neural_input = torch.tensor(neural_input, device=device, dtype=torch.float32)
                
                # Run model with this day transformation
                try:
                    logits = runSingleDecodingStep(neural_input, day_idx, model, model_args, device)
                    
                    # Get predictions
                    pred_seq = np.argmax(logits[0], axis=-1)
                    pred_seq = [int(p) for p in pred_seq if p != 0]  # Remove blanks
                    pred_seq = [pred_seq[i] for i in range(len(pred_seq)) 
                              if i == 0 or pred_seq[i] != pred_seq[i-1]]  # Remove consecutive duplicates
                    pred_phonemes = [LOGIT_TO_PHONEME[p] for p in pred_seq if p < len(LOGIT_TO_PHONEME)]
                    
                    # Compute phoneme error rate
                    per = compute_phoneme_error_rate(pred_phonemes, true_phonemes)
                    per_scores.append(per)
                    
                    # Compute CTC loss
                    try:
                        # Prepare for CTC loss computation
                        logits_tensor = torch.tensor(logits, dtype=torch.float32)
                        targets = torch.tensor(true_seq_ids, dtype=torch.int32)
                        input_lengths = torch.tensor([logits.shape[1]], dtype=torch.int32)
                        target_lengths = torch.tensor([len(true_seq_ids)], dtype=torch.int32)
                        
                        ctc_loss = compute_ctc_loss(logits_tensor, targets, input_lengths, target_lengths)
                        ctc_losses.append(ctc_loss)
                    except Exception as e:
                        print(f"CTC loss computation failed for trial {trial_idx}: {e}")
                        ctc_losses.append(float('inf'))
                    
                except Exception as e:
                    print(f"Error processing trial {trial_idx} with day {day_idx}: {e}")
                    per_scores.append(1.0)  # Maximum error
                    ctc_losses.append(float('inf'))
            
            # Store average results for this day transformation
            avg_per = np.mean(per_scores) if per_scores else 1.0
            avg_ctc_loss = np.mean([l for l in ctc_losses if not np.isinf(l)]) if ctc_losses else float('inf')
            
            results[val_session][day_idx] = {
                'per': avg_per,
                'ctc_loss': avg_ctc_loss,
                'is_correct_day': (day_idx == correct_day_idx)
            }
            
            # Print progress for correct day
            if day_idx == correct_day_idx:
                print(f"  Correct day {day_idx}: PER={avg_per:.4f}, CTC Loss={avg_ctc_loss:.4f}")
    
    return results

def analyze_results(results, model_args):
    """Analyze and visualize the cross-day generalization results"""
    sessions = list(results.keys())
    all_sessions = model_args['dataset']['sessions']
    n_days = len(all_sessions)
    
    print(f"\n{'='*60}")
    print("CROSS-DAY GENERALIZATION ANALYSIS")
    print(f"{'='*60}")
    
    # Create matrices for visualization
    per_matrix = np.full((len(sessions), n_days), np.nan)
    ctc_matrix = np.full((len(sessions), n_days), np.nan)
    
    # Collect statistics
    correct_day_pers = []
    wrong_day_pers = []
    per_degradations = []
    
    for i, val_session in enumerate(sessions):
        val_results = results[val_session]
        correct_day_idx = None
        correct_day_per = None
        
        for day_idx in range(n_days):
            if day_idx in val_results:
                per = val_results[day_idx]['per']
                ctc_loss = val_results[day_idx]['ctc_loss']
                is_correct = val_results[day_idx]['is_correct_day']
                
                per_matrix[i, day_idx] = per
                if not np.isinf(ctc_loss):
                    ctc_matrix[i, day_idx] = ctc_loss
                
                if is_correct:
                    correct_day_idx = day_idx
                    correct_day_per = per
                    correct_day_pers.append(per)
        
        # Compute degradation for wrong days
        if correct_day_per is not None:
            for day_idx in range(n_days):
                if day_idx in val_results and day_idx != correct_day_idx:
                    wrong_per = val_results[day_idx]['per']
                    wrong_day_pers.append(wrong_per)
                    degradation = wrong_per - correct_day_per
                    per_degradations.append(degradation)
    
    # Print summary statistics
    print(f"Average PER on correct day: {np.mean(correct_day_pers):.4f} ± {np.std(correct_day_pers):.4f}")
    print(f"Average PER on wrong days: {np.mean(wrong_day_pers):.4f} ± {np.std(wrong_day_pers):.4f}")
    print(f"Average PER degradation: {np.mean(per_degradations):.4f} ± {np.std(per_degradations):.4f}")
    print(f"Relative performance drop: {np.mean(per_degradations) / np.mean(correct_day_pers) * 100:.1f}%")
    
    # Create detailed session-by-session report
    print(f"\n{'='*60}")
    print("DETAILED RESULTS BY SESSION")
    print(f"{'='*60}")
    
    for i, val_session in enumerate(sessions):
        val_results = results[val_session]
        correct_day_idx = None
        correct_per = None
        
        # Find correct day performance
        for day_idx, result in val_results.items():
            if result['is_correct_day']:
                correct_day_idx = day_idx
                correct_per = result['per']
                break
        
        print(f"\n{val_session}:")
        print(f"  Correct day {correct_day_idx}: PER = {correct_per:.4f}")
        
        # Find best and worst wrong day performance
        wrong_day_results = [(day_idx, res['per']) for day_idx, res in val_results.items() 
                           if not res['is_correct_day']]
        
        if wrong_day_results:
            wrong_day_results.sort(key=lambda x: x[1])  # Sort by PER
            best_wrong = wrong_day_results[0]
            worst_wrong = wrong_day_results[-1]
            
            print(f"  Best wrong day {best_wrong[0]}: PER = {best_wrong[1]:.4f} (Δ = {best_wrong[1] - correct_per:+.4f})")
            print(f"  Worst wrong day {worst_wrong[0]}: PER = {worst_wrong[1]:.4f} (Δ = {worst_wrong[1] - correct_per:+.4f})")
    
    # Create visualization
    create_heatmap_visualization(per_matrix, sessions, all_sessions, "Phoneme Error Rate")
    create_degradation_histogram(per_degradations)
    
    return {
        'per_matrix': per_matrix,
        'ctc_matrix': ctc_matrix,
        'correct_day_pers': correct_day_pers,
        'wrong_day_pers': wrong_day_pers,
        'per_degradations': per_degradations
    }

def create_heatmap_visualization(matrix, val_sessions, all_sessions, title):
    """Create heatmap visualization of the results"""
    plt.figure(figsize=(20, 8))
    
    # Create heatmap
    im = plt.imshow(matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label(title, rotation=270, labelpad=20)
    
    # Set ticks and labels
    plt.xticks(range(len(all_sessions)), [s.split('.')[-1] for s in all_sessions], rotation=45, ha='right')
    plt.yticks(range(len(val_sessions)), [s.split('.')[-1] for s in val_sessions])
    
    plt.xlabel('Day Index Used for Transformation')
    plt.ylabel('Validation Session')
    plt.title(f'Cross-Day Generalization: {title}')
    
    # Add text annotations for correct days
    for i, val_session in enumerate(val_sessions):
        correct_day_idx = all_sessions.index(val_session)
        plt.plot(correct_day_idx, i, 'ko', markersize=8, markerfacecolor='none', markeredgewidth=2)
    
    plt.tight_layout()
    plt.savefig(f'cross_day_generalization_{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_degradation_histogram(degradations):
    """Create histogram of performance degradations"""
    plt.figure(figsize=(10, 6))
    
    plt.hist(degradations, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(degradations), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(degradations):.4f}')
    plt.axvline(0, color='black', linestyle='-', alpha=0.5, label='No degradation')
    
    plt.xlabel('PER Degradation (Wrong Day - Correct Day)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Performance Degradation When Using Wrong Day Transformation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('per_degradation_histogram.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main analysis function"""
    print("Loading pretrained RNN model...")
    model, model_args, device = load_pretrained_model()
    
    print("Model loaded successfully!")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Load validation data
    csv_path = '../data/t15_copyTaskData_description.csv'
    val_data = load_validation_data(model_args, csv_path)
    
    if len(val_data) == 0:
        print("No validation data found!")
        return
    
    # Run cross-day generalization test
    results = run_cross_day_test(model, model_args, device, val_data)
    
    # Analyze results
    analysis = analyze_results(results, model_args)
    
    print(f"\n{'='*60}")
    print("CONCLUSION")
    print(f"{'='*60}")
    
    avg_degradation = np.mean(analysis['per_degradations'])
    relative_drop = avg_degradation / np.mean(analysis['correct_day_pers']) * 100
    
    if avg_degradation > 0.05:  # 5% absolute degradation
        print("🚨 SIGNIFICANT DAY-SPECIFIC EFFECT DETECTED!")
        print(f"   Average performance drops by {avg_degradation:.4f} PER ({relative_drop:.1f}%)")
        print("   This suggests the model is overly dependent on day-specific transformations.")
        print("   Removing day-specific layers could improve generalization.")
    elif avg_degradation > 0.02:  # 2% absolute degradation
        print("⚠️  MODERATE DAY-SPECIFIC EFFECT DETECTED")
        print(f"   Average performance drops by {avg_degradation:.4f} PER ({relative_drop:.1f}%)")
        print("   Day-specific transformations provide some benefit but may limit generalization.")
    else:
        print("✅ MINIMAL DAY-SPECIFIC EFFECT")
        print(f"   Average performance drops by only {avg_degradation:.4f} PER ({relative_drop:.1f}%)")
        print("   Day-specific transformations appear to have minimal impact.")
    
    print("\nAnalysis complete! Check the generated visualization files:")
    print("- cross_day_generalization_phoneme_error_rate.png")
    print("- per_degradation_histogram.png")

if __name__ == "__main__":
    main()
