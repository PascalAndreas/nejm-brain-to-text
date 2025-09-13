#!/usr/bin/env python3
"""
Analysis script to investigate the effect of day-specific transformations in the pretrained RNN.
This script loads the pretrained model and tests how much the day index affects outputs.
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

# Add model_training to path to import modules
import sys
sys.path.append('../model_training')
from rnn_model import GRUDecoder
from evaluate_model_helpers import load_h5py_file, runSingleDecodingStep, LOGIT_TO_PHONEME

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

def analyze_day_weights(model, model_args):
    """Analyze the learned day-specific weight matrices"""
    print("\n=== Day-Specific Weight Analysis ===")
    
    n_days = len(model_args['dataset']['sessions'])
    sessions = model_args['dataset']['sessions']
    
    # Extract day weights and biases
    day_weights = [w.detach().cpu().numpy() for w in model.day_weights]
    day_biases = [b.detach().cpu().numpy() for b in model.day_biases]
    
    # Compute statistics
    print(f"Number of days: {n_days}")
    print(f"Weight matrix shape: {day_weights[0].shape}")
    print(f"Bias vector shape: {day_biases[0].shape}")
    
    # Analyze how much each day's weights deviate from identity
    identity = np.eye(day_weights[0].shape[0])
    deviations_from_identity = []
    frobenius_norms = []
    
    for i, (session, weights) in enumerate(zip(sessions, day_weights)):
        deviation = np.linalg.norm(weights - identity, 'fro')
        frobenius_norm = np.linalg.norm(weights, 'fro')
        deviations_from_identity.append(deviation)
        frobenius_norms.append(frobenius_norm)
        
        print(f"Day {i} ({session}): Deviation from identity = {deviation:.4f}, Frobenius norm = {frobenius_norm:.4f}")
    
    # Plot deviations
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.bar(range(n_days), deviations_from_identity)
    plt.xlabel('Day Index')
    plt.ylabel('Deviation from Identity (Frobenius norm)')
    plt.title('Day Weight Deviations from Identity Matrix')
    plt.xticks(range(0, n_days, 5), rotation=45)
    
    plt.subplot(1, 3, 2)
    plt.bar(range(n_days), frobenius_norms)
    plt.xlabel('Day Index')
    plt.ylabel('Weight Matrix Frobenius Norm')
    plt.title('Day Weight Matrix Norms')
    plt.xticks(range(0, n_days, 5), rotation=45)
    
    # Bias analysis
    bias_norms = [np.linalg.norm(b) for b in day_biases]
    plt.subplot(1, 3, 3)
    plt.bar(range(n_days), bias_norms)
    plt.xlabel('Day Index')
    plt.ylabel('Bias Vector L2 Norm')
    plt.title('Day Bias Vector Norms')
    plt.xticks(range(0, n_days, 5), rotation=45)
    
    plt.tight_layout()
    plt.savefig('day_weights_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show for non-interactive
    
    return deviations_from_identity, frobenius_norms, bias_norms

def test_day_effect_on_outputs(model, model_args, device):
    """Test how much the day index affects model outputs for the same neural input"""
    print("\n=== Day Effect on Outputs Analysis ===")
    
    # Load some sample data
    data_dir = '../data/t15_copyTask_neuralData/hdf5_data_final'
    csv_path = '../data/t15_copyTaskData_description.csv'
    b2txt_csv_df = pd.read_csv(csv_path)
    
    # Find a session with validation data
    sessions = model_args['dataset']['sessions']
    sample_data = None
    sample_session_idx = None
    
    for i, session in enumerate(sessions):
        session_path = os.path.join(data_dir, session)
        if os.path.exists(session_path):
            val_file = os.path.join(session_path, 'data_val.hdf5')
            if os.path.exists(val_file):
                print(f"Loading sample data from {session}")
                sample_data = load_h5py_file(val_file, b2txt_csv_df)
                sample_session_idx = i
                break
    
    if sample_data is None:
        print("No validation data found, using test data instead")
        for i, session in enumerate(sessions):
            session_path = os.path.join(data_dir, session)
            if os.path.exists(session_path):
                test_file = os.path.join(session_path, 'data_test.hdf5')
                if os.path.exists(test_file):
                    print(f"Loading sample data from {session}")
                    sample_data = load_h5py_file(test_file, b2txt_csv_df)
                    sample_session_idx = i
                    break
    
    if sample_data is None:
        print("No suitable data found for analysis")
        return
    
    # Take first few trials as samples
    n_samples = min(5, len(sample_data['neural_features']))
    n_days = len(sessions)
    
    results = []
    
    print(f"Testing {n_samples} neural samples across {n_days} different day indices...")
    
    for sample_idx in range(n_samples):
        neural_input = sample_data['neural_features'][sample_idx]
        neural_input = np.expand_dims(neural_input, axis=0)  # Add batch dim
        neural_input = torch.tensor(neural_input, device=device, dtype=torch.bfloat16)
        
        sample_results = {
            'sample_idx': sample_idx,
            'outputs': [],
            'predictions': [],
            'day_indices': []
        }
        
        # Test this neural input with different day indices
        with torch.no_grad():
            for day_idx in range(n_days):
                # Run model with this day index - convert to float32 to avoid dtype issues
                neural_input_float = neural_input.float()
                logits = runSingleDecodingStep(neural_input_float, day_idx, model, model_args, device)
                
                # Get predictions
                pred_seq = np.argmax(logits[0], axis=-1)
                pred_seq = [int(p) for p in pred_seq if p != 0]  # Remove blanks
                pred_seq = [pred_seq[i] for i in range(len(pred_seq)) if i == 0 or pred_seq[i] != pred_seq[i-1]]  # Remove consecutive duplicates
                pred_phonemes = [LOGIT_TO_PHONEME[p] for p in pred_seq]
                
                sample_results['outputs'].append(logits[0])
                sample_results['predictions'].append(pred_phonemes)
                sample_results['day_indices'].append(day_idx)
        
        results.append(sample_results)
        
        # Print predictions for this sample
        print(f"\nSample {sample_idx} predictions across days:")
        print(f"Original day: {sample_session_idx} ({sessions[sample_session_idx]})")
        for day_idx, pred in enumerate(sample_results['predictions']):
            marker = " <-- ORIGINAL" if day_idx == sample_session_idx else ""
            print(f"  Day {day_idx:2d}: {' '.join(pred[:10])}{'...' if len(pred) > 10 else ''}{marker}")
    
    return results

def analyze_output_similarity(results):
    """Analyze similarity between outputs from different day indices"""
    print("\n=== Output Similarity Analysis ===")
    
    for sample_idx, sample_results in enumerate(results):
        outputs = sample_results['outputs']
        predictions = sample_results['predictions']
        n_days = len(outputs)
        
        # Compute pairwise cosine similarities between logit outputs
        similarities = np.zeros((n_days, n_days))
        
        for i in range(n_days):
            for j in range(n_days):
                # Flatten the logits and compute cosine similarity
                logits_i = outputs[i].flatten()
                logits_j = outputs[j].flatten()
                
                dot_product = np.dot(logits_i, logits_j)
                norm_i = np.linalg.norm(logits_i)
                norm_j = np.linalg.norm(logits_j)
                
                if norm_i > 0 and norm_j > 0:
                    similarities[i, j] = dot_product / (norm_i * norm_j)
                else:
                    similarities[i, j] = 0
        
        # Plot similarity matrix
        plt.figure(figsize=(10, 8))
        im = plt.imshow(similarities, cmap='viridis', aspect='auto')
        plt.colorbar(im, label='Cosine Similarity')
        plt.title(f'Logit Similarity Matrix - Sample {sample_idx}')
        plt.xlabel('Day Index')
        plt.ylabel('Day Index')
        plt.savefig(f'logit_similarity_sample_{sample_idx}.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close instead of show for non-interactive
        
        # Compute statistics
        # Exclude diagonal (self-similarity = 1.0)
        off_diagonal = similarities[np.triu_indices_from(similarities, k=1)]
        
        print(f"Sample {sample_idx}:")
        print(f"  Mean pairwise similarity: {np.mean(off_diagonal):.4f}")
        print(f"  Std pairwise similarity: {np.std(off_diagonal):.4f}")
        print(f"  Min pairwise similarity: {np.min(off_diagonal):.4f}")
        print(f"  Max pairwise similarity: {np.max(off_diagonal):.4f}")
        
        # Count how many predictions are identical
        unique_predictions = set()
        for pred in predictions:
            unique_predictions.add(' '.join(pred))
        
        print(f"  Unique predictions out of {n_days}: {len(unique_predictions)}")
        print(f"  Prediction diversity: {len(unique_predictions) / n_days:.4f}")

def main():
    """Main analysis function"""
    print("Loading pretrained RNN model...")
    model, model_args, device = load_pretrained_model()
    
    print("Model loaded successfully!")
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Analyze day-specific weights
    deviations, norms, bias_norms = analyze_day_weights(model, model_args)
    
    # Test day effect on outputs
    results = test_day_effect_on_outputs(model, model_args, device)
    
    if results:
        # Analyze output similarity
        analyze_output_similarity(results)
    
    print("\n=== Summary ===")
    print(f"Average deviation from identity: {np.mean(deviations):.4f} ± {np.std(deviations):.4f}")
    print(f"Average weight norm: {np.mean(norms):.4f} ± {np.std(norms):.4f}")
    print(f"Average bias norm: {np.mean(bias_norms):.4f} ± {np.std(bias_norms):.4f}")
    
    if np.mean(deviations) < 0.1:
        print("✓ Day-specific weights are close to identity - minimal day effect expected")
    else:
        print("⚠ Day-specific weights deviate significantly from identity - substantial day effect possible")

if __name__ == "__main__":
    main()
