#!/usr/bin/env python3
"""
Quick test of current decoder performance on 50 samples with RNN model.
"""

import sys
import os
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from model_training.rnn_model import GRUDecoder
from model_training.evaluate_model_helpers import load_h5py_file, runSingleDecodingStep
from decoding import Decoder
from nejm_b2txt_utils.general_utils import remove_punctuation
from omegaconf import OmegaConf
import pandas as pd
import jiwer
from tqdm import tqdm


def test_current_performance(num_samples=50):
    """Test current decoder performance on specified number of samples."""
    
    # Load model
    model_path = "data/t15_pretrained_rnn_baseline/t15_pretrained_rnn_baseline"
    data_dir = "data/t15_copyTask_neuralData/hdf5_data_final"
    device = torch.device('cpu')
    
    print(f"🧪 Testing current decoder performance on {num_samples} samples")
    print(f"📁 Model: {model_path}")
    print(f"📊 Data: {data_dir}")
    
    # Load model config and weights
    model_args = OmegaConf.load(os.path.join(model_path, 'checkpoint/args.yaml'))
    csv_path = os.path.join('data', 't15_copyTaskData_description.csv')
    b2txt_csv_df = pd.read_csv(csv_path)
    
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
    
    # Load weights
    checkpoint = torch.load(os.path.join(model_path, 'checkpoint/best_checkpoint'), 
                          weights_only=False, map_location=device)
    state_dict = {}
    for key, value in checkpoint['model_state_dict'].items():
        clean_key = key.replace("module.", "").replace("_orig_mod.", "")
        state_dict[clean_key] = value
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully")
    
    # Initialize decoder with current config
    
    decoder = Decoder()
    
    print("✅ Decoder initialized successfully")
    print(f"⚙️  Decoder type: {decoder.decoder_type}")
    
    # Try to get config info if available
    try:
        if hasattr(decoder.decoder_instance, 'lm_weight'):
            print(f"⚙️  Config: lm_weight={decoder.decoder_instance.lm_weight}, word_score={decoder.decoder_instance.word_score}, beam_size={decoder.decoder_instance.beam_size}")
        else:
            print("⚙️  Using greedy decoder (no beam search parameters)")
    except AttributeError:
        print("⚙️  Decoder config not accessible")
    
    # Load validation data from multiple sessions to get enough samples
    all_data = {'neural_features': [], 'sentence_label': []}
    loaded_sessions = []
    total_loaded = 0
    
    for sess in model_args['dataset']['sessions']:
        if total_loaded >= num_samples:
            break
            
        session_dir = os.path.join(data_dir, sess)
        val_file = os.path.join(session_dir, 'data_val.hdf5')
        if os.path.exists(val_file):
            session_data = load_h5py_file(val_file, b2txt_csv_df)
            
            # Add data from this session
            samples_to_take = min(len(session_data['neural_features']), num_samples - total_loaded)
            all_data['neural_features'].extend(session_data['neural_features'][:samples_to_take])
            all_data['sentence_label'].extend(session_data['sentence_label'][:samples_to_take])
            
            loaded_sessions.append(f"{sess}({samples_to_take})")
            total_loaded += samples_to_take
    
    if total_loaded == 0:
        raise FileNotFoundError("No validation data files found")
    
    print(f"📊 Loaded {total_loaded} validation trials from {len(loaded_sessions)} sessions:")
    print(f"   Sessions: {', '.join(loaded_sessions[:3])}" + (f" + {len(loaded_sessions)-3} more" if len(loaded_sessions) > 3 else ""))
    
    # Use the loaded data
    data = all_data
    num_trials = len(data['neural_features'])
    print(f"🔍 Testing on {num_trials} samples")
    
    predictions = []
    ground_truth = []
    nbest_predictions = []  # Store all n-best for analysis
    input_layer = 0  # Use first session index for simplicity
    
    with torch.no_grad():
        for trial in tqdm(range(num_trials), desc="Processing samples"):
            # Get neural input and run model
            neural_input = data['neural_features'][trial]
            neural_input = np.expand_dims(neural_input, axis=0)
            neural_input = torch.tensor(neural_input, device=device, dtype=torch.float32)
            
            # Get logits
            logits = runSingleDecodingStep(neural_input, input_layer, model, model_args, device)
            logits_tensor = torch.tensor(logits[0], dtype=torch.float32)
            
            # Decode
            result = decoder.decode(logits_tensor)
            pred_sentence = result['sentence'] if isinstance(result, dict) else result.get('sentence', '')
            pred_sentence = remove_punctuation(pred_sentence).strip()
            
            # Ground truth
            true_sentence = remove_punctuation(data['sentence_label'][trial]).strip()
            
            predictions.append(pred_sentence)
            ground_truth.append(true_sentence)
            
            # Extract n-best alternatives for analysis
            trial_nbest = []
            if isinstance(result, dict) and 'nbest' in result:
                for i, nbest_result in enumerate(result['nbest'][:10]):  # Top 10 n-best
                    nbest_sentence = nbest_result.get('sentence', '')
                    nbest_sentence = remove_punctuation(nbest_sentence).strip()
                    trial_nbest.append({
                        'rank': i,
                        'sentence': nbest_sentence,
                        'score': nbest_result.get('score', float('-inf'))
                    })
            nbest_predictions.append(trial_nbest)
            
            # Show first few examples with n-best
            if trial < 3:
                print(f"  Sample {trial+1}: '{true_sentence}' → '{pred_sentence}'")
                if trial_nbest:
                    print(f"    Top 3 alternatives:")
                    for alt in trial_nbest[:3]:
                        print(f"      {alt['rank']+1}. '{alt['sentence']}' (score: {alt['score']:.2f})")
    
    # Calculate metrics
    wer = jiwer.wer(ground_truth, predictions)
    cer = jiwer.cer(ground_truth, predictions)
    
    # Analyze n-best alternatives
    nbest_wers = []
    nbest_cers = []
    
    for rank in range(10):  # Analyze top 10 n-best
        rank_predictions = []
        rank_ground_truth = []
        
        for trial_idx, trial_nbest in enumerate(nbest_predictions):
            if len(trial_nbest) > rank:
                rank_predictions.append(trial_nbest[rank]['sentence'])
                rank_ground_truth.append(ground_truth[trial_idx])
            else:
                # If no n-best at this rank, use the best available
                if trial_nbest:
                    rank_predictions.append(trial_nbest[-1]['sentence'])
                else:
                    rank_predictions.append(predictions[trial_idx])
                rank_ground_truth.append(ground_truth[trial_idx])
        
        if rank_predictions:
            rank_wer = jiwer.wer(rank_ground_truth, rank_predictions)
            rank_cer = jiwer.cer(rank_ground_truth, rank_predictions)
            nbest_wers.append(rank_wer)
            nbest_cers.append(rank_cer)
    
    # Find best WER in n-best
    best_nbest_wer = min(nbest_wers) if nbest_wers else wer
    best_nbest_rank = nbest_wers.index(best_nbest_wer) if nbest_wers else 0
    
    print(f"\n📊 RESULTS ({num_trials} samples)")
    print(f"🎯 1-best WER: {wer:.4f} ({'✅ TARGET MET' if wer < 0.2 else '❌ Above target'})")
    print(f"📝 1-best CER: {cer:.4f}")
    print(f"\n🔍 N-BEST ANALYSIS:")
    print(f"🏆 Best WER in top-10: {best_nbest_wer:.4f} (rank {best_nbest_rank + 1})")
    print(f"📈 WER improvement: {((wer - best_nbest_wer) / wer * 100):.1f}%")
    
    # Show WER for each rank
    print(f"\n📋 WER by rank:")
    for i, rank_wer in enumerate(nbest_wers[:5]):  # Show top 5
        marker = "🏆" if i == best_nbest_rank else "  "
        print(f"{marker} Rank {i+1}: {rank_wer:.4f}")
    
    print(f"🎯 Target: WER < 0.2")
    
    return wer, cer, predictions, ground_truth, best_nbest_wer, nbest_wers


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=50, help='Number of samples to test')
    args = parser.parse_args()
    
    result = test_current_performance(args.num_samples)
    if len(result) == 6:
        wer, cer, predictions, ground_truth, best_nbest_wer, nbest_wers = result
    else:
        wer, cer, predictions, ground_truth = result[:4]
