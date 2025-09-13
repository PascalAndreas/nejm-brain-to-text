#!/usr/bin/env python3
"""
Focused parameter search around the current winning configuration.
"""

import sys
import os
sys.path.append('.')
sys.path.append('model_training')
sys.path.append('decoding')

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
from itertools import product


def load_model_and_data(model_path, data_dir, device, num_samples=100):
    """Load model and validation data."""
    
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
    
    # Load validation data from multiple sessions
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
    
    print(f"📊 Loaded {total_loaded} validation trials from {len(loaded_sessions)} sessions")
    
    return model, model_args, all_data


def test_parameters(model, model_args, data, device, lm_weight, word_score, beam_size):
    """Test specific parameter combination."""
    
    # Load config
    config_path = os.path.join('decoding', 'config.yaml')
    config = OmegaConf.load(config_path)
    
    # Initialize decoder with test parameters
    decoder = Decoder(
        tokens_path=config.artifacts.tokens_txt,
        lexicon_path=config.artifacts.lexicon_txt,
        lm_path=config.language_model.active_model,
        lm_weight=lm_weight,
        word_score=word_score,
        beam_size=beam_size,
        beam_size_token=config.decoder.beam_size_token,
        beam_threshold=config.decoder.beam_threshold,
        nbest=config.decoder.nbest,
        blank_token=config.decoder.blank_token,
        silence_token=config.decoder.silence_token,
        unk_word=config.decoder.unk_word
    )
    
    predictions = []
    ground_truth = []
    input_layer = 0  # Use first session index
    
    with torch.no_grad():
        for trial in range(len(data['neural_features'])):
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
    
    # Calculate metrics
    wer = jiwer.wer(ground_truth, predictions)
    cer = jiwer.cer(ground_truth, predictions)
    
    return wer, cer


def focused_parameter_search():
    """Run focused parameter search around winning configuration."""
    
    print("🔍 FOCUSED PARAMETER SEARCH")
    print("=" * 50)
    
    # Configuration
    model_path = "data/t15_pretrained_rnn_baseline/t15_pretrained_rnn_baseline"
    data_dir = "data/t15_copyTask_neuralData/hdf5_data_final"
    device = torch.device('cpu')
    num_samples = 100
    
    # Load model and data
    print("📦 Loading model and data...")
    model, model_args, data = load_model_and_data(model_path, data_dir, device, num_samples)
    
    # Parameter ranges around winning config (lm_weight=1.4, word_score=-0.2, beam_size=200)
    lm_weights = [1.0, 1.2, 1.4, 1.6, 1.8]  # ±0.4 around 1.4
    word_scores = [-0.6, -0.4, -0.2, 0.0, 0.2]  # ±0.4 around -0.2
    beam_sizes = [150, 200, 250]  # Around 200
    
    print(f"🎯 Testing {len(lm_weights)} × {len(word_scores)} × {len(beam_sizes)} = {len(lm_weights) * len(word_scores) * len(beam_sizes)} combinations")
    print(f"📊 Using {len(data['neural_features'])} validation samples")
    
    results = []
    best_wer = float('inf')
    best_params = None
    
    total_combinations = len(lm_weights) * len(word_scores) * len(beam_sizes)
    
    with tqdm(total=total_combinations, desc="Parameter combinations") as pbar:
        for lm_weight, word_score, beam_size in product(lm_weights, word_scores, beam_sizes):
            try:
                wer, cer = test_parameters(model, model_args, data, device, lm_weight, word_score, beam_size)
                
                result = {
                    'lm_weight': lm_weight,
                    'word_score': word_score, 
                    'beam_size': beam_size,
                    'WER': wer,
                    'CER': cer
                }
                results.append(result)
                
                # Track best result
                if wer < best_wer:
                    best_wer = wer
                    best_params = result.copy()
                    print(f"\n🏆 New best WER: {wer:.4f} with lm_weight={lm_weight}, word_score={word_score}, beam_size={beam_size}")
                
                pbar.set_postfix({'Best WER': f'{best_wer:.4f}', 'Current': f'{wer:.4f}'})
                
            except Exception as e:
                print(f"\n❌ Error with lm_weight={lm_weight}, word_score={word_score}, beam_size={beam_size}: {e}")
            
            pbar.update(1)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv('focused_parameter_search_results.csv', index=False)
    print(f"\n💾 Results saved to: focused_parameter_search_results.csv")
    
    # Display best results
    print(f"\n🏆 BEST PARAMETERS:")
    if best_params:
        for key, value in best_params.items():
            print(f"   {key}: {value}")
        
        target_met = "✅ TARGET MET" if best_params['WER'] < 0.2 else "❌ Above target"
        print(f"\n🎯 Best WER: {best_params['WER']:.4f} ({target_met})")
        print(f"📝 Best CER: {best_params['CER']:.4f}")
    
    # Show top 5 results
    print(f"\n📋 TOP 5 RESULTS:")
    top_results = sorted(results, key=lambda x: x['WER'])[:5]
    for i, result in enumerate(top_results, 1):
        print(f"   {i}. WER={result['WER']:.4f}, lm_weight={result['lm_weight']}, word_score={result['word_score']}, beam_size={result['beam_size']}")
    
    return results_df, best_params


if __name__ == "__main__":
    results_df, best_params = focused_parameter_search()
