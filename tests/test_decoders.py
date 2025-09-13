"""
Tests for decoder functionality.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

from decoding.decoder import Decoder


class TestDecoderBasics:
    """Test basic decoder functionality."""
    
    def test_greedy_decoder_creation(self, greedy_decoder):
        """Test that greedy decoder can be created."""
        assert greedy_decoder is not None
        assert greedy_decoder.decoder_type == 'greedy'
        assert hasattr(greedy_decoder, 'tokens')
        assert len(greedy_decoder.tokens) > 0
    
    def test_flashlight_decoder_creation(self, flashlight_decoder):
        """Test that flashlight decoder can be created if available."""
        assert flashlight_decoder is not None
        assert flashlight_decoder.decoder_type == 'flashlight'
        assert hasattr(flashlight_decoder, 'tokens')
        assert len(flashlight_decoder.tokens) > 0
    
    def test_decoder_tokens_consistency(self, greedy_decoder):
        """Test that decoder has consistent token mappings."""
        tokens = greedy_decoder.tokens
        token_to_idx = greedy_decoder.token_to_idx
        
        # Check that all tokens have indices
        assert len(tokens) == len(token_to_idx)
        
        # Check that indices are consistent
        for i, token in enumerate(tokens):
            assert token_to_idx[token] == i
    
    def test_blank_token_exists(self, greedy_decoder):
        """Test that blank token exists and has correct index."""
        blank_idx = greedy_decoder.blank_idx
        tokens = greedy_decoder.tokens
        
        assert 0 <= blank_idx < len(tokens)
        # Typically blank token should be 'BLANK' or similar
        assert 'BLANK' in tokens[blank_idx].upper() or tokens[blank_idx] == '<blank>'


class TestDecoderFunctionality:
    """Test decoder decoding functionality."""
    
    def test_greedy_decode_sample_logits(self, greedy_decoder, sample_logits):
        """Test greedy decoder on sample logits."""
        logits, logit_lengths = sample_logits
        
        # Test single sequence
        single_result = greedy_decoder.decode(logits[0], logit_lengths[0:1])
        assert isinstance(single_result, dict)
        assert 'words' in single_result
        assert 'tokens' in single_result
        assert 'sentence' in single_result
        assert 'score' in single_result
        assert isinstance(single_result['words'], list)
        assert isinstance(single_result['tokens'], list)
        assert isinstance(single_result['sentence'], str)
        
        # Test batch
        batch_results = greedy_decoder.decode(logits, logit_lengths)
        assert isinstance(batch_results, list)
        assert len(batch_results) == 2
        
        for result in batch_results:
            assert isinstance(result, dict)
            assert 'words' in result
            assert 'tokens' in result
            assert 'sentence' in result
            assert 'score' in result
    
    def test_flashlight_decode_sample_logits(self, flashlight_decoder, sample_logits):
        """Test flashlight decoder on sample logits."""
        logits, logit_lengths = sample_logits
        
        # Test single sequence
        single_result = flashlight_decoder.decode(logits[0], logit_lengths[0:1])
        assert isinstance(single_result, dict)
        assert 'words' in single_result
        assert 'tokens' in single_result
        assert 'sentence' in single_result
        assert 'score' in single_result
        assert 'nbest' in single_result
        assert isinstance(single_result['words'], list)
        assert isinstance(single_result['tokens'], list)
        assert isinstance(single_result['sentence'], str)
        assert isinstance(single_result['nbest'], list)
        
        # Test batch
        batch_results = flashlight_decoder.decode(logits, logit_lengths)
        assert isinstance(batch_results, list)
        assert len(batch_results) == 2
        
        for result in batch_results:
            assert isinstance(result, dict)
            assert 'words' in result
            assert 'tokens' in result
            assert 'sentence' in result
            assert 'score' in result
            assert 'nbest' in result
    
    def test_decoder_consistency(self, greedy_decoder, sample_logits):
        """Test that decoder produces consistent results."""
        logits, logit_lengths = sample_logits
        
        # Run same input multiple times
        result1 = greedy_decoder.decode(logits[0], logit_lengths[0:1])
        result2 = greedy_decoder.decode(logits[0], logit_lengths[0:1])
        
        # Results should be identical for greedy decoder
        assert result1['words'] == result2['words']
        assert result1['tokens'] == result2['tokens']
        assert result1['sentence'] == result2['sentence']
    
    def test_empty_logits_handling(self, greedy_decoder):
        """Test decoder handles empty/zero-length sequences."""
        # Create logits with zero length
        logits = torch.randn(1, 10, 41)
        logit_lengths = torch.tensor([0])
        
        result = greedy_decoder.decode(logits, logit_lengths)
        assert isinstance(result, dict)
        assert result['words'] == []
        assert result['tokens'] == []
        assert result['sentence'] == ''


class TestRNNModelIntegration:
    """Test integration with RNN model for realistic logits."""
    
    def test_rnn_model_loads(self, loaded_rnn_model):
        """Test that RNN model loads successfully."""
        model, model_args = loaded_rnn_model
        assert model is not None
        assert hasattr(model, 'forward')
        assert model_args is not None
    
    def test_rnn_model_forward_pass(self, loaded_rnn_model, sample_neural_data):
        """Test RNN model forward pass produces logits using the proper inference method."""
        model, model_args = loaded_rnn_model
        neural_data, day_indices = sample_neural_data
        
        # Use the same approach as test_rnn_model_phonemes.py
        from model_training.data_augmentations import gauss_smooth
        
        device = torch.device('cpu')
        model.to(device)
        neural_data = neural_data.to(device)
        
        # Test single sample inference (like runSingleDecodingStep)
        single_sample = neural_data[0:1]  # Take first sample, keep batch dim
        day_idx = day_indices[0].item()
        
        with torch.autocast(device_type="cpu", enabled=model_args['use_amp'], dtype=torch.bfloat16):
            # Apply smoothing
            smoothed_data = gauss_smooth(
                inputs=single_sample, 
                device=device,
                smooth_kernel_std=model_args['dataset']['data_transforms']['smooth_kernel_std'],
                smooth_kernel_size=model_args['dataset']['data_transforms']['smooth_kernel_size'],
                padding='valid',
            )
            
            with torch.no_grad():
                logits, _ = model(
                    x=smoothed_data,
                    day_idx=torch.tensor([day_idx], device=device),
                    states=None,
                    return_state=True,
                )
        
        # Convert to float32
        logits = logits.float()
        
        assert isinstance(logits, torch.Tensor)
        assert logits.dim() == 3  # [batch, time, vocab]
        assert logits.shape[0] == 1  # Single sample
        assert logits.shape[2] == model_args['dataset']['n_classes']  # Correct vocab size
    
    def test_decoder_on_rnn_logits(self, loaded_rnn_model, sample_neural_data, greedy_decoder):
        """Test decoder on logits from RNN model."""
        model, model_args = loaded_rnn_model
        neural_data, day_indices = sample_neural_data
        
        from model_training.data_augmentations import gauss_smooth
        
        device = torch.device('cpu')
        model.to(device)
        neural_data = neural_data.to(device)
        
        # Get logits from RNN using proper inference
        single_sample = neural_data[0:1]
        day_idx = day_indices[0].item()
        
        with torch.autocast(device_type="cpu", enabled=model_args['use_amp'], dtype=torch.bfloat16):
            smoothed_data = gauss_smooth(
                inputs=single_sample, 
                device=device,
                smooth_kernel_std=model_args['dataset']['data_transforms']['smooth_kernel_std'],
                smooth_kernel_size=model_args['dataset']['data_transforms']['smooth_kernel_size'],
                padding='valid',
            )
            
            with torch.no_grad():
                logits, _ = model(
                    x=smoothed_data,
                    day_idx=torch.tensor([day_idx], device=device),
                    states=None,
                    return_state=True,
                )
        
        logits = logits.float()
        
        # Decode with greedy decoder
        logit_lengths = torch.tensor([logits.shape[1]])
        result = greedy_decoder.decode(logits, logit_lengths)
        
        assert isinstance(result, dict)
        assert 'words' in result
        assert 'tokens' in result
        assert 'sentence' in result
        # For synthetic data, we just verify the structure is correct
        # (may or may not produce meaningful output depending on the synthetic data)
        assert isinstance(result['words'], list)
        assert isinstance(result['tokens'], list)
        assert isinstance(result['sentence'], str)
    
    def test_flashlight_decoder_on_rnn_logits(self, loaded_rnn_model, sample_neural_data, flashlight_decoder):
        """Test flashlight decoder on logits from RNN model."""
        model, model_args = loaded_rnn_model
        neural_data, day_indices = sample_neural_data
        
        from model_training.data_augmentations import gauss_smooth
        
        device = torch.device('cpu')
        model.to(device)
        neural_data = neural_data.to(device)
        
        # Get logits from RNN using proper inference
        single_sample = neural_data[0:1]
        day_idx = day_indices[0].item()
        
        with torch.autocast(device_type="cpu", enabled=model_args['use_amp'], dtype=torch.bfloat16):
            smoothed_data = gauss_smooth(
                inputs=single_sample, 
                device=device,
                smooth_kernel_std=model_args['dataset']['data_transforms']['smooth_kernel_std'],
                smooth_kernel_size=model_args['dataset']['data_transforms']['smooth_kernel_size'],
                padding='valid',
            )
            
            with torch.no_grad():
                logits, _ = model(
                    x=smoothed_data,
                    day_idx=torch.tensor([day_idx], device=device),
                    states=None,
                    return_state=True,
                )
        
        logits = logits.float()
        
        # Decode with flashlight decoder
        logit_lengths = torch.tensor([logits.shape[1]])
        result = flashlight_decoder.decode(logits, logit_lengths)
        
        assert isinstance(result, dict)
        assert 'words' in result
        assert 'tokens' in result
        assert 'sentence' in result
        assert 'nbest' in result
        # For synthetic data, we just verify the structure is correct
        assert isinstance(result['words'], list)
        assert isinstance(result['tokens'], list)
        assert isinstance(result['sentence'], str)
        assert isinstance(result['nbest'], list)
    
    def test_decoder_comparison_on_rnn_logits(self, loaded_rnn_model, sample_neural_data, 
                                            greedy_decoder, flashlight_decoder):
        """Compare greedy and flashlight decoders on same RNN logits."""
        model, model_args = loaded_rnn_model
        neural_data, day_indices = sample_neural_data
        
        from model_training.data_augmentations import gauss_smooth
        
        device = torch.device('cpu')
        model.to(device)
        neural_data = neural_data.to(device)
        
        # Get logits from RNN using proper inference
        single_sample = neural_data[0:1]
        day_idx = day_indices[0].item()
        
        with torch.autocast(device_type="cpu", enabled=model_args['use_amp'], dtype=torch.bfloat16):
            smoothed_data = gauss_smooth(
                inputs=single_sample, 
                device=device,
                smooth_kernel_std=model_args['dataset']['data_transforms']['smooth_kernel_std'],
                smooth_kernel_size=model_args['dataset']['data_transforms']['smooth_kernel_size'],
                padding='valid',
            )
            
            with torch.no_grad():
                logits, _ = model(
                    x=smoothed_data,
                    day_idx=torch.tensor([day_idx], device=device),
                    states=None,
                    return_state=True,
                )
        
        logits = logits.float()
        logit_lengths = torch.tensor([logits.shape[1]])
        
        # Decode with both decoders
        greedy_result = greedy_decoder.decode(logits, logit_lengths)
        flashlight_result = flashlight_decoder.decode(logits, logit_lengths)
        
        # Both should have valid structure
        assert isinstance(greedy_result['sentence'], str)
        assert isinstance(flashlight_result['sentence'], str)
        
        # Flashlight should have additional features
        assert 'nbest' in flashlight_result
        assert len(flashlight_result['nbest']) >= 0
        
        # Print comparison for manual inspection
        print(f"Decoder comparison:")
        print(f"  Greedy: '{greedy_result['sentence']}'")
        print(f"  Flashlight: '{flashlight_result['sentence']}'")
        if flashlight_result['nbest']:
            print(f"  Flashlight n-best: {[nb['sentence'] for nb in flashlight_result['nbest'][:3]]}")
        
        # For synthetic data, verify structure is correct (may or may not have content)
        assert isinstance(greedy_result['tokens'], list)
        assert isinstance(greedy_result['words'], list)
        assert isinstance(flashlight_result['tokens'], list)
        assert isinstance(flashlight_result['words'], list)
