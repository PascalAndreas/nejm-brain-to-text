"""Unit tests for neural encoder models.

This module tests critical invariants and functionality of the models,
including padding invariance, length mapping, and component integration.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.base import Batch, Emissions
from models.gru_ctc import GRUCTC
from models.blocks import (
    GaussianSmoother,
    DayAdapter,
    PreNet,
    GRUBackbone,
    ProjectionHead,
    compute_output_lengths,
    mask_logits_,
    apply_patch_embedding
)
from models import build_encoder


class TestPaddingInvariance(unittest.TestCase):
    """Test that models are invariant to padding."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.device = 'cpu'
        
        # Create a simple model
        self.model = GRUCTC(
            input_dim=128,
            vocab_size=41,
            num_days=5,
            hidden_size=256,
            num_layers=2,
            dropout=0.0,  # No dropout for deterministic tests
            prenet_config={
                'smoother_config': None,  # Disable for simplicity
                'day_adapter_config': None,
                'patch_config': {'size': 1, 'stride': 1},  # No patching
                'activation': 'relu'
            },
            aux_ctc_config=None
        ).to(self.device)
        
        self.model.eval()
    
    def test_single_vs_batched(self):
        """Test that single sample gives same result as batched."""
        # Create a single sample
        seq_len = 50
        feat_dim = 128
        x_single = torch.randn(1, seq_len, feat_dim)
        
        # Create batch with padding
        x_padded = torch.zeros(3, 100, feat_dim)
        x_padded[0, :seq_len] = x_single[0]
        x_padded[1, :70] = torch.randn(70, feat_dim)
        x_padded[2, :30] = torch.randn(30, feat_dim)
        
        # Create batch objects
        batch_single = Batch(
            x=x_single,
            x_lens=torch.tensor([seq_len]),
            day_id=torch.tensor([0])
        )
        
        batch_padded = Batch(
            x=x_padded,
            x_lens=torch.tensor([seq_len, 70, 30]),
            day_id=torch.tensor([0, 1, 2])
        )
        
        # Forward pass
        with torch.no_grad():
            emissions_single = self.model(batch_single)
            emissions_padded = self.model(batch_padded)
        
        # Compare first sample from padded batch with single sample
        # They should be identical up to the valid length
        single_log_probs = emissions_single.log_probs[0, :emissions_single.out_lens[0]]
        padded_log_probs = emissions_padded.log_probs[0, :emissions_padded.out_lens[0]]
        
        torch.testing.assert_close(single_log_probs, padded_log_probs, rtol=1e-5, atol=1e-5)
        
        # Lengths should also match
        self.assertEqual(emissions_single.out_lens[0].item(), emissions_padded.out_lens[0].item())
    
    def test_different_padding_same_result(self):
        """Test that different amounts of padding give same results."""
        # Create sample
        seq_len = 40
        feat_dim = 128
        x = torch.randn(seq_len, feat_dim)
        
        # Create batches with different padding
        x_pad_60 = torch.zeros(1, 60, feat_dim)
        x_pad_60[0, :seq_len] = x
        
        x_pad_100 = torch.zeros(1, 100, feat_dim)
        x_pad_100[0, :seq_len] = x
        
        batch_60 = Batch(
            x=x_pad_60,
            x_lens=torch.tensor([seq_len]),
            day_id=torch.tensor([0])
        )
        
        batch_100 = Batch(
            x=x_pad_100,
            x_lens=torch.tensor([seq_len]),
            day_id=torch.tensor([0])
        )
        
        # Forward pass
        with torch.no_grad():
            emissions_60 = self.model(batch_60)
            emissions_100 = self.model(batch_100)
        
        # Results should be identical for valid positions
        valid_60 = emissions_60.log_probs[0, :emissions_60.out_lens[0]]
        valid_100 = emissions_100.log_probs[0, :emissions_100.out_lens[0]]
        
        torch.testing.assert_close(valid_60, valid_100, rtol=1e-5, atol=1e-5)


class TestLengthMapping(unittest.TestCase):
    """Test that length computations are correct."""
    
    def test_compute_output_lengths(self):
        """Test output length computation for various operations."""
        # Test patching
        input_lens = torch.tensor([100, 150, 75])
        
        ops = [{'type': 'patch', 'size': 4, 'stride': 2}]
        output_lens = compute_output_lengths(input_lens, ops)
        expected = torch.tensor([49, 74, 36])  # floor((L - 4) / 2) + 1
        torch.testing.assert_close(output_lens, expected)
        
        # Test convolution
        ops = [{'type': 'conv1d', 'kernel_size': 3, 'stride': 1, 'padding': 1}]
        output_lens = compute_output_lengths(input_lens, ops)
        torch.testing.assert_close(output_lens, input_lens)  # Same padding
        
        # Test pooling
        ops = [{'type': 'pool', 'kernel_size': 2, 'stride': 2}]
        output_lens = compute_output_lengths(input_lens, ops)
        expected = torch.tensor([50, 75, 37])  # floor(L / 2)
        torch.testing.assert_close(output_lens, expected)
        
        # Test composition
        ops = [
            {'type': 'patch', 'size': 4, 'stride': 2},
            {'type': 'conv1d', 'kernel_size': 3, 'stride': 1, 'padding': 1}
        ]
        output_lens = compute_output_lengths(input_lens, ops)
        expected = torch.tensor([49, 74, 36])  # Patching then same-padding conv
        torch.testing.assert_close(output_lens, expected)
    
    def test_patch_embedding_lengths(self):
        """Test that patch embedding correctly updates lengths."""
        batch_size = 2
        seq_len = 100
        feat_dim = 64
        
        x = torch.randn(batch_size, seq_len, feat_dim)
        lengths = torch.tensor([100, 80])
        
        # Apply patching
        x_patched, lengths_patched = apply_patch_embedding(x, patch_size=4, patch_stride=2, lengths=lengths)
        
        # Check output shape
        self.assertEqual(x_patched.shape[0], batch_size)
        self.assertEqual(x_patched.shape[2], feat_dim * 4)  # Features multiplied by patch size
        
        # Check lengths
        expected_lengths = torch.tensor([49, 39])  # floor((L - 4) / 2) + 1
        torch.testing.assert_close(lengths_patched, expected_lengths)
    
    def test_model_time_reduction(self):
        """Test that model correctly reports time reduction."""
        # Model with patching
        model = GRUCTC(
            input_dim=128,
            vocab_size=41,
            num_days=5,
            hidden_size=256,
            num_layers=2,
            prenet_config={
                'patch_config': {'size': 4, 'stride': 2},
                'activation': 'relu'
            }
        )
        
        self.assertEqual(model.time_reduction(), 2)
        
        # Model without patching
        model_no_patch = GRUCTC(
            input_dim=128,
            vocab_size=41,
            num_days=5,
            hidden_size=256,
            num_layers=2,
            prenet_config={
                'patch_config': {'size': 1, 'stride': 1},
                'activation': 'relu'
            }
        )
        
        self.assertEqual(model_no_patch.time_reduction(), 1)


class TestMasking(unittest.TestCase):
    """Test masking functionality."""
    
    def test_mask_logits(self):
        """Test that mask_logits correctly masks invalid positions."""
        batch_size = 3
        seq_len = 10
        vocab_size = 5
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        lengths = torch.tensor([5, 7, 3])
        
        # Apply masking
        mask_logits_(logits, lengths)
        
        # Check that masked positions have very negative values
        for i in range(batch_size):
            valid_len = lengths[i].item()
            
            # Valid positions should be unchanged (not -1e9)
            self.assertTrue((logits[i, :valid_len] > -1e8).all())
            
            # Invalid positions should be masked
            if valid_len < seq_len:
                self.assertTrue((logits[i, valid_len:] < -1e8).all())
    
    def test_masked_softmax(self):
        """Test that masked positions have ~0 probability after softmax."""
        batch_size = 2
        seq_len = 8
        vocab_size = 4
        
        logits = torch.randn(batch_size, seq_len, vocab_size)
        lengths = torch.tensor([4, 6])
        
        # Apply masking and softmax
        mask_logits_(logits, lengths)
        probs = torch.softmax(logits, dim=-1)
        
        # Check probabilities
        for i in range(batch_size):
            valid_len = lengths[i].item()
            
            # Valid positions should have non-zero probabilities
            self.assertTrue((probs[i, :valid_len].sum(dim=-1) > 0.99).all())
            
            # Invalid positions should have ~0 probability
            if valid_len < seq_len:
                self.assertTrue((probs[i, valid_len:] < 1e-6).all())


class TestDayAdapter(unittest.TestCase):
    """Test day-specific adaptation (FiLM)."""
    
    def test_identity_initialization(self):
        """Test that FiLM with identity init acts as identity."""
        num_features = 64
        num_days = 3
        
        adapter = DayAdapter(
            num_features=num_features,
            num_days=num_days,
            grouping='full',
            identity_init=True
        )
        
        # Create input
        batch_size = 2
        seq_len = 10
        x = torch.randn(batch_size, seq_len, num_features)
        day_indices = torch.tensor([0, 1])
        
        # Apply adapter
        y = adapter(x, day_indices)
        
        # With identity init, output should equal input
        torch.testing.assert_close(y, x, rtol=1e-5, atol=1e-5)
    
    def test_regularization_loss(self):
        """Test that regularization loss is computed correctly."""
        adapter = DayAdapter(
            num_features=32,
            num_days=2,
            grouping='full',
            identity_init=True,
            l2_tether=0.01,
            l1_reg=0.001
        )
        
        # Perturb parameters from identity
        adapter.gammas[0].data += 0.1
        adapter.betas[0].data += 0.05
        
        # Compute regularization loss
        reg_loss = adapter.regularization_loss()
        
        # Loss should be positive
        self.assertGreater(reg_loss.item(), 0)
    
    def test_group_wise_adaptation(self):
        """Test group-wise FiLM adaptation."""
        num_features = 512  # Divisible by 8
        num_days = 2
        
        adapter = DayAdapter(
            num_features=num_features,
            num_days=num_days,
            grouping='blocks8',
            identity_init=False
        )
        
        # Check parameter shapes
        self.assertEqual(adapter.gammas[0].shape, (8, 64))  # 8 groups of 64
        self.assertEqual(adapter.betas[0].shape, (8, 64))
        
        # Apply to input
        x = torch.randn(1, 10, num_features)
        day_indices = torch.tensor([0])
        
        y = adapter(x, day_indices)
        self.assertEqual(y.shape, x.shape)


class TestModelRegistry(unittest.TestCase):
    """Test model registry and factory."""
    
    def test_build_encoder(self):
        """Test that encoders can be built from registry."""
        config = {
            'input_dim': 128,
            'vocab_size': 41,
            'num_days': 10,
            'hidden_size': 256,
            'num_layers': 3
        }
        
        model = build_encoder('gru_v1', config)
        
        self.assertIsInstance(model, GRUCTC)
        self.assertEqual(model.hidden_size, 256)
        self.assertEqual(model.num_layers, 3)
    
    def test_unknown_encoder(self):
        """Test that unknown encoder raises error."""
        with self.assertRaises(ValueError):
            build_encoder('unknown_model', {})


class TestGRUBackbone(unittest.TestCase):
    """Test GRU backbone functionality."""
    
    def test_packed_sequences(self):
        """Test that packed sequences work correctly."""
        backbone = GRUBackbone(
            input_size=64,
            hidden_size=128,
            num_layers=2,
            use_packed=True
        )
        
        # Create batch with different lengths
        batch_size = 3
        max_len = 20
        feat_dim = 64
        
        x = torch.randn(batch_size, max_len, feat_dim)
        lengths = torch.tensor([20, 15, 10])
        
        # Forward pass
        output, hidden = backbone(x, lengths)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, max_len, 128))
        self.assertEqual(hidden.shape, (2, batch_size, 128))  # 2 layers
    
    def test_learnable_h0(self):
        """Test learnable initial hidden state."""
        backbone = GRUBackbone(
            input_size=32,
            hidden_size=64,
            num_layers=1,
            learnable_h0=True
        )
        
        # Check that h0 is a parameter
        self.assertIsNotNone(backbone.h0)
        self.assertIsInstance(backbone.h0, nn.Parameter)
        
        # Forward pass should use h0
        x = torch.randn(2, 10, 32)
        output, hidden = backbone(x)
        
        # Hidden should be influenced by h0
        self.assertEqual(hidden.shape, (1, 2, 64))


if __name__ == '__main__':
    unittest.main()
