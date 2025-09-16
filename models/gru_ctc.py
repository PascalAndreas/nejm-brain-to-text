"""GRU-CTC neural encoder implementation.

This module implements the main GRU-based CTC encoder by composing
the various building blocks (Smoother, PreNet, GRU backbone, projection heads).
"""

from typing import Dict, Any, Optional, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Batch, Emissions, NeuralEncoder
from .blocks import (
    PreNet,
    GRUBackbone,
    CTCHead,
    AuxiliaryHead,
    TemperatureScaling,
    mask_logits_
)
from .blocks.smoothers import build_smoother, SmootherBase


class GRUCTC(NeuralEncoder):
    """GRU-based neural encoder with CTC output head.
    
    This model implements the complete pipeline:
    1. Smoother: Temporal smoothing (learnable or fixed)
    2. PreNet: Day Adaptation → Activation → Patching
    3. GRU Backbone: Multi-layer GRU with packed sequence support
    4. Projection Head: Linear projection to vocabulary
    5. Optional: Auxiliary CTC head for deep supervision
    6. Optional: Temperature calibration
    
    Args:
        input_dim: Number of input features
        vocab_size: Size of output vocabulary (including CTC blank)
        num_days: Number of recording days for adaptation
        hidden_size: GRU hidden state size
        num_layers: Number of GRU layers
        dropout: Dropout probability
        smoother_config: Configuration for temporal smoother
        prenet_config: Configuration for PreNet
        aux_layer: Layer index for auxiliary CTC (None to disable)
        temperature: Initial temperature for calibration
        blank_idx: Index of CTC blank token
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        vocab_size: int = 41,
        num_days: int = 20,
        hidden_size: int = 768,
        num_layers: int = 5,
        dropout: float = 0.2,
        smoother_config: Optional[Dict[str, Any]] = None,
        prenet_config: Optional[Dict[str, Any]] = None,
        aux_layer: Optional[int] = None,
        temperature: float = 1.0,
        blank_idx: int = 0
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.num_days = num_days
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.blank_idx = blank_idx
        self.aux_layer = aux_layer
        
        # Default smoother configuration
        if smoother_config is None:
            smoother_config = {
                'type': 'depthwise_causal',
                'kernel_size': 11,
                'residual_gate': True,
                'init_std': 2.0
            }
        
        # Create smoother
        smoother_type = smoother_config.pop('type', 'depthwise_causal')
        self.smoother = build_smoother(
            smoother_type=smoother_type,
            num_features=input_dim,
            config=smoother_config
        )
        
        # Default PreNet configuration (no longer includes smoother)
        if prenet_config is None:
            prenet_config = {
                'day_adapter_config': {
                    'grouping': 'blocks8',
                    'identity_init': True,
                    'l2_tether': 1e-4,
                    'l1_reg': 0.0
                },
                'patch_config': {
                    'size': 4,
                    'stride': 2
                },
                'activation': 'softsign'
            }
        
        # Create PreNet (now without smoother)
        self.prenet = PreNet(
            input_dim=input_dim,
            num_days=num_days,
            **prenet_config
        )
        
        # Get PreNet output dimension (changes if patching is used)
        prenet_output_dim = self.prenet.output_dim
        
        # Create GRU backbone
        self.backbone = GRUBackbone(
            input_size=prenet_output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            dropout_type='standard',
            bidirectional=False,
            use_packed=True,
            learnable_h0=True
        )
        
        # Main CTC head (now with built-in masking)
        self.ctc_head = CTCHead(
            input_size=self.backbone.output_size,
            vocab_size=vocab_size,
            blank_idx=blank_idx,
            dropout=dropout / 2  # Less dropout in final layer
        )
        
        # Optional auxiliary CTC head for deep supervision
        self.aux_head = None
        if aux_layer is not None and 0 <= aux_layer < num_layers:
            # Check if backbone supports auxiliary outputs
            if hasattr(self.backbone, 'supports_aux') and self.backbone.supports_aux():
                self.aux_head = AuxiliaryHead(
                    input_size=self.backbone.output_size,
                    vocab_size=vocab_size,
                    hidden_size=None,  # Direct projection
                    dropout=dropout / 2,
                    layer_norm=True
                )
            else:
                print(f"Warning: Backbone {type(self.backbone).__name__} does not support auxiliary outputs")
                self.aux_layer = None
        
        # Temperature calibration
        self.calibrator = TemperatureScaling(
            initial_temperature=temperature,
            learnable=False  # Will be fitted post-training
        )
        
        # Track time reduction for proper length handling
        self._time_reduction = 1
        if prenet_config and 'patch_config' in prenet_config:
            patch_size = prenet_config['patch_config'].get('size', 1)
            patch_stride = prenet_config['patch_config'].get('stride', patch_size)
            if patch_size > 1:
                # This is approximate; exact reduction depends on input length
                self._time_reduction = patch_stride
    
    def forward(self, batch: Batch) -> Emissions:
        """Forward pass through the encoder.
        
        Args:
            batch: Input batch with neural features and metadata
            
        Returns:
            Emissions with log probabilities and lengths
        """
        # Extract inputs
        x = batch.x  # [B, T, F]
        lengths = batch.x_lens  # [B]
        day_indices = batch.day_id  # [B]
        
        # Smoother: temporal smoothing
        x, lengths = self.smoother(x, lengths)
        
        # PreNet: day adaptation, activation, patching
        x, lengths = self.prenet(x, lengths, day_indices)
        
        # GRU backbone with packed sequences
        # Request intermediates if auxiliary head is enabled
        x, hidden, intermediates = self.backbone(
            x, lengths, 
            return_intermediates=(self.aux_head is not None)
        )
        
        # Main CTC head (now handles masking internally)
        if self.training:
            # During training, don't apply temperature
            log_probs = self.ctc_head(x, lengths, temperature=1.0)
        else:
            # During inference, apply calibrated temperature
            log_probs = self.ctc_head(x, lengths, temperature=self.calibrator.temperature.item())
        
        # Prepare auxiliary outputs if using deep supervision
        aux_outputs = {}
        
        if self.aux_head is not None and intermediates is not None:
            # Get features from the specified intermediate layer
            if self.aux_layer < len(intermediates):
                aux_features = intermediates[self.aux_layer]
                
                # Get auxiliary predictions
                aux_logits = self.aux_head(aux_features)
                
                # For auxiliary outputs, lengths should be the same as main output
                # since all GRU layers have the same time dimension (no internal stride)
                # The time reduction happens in PreNet (patching) before the GRU backbone
                aux_lengths = lengths
                
                # Apply masking and log-softmax
                mask_logits_(aux_logits, aux_lengths)
                aux_log_probs = F.log_softmax(aux_logits, dim=-1)
                
                aux_outputs['aux_log_probs'] = aux_log_probs
                aux_outputs['aux_lengths'] = aux_lengths  # Store for verification
                aux_outputs['aux_layer'] = self.aux_layer
        
        # Add regularization loss if applicable
        if self.prenet.day_adapter is not None:
            aux_outputs['film_reg_loss'] = self.prenet.regularization_loss()
        
        # Create emissions
        emissions = Emissions(
            log_probs=log_probs,
            out_lens=lengths,
            aux=aux_outputs if aux_outputs else None
        )
        
        return emissions
    
    def time_reduction(self) -> int:
        """Get the overall time reduction factor.
        
        Returns:
            Time reduction factor from patching
        """
        return self._time_reduction
    
    def supports_packed(self) -> bool:
        """Check if model supports packed sequences.
        
        Returns:
            True (GRU supports packed sequences)
        """
        return True
    
    def eval_calibrate(self, dev_loader: torch.utils.data.DataLoader) -> None:
        """Calibrate temperature scaling on validation data.
        
        Args:
            dev_loader: Validation data loader
        """
        print("Calibrating temperature on validation set...")
        optimal_temp = self.calibrator.fit(self, dev_loader)
        self.calibrator.set_temperature(optimal_temp)
        print(f"Temperature calibration complete: T={optimal_temp:.4f}")
    
    def export_config(self) -> Dict[str, Any]:
        """Export model configuration for reproducibility.
        
        Returns:
            Complete model configuration
        """
        config = super().export_config()
        
        # Smoother configuration
        smoother_config = self.smoother.export_config() if hasattr(self.smoother, 'export_config') else {'type': 'unknown'}
        
        # PreNet detailed configuration
        prenet_config = {
            'activation': self.prenet.activation_name,
            'patch_size': self.prenet.patch_size,
            'patch_stride': self.prenet.patch_stride,
        }
        
        # Day adapter configuration
        if hasattr(self.prenet, 'day_adapter') and self.prenet.day_adapter is not None:
            adapter = self.prenet.day_adapter
            prenet_config['day_adapter'] = {
                'grouping': adapter.grouping,
                'l2_tether': adapter.l2_tether,
                'l1_reg': adapter.l1_reg,
                'num_groups': adapter.num_groups,
                'group_size': adapter.group_size
            }
        else:
            prenet_config['day_adapter'] = None
        
        config.update({
            'input_dim': self.input_dim,
            'vocab_size': self.vocab_size,
            'num_days': self.num_days,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'smoother': smoother_config,
            'prenet': prenet_config,
            'backbone': {
                'type': 'gru',
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.backbone.dropout,
                'bidirectional': self.backbone.bidirectional,
                'use_packed': self.backbone.use_packed,
                'supports_aux': self.backbone.supports_aux()
            },
            'aux_layer': self.aux_layer,
            'temperature': self.calibrator.temperature.item(),
            'blank_idx': self.blank_idx
        })
        return config