"""GRU-CTC neural encoder implementation.

This module implements the main GRU-based CTC encoder by composing
the various building blocks (PreNet, GRU backbone, projection heads).
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
    compute_output_lengths,
    mask_logits_
)


class GRUCTC(NeuralEncoder):
    """GRU-based neural encoder with CTC output head.
    
    This model implements the complete pipeline:
    1. PreNet: Gaussian smoothing → Patching → Day adaptation → Activation
    2. GRU Backbone: Multi-layer GRU with packed sequence support
    3. Projection Head: Linear projection to vocabulary
    4. Optional: Auxiliary CTC head for deep supervision
    5. Optional: Temperature calibration
    
    Args:
        input_dim: Number of input features
        vocab_size: Size of output vocabulary (including CTC blank)
        num_days: Number of recording days for adaptation
        hidden_size: GRU hidden state size
        num_layers: Number of GRU layers
        dropout: Dropout probability
        prenet_config: Configuration for PreNet
        aux_ctc_config: Configuration for auxiliary CTC head
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
        prenet_config: Optional[Dict[str, Any]] = None,
        aux_ctc_config: Optional[Dict[str, Any]] = None,
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
        
        # Default PreNet configuration
        if prenet_config is None:
            prenet_config = {
                'smoother_config': {
                    'kernel_size': 100,
                    'std': 2.0,
                    'trainable': False
                },
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
        
        # Create PreNet
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
        
        # Main CTC head
        self.ctc_head = CTCHead(
            input_size=self.backbone.output_size,
            vocab_size=vocab_size,
            blank_idx=blank_idx,
            dropout=dropout / 2  # Less dropout in final layer
        )
        
        # Optional auxiliary CTC head for deep supervision
        self.aux_head = None
        self.aux_layer = None
        self.aux_weight = 0.0
        
        if aux_ctc_config is not None:
            self.aux_layer = aux_ctc_config.get('layer', num_layers // 2)
            self.aux_weight = aux_ctc_config.get('weight', 0.25)
            
            self.aux_head = AuxiliaryHead(
                input_size=hidden_size,
                vocab_size=vocab_size,
                hidden_size=None,  # Direct projection
                dropout=dropout / 2,
                layer_norm=True
            )
            
            # Register hook to capture intermediate representations
            self._aux_features = None
            self._register_aux_hook()
        
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
    
    def _register_aux_hook(self):
        """Register forward hook to capture intermediate GRU features."""
        if self.aux_head is None:
            return
        
        def hook_fn(module, input, output):
            # For GRU: output is (output_seq, hidden_states)
            if isinstance(output, tuple):
                output_seq = output[0]
            else:
                output_seq = output
            
            # Store intermediate features
            # We'll extract the specific layer output later
            self._aux_features = output_seq
        
        # Register hook on the GRU module
        self.backbone.gru.register_forward_hook(hook_fn)
    
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
        
        # PreNet: smoothing, patching, day adaptation, activation
        x, lengths = self.prenet(x, lengths, day_indices)
        
        # GRU backbone with packed sequences
        x, hidden = self.backbone(x, lengths)
        
        # Main CTC head
        if self.training:
            # During training, don't apply temperature
            log_probs = self.ctc_head(x, lengths, temperature=1.0)
        else:
            # During inference, apply calibrated temperature
            log_probs = self.ctc_head(x, lengths, temperature=self.calibrator.temperature.item())
        
        # Prepare auxiliary outputs if using deep supervision
        aux_outputs = {}
        if self.aux_head is not None and self._aux_features is not None:
            # Get auxiliary predictions from intermediate layer
            aux_logits = self.aux_head(self._aux_features)
            
            # Apply masking and log-softmax
            mask_logits_(aux_logits, lengths)
            aux_log_probs = F.log_softmax(aux_logits, dim=-1)
            
            aux_outputs['aux_log_probs'] = aux_log_probs
            aux_outputs['aux_layer'] = self.aux_layer
            aux_outputs['aux_weight'] = self.aux_weight
            
            # Clear stored features
            self._aux_features = None
        
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
        config.update({
            'input_dim': self.input_dim,
            'vocab_size': self.vocab_size,
            'num_days': self.num_days,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'prenet': {
                'smoother': hasattr(self.prenet, 'smoother') and self.prenet.smoother is not None,
                'patch_size': self.prenet.patch_size,
                'patch_stride': self.prenet.patch_stride,
                'day_adapter': hasattr(self.prenet, 'day_adapter') and self.prenet.day_adapter is not None,
                'activation': str(self.prenet.activation)
            },
            'aux_ctc': {
                'enabled': self.aux_head is not None,
                'layer': self.aux_layer,
                'weight': self.aux_weight
            },
            'temperature': self.calibrator.temperature.item(),
            'blank_idx': self.blank_idx
        })
        return config
    
    def get_intermediate_features(self, layer: int) -> Optional[torch.Tensor]:
        """Get features from a specific GRU layer.
        
        Args:
            layer: Layer index (0-based)
            
        Returns:
            Features from specified layer if available
        """
        # This would require more sophisticated hook management
        # For now, return None
        return None
    
    def compute_ctc_loss(
        self,
        emissions: Emissions,
        targets: torch.LongTensor,
        target_lengths: torch.LongTensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute CTC loss with optional auxiliary loss.
        
        Args:
            emissions: Model emissions
            targets: Target phoneme sequences
            target_lengths: Target sequence lengths
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Import CTC loss
        from torch.nn import CTCLoss
        ctc_criterion = CTCLoss(blank=self.blank_idx, reduction='mean', zero_infinity=True)
        
        # Main CTC loss
        # Note: CTC expects [T, B, V] so we transpose
        log_probs_transposed = emissions.log_probs.transpose(0, 1)
        
        main_loss = ctc_criterion(
            log_probs_transposed,
            targets,
            emissions.out_lens,
            target_lengths
        )
        
        losses = {'ctc_loss': main_loss}
        total_loss = main_loss
        
        # Add auxiliary CTC loss if available
        if emissions.aux and 'aux_log_probs' in emissions.aux:
            aux_log_probs = emissions.aux['aux_log_probs'].transpose(0, 1)
            aux_loss = ctc_criterion(
                aux_log_probs,
                targets,
                emissions.out_lens,
                target_lengths
            )
            
            losses['aux_ctc_loss'] = aux_loss
            total_loss = total_loss + emissions.aux['aux_weight'] * aux_loss
        
        # Add FiLM regularization if available
        if emissions.aux and 'film_reg_loss' in emissions.aux:
            reg_loss = emissions.aux['film_reg_loss']
            losses['film_reg_loss'] = reg_loss
            total_loss = total_loss + reg_loss
        
        return total_loss, losses
