"""RNN backbone implementations for neural encoder models.

This module provides RNN-based encoder backbones with support for
packed sequences to efficiently handle variable-length inputs.
"""

from typing import Optional, Literal, Tuple, Dict, Any, List
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
from .utils import expand_h0, apply_output_zoneout, apply_interlayer_dropout


class GRUBackbone(nn.Module):
    """Multi-layer GRU backbone with packed sequence support.
    
    This module implements a stack of GRU layers with support for:
    - Packed sequences to skip padding computations
    - Intermediate layer outputs for auxiliary supervision
    - Multiple dropout types (standard, variational, zoneout)
    - Learnable initial hidden states
    - Orthogonal weight initialization
    
    Args:
        input_size: Size of input features
        hidden_size: Size of GRU hidden state
        num_layers: Number of GRU layers
        dropout: Dropout probability between layers
        dropout_type: Type of dropout ('standard', 'locked', 'zoneout')
        zoneout: Zoneout probability for recurrent connections (0.0 to disable)
        bidirectional: If True, use bidirectional GRU
        use_packed: If True, use packed sequences
        learnable_h0: If True, use learnable initial hidden state
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 5,
        dropout: float = 0.2,
        dropout_type: Literal['standard', 'locked'] = 'standard',
        zoneout: float = 0.0,
        bidirectional: bool = False,
        use_packed: bool = True,
        learnable_h0: bool = True
    ):
        super().__init__()
        
        # Validate parameters first
        if not 0 <= dropout <= 1:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")
        if not 0 <= zoneout <= 1:
            raise ValueError(f"zoneout must be in [0, 1], got {zoneout}")
        if dropout_type not in ['standard', 'locked']:
            raise ValueError(f"dropout_type must be one of ['standard', 'locked'], got {dropout_type}")
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.dropout_type = dropout_type
        self.zoneout = zoneout
        self.bidirectional = bidirectional
        self.use_packed = use_packed
        self.num_directions = 2 if bidirectional else 1
        self.output_size = hidden_size * self.num_directions
        
        # Create stack of 1-layer GRUs for clean intermediate outputs
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else self.output_size
            
            layer = nn.GRU(
                input_size=layer_input_size,
                hidden_size=hidden_size,
                num_layers=1,  # Single layer for each module
                dropout=0,  # We'll handle dropout separately
                batch_first=True,
                bidirectional=bidirectional
            )
            
            self.layers.append(layer)
        
        # Inter-layer dropout modules (only for standard dropout)
        self.dropout_modules = nn.ModuleList()
        for i in range(num_layers - 1):  # No dropout after last layer
            if dropout_type == 'standard':
                self.dropout_modules.append(nn.Dropout(dropout))
            else:
                # Locked and zoneout dropout handled in forward pass
                self.dropout_modules.append(None)  # Placeholder
        
        # Initialize weights
        self._init_weights()
        
        # Learnable initial hidden states (one per layer)
        self.h0 = None
        if learnable_h0:
            self.h0 = nn.ParameterList([
                nn.Parameter(torch.zeros(self.num_directions, 1, hidden_size))
                for _ in range(num_layers)
            ])
            # Use orthogonal initialization for better stability
            for h in self.h0:
                nn.init.orthogonal_(h)
        
    
    def _init_weights(self):
        """Initialize GRU weights with orthogonal/xavier initialization."""
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(
        self,
        x: torch.FloatTensor,
        lengths: Optional[torch.LongTensor] = None,
        hidden: Optional[List[torch.FloatTensor]] = None,
        return_intermediates: bool = False
    ) -> Tuple[torch.FloatTensor, List[torch.FloatTensor], Optional[List[torch.FloatTensor]]]:
        """Forward pass through GRU layers.
        
        Args:
            x: Input tensor [B, T, F]
            lengths: Sequence lengths for packing [B]
            hidden: Initial hidden states (list of [num_directions, B, H])
            return_intermediates: If True, return intermediate layer outputs
            
        Returns:
            Tuple of:
            - Output tensor [B, T, H * num_directions]
            - Final hidden states (list of [num_directions, B, H])
            - Intermediate outputs (optional, list of [B, T, H * num_directions])
        """
        batch_size = x.shape[0]
        max_len = x.shape[1]
        
        # Initialize hidden states if not provided
        if hidden is None:
            hidden = []
            for i in range(self.num_layers):
                if self.h0 is not None:
                    # Use learnable initial state
                    h = expand_h0(self.h0[i], batch_size, x.device, x.dtype)
                else:
                    # Zero initialization
                    h = torch.zeros(
                        self.num_directions,
                        batch_size,
                        self.hidden_size,
                        device=x.device,
                        dtype=x.dtype
                    )
                hidden.append(h)
        
        # Pack input if using packed sequences
        if self.use_packed and lengths is not None:
            seq = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        else:
            seq = x
        
        # No need for mask lifecycle management - we regenerate per forward pass
        
        # Process through layers
        intermediates = [] if return_intermediates else None
        final_hidden = []
        
        for i, layer in enumerate(self.layers):
            # 1. Forward through layer
            seq, h = layer(seq, hidden[i])
            final_hidden.append(h)
            
            # 2. Apply zoneout (if enabled)
            if self.zoneout > 0:
                seq = apply_output_zoneout(seq, self.zoneout, lengths, max_len, self.bidirectional)
            
            # 3. Apply inter-layer dropout (locked or standard)
            seq = apply_interlayer_dropout(seq, i, batch_size, max_len, self.dropout, self.dropout_type, self.num_layers, self.output_size, self.dropout_modules, self.training)
            
            # 4. Store intermediate output if requested (after regularization)
            if return_intermediates:
                if isinstance(seq, PackedSequence):
                    # Unpack to get tensor
                    out_i, _ = pad_packed_sequence(
                        seq, batch_first=True, total_length=max_len
                    )
                else:
                    out_i = seq
                intermediates.append(out_i)
        
        # Unpack final output
        if isinstance(seq, PackedSequence):
            output, _ = pad_packed_sequence(seq, batch_first=True, total_length=max_len)
        else:
            output = seq
        
        return output, final_hidden, intermediates
    
    def train(self, mode: bool = True):
        """Set training mode."""
        return super().train(mode)
    
    def reset_parameters(self):
        """Reset all parameters to their initial values."""
        self._init_weights()
        if self.h0 is not None:
            for h in self.h0:
                nn.init.orthogonal_(h)
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration for serialization."""
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'dropout_type': self.dropout_type,
            'zoneout': self.zoneout,
            'bidirectional': self.bidirectional,
            'use_packed': self.use_packed,
            'learnable_h0': self.h0 is not None
        }
    
    def extra_repr(self) -> str:
        s = (f'input_size={self.input_size}, hidden_size={self.hidden_size}, '
             f'num_layers={self.num_layers}, dropout={self.dropout}, '
             f'dropout_type={self.dropout_type}, zoneout={self.zoneout}, '
             f'bidirectional={self.bidirectional}, use_packed={self.use_packed}')
        return s


class LSTMBackbone(nn.Module):
    """LSTM backbone with support for intermediate outputs.
    
    Similar to GRUBackbone but using LSTM cells.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 5,
        dropout: float = 0.2,
        dropout_type: Literal['standard', 'locked'] = 'standard',
        zoneout: float = 0.0,
        bidirectional: bool = False,
        use_packed: bool = True,
        learnable_h0: bool = True,
        h0_init: Literal['zeros', 'orthogonal'] = 'orthogonal',
        c0_init: Literal['zeros', 'orthogonal'] = 'orthogonal'
    ):
        super().__init__()
        
        # Validate parameters
        if not 0 <= dropout <= 1:
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")
        if not 0 <= zoneout <= 1:
            raise ValueError(f"zoneout must be in [0, 1], got {zoneout}")
        if dropout_type not in ['standard', 'locked']:
            raise ValueError(f"dropout_type must be one of ['standard', 'locked'], got {dropout_type}")
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.dropout_type = dropout_type
        self.zoneout = zoneout
        self.bidirectional = bidirectional
        self.use_packed = use_packed
        self.h0_init = h0_init
        self.c0_init = c0_init
        self.num_directions = 2 if bidirectional else 1
        self.output_size = hidden_size * self.num_directions
        
        # Create stack of 1-layer LSTMs
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else self.output_size
            
            layer = nn.LSTM(
                input_size=layer_input_size,
                hidden_size=hidden_size,
                num_layers=1,
                dropout=0,
                batch_first=True,
                bidirectional=bidirectional
            )
            
            self.layers.append(layer)
        
        # Inter-layer dropout modules (only for standard dropout)
        self.dropout_modules = nn.ModuleList()
        for i in range(num_layers - 1):  # No dropout after last layer
            if dropout_type == 'standard':
                self.dropout_modules.append(nn.Dropout(dropout))
            else:
                # Locked dropout handled in forward pass
                self.dropout_modules.append(None)  # Placeholder
        
        # Initialize weights with forget gate bias = 1
        self._init_weights()
        
        # Learnable initial states
        self.h0 = None
        self.c0 = None
        if learnable_h0:
            self.h0 = nn.ParameterList([
                nn.Parameter(torch.zeros(self.num_directions, 1, hidden_size))
                for _ in range(num_layers)
            ])
            self.c0 = nn.ParameterList([
                nn.Parameter(torch.zeros(self.num_directions, 1, hidden_size))
                for _ in range(num_layers)
            ])
            
            # Initialize learnable states based on config
            for h, c in zip(self.h0, self.c0):
                if h0_init == 'orthogonal':
                    nn.init.orthogonal_(h)
                else:  # zeros
                    nn.init.zeros_(h)
                    
                if c0_init == 'orthogonal':
                    nn.init.orthogonal_(c)
                else:  # zeros
                    nn.init.zeros_(c)
    
    def _init_weights(self):
        """Initialize LSTM weights with orthogonal/xavier initialization and forget gate bias = 1."""
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias_ih' in name or 'bias_hh' in name:
                    param.data.fill_(0)
                    # PyTorch gate order: (i, f, g, o)
                    n = param.size(0) // 4
                    param.data[n:2*n].fill_(1.0)  # forget gate bias = 1
    
    def forward(
        self,
        x: torch.FloatTensor,
        lengths: Optional[torch.LongTensor] = None,
        hidden: Optional[List[Tuple[torch.FloatTensor, torch.FloatTensor]]] = None,
        return_intermediates: bool = False
    ) -> Tuple[torch.FloatTensor, List[Tuple[torch.FloatTensor, torch.FloatTensor]], 
               Optional[List[torch.FloatTensor]]]:
        """Forward pass through LSTM layers.
        
        Args:
            x: Input tensor [B, T, F]
            lengths: Sequence lengths for packing [B]
            hidden: Initial (hidden, cell) states
            return_intermediates: If True, return intermediate layer outputs
            
        Returns:
            Tuple of:
            - Output tensor [B, T, H * num_directions]
            - Final (hidden, cell) states
            - Intermediate outputs (optional)
        """
        batch_size = x.shape[0]
        max_len = x.shape[1]
        
        # Initialize states if not provided
        if hidden is None:
            hidden = []
            for i in range(self.num_layers):
                if self.h0 is not None:
                    h = expand_h0(self.h0[i], batch_size, x.device, x.dtype)
                    c = expand_h0(self.c0[i], batch_size, x.device, x.dtype)
                    hidden.append((h, c))
                else:
                    hidden.append(None)
        
        # Pack input if using packed sequences
        if self.use_packed and lengths is not None:
            seq = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        else:
            seq = x
        
        # Process through layers
        intermediates = [] if return_intermediates else None
        final_hidden = []
        
        for i, layer in enumerate(self.layers):
            # 1. Forward through layer
            seq, hc = layer(seq, hidden[i])
            final_hidden.append(hc)
            
            # 2. Apply zoneout (if enabled)
            if self.zoneout > 0:
                seq = apply_output_zoneout(seq, self.zoneout, lengths, max_len, self.bidirectional)
            
            # 3. Apply inter-layer dropout (locked or standard)
            seq = apply_interlayer_dropout(seq, i, batch_size, max_len, self.dropout, self.dropout_type, self.num_layers, self.output_size, self.dropout_modules, self.training)
            
            # 4. Store intermediate output if requested (after regularization)
            if return_intermediates:
                if isinstance(seq, PackedSequence):
                    out_i, _ = pad_packed_sequence(
                        seq, batch_first=True, total_length=max_len
                    )
                else:
                    out_i = seq
                intermediates.append(out_i)
        
        # Unpack final output
        if isinstance(seq, PackedSequence):
            output, _ = pad_packed_sequence(seq, batch_first=True, total_length=max_len)
        else:
            output = seq
        
        return output, final_hidden, intermediates