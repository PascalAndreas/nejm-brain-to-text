"""RNN backbone implementations for neural encoder models.

This module provides RNN-based encoder backbones with support for
packed sequences to efficiently handle variable-length inputs.
"""

from typing import Optional, Literal, Tuple, Dict, Any, List
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence


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
        dropout_type: Type of dropout ('standard', 'variational', 'zoneout')
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
        dropout_type: Literal['standard', 'variational', 'zoneout'] = 'standard',
        bidirectional: bool = False,
        use_packed: bool = True,
        learnable_h0: bool = True
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.dropout_type = dropout_type
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
        
        # Inter-layer dropout
        self.dropout_modules = nn.ModuleList()
        for i in range(num_layers - 1):  # No dropout after last layer
            if dropout_type == 'standard':
                self.dropout_modules.append(nn.Dropout(dropout))
            else:
                # For variational/zoneout, use standard for now
                # TODO: Implement proper variational dropout
                self.dropout_modules.append(nn.Dropout(dropout))
        
        # Initialize weights
        self._init_weights()
        
        # Learnable initial hidden states (one per layer)
        self.h0 = None
        if learnable_h0:
            self.h0 = nn.ParameterList([
                nn.Parameter(torch.zeros(self.num_directions, 1, hidden_size))
                for _ in range(num_layers)
            ])
            for h in self.h0:
                nn.init.xavier_uniform_(h)
    
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
                    h = self.h0[i].expand(-1, batch_size, -1).contiguous()
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
        
        # Process through layers
        intermediates = [] if return_intermediates else None
        final_hidden = []
        
        for i, layer in enumerate(self.layers):
            # Forward through layer
            seq, h = layer(seq, hidden[i])
            final_hidden.append(h)
            
            # Store intermediate output if requested
            if return_intermediates:
                if isinstance(seq, PackedSequence):
                    # Unpack to get tensor
                    out_i, _ = pad_packed_sequence(
                        seq, batch_first=True, total_length=max_len
                    )
                else:
                    out_i = seq
                intermediates.append(out_i)
            
            # Apply dropout between layers (not after last layer)
            if i < self.num_layers - 1 and self.training and self.dropout > 0:
                if isinstance(seq, PackedSequence):
                    # Unpack, apply dropout, repack
                    data = seq.data
                    data = self.dropout_modules[i](data)
                    seq = PackedSequence(data, seq.batch_sizes, 
                                       seq.sorted_indices, seq.unsorted_indices)
                else:
                    seq = self.dropout_modules[i](seq)
        
        # Unpack final output
        if isinstance(seq, PackedSequence):
            output, _ = pad_packed_sequence(seq, batch_first=True, total_length=max_len)
        else:
            output = seq
        
        return output, final_hidden, intermediates
    
    def supports_aux(self) -> bool:
        """Check if this backbone supports auxiliary outputs."""
        return True
    
    def extra_repr(self) -> str:
        s = (f'input_size={self.input_size}, hidden_size={self.hidden_size}, '
             f'num_layers={self.num_layers}, dropout={self.dropout}, '
             f'dropout_type={self.dropout_type}, bidirectional={self.bidirectional}, '
             f'use_packed={self.use_packed}')
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
        bidirectional: bool = False,
        use_packed: bool = True,
        learnable_h0: bool = True
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.use_packed = use_packed
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
        
        # Inter-layer dropout
        self.dropout_modules = nn.ModuleList()
        for i in range(num_layers - 1):
            self.dropout_modules.append(nn.Dropout(dropout))
        
        # Initialize weights
        for layer in self.layers:
            for name, param in layer.named_parameters():
                if 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        
        # Learnable initial states
        if learnable_h0:
            self.h0 = nn.ParameterList([
                nn.Parameter(torch.zeros(self.num_directions, 1, hidden_size))
                for _ in range(num_layers)
            ])
            self.c0 = nn.ParameterList([
                nn.Parameter(torch.zeros(self.num_directions, 1, hidden_size))
                for _ in range(num_layers)
            ])
            for h, c in zip(self.h0, self.c0):
                nn.init.xavier_uniform_(h)
                nn.init.xavier_uniform_(c)
        else:
            self.h0 = None
            self.c0 = None
    
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
                    h = self.h0[i].expand(-1, batch_size, -1).contiguous()
                    c = self.c0[i].expand(-1, batch_size, -1).contiguous()
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
            # Forward through layer
            seq, hc = layer(seq, hidden[i])
            final_hidden.append(hc)
            
            # Store intermediate output
            if return_intermediates:
                if isinstance(seq, PackedSequence):
                    out_i, _ = pad_packed_sequence(
                        seq, batch_first=True, total_length=max_len
                    )
                else:
                    out_i = seq
                intermediates.append(out_i)
            
            # Apply dropout between layers
            if i < self.num_layers - 1 and self.training and self.dropout > 0:
                if isinstance(seq, PackedSequence):
                    data = seq.data
                    data = self.dropout_modules[i](data)
                    seq = PackedSequence(data, seq.batch_sizes,
                                       seq.sorted_indices, seq.unsorted_indices)
                else:
                    seq = self.dropout_modules[i](seq)
        
        # Unpack final output
        if isinstance(seq, PackedSequence):
            output, _ = pad_packed_sequence(seq, batch_first=True, total_length=max_len)
        else:
            output = seq
        
        return output, final_hidden, intermediates
    
    def supports_aux(self) -> bool:
        """Check if this backbone supports auxiliary outputs."""
        return True


class VariationalGRU(nn.Module):
    """GRU with variational dropout (same mask across time steps).
    
    Note: This implementation does not support auxiliary outputs or
    packed sequences, and should be considered experimental.
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
        batch_first: bool = True,
        bidirectional: bool = False
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.output_size = hidden_size * (2 if bidirectional else 1)
        
        # Note: Bidirectional not fully implemented
        if bidirectional:
            raise NotImplementedError("Bidirectional VariationalGRU not yet supported")
        
        # Create GRU cells for each layer
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = nn.GRUCell(layer_input_size, hidden_size)
            self.cells.append(cell)
        
        # Variational dropout modules
        self.dropout_modules = nn.ModuleList()
        for layer in range(num_layers - 1):
            self.dropout_modules.append(nn.Dropout(dropout))
    
    def forward(
        self,
        x: torch.FloatTensor,
        hidden: Optional[torch.FloatTensor] = None,
        return_intermediates: bool = False
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, None]:
        """Forward pass with variational dropout.
        
        Note: This implementation does not support packed sequences or
        intermediate outputs.
        
        Args:
            x: Input tensor [B, T, F] if batch_first else [T, B, F]
            hidden: Initial hidden states
            return_intermediates: Not supported, will return None
            
        Returns:
            Tuple of (output, final_hidden, None)
        """
        if return_intermediates:
            print("Warning: VariationalGRU does not support intermediate outputs")
        
        if self.batch_first:
            x = x.transpose(0, 1)  # [T, B, F]
        
        seq_len, batch_size, _ = x.shape
        
        # Initialize hidden states if not provided
        if hidden is None:
            hidden = x.new_zeros(
                self.num_layers,
                batch_size,
                self.hidden_size
            )
        
        # Process through layers
        outputs = []
        for t in range(seq_len):
            x_t = x[t]
            new_hidden = []
            
            for layer in range(self.num_layers):
                h_layer = hidden[layer]
                cell = self.cells[layer]
                h_new = cell(x_t, h_layer)
                new_hidden.append(h_new)
                x_t = h_new
                
                # Apply dropout (except after last layer)
                if layer < self.num_layers - 1 and self.training:
                    if t == 0:
                        # Create dropout mask on first timestep
                        self._dropout_mask = x_t.new_empty(x_t.shape).bernoulli_(1 - self.dropout)
                        self._dropout_mask = self._dropout_mask / (1 - self.dropout)
                    x_t = x_t * self._dropout_mask
            
            outputs.append(x_t.unsqueeze(0))
            hidden = torch.stack(new_hidden)
        
        output = torch.cat(outputs, dim=0)
        
        if self.batch_first:
            output = output.transpose(0, 1)  # [B, T, H]
        
        return output, hidden, None
    
    def supports_aux(self) -> bool:
        """Check if this backbone supports auxiliary outputs."""
        return False