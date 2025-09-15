"""RNN backbone implementations for neural encoder models.

This module provides RNN-based encoder backbones with support for
packed sequences to efficiently handle variable-length inputs.
"""

from typing import Optional, Literal, Tuple, Dict, Any
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class GRUBackbone(nn.Module):
    """Multi-layer GRU backbone with packed sequence support.
    
    This module implements a stack of GRU layers with support for:
    - Packed sequences to skip padding computations
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
        
        # Create GRU layers
        if dropout_type == 'standard':
            # Standard dropout between layers (built into GRU)
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=bidirectional
            )
        else:
            # For variational/zoneout, we need custom implementation
            self.gru = self._build_custom_gru(dropout_type)
        
        # Initialize weights
        self._init_weights()
        
        # Learnable initial hidden state
        self.h0 = None
        if learnable_h0:
            self.h0 = nn.Parameter(
                torch.zeros(num_layers * self.num_directions, 1, hidden_size)
            )
            nn.init.xavier_uniform_(self.h0)
    
    def _build_custom_gru(self, dropout_type: str) -> nn.Module:
        """Build custom GRU for variational/zoneout dropout."""
        if dropout_type == 'variational':
            return VariationalGRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                batch_first=True,
                bidirectional=self.bidirectional
            )
        elif dropout_type == 'zoneout':
            # Zoneout would require custom implementation
            # For now, fall back to standard
            print(f"Warning: Zoneout not implemented, using standard dropout")
            return nn.GRU(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout if self.num_layers > 1 else 0,
                batch_first=True,
                bidirectional=self.bidirectional
            )
        else:
            raise ValueError(f"Unknown dropout type: {dropout_type}")
    
    def _init_weights(self):
        """Initialize GRU weights with orthogonal/xavier initialization."""
        if isinstance(self.gru, nn.GRU):
            for name, param in self.gru.named_parameters():
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
        hidden: Optional[torch.FloatTensor] = None
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Forward pass through GRU layers.
        
        Args:
            x: Input tensor [B, T, F]
            lengths: Sequence lengths for packing [B]
            hidden: Initial hidden state [num_layers * num_directions, B, H]
            
        Returns:
            Tuple of:
            - Output tensor [B, T, H * num_directions]
            - Final hidden state [num_layers * num_directions, B, H]
        """
        batch_size = x.shape[0]
        
        # Initialize hidden state if not provided
        if hidden is None:
            if self.h0 is not None:
                # Use learnable initial state
                hidden = self.h0.expand(-1, batch_size, -1).contiguous()
            else:
                # Zero initialization
                hidden = torch.zeros(
                    self.num_layers * self.num_directions,
                    batch_size,
                    self.hidden_size,
                    device=x.device,
                    dtype=x.dtype
                )
        
        # Use packed sequences if enabled and lengths provided
        if self.use_packed and lengths is not None:
            # Pack the input
            x_packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            
            # Forward through GRU
            output_packed, hidden = self.gru(x_packed, hidden)
            
            # Unpack the output
            output, output_lengths = pad_packed_sequence(
                output_packed, batch_first=True
            )
        else:
            # Standard forward pass
            output, hidden = self.gru(x, hidden)
        
        return output, hidden
    
    def extra_repr(self) -> str:
        s = (f'input_size={self.input_size}, hidden_size={self.hidden_size}, '
             f'num_layers={self.num_layers}, dropout={self.dropout}, '
             f'dropout_type={self.dropout_type}, bidirectional={self.bidirectional}, '
             f'use_packed={self.use_packed}')
        return s


class VariationalGRU(nn.Module):
    """GRU with variational dropout (same mask across time steps).
    
    This implementation applies the same dropout mask across all time steps
    for better regularization in RNNs.
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
        
        # Create GRU cells for each layer
        self.cells = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * (2 if bidirectional else 1)
            
            if bidirectional:
                cell_fw = nn.GRUCell(layer_input_size, hidden_size)
                cell_bw = nn.GRUCell(layer_input_size, hidden_size)
                self.cells.append(nn.ModuleList([cell_fw, cell_bw]))
            else:
                cell = nn.GRUCell(layer_input_size, hidden_size)
                self.cells.append(cell)
        
        # Variational dropout modules
        self.dropout_modules = nn.ModuleList()
        for layer in range(num_layers - 1):  # No dropout after last layer
            self.dropout_modules.append(nn.Dropout(dropout))
    
    def forward(
        self,
        x: torch.FloatTensor,
        hidden: Optional[torch.FloatTensor] = None
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Forward pass with variational dropout.
        
        Args:
            x: Input tensor [B, T, F] if batch_first else [T, B, F]
            hidden: Initial hidden states
            
        Returns:
            Tuple of output and final hidden states
        """
        if self.batch_first:
            x = x.transpose(0, 1)  # [T, B, F]
        
        seq_len, batch_size, _ = x.shape
        
        # Initialize hidden states if not provided
        if hidden is None:
            hidden = x.new_zeros(
                self.num_layers * (2 if self.bidirectional else 1),
                batch_size,
                self.hidden_size
            )
        
        # Process through layers
        outputs = []
        for t in range(seq_len):
            x_t = x[t]
            new_hidden = []
            
            for layer in range(self.num_layers):
                if self.bidirectional:
                    # Bidirectional processing would be more complex
                    # For now, just use forward direction
                    h_layer = hidden[layer * 2]
                    cell = self.cells[layer][0]
                    h_new = cell(x_t, h_layer)
                    new_hidden.append(h_new)
                    new_hidden.append(hidden[layer * 2 + 1])  # Keep backward unchanged
                    x_t = h_new
                else:
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
        
        return output, hidden


class LSTMBackbone(nn.Module):
    """LSTM backbone as an alternative to GRU.
    
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
        
        # Create LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Initialize weights
        for name, param in self.lstm.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        # Learnable initial states
        if learnable_h0:
            self.h0 = nn.Parameter(
                torch.zeros(num_layers * self.num_directions, 1, hidden_size)
            )
            self.c0 = nn.Parameter(
                torch.zeros(num_layers * self.num_directions, 1, hidden_size)
            )
            nn.init.xavier_uniform_(self.h0)
            nn.init.xavier_uniform_(self.c0)
        else:
            self.h0 = None
            self.c0 = None
    
    def forward(
        self,
        x: torch.FloatTensor,
        lengths: Optional[torch.LongTensor] = None,
        hidden: Optional[Tuple[torch.FloatTensor, torch.FloatTensor]] = None
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """Forward pass through LSTM layers.
        
        Args:
            x: Input tensor [B, T, F]
            lengths: Sequence lengths for packing [B]
            hidden: Initial (hidden, cell) states
            
        Returns:
            Tuple of:
            - Output tensor [B, T, H * num_directions]
            - Final (hidden, cell) states
        """
        batch_size = x.shape[0]
        
        # Initialize states if not provided
        if hidden is None:
            if self.h0 is not None:
                h = self.h0.expand(-1, batch_size, -1).contiguous()
                c = self.c0.expand(-1, batch_size, -1).contiguous()
                hidden = (h, c)
            else:
                hidden = None
        
        # Use packed sequences if enabled
        if self.use_packed and lengths is not None:
            x_packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            output_packed, hidden = self.lstm(x_packed, hidden)
            output, _ = pad_packed_sequence(output_packed, batch_first=True)
        else:
            output, hidden = self.lstm(x, hidden)
        
        return output, hidden
