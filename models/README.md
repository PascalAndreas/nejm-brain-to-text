# Neural Encoder Models for Brain-to-Text

This module provides modular, production-ready neural encoder architectures for converting neural implant data to text using CTC loss and beam search decoding.

## 🎯 Key Features

- **Modular Architecture**: Composable blocks (PreNet → Backbone → Head) for easy experimentation
- **Packed Sequences**: Efficient RNN training with proper padding handling
- **Day Adaptation**: FiLM (Feature-wise Linear Modulation) for recording day normalization
- **Deep Supervision**: Auxiliary CTC heads for improved gradient flow
- **EMA Weights**: Exponential Moving Average for better generalization
- **Temperature Calibration**: Post-training calibration for reliable confidence scores
- **Lightning Integration**: Seamless MPS → CUDA migration with PyTorch Lightning

## 📦 Installation

```bash
# Install dependencies
pip install torch pytorch-lightning wandb pyyaml numpy

# Optional: Install for faster decoding
pip install torchmetrics
```

## 🚀 Quick Start

### Training a Model

```bash
# Train with default configuration
python models/train.py --config pipeline/config.yaml

# Resume from checkpoint
python models/train.py --resume models/checkpoints/last.ckpt

# Override hyperparameters
python models/train.py --batch-size 64 --lr 1e-3 --epochs 50
```

### Using the Model API

```python
from models import build_encoder
from models.base import Batch

# Load model from config
model = build_encoder('gru_v1', {
    'input_dim': 512,
    'vocab_size': 41,
    'num_days': 20,
    'hidden_size': 768,
    'num_layers': 5
})

# Create batch
batch = Batch(
    x=neural_features,  # [B, T, F]
    x_lens=lengths,      # [B]
    day_id=day_indices   # [B]
)

# Get emissions
emissions = model(batch)
log_probs = emissions.log_probs  # [B, T, V]
```

## 🏗️ Architecture

### Model Pipeline

```
Input [B, T, 512]
    ↓
┌─────────────────────────┐
│       PreNet            │
├─────────────────────────┤
│ • Gaussian Smoothing    │  kernel=100, std=2
│ • Patching (4→2)        │  Concat 4 frames, stride 2
│ • Day Adapter (FiLM)    │  γ·x + β per day/block
│ • Activation (softsign) │
└─────────────────────────┘
    ↓ [B, T/2, 2048]
┌─────────────────────────┐
│    GRU Backbone         │
├─────────────────────────┤
│ • 5 layers × 768 units  │
│ • Packed sequences      │
│ • Dropout (0.2)         │
│ • Learnable h₀          │
└─────────────────────────┘
    ↓ [B, T/2, 768]
┌─────────────────────────┐
│   Projection Head       │
├─────────────────────────┤
│ • Linear → 41 phones    │
│ • Temperature scaling   │
│ • Log-softmax          │
└─────────────────────────┘
    ↓
Output [B, T/2, 41]
```

### Key Components

#### PreNet
- **Gaussian Smoothing**: Temporal smoothing of neural signals
- **Patching**: Concatenates frames to reduce sequence length (4:2 ratio)
- **Day Adapter**: FiLM-based normalization with identity initialization
- **Activation**: Softsign for bounded non-linearity

#### Backbone
- **GRU Stack**: Multi-layer GRU with orthogonal initialization
- **Packed Sequences**: Efficient handling of variable-length inputs
- **Dropout Types**: Standard, Variational, or Zoneout
- **Bidirectional Option**: Support for bidirectional processing

#### Head
- **Main CTC Head**: Projects to vocabulary with log-softmax
- **Auxiliary Head**: Mid-layer CTC for deep supervision
- **Temperature Calibration**: Post-training calibration on validation set

## 🔧 Configuration

### Model Configuration (`pipeline/config.yaml`)

```yaml
model:
  name: gru_v1
  params:
    input_dim: 512
    vocab_size: 41
    num_days: 20
    hidden_size: 768
    num_layers: 5
    dropout: 0.2
    
    prenet_config:
      smoother_config:
        kernel_size: 100
        std: 2.0
      day_adapter_config:
        grouping: blocks8  # 8 groups of 64 channels
        identity_init: true
        l2_tether: 0.0001
      patch_config:
        size: 4
        stride: 2
      activation: softsign
    
    aux_ctc_config:
      layer: 3
      weight: 0.25
```

### Training Configuration

```yaml
training:
  optimizer:
    lr: 0.0003
    weight_decay: 0.01
  scheduler:
    type: cosine
    warmup_steps: 5000
  use_ema: true
  ema_decay: 0.999
  gradient_clip_val: 1.0
```

## 📊 Emission Caching

Cache model outputs for efficient decoder tuning:

```python
from pipeline.emit import EmissionCache

# Create cache manager
cache = EmissionCache(
    cache_dir='cache/emissions',
    format='npz',  # or 'pt', 'hdf5'
    compress=True
)

# Cache emissions
cache_file = cache.cache_emissions(
    model=model,
    dataloader=val_loader,
    split_name='val',
    device='cuda'
)

# Load cached emissions
emissions = cache.load_emissions('val')
```

## 🧪 Testing

Run unit tests to verify model correctness:

```bash
# Run all tests
python -m pytest tests/test_models.py -v

# Run specific test
python -m pytest tests/test_models.py::TestPaddingInvariance -v
```

### Key Test Coverage

- **Padding Invariance**: Model outputs are consistent regardless of padding
- **Length Mapping**: Correct computation of output lengths through all operations
- **Masking**: Proper masking of invalid positions
- **Day Adaptation**: FiLM parameters correctly applied
- **Packed Sequences**: RNN packing/unpacking preserves outputs

## 🎛️ Advanced Features

### Exponential Moving Average (EMA)

EMA weights often generalize better than raw trained weights:

```python
from models.lightning_module import EMA

# Create EMA tracker
ema = EMA(model, decay=0.999)

# Update after each training step
ema.update()

# Apply EMA weights for evaluation
ema.apply_shadow()
outputs = model(batch)
ema.restore()
```

### Temperature Calibration

Improve confidence calibration post-training:

```python
# Calibrate on validation set
model.eval_calibrate(val_loader)

# Temperature is now optimized for better calibration
calibrated_outputs = model(batch)  # Uses optimal temperature
```

### Multi-Head Ensembles

Use multiple projection heads for uncertainty estimation:

```python
from models.blocks import MultiHeadProjection

multi_head = MultiHeadProjection(
    input_size=768,
    vocab_size=41,
    num_heads=3,
    ensemble_method='mean'
)

# Get ensemble predictions and uncertainty
logits, all_logits = multi_head(hidden_states, return_all=True)
uncertainty = multi_head.get_uncertainty(hidden_states)
```

## 📈 Performance Tips

1. **Use Packed Sequences**: Always provide sequence lengths for RNN models
2. **Enable AMP**: Use automatic mixed precision for faster training
3. **Tune Patching**: Adjust patch size/stride based on your sequence lengths
4. **Monitor FiLM Norms**: Check day adapter deviations aren't too large
5. **Cache Emissions**: Cache outputs for decoder hyperparameter tuning
6. **Use EMA**: Often provides 1-2% improvement over raw weights

## 🔄 Model Variants

The architecture supports easy swapping of components:

```python
# Use LSTM instead of GRU
from models.blocks import LSTMBackbone

backbone = LSTMBackbone(
    input_size=2048,
    hidden_size=768,
    num_layers=5
)

# Different activation functions
from models.blocks import get_activation_fn

activation = get_activation_fn('gelu')  # or 'relu', 'silu', etc.

# No patching for higher temporal resolution
prenet_config = {
    'patch_config': {'size': 1, 'stride': 1}
}
```

## 📝 Citation

If you use this code, please cite:

```bibtex
@software{brain_to_text_2025,
  title = {Modular Neural Encoders for Brain-to-Text},
  year = {2025},
  url = {https://github.com/yourusername/brain-to-text-25}
}
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Follow the existing code structure
2. Add unit tests for new features
3. Update documentation
4. Run linting and tests before submitting

## 📄 License

This project is licensed under the MIT License.
