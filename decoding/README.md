# Pure-Python CTC Decoding Pipeline

This package replaces the legacy Kaldi/SRILM/Redis stack with a pure-Python implementation using TorchAudio CTC decoder + KenLM.

## 🎯 **Quick Start**

### 1. **Build Components**
```bash
# Activate environment
source .venv/bin/activate

# Build tokens.txt (41 tokens from LOGIT_PHONE_DEF)
cd decoding
python build_tokens.py

# Build lexicon.txt (CMUdict + G2P with phone mapping)
python build_lexicon.py

# Create simple test language model
python create_simple_lm.py
```

### 2. **Train Model with W&B Logging**
```bash
cd model_training
python train_model.py --config rnn_args_extended.yaml
```

### 3. **Tune Decoding Parameters**
```bash
python tune_decoding_params.py \
    --model_path ../data/t15_pretrained_rnn_baseline \
    --data_dir ../data/hdf5_data_final \
    --output_dir tuning_results
```

### 4. **Generate Submission**
```bash
python generate_submission.py \
    --model_path ../data/t15_pretrained_rnn_baseline \
    --data_dir ../data/hdf5_data_final \
    --eval_type test \
    --output_dir submission_output
```

## 📁 **Components**

### **Core Decoding**
- `build_tokens.py` - Generates `tokens.txt` from `LOGIT_PHONE_DEF` (41 tokens)
- `build_lexicon.py` - Creates `lexicon.txt` from CMUdict + G2P with phone mapping
- `decode_ctc.py` - TorchAudio CTC decoder with Flashlight backend
- `filter_lm.py` - Optional KenLM vocabulary filtering

### **Training & Evaluation**
- `rnn_args_extended.yaml` - Extended config with decoding + W&B settings
- `tune_decoding_params.py` - Hyperparameter grid search for λ and β
- `generate_submission.py` - Test decoding and CSV generation

## 🔧 **Configuration**

### **Proven Parameters** (from legacy system)
```yaml
decoding:
  lm_weight: 0.35      # acoustic_scale from legacy (0.325-0.35 working)
  word_score: -90.0    # negative blank_penalty from legacy (90.0 working)  
  beam_size: 17        # beam from legacy (17.0 default)
  beam_threshold: 8.0  # lattice_beam from legacy
  nbest: 100          # nbest from legacy
```

### **W&B Logging**
```yaml
wandb:
  project: "nejm-brain-to-text"
  tags: ["ctc", "rnn", "baseline"]
  log_frequency: 100
```

## 🎛️ **Parameter Tuning**

The grid search tests combinations of:
- **lm_weight** (λ): `[0.1, 0.2, 0.35, 0.5, 1.0, 2.0, 3.0]`
- **word_score** (β): `[-100.0, -90.0, -50.0, -10.0, -1.0, -0.5, 0.0, 0.5]`
- **beam_size**: `[17]` (optional: `[10, 17, 50]`)

Results saved to:
- `decoding_param_grid_results.csv` - Full grid results
- `best_decoding_params.yaml` - Optimal parameters

## 📊 **Metrics Logged**

### **Training (W&B)**
- `train/loss`, `train/grad_norm`, `train/lr`
- `val/PER`, `val/loss`, `val/best_PER`
- `val/PER_{session_name}` (per-day performance)

### **Tuning**
- **WER** (primary metric), **CER**, parameter combinations

## 🔄 **Migration from Legacy**

| Legacy (Redis/Kaldi) | New (Python/TorchAudio) |
|----------------------|---------------------------|
| `acoustic_scale=0.35` | `lm_weight=0.35` |
| `blank_penalty=90.0` | `word_score=-90.0` |
| `beam=17.0` | `beam_size=17` |
| `lattice_beam=8.0` | `beam_threshold=8.0` |
| `nbest=100` | `nbest=100` |

## 🚀 **Next Steps**

1. **Get proper 5-gram KenLM model** (user will handle)
2. **Run parameter tuning** on validation data
3. **Train model without day conditioning** (per Phase 1 plan)
4. **Generate final test submission**

## 🛠️ **Dependencies**

- PyTorch + TorchAudio
- flashlight-text (CTC decoder backend)
- KenLM (Python package)
- W&B, jiwer, g2p_en, omegaconf

All installed in `.venv` environment.
