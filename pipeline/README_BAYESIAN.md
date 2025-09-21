# Bayesian Hyperparameter Optimization for Decoder

This directory contains a complete Bayesian optimization system for decoder hyperparameters with corpus-specific profile support.

## Overview

The system provides:

1. **Efficient Bayesian Optimization**: Uses Optuna for smart hyperparameter search
2. **Corpus-Specific Profiles**: Different decoding parameters for different corpus types
3. **Emission Caching**: Avoids repeated model inference during tuning
4. **Comprehensive Evaluation**: Detailed performance analysis by corpus

## Key Components

### 1. `bayesian_tuning.py`
- `BayesianDecoderTuner`: Main optimization class
- `DecodingProfile`: Data structure for hyperparameter profiles
- `run_full_optimization()`: Complete pipeline function

### 2. `corpus_decoder.py`
- `CorpusAwareDecoder`: Decoder that uses corpus-specific profiles
- `CorpusAwareEvaluator`: Performance evaluation with corpus breakdown

### 3. `emit.py` (Updated)
- Enhanced emission caching with corpus metadata
- Support for different storage formats (NPZ, PT, HDF5)

### 4. `run_bayesian_tuning.py`
- Command-line interface for running optimization
- Supports both optimization and evaluation modes

## Quick Start

### 1. Install Dependencies

```bash
# In your .venv
pip install -r requirements_bayesian.txt
```

### 2. Run Full Optimization

```bash
python pipeline/run_bayesian_tuning.py \
    --model_checkpoint models/checkpoints/best_model.ckpt \
    --config pipeline/config.yaml \
    --n_trials 100 \
    --output_dir tuning_results
```

### 3. Evaluate Results

```bash
python pipeline/run_bayesian_tuning.py \
    --evaluate_only tuning_results/profiles_val.json
```

## Detailed Usage

### Step-by-Step Optimization

```python
from pipeline.emit import EmissionCache
from pipeline.bayesian_tuning import BayesianDecoderTuner

# 1. Initialize cache manager
cache_manager = EmissionCache(cache_dir='cache/emissions')

# 2. Initialize tuner
tuner = BayesianDecoderTuner(
    cache_manager=cache_manager,
    split_name='val',
    n_trials=100,
    output_dir='tuning_results'
)

# 3. Optimize global profile
global_profile = tuner.optimize_global_profile()

# 4. Optimize corpus-specific profiles
corpus_profiles = tuner.optimize_corpus_specific_profiles(
    base_profile=global_profile
)

# 5. Save results
profiles_file = tuner.save_profiles()
```

### Using Corpus-Aware Decoder

```python
from pipeline.corpus_decoder import CorpusAwareDecoder
import torch

# Load optimized profiles
decoder = CorpusAwareDecoder('tuning_results/profiles_val.json')

# Decode with corpus information
log_probs = torch.randn(1, 100, 41)  # [batch, time, vocab]
lengths = torch.tensor([100])
corpus = "Switchboard"

result = decoder.decode_single(log_probs, lengths, corpus)
print(f"Decoded: '{result['sentence']}'")
```

### Batch Decoding with Mixed Corpora

```python
# Batch with different corpus types
log_probs = torch.randn(3, 100, 41)  # 3 utterances
lengths = torch.tensor([100, 95, 80])
corpus_info = ["50-Word", "Switchboard", "Harvard"]

results = decoder.decode_batch(log_probs, lengths, corpus_info)
for i, result in enumerate(results):
    print(f"{corpus_info[i]}: '{result['sentence']}'")
```

## Configuration

### Hyperparameter Search Space

The optimization searches over:

- `lm_weight`: Language model weight (0.5 - 5.0)
- `word_score`: Word insertion penalty (-2.0 - 1.0)  
- `sil_score`: Silence token score (-2.0 - 2.0)
- `beam_size`: Beam search width (20 - 200)
- `beam_size_token`: Token-level beam size (5 - 20)
- `beam_threshold`: Beam pruning threshold (5.0 - 50.0)

### Corpus Types

The system recognizes these corpus types from the dataset:

- `50-Word`: 50-word vocabulary tasks
- `Switchboard`: Conversational speech
- `Openwebtext`: Web text
- `Random`: Random sentences
- `Harvard`: Harvard sentences
- `Freq words`: Frequent words

## Performance Tips

### 1. Emission Caching

Cache emissions once and reuse for multiple optimization runs:

```bash
# Cache emissions first
python pipeline/run_bayesian_tuning.py \
    --model_checkpoint models/best.ckpt \
    --n_trials 1  # Just cache, don't optimize

# Then run optimization without re-caching
python pipeline/run_bayesian_tuning.py \
    --skip_caching \
    --n_trials 100
```

### 2. Optimization Strategy

1. **Global First**: Run global optimization with many trials (100+)
2. **Corpus-Specific**: Use fewer trials (20-50) for corpus-specific tuning
3. **Iterative**: Start with small trials, increase if needed

### 3. Memory Management

- Use NPZ format for large datasets (compressed)
- Use PT format for faster loading (uncompressed)
- Use HDF5 for complex metadata requirements

## Output Files

### Profiles JSON
```json
{
  "global": {
    "name": "global",
    "lm_weight": 2.5,
    "word_score": -0.2,
    "performance": {"wer": 0.234, "cer": 0.123}
  },
  "Switchboard": {
    "name": "Switchboard", 
    "lm_weight": 3.0,
    "word_score": -0.1,
    "performance": {"wer": 0.198, "cer": 0.098}
  }
}
```

### Optuna Studies
- Saved as pickle files for detailed analysis
- Can be loaded with `optuna.load_study()`
- Support visualization with `optuna.visualization`

## Advanced Usage

### Custom Evaluation Metrics

```python
from pipeline.corpus_decoder import CorpusAwareEvaluator

evaluator = CorpusAwareEvaluator(corpus_decoder)
results = evaluator.evaluate_on_emissions(emissions_dict)

# Results include per-corpus breakdown
for corpus, metrics in results.items():
    print(f"{corpus}: WER={metrics['wer']:.3f}")
```

### Manual Profile Creation

```python
from pipeline.bayesian_tuning import DecodingProfile

custom_profile = DecodingProfile(
    name='custom',
    lm_weight=2.0,
    word_score=-0.3,
    sil_score=0.1,
    beam_size=80,
    beam_size_token=12,
    beam_threshold=25.0
)

decoder.update_profile('custom_corpus', custom_profile)
```

### Integration with Existing Pipeline

The system integrates seamlessly with existing pipeline components:

```python
from pipeline.generate_submission import generate_submission

# Use corpus-aware decoder in submission generation
# (requires modification of generate_submission.py)
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Install `optuna` and dependencies
2. **Memory Issues**: Use NPZ format and smaller batch sizes
3. **No Ground Truth**: Ensure validation data has sentence labels
4. **Empty Results**: Check decoder configuration and token files

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Monitoring

Monitor optimization progress:

```python
# Access study object for analysis
study = tuner.study
print(f"Best trial: {study.best_trial.number}")
print(f"Best value: {study.best_value}")

# Plot optimization history (requires plotly)
import optuna.visualization as vis
fig = vis.plot_optimization_history(study)
fig.show()
```

## Future Enhancements

1. **Multi-objective Optimization**: Optimize WER and CER simultaneously
2. **Active Learning**: Intelligently select validation samples
3. **Ensemble Decoding**: Combine multiple profiles
4. **Online Adaptation**: Update profiles based on new data
5. **LLM Reranking Integration**: Prepare for future LLM reranking
