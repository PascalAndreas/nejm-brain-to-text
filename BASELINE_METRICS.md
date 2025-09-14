# Pretrained RNN Baseline Metrics

This document provides comprehensive baseline metrics from the pretrained RNN model for reference when developing new NeuralEncoder implementations.

## Model Overview

**Architecture**: 5-layer GRU with day-specific adaptation
- **Hidden units**: 768 per layer
- **Input features**: 512 neural features  
- **Output classes**: 41 phonemes (including BLANK and SIL)
- **Day-specific layers**: 512×512 linear layers with softsign activation
- **Training method**: CTC loss with AdamW optimizer
- **Data augmentation**: Gaussian noise, temporal jitter, smoothing (kernel=100, std=2)

## Training Configuration

- **Total batches**: 120,000
- **Batch size**: 64
- **Training duration**: 213.41 minutes (~3.5 hours)
- **Evaluation frequency**: Every 2,000 batches (61 validation checkpoints)
- **Hardware**: RTX 4090 (as mentioned in README)
- **Sessions**: 42 recording sessions from t15.2023.08.11 to t15.2025.04.13

## Training Log Location

```
/Users/pascalandreas/Documents/repositories/brain-to-text-25/data/t15_pretrained_rnn_baseline/t15_pretrained_rnn_baseline/training_log
```

- **Size**: 361KB, 5,982 lines
- **Date**: July 15, 2024
- **Contains**: Complete training progression with per-batch metrics

## Performance Targets

### Validation Set Metrics (Primary Targets)

**Converged Performance** (final 20 checkpoints, batches 82,000-119,999):
- **Validation PER**: **10.17% ± 0.15%** (range: 10.10-10.25%)
- **Validation CTC Loss**: **22.52 ± 0.25** (range: 22.39-22.75)

**Best Performance Achieved**:
- **Best Average Validation PER**: **10.101%**

### Target Benchmarks for New Implementations

| **Performance Level** | **Validation PER Target** | **Notes** |
|----------------------|---------------------------|-----------|
| **Conservative** | ≤ 10.5% | Should definitely achieve |
| **Competitive** | ≤ 10.2% | Match current baseline |
| **Stretch** | ≤ 10.0% | Improve on baseline |

## Training Convergence Pattern

| **Training Phase** | **Batch Range** | **Validation PER** | **Validation CTC Loss** | **Characteristics** |
|-------------------|-----------------|-------------------|------------------------|-------------------|
| **Initial Learning** | 0 - 2,000 | 121.56% → 22.50% | 714.99 → 22.66 | Rapid initial convergence |
| **Fast Convergence** | 2,000 - 20,000 | 22.50% → 12.10% | 22.66 → 21.67 | Major performance gains |
| **Gradual Improvement** | 20,000 - 60,000 | 12.10% → 10.70% | 21.67 → 23.95 | Steady optimization |
| **Fine-tuning** | 60,000 - 120,000 | 10.70% → 10.11% | 23.95 → 22.39 | Stable convergence |

### Key Convergence Observations

1. **Rapid Initial Learning**: Model drops from random performance (121% PER) to reasonable performance (22% PER) in just 2,000 batches
2. **Major Convergence**: Achieves near-final performance (~12% PER) by batch 20,000 (1/6 of training)
3. **Stable Convergence**: Performance stabilizes around batch 60,000-80,000
4. **Final Stability**: Last 30 checkpoints show very consistent metrics (±0.15% PER variation)

## Detailed Performance Progression

### Early Training (Key Milestones)
```
Batch 0:     PER: 121.56%, CTC Loss: 714.99
Batch 2000:  PER: 22.50%,  CTC Loss: 22.66
Batch 4000:  PER: 15.91%,  CTC Loss: 17.56
Batch 6000:  PER: 13.89%,  CTC Loss: 17.27
Batch 10000: PER: 12.87%,  CTC Loss: 18.78
Batch 20000: PER: 12.10%,  CTC Loss: 21.67
```

### Mid Training (Stabilization)
```
Batch 40000: PER: 11.09%,  CTC Loss: 23.48
Batch 50000: PER: 10.93%,  CTC Loss: 23.80
Batch 60000: PER: 10.54%,  CTC Loss: 23.10
```

### Final Training (Converged Performance)
```
Batch 100000: PER: 10.11%, CTC Loss: 22.45
Batch 110000: PER: 10.17%, CTC Loss: 22.39
Batch 119999: PER: 10.11%, CTC Loss: 22.39  [FINAL]
```

## Implementation Notes

### Expected Training Characteristics
- **Fast initial convergence**: Expect major improvements in first 20K batches
- **Gradual refinement**: Fine-tuning phase typically lasts 40K+ batches  
- **Stability**: Performance should stabilize with <0.2% PER variation in final phase
- **Loss vs PER**: CTC loss may continue optimizing even when PER plateaus

### Validation Strategy
- **Evaluation frequency**: Every 2,000 batches provides good monitoring resolution
- **Early stopping**: Could potentially stop around batch 80,000-100,000 based on convergence
- **Stability check**: Monitor last 10-20 checkpoints for consistent performance

### Architecture Comparison Reference
When implementing new NeuralEncoder architectures, use these metrics to compare:
- **GRU baseline**: 10.17% ± 0.15% validation PER
- **Training efficiency**: 120K batches, 213 minutes
- **Convergence speed**: Major gains by batch 20K, stable by batch 80K

## Related Files

- **Model checkpoint**: `/data/t15_pretrained_rnn_baseline/t15_pretrained_rnn_baseline/checkpoint/best_checkpoint`
- **Model config**: `/data/t15_pretrained_rnn_baseline/t15_pretrained_rnn_baseline/checkpoint/args.yaml`
- **Validation results**: `/model_training/rnn_baseline_submission_file_valsplit.csv` (1,426 predictions)
- **Training code**: `/model_training/` (legacy, to be deprecated)

---

*Generated from training logs on September 14, 2025*
*Use these metrics as reference targets for new NeuralEncoder implementations per REFACTOR_SPEC.md*
