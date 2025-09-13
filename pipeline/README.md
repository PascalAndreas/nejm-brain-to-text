# Pipeline

This directory contains scripts for end-to-end inference and submission generation.

## Scripts

### generate_submission.py

Generate submission CSV files for test data using trained models and optimized decoding parameters.

**Usage:**
```bash
# Basic usage
python pipeline/generate_submission.py \
    --model_path data/t15_pretrained_rnn_baseline/t15_pretrained_rnn_baseline \
    --data_dir data/t15_copyTask_neuralData/hdf5_data_final \
    --output_dir submission_output

# With specific decoder backend
python pipeline/generate_submission.py \
    --model_path data/t15_pretrained_rnn_baseline/t15_pretrained_rnn_baseline \
    --data_dir data/t15_copyTask_neuralData/hdf5_data_final \
    --decoder_backend flashlight \
    --output_dir submission_output

# With custom config
python pipeline/generate_submission.py \
    --model_path data/t15_pretrained_rnn_baseline/t15_pretrained_rnn_baseline \
    --data_dir data/t15_copyTask_neuralData/hdf5_data_final \
    --config_path decoding/config.yaml \
    --output_dir submission_output
```

**Arguments:**
- `--model_path`: Path to trained model directory (required)
- `--data_dir`: Path to neural data directory (required)
- `--output_dir`: Output directory for submission files (default: submission_output)
- `--eval_type`: Evaluation type - 'val' or 'test' (default: test)
- `--config_path`: Path to config file with decoding parameters
- `--decoder_backend`: Decoder backend - 'greedy' or 'flashlight' (default: flashlight)
- `--device`: Device to use - 'cuda' or 'cpu' (default: cuda)

**Output:**
- `submission_<eval_type>_<backend>_<timestamp>.csv` - Main submission file
- `submission_log_<timestamp>.yaml` - Processing log with metadata
- `submission_generation.log` - Detailed execution log

## Features

- **Flexible decoder backends**: Supports both greedy and flashlight decoders
- **Automatic configuration**: Uses decoding/config.yaml by default
- **Comprehensive logging**: Detailed logs of processing time and parameters
- **Text normalization**: Configurable text preprocessing for submissions
- **Chronological ordering**: Ensures proper trial ordering in submission files

## Configuration

The script uses the decoder configuration from `decoding/config.yaml` by default. You can override this with a custom config file that includes:

```yaml
decoding:
  backend: flashlight
  lm_weight: 1.0
  word_score: -0.5
  beam_size: 500
  # ... other decoder parameters

submission:
  lowercase: true
  strip_punct: true
  collapse_space: true
```
