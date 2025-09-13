# Tests

This directory contains pytest-based tests for the NEJM Brain-to-Text decoding system.

## Setup

Install test dependencies:
```bash
pip install -r tests/requirements.txt
```

## Running Tests

Run all tests:
```bash
pytest tests/
```

Run specific test categories:
```bash
# Basic decoder functionality
pytest tests/test_decoders.py::TestDecoderBasics

# RNN model integration tests
pytest tests/test_decoders.py::TestRNNModelIntegration

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=decoding --cov=pipeline
```

## Test Structure

- `conftest.py` - Pytest fixtures and configuration
- `test_decoders.py` - Tests for decoder functionality and RNN integration
- `requirements.txt` - Test-specific dependencies

## Fixtures

The test suite includes several useful fixtures:

- `sample_logits` - Generated sample logits for testing
- `loaded_rnn_model` - Pretrained RNN model loaded from checkpoint
- `real_neural_data_sample` - Real neural data sample for testing
- `greedy_decoder` / `flashlight_decoder` - Decoder instances

## Notes

- Tests automatically skip if required data or models are not available
- RNN model tests use the same loading approach as `test_rnn_model_phonemes.py`
- Real neural data tests require the dataset to be present in `data/`
