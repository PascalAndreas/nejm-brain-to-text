"""
Setup script for NVIDIA TAO language models and lexicon.
Builds the lexicon using NVIDIA TAO vocabulary and configures the decoding pipeline.
"""

import os
import sys
from pathlib import Path
from omegaconf import OmegaConf

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from build_tokens import build_tokens
from build_lexicon import build_lexicon
from decoding import DEFAULT_CONFIG


def setup_nvidia_tao_pipeline():
    """Set up the complete NVIDIA TAO decoding pipeline."""
    print("🚀 Setting up NVIDIA TAO CTC Decoding Pipeline")
    print("=" * 50)
    
    config = DEFAULT_CONFIG
    
    # 1. Build tokens.txt
    print("\n1. Building tokens.txt...")
    tokens_path = config.artifacts.tokens_txt
    build_tokens(tokens_path)
    
    # 2. Build lexicon using NVIDIA TAO vocabulary
    print("\n2. Building lexicon with NVIDIA TAO vocabulary...")
    lexicon_path = config.artifacts.lexicon_txt
    
    # Use NVIDIA TAO vocabulary file
    nvidia_vocab_file = config.language_model.nvidia_tao.vocab_file
    
    if not os.path.exists(nvidia_vocab_file):
        print(f"❌ NVIDIA TAO vocabulary file not found: {nvidia_vocab_file}")
        print("Make sure you've downloaded the NVIDIA TAO model with:")
        print('ngc registry model download-version "nvidia/tao/speechtotext_en_us_lm:deployable_v4.1"')
        return False
    
    # Build lexicon with NVIDIA vocabulary
    num_words, num_pronunciations, oov_rate = build_lexicon(
        output_path=lexicon_path,
        vocab_boost_size=config.lexicon.lm_vocab_size,
        lm_vocab_file=nvidia_vocab_file
    )
    
    print(f"✅ Lexicon built: {num_words} words, {num_pronunciations} pronunciations, {oov_rate:.1%} OOV rate")
    
    # 3. Verify language model files
    print("\n3. Verifying NVIDIA TAO language model files...")
    active_lm = config.language_model.active_model
    
    if os.path.exists(active_lm):
        file_size = os.path.getsize(active_lm) / (1024**2)  # MB
        print(f"✅ Active language model: {active_lm} ({file_size:.1f} MB)")
    else:
        print(f"❌ Active language model not found: {active_lm}")
        return False
    
    # 4. Display available models
    print("\n4. Available NVIDIA TAO models:")
    tao_config = config.language_model.nvidia_tao
    models = [
        ("4-gram ARPA", tao_config.arpa_4gram),
        ("4-gram Binary", tao_config.binary_4gram),
        ("Mixed LM ARPA (5-gram)", tao_config.arpa_mixed),
        ("Mixed LM Binary", tao_config.binary_mixed),
        ("3-gram ARPA", tao_config.arpa_3gram),
    ]
    
    for name, path in models:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024**2)
            status = "✅" if path == active_lm else "  "
            print(f"{status} {name}: {Path(path).name} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {name}: {path} (missing)")
    
    # 5. Display configuration summary
    print("\n5. Configuration Summary:")
    decoder_config = config.decoder
    print(f"   LM Weight: {decoder_config.lm_weight}")
    print(f"   Word Score: {decoder_config.word_score}")
    print(f"   Beam Size: {decoder_config.beam_size}")
    print(f"   N-best: {decoder_config.nbest}")
    
    print("\n🎉 NVIDIA TAO pipeline setup complete!")
    print("\nNext steps:")
    print("1. Train your model: cd model_training && python train_model.py")
    print("2. Tune parameters: cd decoding && python tune_decoding_params.py")
    print("3. Generate submission: cd model_training && python generate_submission.py")
    
    return True


def switch_language_model(model_name: str):
    """
    Switch the active language model.
    
    Args:
        model_name: One of ['4gram', 'mixed', '3gram']
    """
    config_path = Path(__file__).parent / "config.yaml"
    config = OmegaConf.load(config_path)
    
    model_mapping = {
        '4gram': config.language_model.nvidia_tao.binary_4gram,
        'mixed': config.language_model.nvidia_tao.binary_mixed,
        '3gram': config.language_model.nvidia_tao.arpa_3gram,
    }
    
    if model_name not in model_mapping:
        print(f"❌ Invalid model name. Choose from: {list(model_mapping.keys())}")
        return False
    
    new_model = model_mapping[model_name]
    
    if not os.path.exists(new_model):
        print(f"❌ Model file not found: {new_model}")
        return False
    
    # Update config
    config.language_model.active_model = new_model
    OmegaConf.save(config, config_path)
    
    print(f"✅ Switched active model to: {model_name} ({Path(new_model).name})")
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup NVIDIA TAO decoding pipeline")
    parser.add_argument('--switch-model', type=str, choices=['4gram', 'mixed', '3gram'],
                        help='Switch active language model')
    
    args = parser.parse_args()
    
    if args.switch_model:
        switch_language_model(args.switch_model)
    else:
        setup_nvidia_tao_pipeline()
