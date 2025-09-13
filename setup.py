from setuptools import setup, find_packages

setup(
    name="nejm-brain-to-text",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "torchaudio",
        "g2p_en",
        "omegaconf",
        "pyyaml"
    ],
    python_requires=">=3.8",
)
