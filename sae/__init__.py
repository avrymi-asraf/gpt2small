"""
sae/__init__.py

Role: Package entry point for the Sparse Autoencoder (SAE) module. Exports the
main classes and functions for training SAEs on GPT-2 layer activations and
visualizing learned features. This module integrates with the ActivationExtractor
from the parent project and is designed to run within the development container.
"""

from sae.model import SparseAutoencoder
from sae.buffer import ActivationBuffer
from sae.train import train_sae
from sae.visualize import analyze_word, compare_reconstruction

__all__ = [
    "SparseAutoencoder",
    "ActivationBuffer",
    "train_sae",
    "analyze_word",
    "compare_reconstruction",
]
