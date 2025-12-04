"""
sae/model.py

Role: Defines the SparseAutoencoder neural network module used for learning
interpretable features from GPT-2 layer activations. This is the core model
component of the SAE pipeline, providing encode/decode functionality with ReLU
activation for sparsity. Designed to work with 768-dim GPT-2 hidden states and
is trained via sae/train.py within the development container.
"""

import torch
import torch.nn as nn


class SparseAutoencoder(nn.Module):
    """
    Standard ReLU Sparse Autoencoder for activation analysis.
    
    Encodes input activations into a sparse higher-dimensional representation,
    then decodes back to the original dimension. Sparsity is encouraged via
    L1 regularization on the latent activations during training.
    
    Args:
        input_dim: Dimension of input activations (768 for GPT-2 Small).
        expansion: Expansion factor for hidden dimension (default 8 → 6144 hidden).
    """
    
    def __init__(self, input_dim: int = 768, expansion: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = input_dim * expansion
        
        # Encoder: input_dim → hidden_dim
        self.W_enc = nn.Linear(input_dim, self.hidden_dim, bias=True)
        
        # Decoder: hidden_dim → input_dim
        self.W_dec = nn.Linear(self.hidden_dim, input_dim, bias=True)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small random values."""
        nn.init.xavier_uniform_(self.W_enc.weight)
        nn.init.zeros_(self.W_enc.bias)
        nn.init.xavier_uniform_(self.W_dec.weight)
        nn.init.zeros_(self.W_dec.bias)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input activations to sparse latent representation."""
        return torch.relu(self.W_enc(x))
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to activation space."""
        return self.W_dec(z)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass: returns (reconstructed, latents)."""
        latents = self.encode(x)
        reconstructed = self.decode(latents)
        return reconstructed, latents
