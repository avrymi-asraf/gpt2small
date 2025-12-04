"""
sae/train.py

Role: Provides the train_sae() function that orchestrates SAE training on GPT-2
activations. Implements a standard PyTorch training loop with MSE reconstruction
loss and L1 sparsity penalty. Saves checkpoints to sae_checkpoints/. This is the
main entry point for training SAEs within the development container.
"""

import os
import torch
from tqdm import tqdm
from typing import Optional

from sae.model import SparseAutoencoder
from sae.buffer import ActivationBuffer


def train_sae(
    layer_name: str,
    epochs: int = 3,
    lr: float = 1e-4,
    l1_coef: float = 1e-3,
    save_path: Optional[str] = None,
    dataset_name: str = "wikitext",
    buffer_size: int = 100_000,
    batch_size: int = 4096,
    input_dim: int = 768,
    expansion: int = 8,
) -> SparseAutoencoder:
    """
    Train a Sparse Autoencoder on activations from a GPT-2 layer.
    
    Args:
        layer_name: Target layer (e.g., "transformer.h.6").
        epochs: Number of passes through the buffer.
        lr: Learning rate.
        l1_coef: L1 sparsity coefficient.
        save_path: Path to save checkpoint (default: sae_checkpoints/{layer_name}.pt).
        dataset_name: HuggingFace dataset ("wikitext", "openwebtext", etc.).
        buffer_size: Number of activation vectors to buffer.
        batch_size: Training batch size.
        input_dim: Input dimension (768 for GPT-2 Small).
        expansion: SAE expansion factor.
    
    Returns:
        Trained SparseAutoencoder model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create SAE
    sae = SparseAutoencoder(input_dim=input_dim, expansion=expansion).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    
    # Create buffer
    buffer = ActivationBuffer(
        layer_name=layer_name,
        buffer_size=buffer_size,
        batch_size=batch_size,
        dataset_name=dataset_name,
        device=device,
    )
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0.0
        total_mse = 0.0
        total_l1 = 0.0
        num_batches = 0
        
        pbar = tqdm(buffer, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed, latents = sae(batch)
            
            # Loss: MSE + L1
            mse_loss = torch.mean((batch - reconstructed) ** 2)
            l1_loss = torch.mean(torch.abs(latents))
            loss = mse_loss + l1_coef * l1_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_l1 += l1_loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "mse": f"{mse_loss.item():.4f}",
                "l1": f"{l1_loss.item():.4f}",
            })
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1}: avg_loss={avg_loss:.4f}")
    
    # Save checkpoint
    if save_path is None:
        os.makedirs("sae_checkpoints", exist_ok=True)
        save_path = f"sae_checkpoints/{layer_name.replace('.', '_')}.pt"
    
    torch.save({
        "model_state_dict": sae.state_dict(),
        "layer_name": layer_name,
        "input_dim": input_dim,
        "expansion": expansion,
    }, save_path)
    print(f"Saved checkpoint to {save_path}")
    
    return sae
