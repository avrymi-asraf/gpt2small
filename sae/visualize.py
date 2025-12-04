"""
sae/visualize.py

Role: Provides visualization utilities for analyzing SAE features. Includes
functions to show which SAE features activate for a given word and to compare
original vs reconstructed activations. Used interactively in Interface.ipynb
for interpretability analysis within the development container.
"""

import torch
import matplotlib.pyplot as plt
from typing import Optional

from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
sys.path.insert(0, "/workspaces/gpt2small")
from ActivationExtractor import ActivationExtractor
from sae.model import SparseAutoencoder


def analyze_word(
    word: str,
    sae: SparseAutoencoder,
    layer_name: str,
    top_k: int = 10,
    model: Optional[AutoModelForCausalLM] = None,
    tokenizer: Optional[AutoTokenizer] = None,
) -> dict:
    """
    Show top-K activated SAE features for a given word.
    
    Args:
        word: Word to analyze.
        sae: Trained SparseAutoencoder.
        layer_name: Target layer name.
        top_k: Number of top features to show.
        model: GPT-2 model (loaded if not provided).
        tokenizer: GPT-2 tokenizer (loaded if not provided).
    
    Returns:
        Dict with 'top_features' (indices) and 'activations' (values).
    """
    device = next(sae.parameters()).device
    
    if model is None or tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        model.eval()
    
    extractor = ActivationExtractor(model, tokenizer, [layer_name])
    result = extractor.extract(word)
    
    # Get activation for the word (last token if multiple)
    act = result["activations"][layer_name].squeeze(0)[-1].to(device)  # [768]
    
    # Encode with SAE
    with torch.no_grad():
        latents = sae.encode(act)  # [hidden_dim]
    
    # Find top-K features
    top_values, top_indices = torch.topk(latents, top_k)
    
    # Print results
    print(f"Top {top_k} SAE features for '{word}':")
    for i, (idx, val) in enumerate(zip(top_indices.tolist(), top_values.tolist())):
        print(f"  {i+1}. Feature {idx}: {val:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 4))
    plt.bar(range(top_k), top_values.cpu().numpy())
    plt.xticks(range(top_k), [f"F{i}" for i in top_indices.tolist()], rotation=45)
    plt.xlabel("SAE Feature")
    plt.ylabel("Activation")
    plt.title(f"Top {top_k} SAE Features for '{word}'")
    plt.tight_layout()
    plt.show()
    
    extractor.clear_hooks()
    return {"top_features": top_indices.tolist(), "activations": top_values.tolist()}


def compare_reconstruction(
    text: str,
    sae: SparseAutoencoder,
    layer_name: str,
    model: Optional[AutoModelForCausalLM] = None,
    tokenizer: Optional[AutoTokenizer] = None,
) -> dict:
    """
    Plot original vs reconstructed activations for input text.
    
    Args:
        text: Input text.
        sae: Trained SparseAutoencoder.
        layer_name: Target layer name.
        model: GPT-2 model (loaded if not provided).
        tokenizer: GPT-2 tokenizer (loaded if not provided).
    
    Returns:
        Dict with 'mse' (reconstruction error) and 'sparsity' (fraction of zero latents).
    """
    device = next(sae.parameters()).device
    
    if model is None or tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
        model.eval()
    
    extractor = ActivationExtractor(model, tokenizer, [layer_name])
    result = extractor.extract(text)
    
    # Get activations
    act = result["activations"][layer_name].squeeze(0).to(device)  # [seq_len, 768]
    
    # Reconstruct
    with torch.no_grad():
        reconstructed, latents = sae(act)
    
    # Compute metrics
    mse = torch.mean((act - reconstructed) ** 2).item()
    sparsity = (latents == 0).float().mean().item()
    
    print(f"Reconstruction MSE: {mse:.6f}")
    print(f"Latent sparsity: {sparsity:.2%} zeros")
    
    # Plot comparison for first few dimensions
    n_dims = min(50, act.shape[1])
    token_idx = -1  # Last token
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    axes[0].bar(range(n_dims), act[token_idx, :n_dims].cpu().numpy(), alpha=0.7)
    axes[0].set_title("Original Activations (first 50 dims)")
    axes[0].set_xlabel("Dimension")
    
    axes[1].bar(range(n_dims), reconstructed[token_idx, :n_dims].cpu().numpy(), alpha=0.7, color="orange")
    axes[1].set_title("Reconstructed Activations (first 50 dims)")
    axes[1].set_xlabel("Dimension")
    
    plt.tight_layout()
    plt.show()
    
    extractor.clear_hooks()
    return {"mse": mse, "sparsity": sparsity}
