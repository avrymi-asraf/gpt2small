"""
sae/buffer.py

Role: Implements the ActivationBuffer class that streams text from HuggingFace
datasets, runs GPT-2 forward passes to extract activations from target transformer
blocks, and provides shuffled batches for SAE training. Buffers are kept on GPU
for training efficiency. Integrates with ActivationExtractor from the parent project.
"""

import torch
from typing import Iterator, Optional
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

import sys
sys.path.insert(0, "/workspaces/gpt2small")
from ActivationExtractor import ActivationExtractor


class ActivationBuffer:
    """
    Streams activations from GPT-2 for SAE training.
    
    Loads text from HuggingFace datasets, extracts activations from a target
    transformer block, and yields shuffled batches. Buffer is kept on GPU.
    
    Args:
        layer_name: Target layer (e.g., "transformer.h.6").
        buffer_size: Number of activation vectors to buffer before shuffling.
        batch_size: Number of vectors per training batch.
        dataset_name: HuggingFace dataset name.
        device: Device for buffer storage (default: auto-detect GPU).
    """
    
    def __init__(
        self,
        layer_name: str,
        buffer_size: int = 100_000,
        batch_size: int = 4096,
        dataset_name: str = "wikitext",
        device: Optional[str] = None,
    ):
        self.layer_name = layer_name
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelForCausalLM.from_pretrained("gpt2").to(self.device)
        self.model.eval()
        
        # Create extractor for the target layer only
        self.extractor = ActivationExtractor(
            model=self.model,
            tokenizer=self.tokenizer,
            layer_names=[layer_name],
        )
        
        # Load streaming dataset
        if dataset_name == "wikitext":
            self.dataset = load_dataset(
                "wikitext", "wikitext-2-raw-v1", split="train", streaming=True
            )
        elif dataset_name == "openwebtext":
            self.dataset = load_dataset("openwebtext", split="train", streaming=True)
        else:
            self.dataset = load_dataset(dataset_name, split="train", streaming=True)
        
        self.data_iter = iter(self.dataset)
        self.buffer: Optional[torch.Tensor] = None
    
    def _fill_buffer(self) -> torch.Tensor:
        """Fill buffer with activations from dataset."""
        activations_list = []
        total_vectors = 0
        
        while total_vectors < self.buffer_size:
            try:
                sample = next(self.data_iter)
            except StopIteration:
                self.data_iter = iter(self.dataset)
                sample = next(self.data_iter)
            
            text = sample.get("text", "")
            if not text or len(text.strip()) < 10:
                continue
            
            # Extract activations (returned on CPU by default)
            result = self.extractor.extract(text)
            act = result["activations"][self.layer_name]  # [1, seq_len, 768]
            
            # Flatten to [seq_len, 768] and move to GPU
            act = act.squeeze(0).to(self.device)
            activations_list.append(act)
            total_vectors += act.shape[0]
        
        # Concatenate and shuffle
        buffer = torch.cat(activations_list, dim=0)[:self.buffer_size]
        perm = torch.randperm(buffer.shape[0], device=self.device)
        return buffer[perm]
    
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Yield shuffled batches of activations."""
        self.buffer = self._fill_buffer()
        idx = 0
        
        while idx + self.batch_size <= self.buffer.shape[0]:
            yield self.buffer[idx : idx + self.batch_size]
            idx += self.batch_size
        
        # Yield remaining
        if idx < self.buffer.shape[0]:
            yield self.buffer[idx:]
