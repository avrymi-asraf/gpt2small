# GPT2SMALL

A tool for extracting, visualizing, and generating text with GPT-2 model activations, with support for training Sparse Autoencoders (SAEs) on layer activations.

## Features

- **Activation Extraction**: Capture and analyze internal layer activations during model forward passes
- **Text Generation**: Generate text continuations with various sampling strategies (greedy, temperature, top-k, top-p)
- **Generation with Activations**: Collect activations during generation to see how internal states evolve token-by-token
- **Sparse Autoencoder (SAE)**: Train SAEs on layer activations to learn interpretable features
- **SAE Visualization**: Analyze which SAE features activate for specific words
- **Interactive Notebook**: Explore model behavior through Jupyter notebooks
- **Persistent Hooks**: Efficient activation extraction with reusable forward hooks

## Program Flow

### Activation Extraction
1. Initialize `ActivationExtractor` with target layers
2. Register PyTorch forward hooks on specified layers
3. Run single forward pass on input text
4. Collect and return activation tensors

### Text Generation
1. Tokenize input prompt
2. Iteratively predict next tokens using model logits
3. Apply sampling strategies (temperature, top-k, nucleus)
4. Append predicted tokens until max length or EOS
5. Return generated text and tokens

## File Structure

```
/workspaces/gpt2small
├── Interface.ipynb         # Interactive notebook for exploration
├── ActivationExtractor.py  # ActivationExtractor class with extract() and generate()
├── pyproject.toml          # Dependencies (uv-managed)
├── Dockerfile.dev          # Development container configuration
├── sae/                    # Sparse Autoencoder module
│   ├── __init__.py         # Package exports
│   ├── model.py            # SparseAutoencoder neural network
│   ├── buffer.py           # ActivationBuffer for streaming training data
│   ├── train.py            # train_sae() training function
│   └── visualize.py        # analyze_word(), compare_reconstruction()
├── sae_checkpoints/        # Saved SAE model checkpoints
└── README.md               # This file
```

## Usage

### Quick Start (Activation Extraction)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from tools import ActivationExtractor

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Extract activations
extractor = ActivationExtractor(model, tokenizer, ["transformer.h.0"])
result = extractor.extract("The future of AI is")

print(result["tokens"])                    # Input tokens
print(result["activations"]["transformer.h.0"].shape)  # Activation tensor
```

### Quick Start (Text Generation)

```python
# Generate text continuation
result = extractor.generate(
    text="Once upon a time",
    max_new_tokens=50,
    temperature=0.8,
    top_p=0.95
)

print(result["generated_text"])       # Full text (prompt + generated)
print(result["generated_tokens"])     # Only new tokens
# Activations are always returned for each step
print(f"Captured {len(result['activations'])} steps of activations")
```

### Generation with Activation Analysis

```python
# Generate text (activations are collected automatically)
result = extractor.generate(
    text="The future of AI is",
    max_new_tokens=10,
    temperature=0.7
)

print(result["generated_text"])

# Analyze activations for each generated token
for step, (token, activations) in enumerate(zip(result["generated_tokens"], result["activations"])):
    print(f"Step {step+1} - Token '{token}':")
    for layer_name, act in activations.items():
        last_token_act = act[0, -1, :]
        print(f"  {layer_name}: mean={last_token_act.mean():.3f}, std={last_token_act.std():.3f}")
```

### Generation Parameters

- `max_new_tokens`: Maximum tokens to generate (default: 50)
- `temperature`: Sampling randomness - higher = more creative (default: 1.0)
- `top_k`: Sample from top-k most likely tokens (optional)
- `top_p`: Nucleus sampling - sample from smallest set with cumulative prob > p (optional)
- `do_sample`: Use sampling vs. greedy decoding (default: True)
- `eos_token_id`: Stop generation at this token (optional)

### Sparse Autoencoder Training

```python
# Train SAE on layer 6 activations
from sae.train import train_sae

sae = train_sae(
    layer_name="transformer.h.6",
    epochs=3,
    lr=1e-4,
    l1_coef=1e-3,
    dataset_name="wikitext"
)
```

### SAE Feature Analysis

```python
from sae.visualize import analyze_word, compare_reconstruction

# See which SAE features activate for a word
analyze_word("king", sae, "transformer.h.6", top_k=10)

# Compare original vs reconstructed activations
compare_reconstruction("The cat sat on the mat", sae, "transformer.h.6")
```

### SAE Parameters

- `layer_name`: Target transformer block (e.g., "transformer.h.6")
- `epochs`: Number of training passes through buffer (default: 3)
- `lr`: Learning rate (default: 1e-4)
- `l1_coef`: L1 sparsity coefficient (default: 1e-3)
- `dataset_name`: HuggingFace dataset ("wikitext", "openwebtext")
- `buffer_size`: Activation vectors to buffer (default: 100K)
- `expansion`: SAE hidden dimension multiplier (default: 8 → 6144 features)

### Running Examples

```bash
# Interactive notebook (recommended)
jupyter lab Interface.ipynb
```

Notebook-based demos and the interactive `Interface.ipynb` replace the older CLI demo scripts.
Note: legacy sample scripts such as `demo_activations.py`, `demo_generation.py`, and `demo_generation_with_activations.py` were removed from the repository to keep the project focused on the interactive notebook and `tools.py` API. Use `Interface.ipynb` or the `ActivationExtractor` in `tools.py` for programmatic examples.

## Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install torch transformers matplotlib
```

## Docker

```bash
# Build and run with GPU
docker build -f Dockerfile.dev -t gpt2small:dev .
docker run --gpus all --rm -it -p 8888:8888 \
  -v "$PWD":/workspaces/gpt2small gpt2small:dev \
  jupyter lab --ip=0.0.0.0 --no-browser --allow-root
```

