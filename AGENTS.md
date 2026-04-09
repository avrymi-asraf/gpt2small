## Project Goal

* **Description:** The primary objective of this project is to provide a tool for extracting and visualizing the internal activations of specific layers within the GPT-2 Small language model. The project is designed to run inside a Docker container (see `Dockerfile.dev`) — the recommended interface is the interactive Jupyter Notebook (`Interface.ipynb`). Legacy CLI examples such as `main.py` and earlier demo scripts have been deprecated or removed in favor of the notebook-driven workflow.

---

## Build, Lint, and Test Commands

### Package Management (uv)
```bash
# Install dependencies
uv sync

# Add a new dependency
uv add <package>

# Remove a dependency
uv remove <package>

# Run with specific Python (if needed)
uv run python <script.py>
```

### Running the Application
```bash
# Local development with Jupyter
uv sync
jupyter lab Interface.ipynb

# Docker (recommended for GPU/consistency)
docker build -f Dockerfile.dev -t gpt2small:dev .
docker run --gpus all --rm -it -p 8888:8888 \
  -v "$PWD":/workspaces/gpt2small gpt2small:dev \
  jupyter lab --ip=0.0.0.0 --no-browser --allow-root
```

### Testing
This project does not currently have a test suite. When adding tests, use `pytest`:
```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_model.py

# Run a single test function
pytest tests/test_model.py::test_sparse_autoencoder_encode

# Run with verbose output
pytest -v

# Run tests matching a pattern
pytest -k "test_encode"
```

### Linting and Type Checking
```bash
# Ruff (linter + formatter)
ruff check .
ruff format .

# Pyright (static type checker)
pyright .

# Run all checks
ruff check . && ruff format --check . && pyright .
```

---

## Code Style Guidelines

### General Rules
- **No comments** in code (unless explicitly required)
- **No docstrings** in code (unless explicitly required)
- Keep lines under 100 characters when practical
- Use 4 spaces for indentation (not tabs)

### Imports
Organize imports in the following order (separate with blank lines):
1. Standard library (`os`, `json`, `torch`, etc.)
2. Third-party libraries (`transformers`, `matplotlib`, `datasets`)
3. Local project imports (relative imports like `from sae.model import ...`)

```python
# Good
import os
import torch
from typing import List, Dict, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt

from sae.model import SparseAutoencoder
from sae.buffer import ActivationBuffer
```

### Naming Conventions
- **Classes**: PascalCase (e.g., `SparseAutoencoder`, `ActivationExtractor`)
- **Functions/variables**: snake_case (e.g., `train_sae`, `layer_name`)
- **Constants**: SCREAMING_SNAKE_CASE (e.g., `BUFFER_SIZE`)
- **Private methods**: leading underscore (e.g., `_init_weights`)

### Type Hints
- Use type hints for all function parameters and return types
- Use `Optional[X]` instead of `X | None` for compatibility
- Use built-in collection types (`list`, `dict`, `tuple`) not capitalized

```python
# Good
def train_sae(
    layer_name: str,
    epochs: int = 3,
    lr: float = 1e-4,
    save_path: Optional[str] = None,
) -> SparseAutoencoder:
    ...

# Avoid
def train_sae(layer_name, epochs=3, lr=1e-4, save_path=None):
    ...
```

### Error Handling
- Use specific exception types
- Include informative error messages
- Handle exceptions at the appropriate level

```python
# Good
if not os.path.exists(meta_file):
    return None
with open(meta_file, "r", encoding="utf-8") as fp:
    meta = json.load(fp)

# Avoid bare except
```

### PyTorch Conventions
- Use `super().__init__()` for parent class initialization
- Use `torch.no_grad()` for inference
- Move tensors to device explicitly: `.to(device)`
- Use `torch.nn.Module` as base for models
- Detach before CPU transfer: `.detach().cpu()`

### File Headers (Mandatory)
Every code file must include a header describing its role in the project architecture and code flow:

```python
"""
sae/model.py

Role: Defines the SparseAutoencoder neural network module used for learning
interpretable features from GPT-2 layer activations. This is the core model
component of the SAE pipeline, providing encode/decode functionality with ReLU
activation for sparsity. Designed to work with 768-dim GPT-2 hidden states and
is trained via sae/train.py within the development container.
"""
```

---

## Project Structure

* **Architecture:** The project is built around a modular design where the core logic for model interaction and data extraction is encapsulated in the `ActivationExtractor` class within `tools.py`. The `Interface.ipynb` notebook is the recommended interface for interactive exploration and visualization; the previously included `main.py` served as a CLI example but is now replaced with the notebook-based workflow.
* **Code Flow:**
    1.  **Initialization:** `Interface.ipynb` detects the available hardware (CPU/GPU) and loads the pre-trained GPT-2 model and tokenizer (the notebook contains a self-contained startup cell). `main.py` was previously used to demonstrate the flow and is deprecated.
    2.  **Setup:** An instance of `ActivationExtractor` is created, specifying the target layers to monitor (e.g., `transformer.h.0`).
    3.  **Extraction:** The `extract` method is called with a text prompt. This registers PyTorch forward hooks on the target layers.
    4.  **Execution:** The model performs a forward pass on the tokenized input. The hooks capture the output tensors (activations) of the specified layers.
    5.  **Cleanup & Output:** Hooks are removed to prevent memory leaks. The captured activations and corresponding tokens are returned to the caller (e.g., the interactive notebook or calling script) for display or analysis.
    6.  **SAE Training (optional):** The `sae/train.py` module can train a Sparse Autoencoder on collected activations to learn interpretable features.
    7.  **SAE Analysis:** Use `sae/visualize.py` to analyze which learned features activate for specific inputs.

---

## File Structure

```
/workspaces/gpt2small
├── Interface.ipynb       # Interactive notebook used as the primary interface
├── ActivationExtractor.py  # Contains the ActivationExtractor class
├── pyproject.toml        # Project configuration and dependency definitions (using uv)
├── Dockerfile.dev        # Docker configuration for the development environment
├── Dockerfile.base       # Base Docker image configuration
├── README.md             # Project documentation
├── sae/                  # Sparse Autoencoder module for interpretability
│   ├── __init__.py       # Package exports
│   ├── model.py          # SparseAutoencoder(nn.Module) - ReLU SAE with 8x expansion
│   ├── buffer.py         # ActivationBuffer - streams activations from HuggingFace datasets
│   ├── train.py          # train_sae() - training loop with MSE + L1 loss
│   └── visualize.py      # analyze_word(), compare_reconstruction() for visualization
├── sae_checkpoints/      # Saved SAE model checkpoints
└── todo                  # Project todo list
```

* `Interface.ipynb`: The primary, interactive interface for experimentation and activation visualization.
* `ActivationExtractor.py`: Utility module defining the `ActivationExtractor` class for PyTorch hooks.
* `sae/`: Sparse Autoencoder module for training SAEs on GPT-2 activations:
  - `model.py`: Defines `SparseAutoencoder` with encode/decode for learning interpretable features.
  - `buffer.py`: Streams activations from WikiText/OpenWebText datasets, buffers on GPU.
  - `train.py`: Training loop with MSE reconstruction + L1 sparsity loss.
  - `visualize.py`: Functions to analyze SAE feature activations and reconstruction quality.
* `pyproject.toml`: Defines the Python dependencies and project metadata.

---

## Building and Running

**Prerequisites:**
*   Docker (to run the container defined in `Dockerfile.dev`)
*   Python 3.12+
*   `uv` package manager
*   PyTorch with CUDA support (optional, for GPU acceleration)
*   `transformers` library

**Build Steps (if applicable):**
1.  Ensure `uv` is installed.
2.  Install dependencies defined in `pyproject.toml`:
    ```bash
    uv sync
    ```

**Running the Application (recommended):**
1.  Run the interactive notebook locally:
    ```bash
    # Activate virtualenv and install deps, or use the container
    uv sync
    jupyter lab Interface.ipynb
    ```
2.  Run inside Docker (recommended for GPU/consistency):
    ```bash
    # Build the dev image
    docker build -f Dockerfile.dev -t gpt2small:dev .

    # Run interactively and map a port so you can open the notebook from your host
    docker run --gpus all --rm -it -p 8888:8888 \
      -v "$PWD":/workspaces/gpt2small gpt2small:dev \
      jupyter lab --ip=0.0.0.0 --no-browser --allow-root
    ```

    Then open the hosted URL on your host machine to reach the Jupyter UI. Replace `--gpus all` with `--gpus "device=0"` or omit if no GPU.

---

## Code Writing Rules 📝
Do not create new documentation files (unless explicitly requested). Only update documentation via the `README` if necessary.

### File Header (Mandatory)
In the header of every code file, you **must** describe how that file relates to the **overall project architecture** and **code flow**.

Each code file **must** include a short description (no more than 4–5 sentences) that explains the following:
- Its role in the **big picture** (as defined in the **Project Structure** section).
- Its connection to the main **code flow** of the project.
- The intended **execution environment** (where this code will run, as defined in the **Project Goal** section).
