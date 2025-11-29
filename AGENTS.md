## Project Goal

* **Description:** The primary objective of this project is to provide a tool for extracting and visualizing the internal activations of specific layers within the GPT-2 Small language model. The project is designed to run inside a Docker container (see `Dockerfile.dev`) — the recommended interface is the interactive Jupyter Notebook (`Interface.ipynb`). Legacy CLI examples such as `main.py` and earlier demo scripts have been deprecated or removed in favor of the notebook-driven workflow.

---

## Project Structure

* **Architecture:** The project is built around a modular design where the core logic for model interaction and data extraction is encapsulated in the `ActivationExtractor` class within `tools.py`. The `Interface.ipynb` notebook is the recommended interface for interactive exploration and visualization; the previously included `main.py` served as a CLI example but is now replaced with the notebook-based workflow.
* **Code Flow:**
    1.  **Initialization:** `Interface.ipynb` detects the available hardware (CPU/GPU) and loads the pre-trained GPT-2 model and tokenizer (the notebook contains a self-contained startup cell). `main.py` was previously used to demonstrate the flow and is deprecated.
    2.  **Setup:** An instance of `ActivationExtractor` is created, specifying the target layers to monitor (e.g., `transformer.h.0`).
    3.  **Extraction:** The `extract` method is called with a text prompt. This registers PyTorch forward hooks on the target layers.
    4.  **Execution:** The model performs a forward pass on the tokenized input. The hooks capture the output tensors (activations) of the specified layers.
    5.  **Cleanup & Output:** Hooks are removed to prevent memory leaks. The captured activations and corresponding tokens are returned to the caller (e.g., the interactive notebook or calling script) for display or analysis.

---

## File Structure

```
/workspaces/gpt2small
├── Interface.ipynb   # Interactive notebook used as the primary interface for activation extraction
├── tools.py          # Contains the ActivationExtractor class and utility logic
├── pyproject.toml    # Project configuration and dependency definitions (using uv)
├── Dockerfile.dev    # Docker configuration for the development environment
├── Dockerfile.base   # Base Docker image configuration
├── README.md         # Project documentation
└── todo              # Project todo list
```

* `Interface.ipynb`: The primary, interactive interface that orchestrates model initialization and visualization; intended for experimentation and activation visualization. Legacy demo scripts (e.g., `demo_activations.py`) have been removed — please use the notebook or `tools.py` for programmatic access.
* `tools.py`: A utility module defining the `ActivationExtractor` class, which handles the low-level PyTorch hooks.
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
**This section must be added to the `AGENTS.md` file exactly as written below:**

### File Header (Mandatory)
In the header of every code file, you **must** describe how that file relates to the **overall project architecture** and **code flow**.

Each code file **must** include a short description (no more than 4–5 sentences) that explains the following:
- Its role in the **big picture** (as defined in the **Project Structure** section).
- Its connection to the main **code flow** of the project.
- The intended **execution environment** (where this code will run, as defined in the **Project Goal** section).
