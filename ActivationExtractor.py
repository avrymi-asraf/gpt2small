"""
tools.py

Role: Implements the ActivationExtractor tool used by the interactive notebook
(`Interface.ipynb`) for extracting and visualizing internal activations of a
GPT-2 model. It registers forward hooks on target modules, collects
activations, and provides utilities to persist the extractor configuration and
model artifacts. This file is designed to be executed within the project's
development container (see Dockerfile.dev) or locally in a Python environment
with the `transformers` and `torch` libraries installed.
"""

import json
import os
import torch
from typing import List, Dict, Any, Optional
from transformers import PreTrainedModel, PreTrainedTokenizer
import matplotlib.pyplot as plt
import matplotlib


class ActivationExtractor:
    """
    A tool to extract activations from specific layers of a model.

        Notes:
            - Hooks are persisted across `extract` calls by default; hooks are not
              removed after an extraction and will be reused on subsequent
              calls. Use `register_hooks` and `clear_hooks` to control hook
              lifetimes manually.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        layer_names: List[str],
    ):
        self.model = model
        # Ensure only single model is passed
        if isinstance(self.model, (list, tuple)):
            raise ValueError(
                "ActivationExtractor expects a single `PreTrainedModel`, not a list or tuple."
            )
        self.tokenizer = tokenizer
        self.layer_names = layer_names
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks: List[Any] = []
        # Whether hooks are registered and persisted until `clear_hooks` is called
        self.hooks_registered: bool = False
        # Hooks are registered once and remain active until `clear_hooks`
        # is called. Backwards compatibility where hooks were removed at the
        # end of extraction is not supported.
        self.persist_hooks: bool = True
        # Register hooks immediately and persist them — they will remain
        # registered until `clear_hooks()` is explicitly called.
        self.register_hooks()

    def _get_module_by_name(self, module_name: str):
        """Retrieve a module from the model by its name."""
        for name, module in self.model.named_modules():
            if name == module_name:
                return module
        raise ValueError(
            f"Module '{module_name}' not found in model. Available modules: {list(dict(self.model.named_modules()).keys())}"
        )

    def _hook_fn_factory(self, layer_name: str):
        """Create a hook function for a specific layer."""

        def hook_fn(module, input, output):
            # Handle HuggingFace output tuples (usually (hidden_states, ...))
            if isinstance(output, tuple):
                data = output[0]
            else:
                data = output

            # Store on CPU to save GPU memory
            self.activations[layer_name] = data.detach().cpu()

        return hook_fn

    def extract(self, text: str) -> Dict[str, Any]:
        """
        Run the model on the input text and extract activations for the specified layers.

        Args:
            text: The input text string.

        Returns:
            A dictionary containing:
            - 'tokens': List of tokens corresponding to the input.
            - 'activations': Dictionary mapping layer_name to the activation tensor.
        """
        self.activations = {}

        # Register hooks once and keep them until clear_hooks() is explicitly
        # called by the user.
        if not self.hooks_registered:
            for name in self.layer_names:
                module = self._get_module_by_name(name)
                hook = module.register_forward_hook(self._hook_fn_factory(name))
                self.hooks.append(hook)
            self.hooks_registered = True
        # Prepare input
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"][0]

        # Get tokens for reference
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        # Run forward pass (hooks are persistent)
        with torch.no_grad():
            self.model(**inputs)

        return {"tokens": tokens, "activations": self.activations}

        # Hooks are persisted by design. To remove hooks, the user must call
        # `extractor.clear_hooks()` explicitly.

    def generate(
        self,
        text: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate text continuation using the model.

        This method performs autoregressive text generation by iteratively
        predicting and appending new tokens to the input sequence.

        Args:
            text: The input prompt text to continue from.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature. Higher values (e.g., 1.5) make output
                more random, lower values (e.g., 0.7) make it more deterministic.
            top_k: If set, only sample from the top k most likely tokens.
            top_p: If set, use nucleus sampling - sample from smallest set of tokens
                whose cumulative probability exceeds top_p.
            do_sample: If True, use sampling; if False, use greedy decoding (argmax).
            eos_token_id: Token ID that signals end of generation. If None, uses
                tokenizer's eos_token_id.

        Returns:
            A dictionary containing:
            - 'prompt': The original input text.
            - 'generated_text': The full text (prompt + generated continuation).
            - 'generated_tokens': List of newly generated tokens.
            - 'full_tokens': List of all tokens (input + generated).
            - 'activations': List of dicts, where each dict maps layer_name to
                activation tensor for that generation step.
        """
        # Prepare input
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_ids = inputs["input_ids"]

        # Store original length
        original_length = input_ids.shape[1]

        # Set EOS token
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        # Track activations
        activation_history = []

        # Generation loop
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Clear activations for this step
                self.activations = {}

                # Forward pass
                outputs = self.model(input_ids)
                next_token_logits = outputs.logits[:, -1, :]

                # Store activations for this step
                # Make a copy of activations before they get overwritten
                step_activations = {
                    layer: act.clone() for layer, act in self.activations.items()
                }
                activation_history.append(step_activations)

                # Apply temperature
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = (
                        next_token_logits
                        < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    )
                    next_token_logits[indices_to_remove] = float("-inf")

                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep the first token above threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float("-inf")

                # Sample or select next token
                if do_sample:
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                # Check for EOS token
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break

        # Decode results
        full_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        generated_ids = input_ids[0, original_length:]
        generated_tokens = self.tokenizer.convert_ids_to_tokens(generated_ids)
        full_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        result = {
            "prompt": text,
            "generated_text": full_text,
            "generated_tokens": generated_tokens,
            "full_tokens": full_tokens,
            "activations": activation_history,
        }

        return result

    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.hooks_registered = False

    def register_hooks(self, force: bool = False):
        """Register forward hooks for the configured layer_names.

        Args:
            force: If True, existing hooks will be removed and re-registered.
        """
        if self.hooks_registered and not force:
            return

        if force:
            self.clear_hooks()

        for name in self.layer_names:
            module = self._get_module_by_name(name)
            hook = module.register_forward_hook(self._hook_fn_factory(name))
            self.hooks.append(hook)

        self.hooks_registered = True

    def save_model_with_hooks(self, path: str, *, save_tokenizer: bool = False) -> None:
        """Save the model state and metadata about hooks.

        The hooks themselves are bound to module objects and typically are not
        picklable — instead this writes the model state (state_dict) and a small
        JSON metadata file listing `layer_names` and hook persistence setting. If
        `save_tokenizer` is True, the tokenizer will be saved to the same
        directory using `tokenizer.save_pretrained`.

        Args:
            path: Directory or file path where to save the model state. If a file
                path is provided it will create the parent directory.
            save_tokenizer: Whether to save the tokenizer alongside the model.
        """
        # Create directory
        root = path
        if os.path.splitext(path)[1] != "":
            # path has an extension; store in parent dir
            root = os.path.dirname(path)

        os.makedirs(root, exist_ok=True)

        # Save state dict for portability
        model_file = os.path.join(root, "model_state.pt")
        torch.save(self.model.state_dict(), model_file)

        # Save metadata about hooks -> layer selection and persistence setting
        meta = {
            "layer_names": self.layer_names,
            "persist_hooks": self.persist_hooks,
        }
        with open(os.path.join(root, "hooks_meta.json"), "w", encoding="utf-8") as fp:
            json.dump(meta, fp, indent=2)

        # Save tokenizer if requested
        if save_tokenizer:
            try:
                self.tokenizer.save_pretrained(root)
            except Exception:
                # Avoid failing if the tokenizer cannot be serialized
                pass

    def load_hooks_metadata(self, path: str) -> Optional[Dict[str, Any]]:
        """Load hook metadata written by `save_model_with_hooks`.

        Returns the metadata dict (or None when no metadata file exists).
        """
        meta_file = os.path.join(path, "hooks_meta.json")
        if not os.path.exists(meta_file):
            return None

        with open(meta_file, "r", encoding="utf-8") as fp:
            meta = json.load(fp)

        # Update extractor using loaded metadata
        self.layer_names = meta.get("layer_names", self.layer_names)
        self.persist_hooks = meta.get("persist_hooks", self.persist_hooks)
        # Re-register hooks for the (possibly) new layer list
        self.register_hooks(force=True)
        return meta
