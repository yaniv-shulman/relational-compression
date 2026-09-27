"""Random-number-generator state helpers for resumable experiments."""

import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, Torch CPU, and available CUDA RNG state."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a state captured by :func:`capture_rng_state`."""
    required_keys = ("python", "numpy", "torch")
    missing_keys = [key for key in required_keys if key not in state]
    if missing_keys:
        missing_entries = ", ".join(missing_keys)
        raise ValueError(f"RNG state is missing required entries: {missing_entries}")

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    cuda_state = state.get("torch_cuda")
    if cuda_state is None:
        if torch.cuda.is_available():
            raise ValueError("Checkpoint does not contain CUDA RNG state for the current CUDA run")
        return

    if not torch.cuda.is_available():
        raise ValueError("Checkpoint contains CUDA RNG state but CUDA is unavailable")

    torch.cuda.set_rng_state_all(cuda_state)
