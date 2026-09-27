"""Random-number-generator state helpers for resumable experiments."""

import random
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import torch


def capture_rng_state(*, include_cuda: bool) -> dict[str, Any]:
    """Capture Python, NumPy, Torch CPU, and optionally CUDA RNG state."""
    bit_generator, keys, position, has_gauss, cached_gaussian = cast(
        tuple[str, Any, int, int, float], np.random.get_state()
    )
    state = {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": bit_generator,
            "keys": torch.from_numpy(np.asarray(keys, dtype=np.uint32).copy()),
            "position": int(position),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached_gaussian),
        },
        "torch": torch.get_rng_state(),
    }
    if include_cuda:
        if not torch.cuda.is_available():
            raise ValueError("CUDA RNG state was requested but CUDA is unavailable")
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any], *, restore_cuda: bool) -> None:
    """Restore a state captured by :func:`capture_rng_state`."""
    required_keys = ("python", "numpy", "torch")
    missing_keys = [key for key in required_keys if key not in state]
    if missing_keys:
        missing_entries = ", ".join(missing_keys)
        raise ValueError(f"RNG state is missing required entries: {missing_entries}")

    random.setstate(state["python"])
    numpy_state = state["numpy"]
    if not isinstance(numpy_state, Mapping):
        raise ValueError("RNG state has an unsupported NumPy format")
    keys = numpy_state.get("keys")
    if not isinstance(keys, torch.Tensor):
        raise ValueError("RNG state is missing tensor-valued NumPy keys")
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            keys.detach().cpu().numpy().astype(np.uint32, copy=False),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch"])

    if not restore_cuda:
        return
    if not torch.cuda.is_available():
        raise ValueError("CUDA RNG state was requested but CUDA is unavailable")
    cuda_state = state.get("torch_cuda")
    if cuda_state is None:
        raise ValueError("Checkpoint does not contain CUDA RNG state for the current CUDA run")
    torch.cuda.set_rng_state_all(cuda_state)
