import random

import numpy as np
import pytest
import torch

from relational_compression.randomness import capture_rng_state, restore_rng_state


def test_rng_state_round_trip_restores_python_numpy_and_torch_cpu() -> None:
    original_state = capture_rng_state()
    try:
        random.seed(17)
        np.random.seed(23)
        torch.manual_seed(29)
        saved_state = capture_rng_state()
        expected_python = random.random()
        expected_numpy = np.random.random(3)
        expected_torch = torch.rand(3)

        random.random()
        np.random.random(3)
        torch.rand(3)
        restore_rng_state(state=saved_state)

        assert random.random() == expected_python
        assert np.array_equal(np.random.random(3), expected_numpy)
        assert torch.equal(torch.rand(3), expected_torch)
    finally:
        restore_rng_state(state=original_state)


def test_rng_state_round_trip_restores_cuda_state_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    expected_cuda_state = [torch.tensor([3, 1, 4], dtype=torch.uint8)]
    restored_cuda_state: list[torch.Tensor] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: expected_cuda_state)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", restored_cuda_state.extend)

    state = capture_rng_state()
    restore_rng_state(state=state)

    assert len(restored_cuda_state) == 1
    assert torch.equal(restored_cuda_state[0], expected_cuda_state[0])


def test_restore_rng_state_rejects_incomplete_checkpoint_state() -> None:
    with pytest.raises(ValueError, match="missing required entries"):
        restore_rng_state(state={})
