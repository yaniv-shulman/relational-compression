import random
from pathlib import Path

import numpy as np
import pytest
import torch

from relational_compression.randomness import capture_rng_state, restore_rng_state


def test_rng_state_round_trip_restores_python_numpy_and_torch_cpu() -> None:
    original_state = capture_rng_state(include_cuda=False)
    try:
        random.seed(17)
        np.random.seed(23)
        torch.manual_seed(29)
        saved_state = capture_rng_state(include_cuda=False)
        expected_python = random.random()
        expected_numpy = np.random.random(3)
        expected_torch = torch.rand(3)

        random.random()
        np.random.random(3)
        torch.rand(3)
        restore_rng_state(state=saved_state, restore_cuda=False)

        assert random.random() == expected_python
        assert np.array_equal(np.random.random(3), expected_numpy)
        assert torch.equal(torch.rand(3), expected_torch)
    finally:
        restore_rng_state(state=original_state, restore_cuda=False)


def test_rng_state_can_be_loaded_with_weights_only(tmp_path: Path) -> None:
    """Ensure checkpointed RNG state uses the safe Torch serialization subset."""
    checkpoint_path = tmp_path / "rng_state.pt"
    torch.save({"rng_state": capture_rng_state(include_cuda=False)}, checkpoint_path)

    loaded = torch.load(checkpoint_path, weights_only=True)

    assert isinstance(loaded["rng_state"]["numpy"]["keys"], torch.Tensor)


def test_cpu_capture_excludes_cuda_state_when_cuda_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure CPU checkpoints do not inherit ambient CUDA state."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: pytest.fail("CUDA state should not be captured"))

    state = capture_rng_state(include_cuda=False)

    assert "torch_cuda" not in state


def test_cpu_restore_ignores_cuda_state_when_cuda_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure CPU resumes ignore stale CUDA state."""
    state = capture_rng_state(include_cuda=False)
    state["torch_cuda"] = [torch.tensor([3, 1, 4], dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda _: pytest.fail("CUDA state should not be restored"))

    restore_rng_state(state=state, restore_cuda=False)


def test_rng_state_round_trip_restores_cuda_state_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure CUDA runs capture and restore CUDA state."""
    expected_cuda_state = [torch.tensor([3, 1, 4], dtype=torch.uint8)]
    restored_cuda_state: list[torch.Tensor] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: expected_cuda_state)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", restored_cuda_state.extend)

    state = capture_rng_state(include_cuda=True)
    restore_rng_state(state=state, restore_cuda=True)

    assert len(restored_cuda_state) == 1
    assert torch.equal(restored_cuda_state[0], expected_cuda_state[0])


def test_cuda_restore_requires_checkpoint_cuda_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject CUDA resumes that cannot restore CUDA randomness exactly."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    state = capture_rng_state(include_cuda=False)

    with pytest.raises(ValueError, match="does not contain CUDA RNG state"):
        restore_rng_state(state=state, restore_cuda=True)


def test_restore_rng_state_rejects_non_tensor_torch_state() -> None:
    """Reject checkpoint CPU RNG state that cannot be restored safely."""
    state = capture_rng_state(include_cuda=False)
    state["torch"] = "not-a-tensor"

    with pytest.raises(ValueError, match="tensor-valued Torch CPU state"):
        restore_rng_state(state=state, restore_cuda=False)


def test_cuda_restore_rejects_non_tensor_cuda_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject checkpoint CUDA RNG state that cannot be restored safely."""
    state = capture_rng_state(include_cuda=False)
    state["torch_cuda"] = ["not-a-tensor"]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with pytest.raises(ValueError, match="unsupported CUDA RNG state"):
        restore_rng_state(state=state, restore_cuda=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_restore_rng_state_normalizes_cuda_mapped_rng_tensors() -> None:
    """Restore RNG state tensors mapped to CUDA back through CPU APIs."""
    original_state = capture_rng_state(include_cuda=True)
    mapped_state = capture_rng_state(include_cuda=True)
    expected_cpu = torch.rand(3)
    expected_cuda = torch.rand(3, device="cuda")
    torch.rand(3)
    torch.rand(3, device="cuda")
    mapped_state["torch"] = mapped_state["torch"].cuda()
    mapped_state["torch_cuda"] = [value.cuda() for value in mapped_state["torch_cuda"]]

    try:
        restore_rng_state(state=mapped_state, restore_cuda=True)
        assert torch.equal(torch.rand(3), expected_cpu)
        assert torch.equal(torch.rand(3, device="cuda"), expected_cuda)
    finally:
        restore_rng_state(state=original_state, restore_cuda=True)


@pytest.mark.parametrize("missing_key", ("python", "numpy", "torch"))
def test_restore_rng_state_rejects_incomplete_checkpoint_state(missing_key: str) -> None:
    """Reject checkpoint RNG states missing any required CPU state."""
    state = capture_rng_state(include_cuda=False)
    del state[missing_key]

    with pytest.raises(ValueError, match="missing required entries"):
        restore_rng_state(state=state, restore_cuda=False)
