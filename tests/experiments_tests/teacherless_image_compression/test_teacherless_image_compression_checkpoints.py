from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from relational_compression.randomness import capture_rng_state


def _resume_config() -> SimpleNamespace:
    return SimpleNamespace(
        dataset_root_dir=Path("/tmp/data"),
        experiments_dir=Path("/tmp/experiments"),
        image_size=128,
        batch_size=32,
        num_workers=0,
        learning_rate=2e-4,
        weight_decay=0.0,
        bits=16,
        hidden=128,
        downsample_layers=3,
        residual_layers=2,
        residual_hidden=64,
        grouping="per_channel",
        concentration_weight=1.0,
        seed=1337,
        diagnostic_pair_samples=8192,
        num_epochs=3,
        max_train_batches=None,
        max_val_batches=None,
        min_learning_rate=0.0,
        learning_rate_warmup_steps=0,
        learning_rate_warmup_start_factor=0.1,
        separation_weight=0.0,
        separation_temperature=1.0,
        separation_margin_gain=3.0,
        threshold_scale_ema_decay=0.99,
        threshold_scale_min=1e-3,
    )


def _checkpoint_config(experiment_config: SimpleNamespace) -> dict[str, object]:
    from relational_compression.experiments.teacherless_image_compression.run_scripts.run_flowers_exp import (
        _as_experiment_config,
        _checkpoint_config_dict,
    )

    model_config = _as_experiment_config(experiment_config)
    return _checkpoint_config_dict(experiment_config, model_config=model_config)


def test_checkpoint_config_supports_weights_only_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure image checkpoint configuration contains only safe serialized values."""
    monkeypatch.setenv("RELCO_DATA_DIR", "/tmp")
    monkeypatch.setenv("RELCO_OUT_DIR", "/tmp")
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(
        {"config": _checkpoint_config(_resume_config()), "rng_state": capture_rng_state(include_cuda=False)},
        checkpoint_path,
    )

    loaded = torch.load(checkpoint_path, weights_only=True)

    assert loaded["config"]["dataset_root"] == "/tmp/data"


@pytest.mark.parametrize(
    ("field", "changed_value"),
    [
        ("num_epochs", 4),
        ("max_train_batches", 3),
        ("max_val_batches", 2),
        ("min_learning_rate", 1e-5),
        ("learning_rate_warmup_steps", 10),
        ("learning_rate_warmup_start_factor", 0.5),
        ("separation_weight", 0.1),
        ("separation_temperature", 0.5),
        ("separation_margin_gain", 2.0),
        ("threshold_scale_ema_decay", 0.9),
        ("threshold_scale_min", 1e-2),
    ],
)
def test_checkpoint_config_detects_training_objective_and_scheduler_changes(
    field: str,
    changed_value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RELCO_DATA_DIR", "/tmp")
    monkeypatch.setenv("RELCO_OUT_DIR", "/tmp")
    base = _resume_config()
    changed = SimpleNamespace(**(vars(base) | {field: changed_value}))

    assert _checkpoint_config(base) != _checkpoint_config(changed)
