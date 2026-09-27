from pathlib import Path

from relational_compression.paths import (
    dataset_local_root_dir,
    get_checkpoints_dir,
    get_data_dir,
    get_experiment_name,
)


def test_dataset_local_root_dir_preserves_existing_datasets_root() -> None:
    root = dataset_local_root_dir(
        dataset_name="flowers102",
        dataset_version="1.0",
        derived=False,
        data_dir=Path("/mnt/data/datasets"),
    )

    assert root == Path("/mnt/data/datasets/raw/flowers102/1.0")


def test_dataset_local_root_dir_adds_datasets_component_when_needed() -> None:
    root = dataset_local_root_dir(
        dataset_name="flowers102",
        dataset_version=None,
        derived=True,
        data_dir=Path("/mnt/data"),
    )

    assert root == Path("/mnt/data/datasets/derived/flowers102")


def test_get_checkpoints_dir_uses_environment(monkeypatch) -> None:
    monkeypatch.setenv("RELCO_CHECKPOINT_DIR", "/mnt/checkpoints")
    assert get_checkpoints_dir() == Path("/mnt/checkpoints")


def test_get_data_dir_uses_environment(monkeypatch) -> None:
    monkeypatch.setenv("RELCO_DATA_DIR", "/mnt/data")
    assert get_data_dir() == Path("/mnt/data")


def test_get_experiment_name_uses_explicit_postfix() -> None:
    assert get_experiment_name("flowers", "smoke") == "flowers_smoke"
