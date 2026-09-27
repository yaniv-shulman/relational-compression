"""Local filesystem paths and naming helpers for experiment artifacts."""

import os
from datetime import datetime as dt
from datetime import timezone
from pathlib import Path

dt_now = dt.now


def get_out_dir() -> Path:
    """Return the absolute directory used for generated artifacts."""
    return Path(os.environ["RELCO_OUT_DIR"]).absolute()


def get_data_dir() -> Path:
    """Return the absolute directory containing local datasets."""
    return Path(os.environ["RELCO_DATA_DIR"]).absolute()


def get_checkpoints_dir() -> Path:
    """Return the absolute directory used for reusable checkpoints."""
    return Path(os.environ["RELCO_CHECKPOINT_DIR"]).absolute()


def get_experiments_dir(out_dir: Path | None = None) -> Path:
    """Return the experiment-output directory below the configured output root.

    Args:
        out_dir: Optional output root that overrides the environment setting.

    Returns:
        Absolute experiment-output directory.

    """
    if out_dir is None:
        out_dir = get_out_dir()

    return out_dir.joinpath("experiments").absolute()


def get_experiment_name(experiment_base_name: str, unique_postfix: str | None = None) -> str:
    """Construct an experiment name from a base name and optional postfix.

    Args:
        experiment_base_name: Stable name describing the experiment family.
        unique_postfix: Optional stable suffix; the current UTC timestamp is used when absent.

    Returns:
        Experiment directory name.

    """
    postfix = unique_postfix if unique_postfix is not None else str(int(dt_now(timezone.utc).timestamp()))
    separator = "_" if len(postfix) > 0 else ""
    return f"{experiment_base_name}{separator}{postfix}"


def get_experiment_dir(experiment_name: str, experiments_dir: Path | None = None) -> Path:
    """Return the absolute directory for one named experiment.

    Args:
        experiment_name: Name of the experiment run.
        experiments_dir: Optional root that overrides the configured experiment directory.

    Returns:
        Absolute experiment directory.

    """
    if experiments_dir is None:
        experiments_dir = get_experiments_dir()

    return experiments_dir.joinpath(experiment_name).absolute()


def dataset_local_root_dir(
    dataset_name: str,
    dataset_version: str | None,
    derived: bool,
    data_dir: Path | None = None,
) -> Path:
    """Return the local dataset directory for a raw or derived dataset version.

    Args:
        dataset_name: Dataset identifier.
        dataset_version: Optional version component.
        derived: Whether to select derived rather than raw data.
        data_dir: Optional data root that overrides the environment setting.

    Returns:
        Absolute dataset directory.

    """
    if data_dir is None:
        data_dir = get_data_dir()

    datasets_dir = data_dir if data_dir.name == "datasets" else data_dir.joinpath("datasets")
    split_dir = "derived" if derived else "raw"
    dataset_dir = datasets_dir.joinpath(split_dir, dataset_name)

    if dataset_version is not None:
        dataset_dir = dataset_dir.joinpath(dataset_version)

    return dataset_dir.absolute()
