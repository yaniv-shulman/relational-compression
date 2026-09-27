"""Experiment support code for relational compression studies."""

from dataclasses import dataclass
from pathlib import Path

from relational_compression.paths import get_data_dir, get_out_dir


@dataclass(frozen=True)
class ExperimentConfig:
    """Configure reconstruction-trained image experiments."""

    dataset_root: Path = get_data_dir().joinpath("raw").absolute()
    output_root: Path = get_out_dir().joinpath("teacherless_image_compression")
    image_size: int = 128
    validation_fraction: float = 0.1
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.0

    bits: int = 16
    hidden: int = 128
    downsample_layers: int = 3
    residual_layers: int = 2
    residual_hidden: int = 64

    grouping: str = "per_channel"
    concentration_weight: float = 1.0

    seed: int = 1337
    diagnostic_pair_samples: int = 8192
    stochastic_eval_samples: int = 4


config = ExperimentConfig()
