"""Configuration values for a reproducible experiment variant."""

from pathlib import Path
from typing import Any

from relational_compression.experiments.teacherless_image_compression.configs.flowers102 import (
    flowers102_mean_group as baseline,
)

config_file: Path = Path(__file__).absolute()

unique_postfix: str = f"{baseline.unique_postfix}_stronger_1e-5_separation"
separation_weight: float = 1e-5


def __getattr__(name: str) -> Any:
    """Delegate unresolved configuration attributes to the base module."""
    return getattr(baseline, name)
