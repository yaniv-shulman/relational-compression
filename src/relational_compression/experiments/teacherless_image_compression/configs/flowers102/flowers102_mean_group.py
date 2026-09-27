"""Configuration values for a reproducible experiment variant."""

from pathlib import Path
from typing import Any

from relational_compression.experiments.teacherless_image_compression.configs.flowers102 import _flowers102_base as base

config_file: Path = Path(__file__).absolute()

if base.unique_postfix is None:
    raise ValueError("No unique postfix")

unique_postfix: str = f"{base.unique_postfix}_mean_group"
modes: tuple[str, ...] = ("deterministic",)
concentration_weight: float = 10.0
separation_weight: float = 1e-6
separation_temperature: float = 1.0


def __getattr__(name: str) -> Any:
    """Delegate unresolved configuration attributes to the base module."""
    return getattr(base, name)
