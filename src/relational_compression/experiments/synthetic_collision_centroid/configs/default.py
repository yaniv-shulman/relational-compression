"""Configuration values for a reproducible experiment variant."""

from datetime import datetime as dt
from datetime import timezone
from pathlib import Path

from relational_compression.paths import get_experiments_dir

config_file: Path = Path(__file__).absolute()

task_model_name: str = "synthetic_collision_centroid"
dataset_name: str = "imbalanced_gaussian_mixture"
dataset_version: str = "1.0"
num_experiments: int = 1

experiments_dir: Path = get_experiments_dir()
experiment_base_name: str = f"{task_model_name}_{dataset_name}_{dataset_version.replace('.', '_')}"
experiment_name: str | None = None
unique_postfix: str | None = str(int(dt.now(timezone.utc).timestamp()))

num_points: int = 768
component_weights: tuple[float, ...] = (0.30, 0.22, 0.17, 0.13, 0.10, 0.08)
component_means: tuple[tuple[float, float], ...] = (
    (-3.5, -2.0),
    (-1.0, 2.8),
    (2.8, 2.4),
    (3.8, -1.5),
    (0.8, -3.5),
    (-4.0, 2.5),
)
component_covariances: tuple[tuple[tuple[float, float], tuple[float, float]], ...] = (
    ((0.45, 0.15), (0.15, 0.80)),
    ((0.80, -0.25), (-0.25, 0.45)),
    ((0.55, 0.10), (0.10, 0.55)),
    ((0.75, 0.25), (0.25, 0.45)),
    ((0.90, -0.20), (-0.20, 0.50)),
    ((0.35, 0.05), (0.05, 0.75)),
)

bits: int = 3
hidden_dim: int = 32
code_temperature: float = 0.7
num_steps: int = 300
learning_rate: float = 0.02
momentum: float = 0.9

random_state_samples: int = 128
random_state_scale_min: float = 0.25
random_state_scale_max: float = 4.0

decoder_steps: int = 200
decoder_learning_rate: float = 0.15
decoder_initial_scale: float = 3.0

seed: int = 1337
log_to_tensorboard_global: bool = True
tensorboard_log_steps: int = 5
figure_dpi: int = 180
