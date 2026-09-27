"""Configuration values for a reproducible experiment variant."""

from datetime import datetime as dt
from datetime import timezone
from pathlib import Path

from relational_compression.paths import dataset_local_root_dir, get_experiments_dir

config_file: Path = Path(__file__).absolute()

task_model_name: str = "teacherless_image_compression"
dataset_name: str = "Flowers102"
dataset_version: str = "1.0"
num_experiments: int = 1

dataset_root_dir: Path = dataset_local_root_dir(
    dataset_name="flowers102",
    dataset_version=dataset_version,
    derived=False,
).absolute()

experiments_dir: Path = get_experiments_dir()
experiment_base_name: str = f"{task_model_name}_{dataset_name.lower()}_{dataset_version.replace('.', '_')}"
unique_postfix: str | None = str(int(dt.now(timezone.utc).timestamp()))

num_epochs: int = 150
image_size: int = 256
batch_size: int = 64
num_workers: int = 1
learning_rate: float = 5e-4
min_learning_rate: float = 1e-6
learning_rate_warmup_steps: int = 100
learning_rate_warmup_start_factor: float = 0.1
weight_decay: float = 0.0

bits: int = 16
hidden: int = 128
downsample_layers: int = 3
residual_layers: int = 2
residual_hidden: int = 64

grouping: str = "per_channel"
separation_temperature: float = 1.0
separation_margin_gain: float = 3.0
threshold_scale_ema_decay: float = 0.99
threshold_scale_min: float = 1e-3

download: bool = True
modes: tuple[str, ...] = ("deterministic", "stochastic")
seed: int = 1337
diagnostic_pair_samples: int = 8192

log_to_tensorboard_global: bool = True
tensorboard_log_steps: int = 20

max_train_batches: int | None = None
max_val_batches: int | None = None
