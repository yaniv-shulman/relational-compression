"""Configuration values for a reproducible experiment variant."""

from datetime import datetime as dt
from datetime import timezone
from pathlib import Path

from relational_compression.experiments.teacher_image_compression.models import (
    DINO_VITS8_CHECKPOINT_URL,
    DINO_VITS8_MODEL,
    DINO_VITS8_REPO,
)
from relational_compression.paths import dataset_local_root_dir, get_experiments_dir, get_out_dir

config_file: Path = Path(__file__).absolute()

task_model_name: str = "teacher_image_compression"
dataset_name: str = "Flowers102"
dataset_version: str = "1.0"
num_experiments: int = 1

dataset_root_dir: Path = dataset_local_root_dir(
    dataset_name="flowers102",
    dataset_version=dataset_version,
    derived=False,
).absolute()

experiments_dir: Path = get_experiments_dir()
experiment_base_name: str = f"{task_model_name}_{dataset_name.lower()}_{dataset_version.replace('.', '_')}_dino_vits8"
experiment_name: str | None = None
unique_postfix: str | None = str(int(dt.now(timezone.utc).timestamp()))

num_epochs: int = 150
image_size: int = 256
batch_size: int = 32
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

alignment_weight: float = 1.0
code_temperature: float = 0.25
teacher_temperature: float = 0.1

sampled_tokens: int = 512
eval_sampled_tokens: int = 1024
validation_sampling_seed: int | None = None
test_sampling_seed: int | None = None
prefer_cross_image_neighbors: bool = True

teacher_backend: str = "dino_vits8"
teacher_repo: str = DINO_VITS8_REPO
teacher_model_name: str = DINO_VITS8_MODEL
teacher_checkpoint_url: str = DINO_VITS8_CHECKPOINT_URL
teacher_cache_dir: Path = get_out_dir().joinpath("model_cache", "teacher_image_compression", "dino_vits8")
teacher_download: bool = True
fixed_random_teacher_embedding_dim: int = 64

decoder_num_epochs: int = 0
decoder_learning_rate: float = 5e-4
decoder_weight_decay: float = 0.0
decoder_reconstruction_log_epochs: int = 1

download: bool = True
seed: int = 1337
log_to_tensorboard_global: bool = True
tensorboard_log_steps: int = 20
similarity_diagnostic_epochs: int = 1
similarity_diagnostic_heatmap_size: int = 512

max_train_batches: int | None = None
max_val_batches: int | None = None
