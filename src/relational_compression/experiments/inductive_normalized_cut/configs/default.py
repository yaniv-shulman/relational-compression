"""Configuration values for a reproducible experiment variant."""

from datetime import datetime as dt
from datetime import timezone
from pathlib import Path

from relational_compression.paths import dataset_local_root_dir, get_experiments_dir

config_file: Path = Path(__file__).absolute()

task_model_name: str = "inductive_normalized_cut"
dataset_name: str = "malnet-tiny"
dataset_version: str = "1.0_relational_compression"
num_experiments: int = 1

dataset_root_dir: Path = dataset_local_root_dir(
    dataset_name=dataset_name,
    dataset_version=dataset_version,
    derived=True,
).absolute()

preprocessing_version: str = "lcc_structural_directed_relative_pagerank_hutch_rwse_v5"
min_nodes_per_graph: int = 512
max_nodes_per_graph: int | None = None
random_walk_steps: int = 16
random_walk_num_probes: int = 8
positional_encoding_seed: int = 1337
include_directed_features: bool = True
pagerank_alpha: float = 0.85
pagerank_max_iter: int = 50
pagerank_tolerance: float = 1e-6
dataset_backend: str = "malnet_tiny"

experiments_dir: Path = get_experiments_dir()

experiment_base_name: str = (
    f"{task_model_name}_{dataset_name.replace('-', '_')}_{dataset_version}"
    "_gps_efficient_attention_directed_sep010_batch80"
)

experiment_name: str | None = None
unique_postfix: str | None = str(int(dt.now(timezone.utc).timestamp()))

num_epochs: int = 250
batch_size: int = 80
num_workers: int = 0
learning_rate: float = 5e-4
min_learning_rate: float = 1e-5
learning_rate_warmup_steps: int = 50
learning_rate_warmup_start_factor: float = 0.1
weight_decay: float = 0.0
gradient_clip_norm: float | None = None

num_partitions: int = 8
model_name: str = "graphgps_efficient_attention"
hidden_dim: int = 128
num_layers: int = 4
assignment_temperature: float = 1.0
use_graph_context: bool = True
activation_name: str = "gelu"
output_head_hidden_multiplier: int = 0
gps_heads: int = 4
gps_dropout: float = 0.0
gps_ffn_multiplier: int = 2
q_z_floor: float = 1e-12
separation_weight: float = 0.10

run_spectral_baseline: bool = True
spectral_seed: int = 1337
spectral_n_init: int = 8
spectral_max_iter: int = 100
spectral_tolerance: float = 1e-5
spectral_cache_version: str = "spectral_ncut_v1"
spectral_absolute_margin: float = 0.25
spectral_relative_margin: float = 0.10

seed: int = 1337
log_to_tensorboard_global: bool = True
tensorboard_log_steps: int = 100

max_train_batches: int | None = None
max_val_batches: int | None = None
device: str = "cpu"
