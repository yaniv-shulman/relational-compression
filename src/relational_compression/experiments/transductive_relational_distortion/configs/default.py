"""Configuration values for a reproducible experiment variant."""

import os
from datetime import datetime as dt
from datetime import timezone
from pathlib import Path


def _default_dataset_root() -> Path:
    """Compute default dataset root."""
    if "RELCO_DATA_DIR" in os.environ:
        data_dir = Path(os.environ["RELCO_DATA_DIR"])
        datasets_dir = data_dir if data_dir.name == "datasets" else data_dir / "datasets"
        return (datasets_dir / "derived" / "malnet-tiny" / "1.0_relational_compression").absolute()
    return Path("data/datasets/derived/malnet-tiny/1.0_relational_compression").absolute()


def _default_experiments_dir() -> Path:
    """Compute default experiments dir."""
    if "RELCO_OUT_DIR" in os.environ:
        return (Path(os.environ["RELCO_OUT_DIR"]) / "experiments").absolute()
    return Path("out/experiments").absolute()


config_file: Path = Path(__file__).absolute()

task_model_name: str = "transductive_relational_distortion"
dataset_name: str = "mixed-graphs"
dataset_version: str = "1.0_relational_compression"
dataset_root_dir: Path = _default_dataset_root()
tu_dataset_root_dir: Path = (
    Path(os.environ["RELCO_TU_DATA_DIR"]).absolute()
    if "RELCO_TU_DATA_DIR" in os.environ
    else Path("data/datasets/derived/tu-datasets/1.0_relational_compression").absolute()
)

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
source_collections: tuple[str, ...] = ("malnet_unweighted", "malnet_weighted", "collab", "proteins")
malnet_split: str = "test"
graphs_per_collection: int = 20
graph_seed: int = 1337
weight_seed: int = 91337
weight_sigma: float = 0.75
tu_min_nodes_per_graph: int = 16
tu_max_nodes_per_graph: int | None = None

experiments_dir: Path = _default_experiments_dir()
experiment_base_name: str = f"{task_model_name}_{dataset_name.replace('-', '_')}_{dataset_version}_final"
experiment_name: str | None = None
unique_postfix: str | None = str(int(dt.now(timezone.utc).timestamp()))

criteria: tuple[str, ...] = ("edge", "fourier", "collision", "collision_entropy")
lambda_org_values: tuple[float, ...] = (
    0.0,
    0.03,
    0.05,
    0.07,
    0.08,
    0.09,
    0.10,
    0.11,
    0.12,
    0.14,
    0.16,
    0.18,
    0.20,
    0.30,
    0.50,
)
splits: tuple[str, ...] = ("test",)
max_graphs_per_split: int | None = None

num_partitions: int = 8
assignment_temperature: float = 1.0
learning_rate: float = 0.05
optimization_steps: int = 600
init_scale: float = 1e-2
restart_seeds: tuple[int, ...] = (1337, 2024, 31415, 2718, 1618, 9001, 42, 73, 101, 211)
include_collapse_restart: bool = True
collapse_init_bias: float = 4.0
cross_warm_start: bool = True
cross_warm_start_closure_rounds: int = 1
seed: int = 1337
device: str = "cuda"

source_geometry_cache_version: str = "trd_source_geometry_v3"
source_geometry_cache_dir: Path | None = None
source_geometry_solve_batch_size: int = 256
random_partition_count: int = 256
random_partition_seed: int = 1337

write_plots: bool = True
