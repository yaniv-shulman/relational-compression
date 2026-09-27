"""Prepare the cached MalNet-Tiny data used by inductive normalized-cut runs."""

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from relational_compression.experiments.inductive_normalized_cut.data import PreprocessConfig, prepare_malnet_tiny_cache


def _get_config() -> Any:
    """Load the selected preprocessing configuration and CLI overrides."""
    parser = argparse.ArgumentParser(description="Prepare the Relational Compression MalNet-Tiny normalized-cut cache")
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Config module below relational_compression.experiments.inductive_normalized_cut.configs.",
    )
    parser.add_argument("--dataset-root-dir", type=Path, default=None)
    parser.add_argument("--min-nodes-per-graph", type=int, default=None)
    parser.add_argument("--max-nodes-per-graph", type=int, default=None)
    parser.add_argument("--random-walk-steps", type=int, default=None)
    parser.add_argument("--random-walk-num-probes", type=int, default=None)
    parser.add_argument("--positional-encoding-seed", type=int, default=None)
    parser.add_argument("--include-directed-features", action="store_true")
    parser.add_argument("--pagerank-alpha", type=float, default=None)
    parser.add_argument("--pagerank-max-iter", type=int, default=None)
    parser.add_argument("--pagerank-tolerance", type=float, default=None)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    cfg: Any = importlib.import_module(
        f"relational_compression.experiments.inductive_normalized_cut.configs.{args.config}"
    )
    if args.dataset_root_dir is not None:
        cfg.dataset_root_dir = args.dataset_root_dir.absolute()
    if args.min_nodes_per_graph is not None:
        cfg.min_nodes_per_graph = args.min_nodes_per_graph
    if args.max_nodes_per_graph is not None:
        cfg.max_nodes_per_graph = args.max_nodes_per_graph
    if args.random_walk_steps is not None:
        cfg.random_walk_steps = args.random_walk_steps
    if args.random_walk_num_probes is not None:
        cfg.random_walk_num_probes = args.random_walk_num_probes
    if args.positional_encoding_seed is not None:
        cfg.positional_encoding_seed = args.positional_encoding_seed
    if args.include_directed_features:
        cfg.include_directed_features = True
    if args.pagerank_alpha is not None:
        cfg.pagerank_alpha = args.pagerank_alpha
    if args.pagerank_max_iter is not None:
        cfg.pagerank_max_iter = args.pagerank_max_iter
    if args.pagerank_tolerance is not None:
        cfg.pagerank_tolerance = args.pagerank_tolerance
    cfg.download = not args.no_download
    return cfg


def main() -> None:
    """Create the configured reusable MalNet-Tiny cache."""
    cfg = _get_config()
    preprocess_config = PreprocessConfig(
        dataset_root_dir=Path(cfg.dataset_root_dir),
        min_nodes_per_graph=int(cfg.min_nodes_per_graph),
        max_nodes_per_graph=cfg.max_nodes_per_graph,
        random_walk_steps=int(cfg.random_walk_steps),
        random_walk_num_probes=int(cfg.random_walk_num_probes),
        positional_encoding_seed=int(cfg.positional_encoding_seed),
        preprocessing_version=str(cfg.preprocessing_version),
        include_directed_features=bool(getattr(cfg, "include_directed_features", False)),
        pagerank_alpha=float(getattr(cfg, "pagerank_alpha", 0.85)),
        pagerank_max_iter=int(getattr(cfg, "pagerank_max_iter", 50)),
        pagerank_tolerance=float(getattr(cfg, "pagerank_tolerance", 1e-6)),
    )
    if not bool(getattr(cfg, "download", True)) and not Path(cfg.dataset_root_dir).exists():
        raise FileNotFoundError(
            f"MalNet-Tiny root does not exist: {cfg.dataset_root_dir}. "
            "Rerun without --no-download to let PyG download the official dataset."
        )
    summary = prepare_malnet_tiny_cache(preprocess_config, download=bool(getattr(cfg, "download", True)))
    print(json.dumps(summary, indent=2))
    print(f"Wrote Relational Compression MalNet-Tiny cache to {preprocess_config.cache_dir}")


if __name__ == "__main__":
    main()
