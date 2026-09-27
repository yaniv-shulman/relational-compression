"""Command-line utilities for reproducible experiment workflows."""

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
from scipy.stats import rankdata
from torch import Tensor
from torch_geometric.data import Data

from relational_compression.experiments.graph_geometry import (
    EdgeResistanceData,
    _laplacian_from_edges,
    _undirected_weighted_edges,
    compute_edge_effective_resistances,
    fourier_dirichlet_distortion,
)
from relational_compression.experiments.inductive_normalized_cut.baselines import (
    SpectralNormalizedCutConfig,
    evaluate_spectral_normalized_cut,
)
from relational_compression.experiments.inductive_normalized_cut.data import PreprocessConfig, load_cached_splits
from relational_compression.experiments.inductive_normalized_cut.metrics import hard_normalized_cut, soft_normalized_cut
from relational_compression.experiments.inductive_normalized_cut.models import make_model

__all__ = [
    "EdgeResistanceData",
    "_laplacian_from_edges",
    "_undirected_weighted_edges",
    "compute_edge_effective_resistances",
    "fourier_dirichlet_distortion",
]

DEFAULT_RUN_ROOT = Path("out/reported_runs")
DEFAULT_SWEEP_ID = "phase2_uniformity_seed_sweep_1788228910"
_RUN_PATTERN = re.compile(
    rf"inductive_normalized_cut_malnet_tiny_1\.0_relational_compression_{DEFAULT_SWEEP_ID}_uni(\d+)_seed(\d+)$"
)
_RESISTANCE_CACHE_PATTERN = re.compile(r".*_n(\d+)_e\d+\.npz$")


@dataclass(frozen=True)
class RunSpec:
    """Store Run Spec values."""

    run_name: str
    run_path: Path
    separation_weight: float
    seed: int


def _repo_root() -> Path:
    """Compute repo root."""
    return Path(__file__).resolve().parents[5]


def _load_json(path: Path) -> dict[str, Any]:
    """Load json."""
    return cast(dict[str, Any], json.loads(path.read_text()))


def _load_run_config(run_path: Path) -> SimpleNamespace:
    """Load run config."""
    config = _load_json(run_path / "effective_config.json")
    if "separation_weight" not in config and "uniformity_weight" in config:
        config["separation_weight"] = config["uniformity_weight"]
    return SimpleNamespace(**config)


def discover_reported_runs(run_root: Path) -> list[RunSpec]:
    """Discover reported runs."""
    runs: list[RunSpec] = []
    for path in sorted(run_root.iterdir()):
        if not path.is_dir():
            continue
        match = _RUN_PATTERN.match(path.name)
        if match is None:
            continue
        suffix, seed = match.groups()
        runs.append(
            RunSpec(
                run_name=path.name,
                run_path=path,
                separation_weight=float(int(suffix)) / 100.0,
                seed=int(seed),
            )
        )
    if not runs:
        raise FileNotFoundError(f"No reported MalNet sweep runs found under {run_root}")
    return runs


def _preprocess_config(config: Any) -> PreprocessConfig:
    """Compute preprocess config."""
    return PreprocessConfig(
        dataset_root_dir=Path(config.dataset_root_dir),
        min_nodes_per_graph=int(config.min_nodes_per_graph),
        max_nodes_per_graph=config.max_nodes_per_graph,
        random_walk_steps=int(config.random_walk_steps),
        random_walk_num_probes=int(config.random_walk_num_probes),
        positional_encoding_seed=int(config.positional_encoding_seed),
        preprocessing_version=str(config.preprocessing_version),
        include_directed_features=bool(getattr(config, "include_directed_features", False)),
        pagerank_alpha=float(getattr(config, "pagerank_alpha", 0.85)),
        pagerank_max_iter=int(getattr(config, "pagerank_max_iter", 50)),
        pagerank_tolerance=float(getattr(config, "pagerank_tolerance", 1e-6)),
    )


def _spectral_config(config: Any) -> SpectralNormalizedCutConfig:
    """Compute spectral config."""
    return SpectralNormalizedCutConfig(
        num_partitions=int(config.num_partitions),
        seed=int(config.spectral_seed),
        n_init=int(config.spectral_n_init),
        max_iter=int(config.spectral_max_iter),
        tolerance=float(config.spectral_tolerance),
        cache_version=str(config.spectral_cache_version),
    )


def _load_best_model(run: RunSpec, config: Any, input_dim: int, device: torch.device) -> torch.nn.Module:
    """Load best model."""
    model = make_model(config, input_dim=input_dim).to(device)
    checkpoint_path = run.run_path / "checkpoints" / "run_00_categorical" / "best_model_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _edge_resistance_cache_key(data: Data) -> str:
    """Compute edge resistance cache key."""
    split = str(getattr(data, "original_split", "unknown"))
    index = int(getattr(data, "original_graph_index", -1))
    nodes = int(data.num_nodes)
    edges = int(data.edge_index.shape[1] // 2)
    return f"{split}_{index:06d}_n{nodes}_e{edges}.npz"


def load_or_compute_edge_resistances(
    data: Data,
    *,
    cache_dir: Path,
    solve_batch_size: int,
) -> EdgeResistanceData:
    """Load or compute edge resistances."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / _edge_resistance_cache_key(data)
    if cache_path.exists():
        loaded = np.load(cache_path)
        return EdgeResistanceData(
            edges=loaded["edges"],
            weights=loaded["weights"],
            resistances=loaded["resistances"],
            foster_sum=float(loaded["foster_sum"]),
        )
    edge_resistance = compute_edge_effective_resistances(data, solve_batch_size=solve_batch_size)
    np.savez_compressed(
        cache_path,
        edges=edge_resistance.edges,
        weights=edge_resistance.weights,
        resistances=edge_resistance.resistances,
        foster_sum=np.asarray(edge_resistance.foster_sum, dtype=np.float64),
    )
    return edge_resistance


def _hard_code_complexity(assignments: Tensor, data: Data, *, num_partitions: int) -> dict[str, float]:
    """Compute hard code complexity."""
    weights = torch.ones(data.edge_index.shape[1], dtype=torch.float32)
    degree = torch.zeros(int(data.num_nodes), dtype=torch.float32)
    degree.index_add_(0, data.edge_index[0].cpu(), weights)
    volume = degree.sum().clamp_min(torch.finfo(degree.dtype).tiny)
    volume_by_partition = torch.zeros(int(num_partitions), dtype=torch.float32)
    volume_by_partition.index_add_(0, assignments.cpu().long(), degree)
    q_z = volume_by_partition / volume
    collision = q_z.square().sum().clamp_min(torch.finfo(q_z.dtype).tiny)
    return {
        "hard_h2": float(-collision.log()),
        "hard_k_eff": float(collision.reciprocal()),
    }


def _data_metadata(data: Data, dataset_index: int) -> dict[str, int | str]:
    """Compute data metadata."""
    return {
        "dataset_index": int(dataset_index),
        "original_graph_index": int(getattr(data, "original_graph_index", dataset_index)),
        "num_nodes": int(data.num_nodes),
        "num_undirected_edges": int(data.edge_index.shape[1] // 2),
    }


def _sanitize(value: float) -> float:
    """Compute sanitize."""
    return float(value) if math.isfinite(float(value)) else float("nan")


def _append_graph_row(
    rows: list[dict[str, Any]],
    *,
    split: str,
    partition_source: str,
    run: RunSpec | None,
    data: Data,
    dataset_index: int,
    assignments: Tensor,
    d_f: float,
    hard_metrics: Any,
    soft_metrics: Any | None = None,
) -> None:
    """Append graph row."""
    num_partitions = int(hard_metrics.volume_fractions.shape[-1])
    row: dict[str, Any] = {
        "split": split,
        "partition_source": partition_source,
        "run_name": "" if run is None else run.run_name,
        "run_path": "" if run is None else str(run.run_path),
        "separation_weight": "" if run is None else run.separation_weight,
        "seed": "" if run is None else run.seed,
        **_data_metadata(data=data, dataset_index=dataset_index),
        "d_f": _sanitize(d_f),
        "hard_ncut": float(hard_metrics.ncut_per_graph[0]),
        "hard_nassoc": float(hard_metrics.nassoc_per_graph[0]),
        "hard_active_partitions": float(hard_metrics.active_partitions_per_graph[0]),
        "hard_max_volume_fraction": float(hard_metrics.max_partition_volume_fraction_per_graph[0]),
        "hard_min_volume_fraction": float(hard_metrics.min_partition_volume_fraction_per_graph[0]),
        "hard_within_edge_fraction": float(hard_metrics.within_edge_fraction_per_graph[0]),
        **_hard_code_complexity(assignments=assignments, data=data, num_partitions=num_partitions),
    }
    if soft_metrics is not None:
        soft_collision = soft_metrics.q_z.square().sum(dim=-1).clamp_min(torch.finfo(soft_metrics.q_z.dtype).tiny)
        row.update(
            {
                "soft_ncut": float(soft_metrics.ncut_per_graph[0]),
                "soft_h2": float(-soft_collision.log()[0]),
                "soft_k_eff": float(soft_metrics.effective_partitions_per_graph[0]),
                "soft_separation_d2": float(soft_metrics.separation_d2_per_graph[0]),
            }
        )
    else:
        row.update({"soft_ncut": "", "soft_h2": "", "soft_k_eff": "", "soft_separation_d2": ""})
    rows.append(row)


def _summarize(values: list[float]) -> dict[str, float]:
    """Summarize the requested values."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    ddof = 1 if array.size > 1 else 0
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=ddof)),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _pearson(x_values: list[float], y_values: list[float]) -> float:
    """Compute pearson."""
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if x.size < 2 or y.size < 2 or np.isclose(x.std(), 0.0) or np.isclose(y.std(), 0.0):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x_values: list[float], y_values: list[float]) -> float:
    """Compute spearman."""
    return _pearson(
        x_values=rankdata(np.asarray(x_values, dtype=np.float64)).tolist(), y_values=rankdata(y_values).tolist()
    )


def _metric_summary(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    """Compute metric summary."""
    return _summarize([float(row[metric]) for row in rows if row[metric] != ""])


def _correlation_summary(rows: list[dict[str, Any]], x_metric: str, y_metric: str) -> dict[str, float | int]:
    """Compute correlation summary."""
    paired = [
        (float(row[x_metric]), float(row[y_metric])) for row in rows if row[x_metric] != "" and row[y_metric] != ""
    ]
    if not paired:
        return {"count": 0, "pearson": float("nan"), "spearman": float("nan")}
    x_values = [x for x, _ in paired]
    y_values = [y for _, y in paired]
    return {
        "count": len(paired),
        "pearson": _pearson(x_values=x_values, y_values=y_values),
        "spearman": _spearman(x_values=x_values, y_values=y_values),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write csv."""
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _read_best_epoch(run: RunSpec) -> int | None:
    """Read best epoch."""
    result_path = run.run_path / "all_run_results.json"
    if not result_path.exists():
        return None
    result = json.loads(result_path.read_text())[0]
    best_epoch = result.get("best_epoch")
    return None if best_epoch is None else int(best_epoch)


def _evaluate_spectral_split(
    *,
    split: str,
    dataset: Any,
    spectral_config: SpectralNormalizedCutConfig,
    spectral_cache_dir: Path,
    resistance_cache_dir: Path,
    solve_batch_size: int,
    max_graphs: int | None,
) -> list[dict[str, Any]]:
    """Evaluate spectral split."""
    rows: list[dict[str, Any]] = []
    graph_count = len(dataset) if max_graphs is None else min(len(dataset), int(max_graphs))
    for dataset_index in range(graph_count):
        data = dataset[dataset_index]
        edge_resistance = load_or_compute_edge_resistances(
            data,
            cache_dir=resistance_cache_dir / split,
            solve_batch_size=solve_batch_size,
        )
        spectral = evaluate_spectral_normalized_cut(data, cache_dir=spectral_cache_dir, config=spectral_config)
        assignments = spectral["labels"].detach().cpu()
        hard_metrics = hard_normalized_cut(
            assignments=assignments, edge_index=data.edge_index, num_partitions=int(spectral_config.num_partitions)
        )
        d_f = fourier_dirichlet_distortion(
            assignments=assignments, edge_resistance=edge_resistance, num_nodes=int(data.num_nodes)
        )
        _append_graph_row(
            rows,
            split=split,
            partition_source="spectral",
            run=None,
            data=data,
            dataset_index=dataset_index,
            assignments=assignments,
            d_f=d_f,
            hard_metrics=hard_metrics,
        )
    return rows


def _evaluate_learned_run_split(
    *,
    split: str,
    dataset: Any,
    run: RunSpec,
    config: Any,
    device: torch.device,
    resistance_cache_dir: Path,
    solve_batch_size: int,
    max_graphs: int | None,
) -> list[dict[str, Any]]:
    """Evaluate learned run split."""
    model = _load_best_model(run=run, config=config, input_dim=int(dataset[0].x.shape[-1]), device=device)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        graph_count = len(dataset) if max_graphs is None else min(len(dataset), int(max_graphs))
        for dataset_index in range(graph_count):
            data = dataset[dataset_index]
            edge_resistance = load_or_compute_edge_resistances(
                data,
                cache_dir=resistance_cache_dir / split,
                solve_batch_size=solve_batch_size,
            )
            model_data = data.to(device)
            output = model(model_data)
            hard_metrics = hard_normalized_cut(
                assignments=output.hard_ids,
                edge_index=model_data.edge_index,
                num_partitions=int(config.num_partitions),
            )
            soft_metrics = soft_normalized_cut(
                probabilities=output.probabilities,
                edge_index=model_data.edge_index,
                q_z_floor=float(config.q_z_floor),
                separation_weight=0.0,
            )
            assignments = output.hard_ids.detach().cpu()
            d_f = fourier_dirichlet_distortion(
                assignments=assignments, edge_resistance=edge_resistance, num_nodes=int(data.num_nodes)
            )
            _append_graph_row(
                rows,
                split=split,
                partition_source="learned",
                run=run,
                data=data,
                dataset_index=dataset_index,
                assignments=assignments,
                d_f=d_f,
                hard_metrics=hard_metrics,
                soft_metrics=soft_metrics,
            )
    return rows


def _aggregate_rows(rows: list[dict[str, Any]], runs: list[RunSpec]) -> dict[str, Any]:
    """Aggregate rows."""
    per_run: list[dict[str, Any]] = []
    for row_split in sorted({str(row["split"]) for row in rows}):
        spectral_rows = [row for row in rows if row["split"] == row_split and row["partition_source"] == "spectral"]
        if spectral_rows:
            per_run.append(
                {
                    "partition_source": "spectral",
                    "split": row_split,
                    "run_name": "spectral_reference",
                    "separation_weight": None,
                    "seed": None,
                    "num_graphs": len(spectral_rows),
                    "d_f": _metric_summary(rows=spectral_rows, metric="d_f"),
                    "hard_ncut": _metric_summary(rows=spectral_rows, metric="hard_ncut"),
                    "hard_h2": _metric_summary(rows=spectral_rows, metric="hard_h2"),
                    "hard_k_eff": _metric_summary(rows=spectral_rows, metric="hard_k_eff"),
                }
            )
        for run in runs:
            run_rows = [
                row
                for row in rows
                if row["split"] == row_split
                and row["partition_source"] == "learned"
                and row["run_name"] == run.run_name
            ]
            if not run_rows:
                continue
            per_run.append(
                {
                    "partition_source": "learned",
                    "split": row_split,
                    "run_name": run.run_name,
                    "separation_weight": run.separation_weight,
                    "seed": run.seed,
                    "best_epoch": _read_best_epoch(run),
                    "num_graphs": len(run_rows),
                    "d_f": _metric_summary(rows=run_rows, metric="d_f"),
                    "hard_ncut": _metric_summary(rows=run_rows, metric="hard_ncut"),
                    "hard_h2": _metric_summary(rows=run_rows, metric="hard_h2"),
                    "hard_k_eff": _metric_summary(rows=run_rows, metric="hard_k_eff"),
                    "hard_max_volume_fraction": _metric_summary(rows=run_rows, metric="hard_max_volume_fraction"),
                    "soft_h2": _metric_summary(rows=run_rows, metric="soft_h2"),
                    "soft_k_eff": _metric_summary(rows=run_rows, metric="soft_k_eff"),
                    "soft_separation_d2": _metric_summary(rows=run_rows, metric="soft_separation_d2"),
                }
            )

    by_setting: list[dict[str, Any]] = []
    for row_split in sorted({str(row["split"]) for row in rows}):
        for separation_weight in sorted({run.separation_weight for run in runs}):
            setting_runs = [
                item
                for item in per_run
                if item["partition_source"] == "learned"
                and item["split"] == row_split
                and float(item["separation_weight"]) == float(separation_weight)
            ]
            if not setting_runs:
                continue
            by_setting.append(
                {
                    "split": row_split,
                    "separation_weight": separation_weight,
                    "num_seeds": len(setting_runs),
                    "d_f_mean_across_seeds": _summarize([float(item["d_f"]["mean"]) for item in setting_runs]),
                    "hard_ncut_mean_across_seeds": _summarize(
                        [float(item["hard_ncut"]["mean"]) for item in setting_runs]
                    ),
                    "hard_h2_mean_across_seeds": _summarize([float(item["hard_h2"]["mean"]) for item in setting_runs]),
                    "hard_k_eff_mean_across_seeds": _summarize(
                        [float(item["hard_k_eff"]["mean"]) for item in setting_runs]
                    ),
                    "hard_max_volume_fraction_mean_across_seeds": _summarize(
                        [float(item["hard_max_volume_fraction"]["mean"]) for item in setting_runs]
                    ),
                    "soft_h2_mean_across_seeds": _summarize([float(item["soft_h2"]["mean"]) for item in setting_runs]),
                    "soft_k_eff_mean_across_seeds": _summarize(
                        [float(item["soft_k_eff"]["mean"]) for item in setting_runs]
                    ),
                }
            )
    return {"per_run": per_run, "by_setting": by_setting}


def _aggregate_csv_rows(aggregate: dict[str, Any]) -> list[dict[str, Any]]:
    """Aggregate csv rows."""
    rows: list[dict[str, Any]] = []
    for item in aggregate["per_run"]:
        rows.append(
            {
                "level": "per_run",
                "partition_source": item["partition_source"],
                "split": item["split"],
                "run_name": item["run_name"],
                "separation_weight": "" if item["separation_weight"] is None else item["separation_weight"],
                "seed": "" if item["seed"] is None else item["seed"],
                "num_graphs": item["num_graphs"],
                "d_f_mean": item["d_f"]["mean"],
                "d_f_std": item["d_f"]["std"],
                "hard_ncut_mean": item["hard_ncut"]["mean"],
                "hard_ncut_std": item["hard_ncut"]["std"],
                "hard_h2_mean": item["hard_h2"]["mean"],
                "hard_k_eff_mean": item["hard_k_eff"]["mean"],
                "hard_max_volume_fraction_mean": item.get("hard_max_volume_fraction", {}).get("mean", ""),
                "soft_h2_mean": item.get("soft_h2", {}).get("mean", ""),
                "soft_k_eff_mean": item.get("soft_k_eff", {}).get("mean", ""),
            }
        )
    for item in aggregate["by_setting"]:
        rows.append(
            {
                "level": "by_setting",
                "partition_source": "learned",
                "split": item["split"],
                "run_name": "",
                "separation_weight": item["separation_weight"],
                "seed": "",
                "num_graphs": item["num_seeds"],
                "d_f_mean": item["d_f_mean_across_seeds"]["mean"],
                "d_f_std": item["d_f_mean_across_seeds"]["std"],
                "hard_ncut_mean": item["hard_ncut_mean_across_seeds"]["mean"],
                "hard_ncut_std": item["hard_ncut_mean_across_seeds"]["std"],
                "hard_h2_mean": item["hard_h2_mean_across_seeds"]["mean"],
                "hard_k_eff_mean": item["hard_k_eff_mean_across_seeds"]["mean"],
                "hard_max_volume_fraction_mean": item["hard_max_volume_fraction_mean_across_seeds"]["mean"],
                "soft_h2_mean": item["soft_h2_mean_across_seeds"]["mean"],
                "soft_k_eff_mean": item["soft_k_eff_mean_across_seeds"]["mean"],
            }
        )
    return rows


def _graph_key(row: dict[str, Any]) -> tuple[str, int]:
    """Compute graph key."""
    return str(row["split"]), int(row["dataset_index"])


def _mean_rows_by_graph(
    rows: list[dict[str, Any]], metrics: tuple[str, ...]
) -> dict[tuple[str, int], dict[str, float]]:
    """Compute mean rows by graph."""
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(_graph_key(row), []).append(row)
    means: dict[tuple[str, int], dict[str, float]] = {}
    for key, group in grouped.items():
        means[key] = {metric: float(np.mean([float(row[metric]) for row in group])) for metric in metrics}
    return means


def _delta_summary(values: list[float]) -> dict[str, float | int]:
    """Compute delta summary."""
    summary = _summarize(values)
    summary["positive_fraction"] = (
        float(np.mean(np.asarray(values, dtype=np.float64) > 0.0)) if values else float("nan")
    )
    summary["positive_count"] = int(np.count_nonzero(np.asarray(values, dtype=np.float64) > 0.0))
    summary["count"] = len(values)
    return summary


def _paired_delta_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute paired delta diagnostics."""
    learned_test = [row for row in rows if row["split"] == "test" and row["partition_source"] == "learned"]
    spectral_test = [row for row in rows if row["split"] == "test" and row["partition_source"] == "spectral"]

    learned_003 = _mean_rows_by_graph(
        rows=[row for row in learned_test if float(row["separation_weight"]) == 0.03],
        metrics=("d_f", "hard_ncut", "hard_k_eff", "hard_max_volume_fraction"),
    )
    learned_020 = _mean_rows_by_graph(
        rows=[row for row in learned_test if float(row["separation_weight"]) == 0.20],
        metrics=("d_f", "hard_ncut", "hard_k_eff", "hard_max_volume_fraction"),
    )
    common_003_020 = sorted(set(learned_003) & set(learned_020))

    learned_000 = _mean_rows_by_graph(
        rows=[row for row in learned_test if float(row["separation_weight"]) == 0.0],
        metrics=("d_f", "hard_ncut", "hard_k_eff", "hard_max_volume_fraction"),
    )
    spectral = _mean_rows_by_graph(
        rows=spectral_test,
        metrics=("d_f", "hard_ncut", "hard_k_eff", "hard_max_volume_fraction"),
    )
    common_000_spectral = sorted(set(learned_000) & set(spectral))

    return {
        "lambda_020_minus_003_test_graph_means": {
            metric: _delta_summary([learned_020[key][metric] - learned_003[key][metric] for key in common_003_020])
            for metric in ("d_f", "hard_ncut", "hard_k_eff", "hard_max_volume_fraction")
        },
        "learned_lambda_000_minus_spectral_test_graph_means": {
            metric: _delta_summary([learned_000[key][metric] - spectral[key][metric] for key in common_000_spectral])
            for metric in ("d_f", "hard_ncut", "hard_k_eff", "hard_max_volume_fraction")
        },
    }


def _correlation_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute correlation diagnostics."""
    metrics = ("hard_ncut", "hard_k_eff", "hard_max_volume_fraction")
    per_run: list[dict[str, Any]] = []
    for split in sorted({str(row["split"]) for row in rows}):
        spectral_rows = [row for row in rows if row["split"] == split and row["partition_source"] == "spectral"]
        if spectral_rows:
            per_run.append(
                {
                    "partition_source": "spectral",
                    "split": split,
                    "run_name": "spectral_reference",
                    "separation_weight": None,
                    "seed": None,
                    "correlations": {
                        f"d_f_vs_{metric}": _correlation_summary(rows=spectral_rows, x_metric="d_f", y_metric=metric)
                        for metric in metrics
                    },
                }
            )
        learned_run_keys = sorted(
            {
                (str(row["run_name"]), float(row["separation_weight"]), int(row["seed"]))
                for row in rows
                if row["split"] == split and row["partition_source"] == "learned"
            },
            key=lambda item: (item[1], item[2]),
        )
        for run_name, separation_weight, seed in learned_run_keys:
            run_rows = [
                row
                for row in rows
                if row["split"] == split and row["partition_source"] == "learned" and row["run_name"] == run_name
            ]
            per_run.append(
                {
                    "partition_source": "learned",
                    "split": split,
                    "run_name": run_name,
                    "separation_weight": separation_weight,
                    "seed": seed,
                    "correlations": {
                        f"d_f_vs_{metric}": _correlation_summary(rows=run_rows, x_metric="d_f", y_metric=metric)
                        for metric in metrics
                    },
                }
            )

    by_setting_graph_means: list[dict[str, Any]] = []
    for split in sorted({str(row["split"]) for row in rows}):
        for separation_weight in sorted(
            {
                float(row["separation_weight"])
                for row in rows
                if row["split"] == split and row["partition_source"] == "learned"
            }
        ):
            setting_rows = [
                row
                for row in rows
                if row["split"] == split
                and row["partition_source"] == "learned"
                and float(row["separation_weight"]) == separation_weight
            ]
            graph_means = _mean_rows_by_graph(rows=setting_rows, metrics=("d_f", *metrics))
            mean_rows = list(graph_means.values())
            by_setting_graph_means.append(
                {
                    "split": split,
                    "separation_weight": separation_weight,
                    "num_graphs": len(mean_rows),
                    "correlations": {
                        f"d_f_vs_{metric}": _correlation_summary(rows=mean_rows, x_metric="d_f", y_metric=metric)
                        for metric in metrics
                    },
                }
            )
    return {"per_run": per_run, "by_setting_graph_means": by_setting_graph_means}


def _high_d_f_tail(rows: list[dict[str, Any]], *, limit: int = 20) -> list[dict[str, Any]]:
    """Compute high d f tail."""
    test_rows = [row for row in rows if row["split"] == "test"]
    sorted_rows = sorted(test_rows, key=lambda row: float(row["d_f"]), reverse=True)
    fields = (
        "partition_source",
        "run_name",
        "separation_weight",
        "seed",
        "dataset_index",
        "original_graph_index",
        "num_nodes",
        "num_undirected_edges",
        "d_f",
        "hard_ncut",
        "hard_k_eff",
        "hard_max_volume_fraction",
    )
    return [{field: row[field] for field in fields} for row in sorted_rows[:limit]]


def _sanity_checks(rows: list[dict[str, Any]], resistance_cache_dir: Path) -> dict[str, Any]:
    """Compute sanity checks."""
    d_f_values = np.asarray([float(row["d_f"]) for row in rows], dtype=np.float64)
    foster_errors: list[float] = []
    foster_relative_errors: list[float] = []
    for cache_path in sorted(resistance_cache_dir.rglob("*.npz")):
        match = _RESISTANCE_CACHE_PATTERN.match(cache_path.name)
        if match is None:
            continue
        num_nodes = int(match.group(1))
        loaded = np.load(cache_path)
        foster_sum = float(loaded["foster_sum"])
        expected = float(num_nodes - 1)
        absolute_error = abs(foster_sum - expected)
        foster_errors.append(absolute_error)
        foster_relative_errors.append(absolute_error / max(expected, 1.0))
    return {
        "d_f_count": int(d_f_values.size),
        "d_f_min": float(d_f_values.min()) if d_f_values.size else float("nan"),
        "d_f_max": float(d_f_values.max()) if d_f_values.size else float("nan"),
        "d_f_negative_count_tolerance_1e-8": int(np.count_nonzero(d_f_values < -1e-8)),
        "d_f_above_one_count_tolerance_1e-8": int(np.count_nonzero(d_f_values > 1.0 + 1e-8)),
        "effective_resistance_graph_count": len(foster_errors),
        "foster_absolute_error_max": float(np.max(foster_errors)) if foster_errors else float("nan"),
        "foster_relative_error_max": float(np.max(foster_relative_errors)) if foster_relative_errors else float("nan"),
        "foster_relative_error_mean": float(np.mean(foster_relative_errors))
        if foster_relative_errors
        else float("nan"),
    }


def _print_summary(aggregate: dict[str, Any]) -> None:
    """Compute print summary."""
    print("\nFourier/Dirichlet distortion summary")
    print("source    split       lambda  seeds  D_F mean +- sd     hard Ncut mean +- sd  soft K_eff")
    for item in aggregate["per_run"]:
        if item["partition_source"] != "spectral":
            continue
        print(
            f"spectral  {item['split']:<10} {'-':>6}  {'-':>5}  "
            f"{item['d_f']['mean']:.4f} +- {item['d_f']['std']:.4f}  "
            f"{item['hard_ncut']['mean']:.4f} +- {item['hard_ncut']['std']:.4f}  {'-':>10}"
        )
    for item in aggregate["by_setting"]:
        print(
            f"learned   {item['split']:<10} {float(item['separation_weight']):>6.2f}  "
            f"{int(item['num_seeds']):>5}  "
            f"{item['d_f_mean_across_seeds']['mean']:.4f} +- {item['d_f_mean_across_seeds']['std']:.4f}  "
            f"{item['hard_ncut_mean_across_seeds']['mean']:.4f} +- "
            f"{item['hard_ncut_mean_across_seeds']['std']:.4f}  "
            f"{item['soft_k_eff_mean_across_seeds']['mean']:.2f}"
        )


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Compute post-hoc graph-Fourier/Dirichlet distortion for reported MalNet partitions."
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--spectral-cache-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--splits", nargs="+", default=("validation", "test"), choices=("validation", "test"))
    parser.add_argument("--solve-batch-size", type=int, default=256)
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    runs = discover_reported_runs(run_root)
    if args.max_runs is not None:
        runs = runs[: int(args.max_runs)]
    reference_config = _load_run_config(runs[0].run_path)
    preprocess_config = _preprocess_config(reference_config)
    datasets = load_cached_splits(preprocess_config.cache_dir)
    split_names = ["val" if split == "validation" else split for split in args.splits]

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = run_root / "malnet_fourier_dirichlet_posthoc"
    output_dir = output_dir.resolve()
    resistance_cache_dir = output_dir / "effective_resistance_cache"
    spectral_cache_dir = args.spectral_cache_dir
    if spectral_cache_dir is None:
        spectral_cache_dir = output_dir / "spectral_cache"
    spectral_cache_dir = spectral_cache_dir.resolve()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    rows: list[dict[str, Any]] = []
    spectral_config = _spectral_config(reference_config)
    print(f"Discovered {len(runs)} learned runs under {run_root}")
    print(f"Using cleaned cache {preprocess_config.cache_dir}")
    print(f"Writing post-hoc results to {output_dir}")
    print(f"Model inference device: {device}")

    for split_name in split_names:
        dataset = datasets[split_name]
        public_split = "validation" if split_name == "val" else split_name
        print(f"[spectral {public_split}] graphs={len(dataset)}", flush=True)
        rows.extend(
            _evaluate_spectral_split(
                split=public_split,
                dataset=dataset,
                spectral_config=spectral_config,
                spectral_cache_dir=spectral_cache_dir,
                resistance_cache_dir=resistance_cache_dir,
                solve_batch_size=int(args.solve_batch_size),
                max_graphs=args.max_graphs,
            )
        )

    for run_index, run in enumerate(runs, start=1):
        config = _load_run_config(run.run_path)
        for split_name in split_names:
            dataset = datasets[split_name]
            public_split = "validation" if split_name == "val" else split_name
            print(
                f"[learned {run_index}/{len(runs)} {public_split}] "
                f"lambda={run.separation_weight:.2f} seed={run.seed} graphs={len(dataset)}",
                flush=True,
            )
            rows.extend(
                _evaluate_learned_run_split(
                    split=public_split,
                    dataset=dataset,
                    run=run,
                    config=config,
                    device=device,
                    resistance_cache_dir=resistance_cache_dir,
                    solve_batch_size=int(args.solve_batch_size),
                    max_graphs=args.max_graphs,
                )
            )

    aggregate = _aggregate_rows(rows=rows, runs=runs)
    diagnostics = {
        "correlations": _correlation_diagnostics(rows),
        "paired_deltas": _paired_delta_diagnostics(rows),
        "high_d_f_tail": _high_d_f_tail(rows),
    }
    sanity = _sanity_checks(rows=rows, resistance_cache_dir=resistance_cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(path=output_dir / "per_graph_fourier_dirichlet.csv", rows=rows)
    aggregate_json = {
        "metric": "D_F = sum_{cut edge e} w_e R_eff^G(e) / (n - 1)",
        "run_root": str(run_root),
        "preprocess_cache_dir": str(preprocess_config.cache_dir),
        "spectral_cache_dir": str(spectral_cache_dir),
        "splits": list(args.splits),
        "num_runs": len(runs),
        "sanity_checks": sanity,
        "diagnostics": diagnostics,
        **aggregate,
    }
    (output_dir / "aggregate_fourier_dirichlet.json").write_text(json.dumps(aggregate_json, indent=2))
    _write_csv(path=output_dir / "aggregate_fourier_dirichlet.csv", rows=_aggregate_csv_rows(aggregate))
    (output_dir / "diagnostic_fourier_dirichlet.json").write_text(json.dumps(diagnostics, indent=2))
    _write_csv(path=output_dir / "high_d_f_tail.csv", rows=cast(list[dict[str, Any]], diagnostics["high_d_f_tail"]))
    _print_summary(aggregate)
    print(
        "Sanity: "
        f"D_F range [{sanity['d_f_min']:.6f}, {sanity['d_f_max']:.6f}], "
        f"out_of_range=({sanity['d_f_negative_count_tolerance_1e-8']}, "
        f"{sanity['d_f_above_one_count_tolerance_1e-8']}), "
        f"max Foster relative error={sanity['foster_relative_error_max']:.3e}"
    )


if __name__ == "__main__":
    main()
