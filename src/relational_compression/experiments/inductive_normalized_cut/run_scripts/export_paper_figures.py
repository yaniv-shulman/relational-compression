"""Command-line utilities for reproducible experiment workflows."""

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence, cast

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from torch_geometric.data import Data

from relational_compression.experiments.inductive_normalized_cut.baselines import (
    SpectralNormalizedCutConfig,
    evaluate_spectral_normalized_cut,
)
from relational_compression.experiments.inductive_normalized_cut.data import PreprocessConfig, load_cached_splits
from relational_compression.experiments.inductive_normalized_cut.metrics import hard_normalized_cut
from relational_compression.experiments.inductive_normalized_cut.models import make_model


@dataclass(frozen=True)
class SweepRun:
    """Store Sweep Run values."""

    label: str
    separation_weight: float
    seed: int
    path: Path


@dataclass(frozen=True)
class PartitionExample:
    """Store Partition Example values."""

    label: str
    dataset_index: int


DEFAULT_SWEEP_ID = "phase2_uniformity_seed_sweep_1788228910"
DEFAULT_RUN_ROOT = Path("out/reported_runs")
DEFAULT_FOURIER_POSTHOC_DIR = DEFAULT_RUN_ROOT / "malnet_fourier_dirichlet_posthoc"
DEFAULT_FOURIER_SUMMARY_PATH = DEFAULT_FOURIER_POSTHOC_DIR / "aggregate_fourier_dirichlet.json"
DEFAULT_FOURIER_PER_GRAPH_PATH = DEFAULT_FOURIER_POSTHOC_DIR / "per_graph_fourier_dirichlet.csv"
DEFAULT_SWEEP_RUNS = tuple(
    SweepRun(
        label=label,
        separation_weight=weight,
        seed=seed,
        path=Path(
            f"inductive_normalized_cut_malnet_tiny_1.0_relational_compression_{DEFAULT_SWEEP_ID}_uni{suffix}_seed{seed}"
        ),
    )
    for label, suffix, weight in (
        ("0", "000", 0.0),
        ("0.03", "003", 0.03),
        ("0.10", "010", 0.10),
        ("0.20", "020", 0.20),
        ("0.50", "050", 0.50),
    )
    for seed in (1337, 2024, 31415, 27182, 16180)
)

DEFAULT_PARTITION_RUN = Path(
    f"inductive_normalized_cut_malnet_tiny_1.0_relational_compression_{DEFAULT_SWEEP_ID}_uni020_seed16180"
)


def _repo_root() -> Path:
    """Compute repo root."""
    return Path(__file__).resolve().parents[5]


def _resolve_run_path(run_path: Path, repo_root: Path, run_root: Path) -> Path:
    """Resolve run path."""
    if run_path.is_absolute():
        return run_path
    candidate = run_root / run_path
    if candidate.exists():
        return candidate
    return repo_root / run_path


def _load_result(run: SweepRun, repo_root: Path, run_root: Path) -> dict[str, Any]:
    """Load result."""
    result_path = _resolve_run_path(run_path=run.path, repo_root=repo_root, run_root=run_root) / "all_run_results.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing result JSON for {run.label}: {result_path}")
    return cast(dict[str, Any], json.loads(result_path.read_text())[0])


def _mean_std(values: list[float]) -> tuple[float, float]:
    """Compute mean std."""
    array = np.asarray(values, dtype=np.float64)
    if array.size < 2:
        return float(array.mean()), 0.0
    return float(array.mean()), float(array.std(ddof=1))


def _load_fourier_summary(summary_path: Path) -> dict[str, Any]:
    """Load fourier summary."""
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing Fourier/Dirichlet summary JSON: {summary_path}")
    return cast(dict[str, Any], json.loads(summary_path.read_text()))


def _fourier_key(value: float) -> str:
    """Compute fourier key."""
    return f"{float(value):.8g}"


def _metric_value(metrics: dict[str, Any], key: str, *, fallback_key: str | None = None) -> float:
    """Compute metric value."""
    if key in metrics:
        return float(metrics[key])
    if fallback_key is not None and fallback_key in metrics:
        return float(metrics[fallback_key])
    raise KeyError(key)


def _plot_separation_sweep(
    repo_root: Path,
    output_dir: Path,
    *,
    run_root: Path = DEFAULT_RUN_ROOT,
    fourier_summary_path: Path = DEFAULT_FOURIER_SUMMARY_PATH,
) -> Path:
    """Plot separation sweep."""
    fourier_summary = _load_fourier_summary(fourier_summary_path)
    test_fourier_by_weight = {
        _fourier_key(float(item["separation_weight"])): item["d_f_mean_across_seeds"]
        for item in fourier_summary["by_setting"]
        if item["split"] == "test"
    }
    spectral_fourier = next(
        item
        for item in fourier_summary["per_run"]
        if item["partition_source"] == "spectral" and item["split"] == "test"
    )
    spectral_reference_ncut = float(spectral_fourier["hard_ncut"]["mean"])
    spectral_reference_d_f = float(spectral_fourier["d_f"]["mean"])

    per_seed_rows: list[dict[str, float | int | str]] = []
    for run in DEFAULT_SWEEP_RUNS:
        result = _load_result(run=run, repo_root=repo_root, run_root=run_root)
        per_seed_rows.append(
            {
                "label": run.label,
                "separation_weight": run.separation_weight,
                "seed": run.seed,
                "best_epoch": int(result["best_epoch"]),
                "validation_hard_ncut": float(result["best_validation"]["hard_ncut_mean"]),
                "test_hard_ncut": float(result["test"]["hard_ncut_mean"]),
                "test_hard_ncut_median": float(result["test"]["hard_ncut_median"]),
                "max_volume": float(result["test"]["hard_max_volume_fraction_mean"]),
                "within_edge": float(result["test"]["hard_within_edge_fraction_mean"]),
                "effective_partitions": float(result["test"]["effective_partitions_mean"]),
                "separation_d2": _metric_value(
                    metrics=result["test"], key="separation_d2_mean", fallback_key="uniformity_d2_mean"
                ),
            }
        )

    rows: list[dict[str, float | str]] = []
    labels = sorted({str(row["label"]) for row in per_seed_rows}, key=lambda value: float(value))
    for label in labels:
        group = [row for row in per_seed_rows if row["label"] == label]
        row: dict[str, float | str] = {
            "label": label,
            "separation_weight": float(group[0]["separation_weight"]),
        }
        for source_key, output_key in (
            ("validation_hard_ncut", "validation_hard_ncut"),
            ("test_hard_ncut", "test_hard_ncut"),
            ("test_hard_ncut_median", "test_hard_ncut_median"),
            ("max_volume", "max_volume"),
            ("within_edge", "within_edge"),
            ("effective_partitions", "effective_partitions"),
            ("separation_d2", "separation_d2"),
        ):
            mean, std = _mean_std([float(item[source_key]) for item in group])
            row[f"{output_key}_mean"] = mean
            row[f"{output_key}_std"] = std
        d_f_summary = test_fourier_by_weight[_fourier_key(float(group[0]["separation_weight"]))]
        row["d_f_mean"] = float(d_f_summary["mean"])
        row["d_f_std"] = float(d_f_summary["std"])
        row["best_epoch_mean"], row["best_epoch_std"] = _mean_std([float(item["best_epoch"]) for item in group])
        rows.append(row)

    lambdas = np.asarray([float(row["separation_weight"]) for row in rows])
    test_ncut = np.asarray([float(row["test_hard_ncut_mean"]) for row in rows])
    test_ncut_std = np.asarray([float(row["test_hard_ncut_std"]) for row in rows])
    max_volume = np.asarray([float(row["max_volume_mean"]) for row in rows])
    max_volume_std = np.asarray([float(row["max_volume_std"]) for row in rows])
    d_f = np.asarray([float(row["d_f_mean"]) for row in rows])
    d_f_std = np.asarray([float(row["d_f_std"]) for row in rows])
    effective_partitions = np.asarray([float(row["effective_partitions_mean"]) for row in rows])
    effective_partitions_std = np.asarray([float(row["effective_partitions_std"]) for row in rows])

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(7.2, 2.6), constrained_layout=True)
    line_color = "black"

    axes[0].errorbar(
        lambdas,
        test_ncut,
        yerr=test_ncut_std,
        marker="o",
        capsize=3,
        color=line_color,
        ecolor=line_color,
        label="Inductive encoder",
    )
    axes[0].axhline(
        spectral_reference_ncut,
        linestyle="--",
        color=line_color,
        label="Spectral reference",
    )
    axes[0].set_xlabel(r"$\lambda_{\mathrm{org}}$")
    axes[0].set_ylabel("Test hard Ncut")
    axes[0].set_title("(a) Cut objective", fontsize=10)
    axes[0].grid(alpha=0.25)

    axes[1].errorbar(
        lambdas,
        d_f,
        yerr=d_f_std,
        marker="o",
        capsize=3,
        color=line_color,
        ecolor=line_color,
        label="Inductive encoder",
    )
    axes[1].axhline(
        spectral_reference_d_f,
        linestyle="--",
        color=line_color,
        label="Spectral reference",
    )
    axes[1].set_xlabel(r"$\lambda_{\mathrm{org}}$")
    axes[1].set_ylabel(r"Test $D_F$")
    axes[1].set_title("(b) Dirichlet distortion", fontsize=10)
    axes[1].grid(alpha=0.25)

    axes[2].errorbar(
        lambdas,
        max_volume,
        yerr=max_volume_std,
        marker="o",
        capsize=3,
        color=line_color,
        ecolor=line_color,
        label="Largest volume",
    )
    axes[2].set_xlabel(r"$\lambda_{\mathrm{org}}$")
    axes[2].set_ylabel("Mean test fraction")
    axes[2].set_title("(c) Dominant partition", fontsize=10)
    axes[2].set_ylim(0.35, 1.0)
    axes[2].grid(alpha=0.25)

    axes[3].errorbar(
        lambdas,
        effective_partitions,
        yerr=effective_partitions_std,
        marker="o",
        capsize=3,
        color=line_color,
        ecolor=line_color,
        label=r"$K_{\mathrm{eff}}$",
    )
    axes[3].set_xlabel(r"$\lambda_{\mathrm{org}}$")
    axes[3].set_ylabel("Effective partitions")
    axes[3].set_title("(d) Collision utilization", fontsize=10)
    axes[3].set_ylim(1.0, 4.0)
    axes[3].grid(alpha=0.25)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "malnet_ncut_separation_sweep.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    metadata = {
        "run_root": str(run_root),
        "fourier_summary_path": str(fourier_summary_path),
        "spectral_reference_hard_ncut": spectral_reference_ncut,
        "spectral_reference_d_f": spectral_reference_d_f,
        "spectral_reference": {
            "hard_ncut": spectral_fourier["hard_ncut"],
            "d_f": spectral_fourier["d_f"],
        },
        "rows": rows,
        "per_seed_rows": per_seed_rows,
    }
    (output_dir / "malnet_ncut_separation_sweep.json").write_text(json.dumps(metadata, indent=2))
    return output_path


def _load_config(run_path: Path) -> SimpleNamespace:
    """Load config."""
    config_path = run_path / "effective_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing effective config: {config_path}")
    return SimpleNamespace(**json.loads(config_path.read_text()))


def _load_test_dataset(config: Any) -> tuple[Sequence[Data], Path]:
    """Load test dataset."""
    preprocess_config = PreprocessConfig(
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
    datasets = load_cached_splits(preprocess_config.cache_dir)
    return cast(Sequence[Data], datasets["test"]), preprocess_config.cache_dir


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


def _load_best_model(run_path: Path, config: Any, input_dim: int) -> torch.nn.Module:
    """Load best model."""
    model = make_model(config, input_dim=input_dim)
    checkpoint_path = run_path / "checkpoints" / "run_00_categorical" / "best_model_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _choose_example_graph(dataset: Sequence[Data], *, dataset_index: int) -> int:
    """Compute choose example graph."""
    if 0 <= int(dataset_index) < len(dataset):
        return int(dataset_index)
    raise ValueError(f"Dataset index out of range: {dataset_index}")


def _display_permuted_labels(labels: torch.Tensor, *, num_partitions: int, seed: int) -> tuple[torch.Tensor, list[int]]:
    """Compute display permuted labels."""
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(int(num_partitions)).astype(np.int64)
    labels_np = labels.detach().cpu().numpy()
    mapped = torch.tensor([int(permutation[int(label)]) for label in labels_np], dtype=torch.long)
    return mapped, [int(value) for value in permutation.tolist()]


def _load_spectral_test_ncuts(repo_root: Path) -> np.ndarray:
    """Load spectral test ncuts."""
    del repo_root
    if not DEFAULT_FOURIER_PER_GRAPH_PATH.exists():
        raise FileNotFoundError(f"Missing Fourier/Dirichlet per-graph CSV: {DEFAULT_FOURIER_PER_GRAPH_PATH}")
    rows: list[tuple[int, float]] = []
    with DEFAULT_FOURIER_PER_GRAPH_PATH.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["partition_source"] == "spectral" and row["split"] == "test":
                rows.append((int(row["dataset_index"]), float(row["hard_ncut"])))
    if not rows:
        raise ValueError("Fourier/Dirichlet per-graph CSV does not contain spectral test rows")
    return np.asarray([value for _, value in sorted(rows)], dtype=np.float64)


def _auto_select_partition_examples(
    repo_root: Path,
    dataset: Sequence[Data],
    run_path: Path,
    config: Any,
    *,
    max_nodes: int = 900,
) -> tuple[PartitionExample, ...]:
    """Compute auto select partition examples."""
    model = _load_best_model(run_path=run_path, config=config, input_dim=int(dataset[0].x.shape[-1]))
    spectral_ncuts = _load_spectral_test_ncuts(repo_root)
    if spectral_ncuts.shape[0] != len(dataset):
        raise ValueError("Spectral reference length does not match the test dataset")

    records: list[dict[str, float | int]] = []
    with torch.no_grad():
        for index, data in enumerate(dataset):
            if int(data.num_nodes) > int(max_nodes):
                continue
            output = model(data)
            hard = hard_normalized_cut(
                assignments=output.hard_ids, edge_index=data.edge_index, num_partitions=int(config.num_partitions)
            )
            learned_ncut = float(hard.ncut_per_graph[0])
            spectral_ncut = float(spectral_ncuts[index])
            records.append(
                {
                    "dataset_index": index,
                    "num_nodes": int(data.num_nodes),
                    "gap": learned_ncut - spectral_ncut,
                }
            )
    if len(records) < 4:
        raise ValueError("Not enough small test graphs to select partition examples")

    signed_gaps = np.asarray([float(record["gap"]) for record in records])
    quantiles = (
        ("25th percentile", 0.25),
        ("Median", 0.50),
        ("75th percentile", 0.75),
        ("95th percentile", 0.95),
    )
    examples: list[PartitionExample] = []
    for label, quantile in quantiles:
        target = float(np.quantile(a=signed_gaps, q=quantile))
        record = min(
            records,
            key=lambda candidate: (
                abs(float(candidate["gap"]) - target),
                int(candidate["num_nodes"]),
            ),
        )
        examples.append(PartitionExample(label, int(record["dataset_index"])))
    return tuple(examples)


def _select_example_records(
    dataset: Sequence[Data],
    run_path: Path,
    config: Any,
    examples: tuple[PartitionExample, ...],
) -> list[dict[str, Any]]:
    """Select example records."""
    model = _load_best_model(run_path=run_path, config=config, input_dim=int(dataset[0].x.shape[-1]))
    spectral_cache = run_path / "run_00_categorical" / "spectral_cache"
    spectral_config = _spectral_config(config)
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for example in examples:
            index = _choose_example_graph(dataset, dataset_index=example.dataset_index)
            data = dataset[index]
            output = model(data)
            hard = hard_normalized_cut(
                assignments=output.hard_ids, edge_index=data.edge_index, num_partitions=int(config.num_partitions)
            )
            learned_ncut = float(hard.ncut_per_graph[0])
            learned_max_volume = float(hard.max_partition_volume_fraction_per_graph[0])
            spectral = evaluate_spectral_normalized_cut(data, cache_dir=spectral_cache, config=spectral_config)
            spectral_ncut = float(spectral["hard_ncut"])
            spectral_labels = spectral["labels"].detach().cpu()
            learned_labels = output.hard_ids.detach().cpu()
            spectral_display_labels, spectral_display_permutation = _display_permuted_labels(
                spectral_labels,
                num_partitions=int(config.num_partitions),
                seed=10_000 + int(index),
            )
            learned_display_labels, learned_display_permutation = _display_permuted_labels(
                learned_labels,
                num_partitions=int(config.num_partitions),
                seed=20_000 + int(index),
            )
            records.append(
                {
                    "label": example.label,
                    "dataset_index": index,
                    "data": data,
                    "spectral_labels": spectral_labels,
                    "spectral_display_labels": spectral_display_labels,
                    "spectral_display_label_permutation": spectral_display_permutation,
                    "learned_labels": learned_labels,
                    "learned_display_labels": learned_display_labels,
                    "learned_display_label_permutation": learned_display_permutation,
                    "spectral_hard_ncut": spectral_ncut,
                    "spectral_active_partitions": float(spectral["active_partitions"]),
                    "spectral_hard_max_volume_fraction": float(spectral["hard_max_volume_fraction"]),
                    "learned_hard_ncut": learned_ncut,
                    "learned_hard_max_volume_fraction": learned_max_volume,
                    "learned_minus_spectral_hard_ncut_gap": learned_ncut - spectral_ncut,
                    "absolute_hard_ncut_gap": abs(learned_ncut - spectral_ncut),
                }
            )
    return records


def _to_networkx_graph(edge_index: torch.Tensor, num_nodes: int) -> nx.Graph:
    """Compute to networkx graph."""
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    edges = {
        (int(min(source, target)), int(max(source, target)))
        for source, target in edge_index.t().tolist()
        if int(source) != int(target)
    }
    graph.add_edges_from(sorted(edges))
    return graph


def _plot_partition_example(
    repo_root: Path,
    output_dir: Path,
    *,
    run_path: Path,
    examples: tuple[PartitionExample, ...],
    layout_seed: int,
) -> Path:
    """Plot partition example."""
    config = _load_config(run_path)
    dataset, _ = _load_test_dataset(config)
    if not examples:
        examples = _auto_select_partition_examples(
            repo_root=repo_root, dataset=dataset, run_path=run_path, config=config
        )
    records = _select_example_records(dataset=dataset, run_path=run_path, config=config, examples=examples)

    fig, axes = plt.subplots(nrows=len(records), ncols=2, figsize=(7.2, 9.0), constrained_layout=True)
    if len(records) == 1:
        axes = np.asarray([axes])

    column_titles = ("Spectral reference", r"Inductive encoder ($\lambda_{\mathrm{org}}=0.20$)")
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, fontsize=10)

    for row, record in enumerate(records):
        data = record["data"]
        graph = _to_networkx_graph(edge_index=data.edge_index.cpu(), num_nodes=int(data.num_nodes))
        pos = nx.spring_layout(
            graph,
            seed=int(layout_seed) + row,
            iterations=120,
            k=1.4 / np.sqrt(max(1, graph.number_of_nodes())),
        )
        row_panels = (
            (
                axes[row, 0],
                record["spectral_display_labels"],
                float(record["spectral_hard_ncut"]),
                float(record["spectral_hard_max_volume_fraction"]),
                "tab10",
            ),
            (
                axes[row, 1],
                record["learned_display_labels"],
                float(record["learned_hard_ncut"]),
                float(record["learned_hard_max_volume_fraction"]),
                "tab10",
            ),
        )
        for column, (axis, labels, ncut, max_volume, palette_name) in enumerate(row_panels):
            base_colors = plt.get_cmap(palette_name)(np.linspace(start=0, stop=1, num=int(config.num_partitions)))
            row_cmap = ListedColormap(np.roll(base_colors, shift=2 * row + column, axis=0))
            nx.draw_networkx_edges(graph, pos, ax=axis, width=0.22, edge_color="#a7a7a7", alpha=0.16)
            nx.draw_networkx_nodes(
                graph,
                pos,
                ax=axis,
                node_color=labels.numpy(),
                cmap=row_cmap,
                vmin=0,
                vmax=max(1, int(config.num_partitions) - 1),
                node_size=7,
                linewidths=0.0,
            )
            axis.text(
                0.01,
                0.01,
                f"Ncut {ncut:.3f}; max vol. {max_volume:.3f}",
                transform=axis.transAxes,
                fontsize=7,
                ha="left",
                va="bottom",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
            )
            axis.set_axis_off()
        axes[row, 0].text(
            -0.02,
            0.5,
            str(record["label"]),
            transform=axes[row, 0].transAxes,
            rotation=90,
            fontsize=9,
            ha="right",
            va="center",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "malnet_ncut_partition_example.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    metadata_examples: list[dict[str, Any]] = []
    for record in records:
        data = record["data"]
        graph = _to_networkx_graph(edge_index=data.edge_index.cpu(), num_nodes=int(data.num_nodes))
        if hasattr(data, "original_graph_index"):
            original_graph_index = int(data.original_graph_index)
        else:
            dataset_record = cast(Any, dataset).index[int(record["dataset_index"])]
            original_graph_index = int(dataset_record["original_graph_index"])
        metadata_examples.append(
            {
                "label": record["label"],
                "split": "test",
                "dataset_index": int(record["dataset_index"]),
                "original_graph_index": original_graph_index,
                "num_nodes": graph.number_of_nodes(),
                "num_undirected_edges": graph.number_of_edges(),
                "spectral_hard_ncut": float(record["spectral_hard_ncut"]),
                "spectral_active_partitions": float(record["spectral_active_partitions"]),
                "spectral_hard_max_volume_fraction": float(record["spectral_hard_max_volume_fraction"]),
                "learned_hard_ncut": float(record["learned_hard_ncut"]),
                "learned_hard_max_volume_fraction": float(record["learned_hard_max_volume_fraction"]),
                "learned_minus_spectral_hard_ncut_gap": float(record["learned_minus_spectral_hard_ncut_gap"]),
                "absolute_hard_ncut_gap": float(record["absolute_hard_ncut_gap"]),
                "spectral_display_label_permutation": list(record["spectral_display_label_permutation"]),
                "learned_display_label_permutation": list(record["learned_display_label_permutation"]),
                "row_palette_shift": int(2 * len(metadata_examples)),
            }
        )

    metadata = {
        "run_path": str(run_path.relative_to(repo_root) if run_path.is_relative_to(repo_root) else run_path),
        "display_label_permutation": "spectral and learned partition IDs are independently permuted for display only; metrics use the original labels",
        "palette": "both panels use high-contrast tab10 palettes with independent label permutations and row-specific shifts; colors are not comparable across panels or rows",
        "examples": metadata_examples,
    }
    (output_dir / "malnet_ncut_partition_example.json").write_text(json.dumps(metadata, indent=2))
    return output_path


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="Export paper figures for the inductive normalized-cut experiment.")
    parser.add_argument("--repo-root", type=Path, default=_repo_root())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--fourier-summary-path", type=Path, default=DEFAULT_FOURIER_SUMMARY_PATH)
    parser.add_argument(
        "--partition-run",
        type=Path,
        default=DEFAULT_PARTITION_RUN,
    )
    parser.add_argument(
        "--example-dataset-indices",
        type=int,
        nargs="+",
        default=None,
    )
    parser.add_argument("--layout-seed", type=int, default=1337)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir or repo_root / "paper" / "figures"
    output_dir = output_dir.resolve()
    partition_run = args.partition_run
    if not partition_run.is_absolute():
        partition_run = _resolve_run_path(run_path=partition_run, repo_root=repo_root, run_root=args.run_root.resolve())

    sweep_path = _plot_separation_sweep(
        repo_root=repo_root,
        output_dir=output_dir,
        run_root=args.run_root.resolve(),
        fourier_summary_path=args.fourier_summary_path.resolve(),
    )
    partition_path = _plot_partition_example(
        repo_root=repo_root,
        output_dir=output_dir,
        run_path=partition_run,
        examples=()
        if args.example_dataset_indices is None
        else tuple(
            PartitionExample(
                ("Low gap", "Near median gap", "High gap")[index] if index < 3 else f"Example {index + 1}",
                dataset_index,
            )
            for index, dataset_index in enumerate(args.example_dataset_indices)
        ),
        layout_seed=int(args.layout_seed),
    )
    print(f"Wrote {sweep_path}")
    print(f"Wrote {partition_path}")


if __name__ == "__main__":
    main()
