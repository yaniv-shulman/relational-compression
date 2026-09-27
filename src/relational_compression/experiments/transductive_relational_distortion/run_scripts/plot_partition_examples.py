"""Command-line utilities for reproducible experiment workflows."""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import torch
from torch_geometric.data import Data

from relational_compression.experiments.graph_geometry import _undirected_weighted_edges
from relational_compression.experiments.transductive_relational_distortion.configs import default as default_config
from relational_compression.experiments.transductive_relational_distortion.data_sources import load_source_graph_records

CRITERIA = ("edge", "fourier", "collision", "collision_entropy")
CRITERION_LABELS = {
    "edge": r"$D_E$",
    "fourier": r"$D_F$",
    "collision": r"$D_C$",
    "collision_entropy": r"$D_{H_2}$",
}
OWN_DISTORTION_FIELD = {
    "edge": "hard_D_E",
    "fourier": "hard_D_F",
    "collision": "hard_D_C",
    "collision_entropy": "hard_D_H2",
}
EXAMPLES: tuple[dict[str, str | int | float], ...] = (
    {"collection": "proteins", "graph_index": 17, "lambda_org": 0.50, "row_label": "PROTEINS"},
    {"collection": "collab", "graph_index": 12, "lambda_org": 0.50, "row_label": "COLLAB"},
)
PALETTE = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
)


def _default_values() -> dict[str, Any]:
    """Compute default values."""
    values: dict[str, Any] = {}
    for name in dir(default_config):
        if name.startswith("_"):
            continue
        value = getattr(default_config, name)
        if callable(value) or getattr(value, "__name__", None) == "annotations":
            continue
        values[name] = value
    return values


def _config_for_collection(collection: str, experiment_dir: Path) -> SimpleNamespace:
    """Compute config for collection."""
    values = _default_values()
    config_path = experiment_dir / "effective_config.json"
    if config_path.exists():
        loaded = cast(dict[str, Any], json.loads(config_path.read_text()))
        for key, value in loaded.items():
            if key.endswith("_dir") or key.endswith("_file"):
                values[key] = Path(value)
            else:
                values[key] = tuple(value) if isinstance(value, list) else value
    values["source_collections"] = (collection,)
    values["unique_postfix"] = None
    return SimpleNamespace(**values)


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Read rows."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _matching_row(
    rows: list[dict[str, str]],
    *,
    collection: str,
    graph_index: int,
    lambda_org: float,
    criterion: str,
) -> dict[str, str]:
    """Compute matching row."""
    matches = [
        row
        for row in rows
        if row["collection"] == collection
        and int(row["graph_index"]) == int(graph_index)
        and np.isclose(float(row["lambda_org"]), float(lambda_org))
        and row["criterion"] == criterion
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected one row for "
            f"collection={collection}, graph={graph_index}, lambda={lambda_org}, criterion={criterion}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _load_assignments(experiment_dir: Path, row: dict[str, str]) -> np.ndarray:
    """Load assignments."""
    artifact = experiment_dir / row["partition_artifact"]
    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    return np.asarray(payload["assignments"].cpu(), dtype=np.int64)


def _load_example_data(experiment_dir: Path, *, collection: str, graph_index: int) -> Data:
    """Load example data."""
    config = _config_for_collection(collection=collection, experiment_dir=experiment_dir)
    records, _ = load_source_graph_records(config)
    if int(graph_index) >= len(records):
        raise IndexError(f"Graph index {graph_index} is unavailable for collection {collection}")
    return records[int(graph_index)].data


def _layout_for_graph(data: Data, *, seed: int) -> dict[int, np.ndarray]:
    """Compute layout for graph."""
    import networkx as nx

    edges, _ = _undirected_weighted_edges(
        edge_index=data.edge_index, num_nodes=int(data.num_nodes), edge_weight=getattr(data, "edge_weight", None)
    )
    graph = nx.Graph()
    graph.add_nodes_from(range(int(data.num_nodes)))
    graph.add_edges_from((int(source), int(target)) for source, target in edges.tolist())
    try:
        return cast(dict[int, np.ndarray], nx.kamada_kawai_layout(graph))
    except (ValueError, ZeroDivisionError):
        return cast(dict[int, np.ndarray], nx.spring_layout(graph, seed=int(seed), iterations=250, k=None))


def _panel_colors(assignments: np.ndarray, *, seed: int) -> list[str]:
    """Compute panel colors."""
    rng = np.random.default_rng(int(seed))
    palette = list(PALETTE)
    rng.shuffle(palette)
    labels = sorted(int(label) for label in np.unique(assignments))
    mapping = {label: palette[index % len(palette)] for index, label in enumerate(labels)}
    return [mapping[int(label)] for label in assignments.tolist()]


def _draw_panel(
    axis: Any,
    data: Data,
    positions: dict[int, np.ndarray],
    assignments: np.ndarray,
    row: dict[str, str],
    *,
    color_seed: int,
) -> None:
    """Compute draw panel."""
    import networkx as nx

    edges, _ = _undirected_weighted_edges(
        edge_index=data.edge_index, num_nodes=int(data.num_nodes), edge_weight=getattr(data, "edge_weight", None)
    )
    graph = nx.Graph()
    graph.add_nodes_from(range(int(data.num_nodes)))
    graph.add_edges_from((int(source), int(target)) for source, target in edges.tolist())
    own_distortion = float(row[OWN_DISTORTION_FIELD[row["criterion"]]])
    k_eff = float(row["hard_k_eff"])
    node_size = 56 if int(data.num_nodes) <= 20 else 38
    nx.draw_networkx_edges(graph, positions, ax=axis, edge_color="#9ca3af", alpha=0.34, width=0.55)
    nx.draw_networkx_nodes(
        graph,
        positions,
        ax=axis,
        node_color=_panel_colors(assignments, seed=color_seed),
        node_size=node_size,
        linewidths=0.45,
        edgecolors="white",
    )
    axis.set_title(
        f"{CRITERION_LABELS[row['criterion']]}\n"
        rf"$D={own_distortion:.3f}$, $K_{{\mathrm{{eff}}}}={k_eff:.2f}$",
        fontsize=8,
        pad=4,
    )
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_aspect("equal")


def plot_partition_examples(*, experiment_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Plot partition examples."""
    import matplotlib.pyplot as plt

    rows = _read_rows(experiment_dir / "per_graph_results.csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(nrows=len(EXAMPLES), ncols=len(CRITERIA), figsize=(7.2, 4.15), constrained_layout=True)
    metadata: dict[str, Any] = {"experiment_dir": str(experiment_dir), "examples": []}

    for row_index, example in enumerate(EXAMPLES):
        collection = str(example["collection"])
        graph_index = int(example["graph_index"])
        lambda_org = float(example["lambda_org"])
        data = _load_example_data(experiment_dir, collection=collection, graph_index=graph_index)
        positions = _layout_for_graph(data, seed=9100 + row_index)
        example_metadata: dict[str, Any] = {
            "collection": collection,
            "row_label": str(example["row_label"]),
            "graph_index": graph_index,
            "lambda_org": lambda_org,
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.edge_index.shape[1] // 2),
            "criteria": {},
        }
        axes[row_index, 0].text(
            -0.17,
            0.5,
            f"{example['row_label']}\n"
            rf"$n={int(data.num_nodes)}$, $m={int(data.edge_index.shape[1] // 2)}$",
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=8,
        )
        for col_index, criterion in enumerate(CRITERIA):
            row = _matching_row(
                rows,
                collection=collection,
                graph_index=graph_index,
                lambda_org=lambda_org,
                criterion=criterion,
            )
            assignments = _load_assignments(experiment_dir=experiment_dir, row=row)
            _draw_panel(
                axis=axes[row_index, col_index],
                data=data,
                positions=positions,
                assignments=assignments,
                row=row,
                color_seed=1000 + 97 * row_index + 17 * col_index,
            )
            example_metadata["criteria"][criterion] = {
                "partition_artifact": row["partition_artifact"],
                "hard_D_E": float(row["hard_D_E"]),
                "hard_D_F": float(row["hard_D_F"]),
                "hard_D_C": float(row["hard_D_C"]),
                "hard_D_H2": float(row["hard_D_H2"]),
                "hard_k_eff": float(row["hard_k_eff"]),
                "hard_max_volume_fraction": float(row["hard_max_volume_fraction"]),
                "hard_active_partitions": int(float(row["hard_active_partitions"])),
                "assignments": assignments.tolist(),
            }
        metadata["examples"].append(example_metadata)

    figure_path = output_dir / "transductive_relational_distortion_partition_examples.png"
    metadata_path = output_dir / "transductive_relational_distortion_partition_examples.json"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment_dir",
        type=Path,
        nargs="?",
        default=Path("out/experiments/trd_corrected_lr0p05_s600_four_collections"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("paper/graphics"))
    args = parser.parse_args()
    plot_partition_examples(experiment_dir=args.experiment_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
