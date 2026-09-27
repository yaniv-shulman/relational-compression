"""Experiment support code for relational compression studies."""

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from torch import Tensor
from torch_geometric.data import Data

from relational_compression.experiments.inductive_normalized_cut.metrics import hard_normalized_cut
from relational_compression.experiments.transductive_relational_distortion.objectives import (
    hard_evaluation,
    soft_objective,
)
from relational_compression.experiments.transductive_relational_distortion.optimize import OptimizedPartition
from relational_compression.experiments.transductive_relational_distortion.source_geometry import (
    COLLISION,
    COLLISION_ENTROPY,
    EDGE,
    FOURIER,
    SOURCE_CRITERIA,
    SourceGeometry,
)


def _to_float(value: Tensor | float | int) -> float:
    """Compute to float."""
    if isinstance(value, Tensor):
        return float(value.detach().cpu())
    return float(value)


def _json_default(value: Any) -> Any:
    """Serialize supported nonstandard values for JSON output."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def evaluate_optimized_partition(
    *,
    data: Data,
    geometry: SourceGeometry,
    optimized: OptimizedPartition,
    split: str,
    graph_index: int,
    num_partitions: int,
    optimization_steps: int,
    temperature: float,
    artifact_path: Path | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate optimized partition."""
    assignments = optimized.assignments.cpu()
    hard_source = hard_evaluation(assignments=assignments, geometry=geometry, num_partitions=int(num_partitions))
    edge_weight = getattr(data, "edge_weight", None)
    hard_ncut = hard_normalized_cut(
        assignments=assignments,
        edge_index=data.edge_index.cpu(),
        num_partitions=int(num_partitions),
        edge_weight=None if edge_weight is None else edge_weight.cpu(),
    )
    soft = optimized.soft_result
    objective_values: dict[str, float] = {}
    for criterion in SOURCE_CRITERIA:
        if criterion in geometry.defined_criteria:
            objective_values[criterion] = _to_float(
                soft_objective(
                    logits=optimized.logits,
                    geometry=geometry,
                    criterion=criterion,
                    lambda_org=float(optimized.lambda_org),
                    temperature=float(temperature),
                ).loss
            )
        else:
            objective_values[criterion] = float("nan")
    q_bar = soft.q_bar.detach().cpu()
    restart = optimized.restart_summaries[optimized.selected_restart]
    row = {
        "collection": str(getattr(data, "collection", "unknown")),
        "source_variant": str(getattr(data, "source_variant", "unknown")),
        "split": split,
        "dataset_index": int(getattr(data, "dataset_index", graph_index)),
        "graph_index": int(graph_index),
        "original_graph_index": int(getattr(data, "original_graph_index", graph_index)),
        "weight_seed": int(getattr(data, "weight_seed", -1)),
        "weight_sigma": _to_float(getattr(data, "weight_sigma", float("nan"))),
        "criterion": optimized.criterion,
        "lambda_org": float(optimized.lambda_org),
        "selected_restart": int(optimized.selected_restart),
        "selected_restart_kind": str(restart.init_kind),
        "selected_seed": int(optimized.selected_seed),
        "num_nodes": int(data.num_nodes),
        "num_edges": int(geometry.edge_index.shape[1]),
        "soft_total_loss": _to_float(soft.loss),
        "soft_own_distortion": _to_float(soft.own_distortion),
        "hard_D_E": float(hard_source.edge_distortion),
        "hard_D_F": float(hard_source.fourier_distortion),
        "hard_D_C": float(hard_source.collision_distortion),
        "hard_D_H2": float(hard_source.collision_entropy_distortion),
        "hard_ncut": _to_float(hard_ncut.ncut_per_graph[0]),
        "hard_nassoc": _to_float(hard_ncut.nassoc_per_graph[0]),
        "soft_D_E": _to_float(soft.edge_distortion),
        "soft_D_F": _to_float(soft.fourier_distortion),
        "soft_D_C": _to_float(soft.collision_distortion),
        "soft_D_H2": _to_float(soft.collision_entropy_distortion),
        "soft_total_loss_edge_objective": objective_values[EDGE],
        "soft_total_loss_fourier_objective": objective_values[FOURIER],
        "soft_total_loss_collision_objective": objective_values[COLLISION],
        "soft_total_loss_collision_entropy_objective": objective_values[COLLISION_ENTROPY],
        "hard_h2": float(hard_source.hard_h2),
        "hard_k_eff": float(hard_source.hard_k_eff),
        "soft_h2": _to_float(soft.soft_h2),
        "soft_k_eff": _to_float(soft.soft_k_eff),
        "marginal_d2": _to_float(soft.marginal_d2),
        "hard_active_partitions": int(hard_source.active_partitions),
        "hard_max_volume_fraction": float(hard_source.max_volume_fraction),
        "hard_min_volume_fraction": float(hard_source.min_volume_fraction),
        "hard_within_edge_fraction": _to_float(hard_ncut.within_edge_fraction_per_graph[0]),
        "assignment_confidence": _to_float(soft.assignment_confidence),
        "assignment_entropy": _to_float(soft.assignment_entropy),
        "optimization_steps": int(optimization_steps),
        "final_temperature": float(temperature),
        "selected_restart_final_loss": float(restart.final_loss),
        "selected_restart_best_loss": float(restart.best_loss),
        "selected_restart_best_step": int(restart.best_step),
        "selected_restart_encountered_nonfinite": int(restart.encountered_nonfinite),
        "selected_restart_final_marginal_d2": float(restart.final_marginal_d2),
        "selected_restart_best_marginal_d2": float(restart.best_marginal_d2),
        "source_foster_sum": float(geometry.foster_sum),
        "source_foster_expected": float(max(0, geometry.num_nodes - 1)),
        "source_edge_denominator": float(geometry.metadata["edge_denominator"]),
        "source_collision_trace": float(geometry.collision_trace),
        "source_collision_entropy_denominator": float(geometry.collision_entropy_denominator),
        "source_collision_entropy_defined": int(geometry.rho_collision_entropy is not None),
        "partition_artifact": "" if artifact_path is None else str(artifact_path),
        "soft_q_bar": json.dumps(q_bar.tolist()),
        "hard_volume_fractions": json.dumps(hard_source.volume_fractions.tolist()),
        "restart_summaries": json.dumps([summary.__dict__ for summary in optimized.restart_summaries]),
    }
    if extra_fields is not None:
        row.update(extra_fields)
    return row


def summarize_values(values: list[float]) -> dict[str, float]:
    """Summarize values."""
    finite = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
    if finite.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    ddof = 1 if finite.size > 1 else 0
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=ddof)),
        "median": float(np.median(finite)),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate rows."""
    metrics = (
        "soft_total_loss",
        "soft_own_distortion",
        "hard_D_E",
        "hard_D_F",
        "hard_D_C",
        "hard_D_H2",
        "hard_ncut",
        "soft_D_E",
        "soft_D_F",
        "soft_D_C",
        "soft_D_H2",
        "soft_total_loss_edge_objective",
        "soft_total_loss_fourier_objective",
        "soft_total_loss_collision_objective",
        "soft_total_loss_collision_entropy_objective",
        "hard_h2",
        "hard_k_eff",
        "soft_h2",
        "soft_k_eff",
        "marginal_d2",
        "hard_active_partitions",
        "hard_max_volume_fraction",
        "hard_min_volume_fraction",
        "hard_within_edge_fraction",
        "assignment_confidence",
        "assignment_entropy",
    )
    groups = sorted(
        {
            (
                str(row["collection"]),
                str(row["source_variant"]),
                str(row["split"]),
                str(row["criterion"]),
                float(row["lambda_org"]),
            )
            for row in rows
        }
    )
    aggregated: list[dict[str, Any]] = []
    for collection, source_variant, split, criterion, lambda_org in groups:
        group_rows = [
            row
            for row in rows
            if str(row["collection"]) == collection
            and str(row["source_variant"]) == source_variant
            and str(row["split"]) == split
            and str(row["criterion"]) == criterion
            and float(row["lambda_org"]) == float(lambda_org)
        ]
        item: dict[str, Any] = {
            "collection": collection,
            "source_variant": source_variant,
            "split": split,
            "criterion": criterion,
            "lambda_org": float(lambda_org),
            "num_graphs": len(group_rows),
        }
        for metric in metrics:
            summary = summarize_values([float(row[metric]) for row in group_rows])
            for key, value in summary.items():
                item[f"{metric}_{key}"] = value
        aggregated.append(item)
    return aggregated


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write csv."""
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    """Write json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=_json_default))
