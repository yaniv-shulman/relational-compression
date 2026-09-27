"""Command-line utilities for reproducible experiment workflows."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np

from relational_compression.experiments.transductive_relational_distortion.source_geometry import (
    COLLISION,
    COLLISION_ENTROPY,
    EDGE,
    FOURIER,
)

CRITERION_STEMS = {
    EDGE: "D_E",
    FOURIER: "D_F",
    COLLISION: "D_C",
    COLLISION_ENTROPY: "D_H2",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read csv."""
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object."""
    return cast(dict[str, Any], json.loads(path.read_text()))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write csv."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    """Write json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def _float(value: Any) -> float:
    """Compute float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _finite(values: list[float]) -> np.ndarray:
    """Compute finite."""
    return np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)


def _mean(values: list[float]) -> float:
    """Compute mean."""
    finite = _finite(values)
    return float(finite.mean()) if finite.size else float("nan")


def _std(values: list[float]) -> float:
    """Compute std."""
    finite = _finite(values)
    if not finite.size:
        return float("nan")
    return float(finite.std(ddof=1 if finite.size > 1 else 0))


def _source_label(collection: str, source_variant: str) -> str:
    """Compute source label."""
    if collection == "malnet_unweighted":
        return "MalNet unweighted"
    if collection == "malnet_weighted":
        return "MalNet weighted"
    if collection == "collab":
        return "COLLAB"
    if collection == "proteins":
        return "PROTEINS"
    return f"{collection} {source_variant}".strip()


def _compact_aggregate(aggregate_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Compute compact aggregate."""
    compact: list[dict[str, Any]] = []
    for row in aggregate_rows:
        criterion = str(row["criterion"])
        stem = CRITERION_STEMS[criterion]
        compact.append(
            {
                "source": _source_label(collection=str(row["collection"]), source_variant=str(row["source_variant"])),
                "collection": str(row["collection"]),
                "source_variant": str(row["source_variant"]),
                "split": str(row["split"]),
                "criterion": criterion,
                "lambda_org": _float(row["lambda_org"]),
                "num_graphs": int(_float(row["num_graphs"])),
                "hard_h2_mean": _float(row["hard_h2_mean"]),
                "hard_k_eff_mean": _float(row["hard_k_eff_mean"]),
                "hard_own_distortion_mean": _float(row[f"hard_{stem}_mean"]),
                "hard_own_distortion_std": _float(row[f"hard_{stem}_std"]),
                "soft_h2_mean": _float(row["soft_h2_mean"]),
                "soft_k_eff_mean": _float(row["soft_k_eff_mean"]),
                "soft_own_distortion_mean": _float(row[f"soft_{stem}_mean"]),
                "soft_own_distortion_std": _float(row[f"soft_{stem}_std"]),
                "assignment_confidence_mean": _float(row["assignment_confidence_mean"]),
                "hard_max_volume_fraction_mean": _float(row["hard_max_volume_fraction_mean"]),
            }
        )
    return compact


def _transition_summary(compact_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute transition summary."""
    thresholds = (4.0, 6.0)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in compact_rows:
        groups[(str(row["source"]), str(row["criterion"]))].append(row)
    summary: list[dict[str, Any]] = []
    for (source, criterion), rows in sorted(groups.items()):
        rows.sort(key=lambda item: float(item["lambda_org"]))
        for threshold in thresholds:
            first = next((row for row in rows if float(row["hard_k_eff_mean"]) >= threshold), None)
            summary.append(
                {
                    "source": source,
                    "criterion": criterion,
                    "hard_k_eff_threshold": threshold,
                    "first_lambda_org": "" if first is None else float(first["lambda_org"]),
                }
            )
    return summary


def _pair_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Compute pair summary."""
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            _source_label(collection=str(row["collection"]), source_variant=str(row["source_variant"])),
            str(row["criterion_a"]),
            str(row["criterion_b"]),
            str(row.get("split", "")),
        )
        grouped[key].append(row)
    summary: list[dict[str, Any]] = []
    for (source, criterion_a, criterion_b, split), group in sorted(grouped.items()):
        item: dict[str, Any] = {
            "source": source,
            "split": split,
            "criterion_a": criterion_a,
            "criterion_b": criterion_b,
            "num_graphs": len(group),
        }
        for metric in ("pearson", "spearman", "cosine", "l1_distance", "l2_distance"):
            values = [_float(row.get(metric)) for row in group]
            item[f"{metric}_mean"] = _mean(values)
            item[f"{metric}_std"] = _std(values)
        summary.append(item)
    return summary


def _cross_objective_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    """Compute cross objective summary."""
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                _source_label(collection=str(row["collection"]), source_variant=str(row["source_variant"])),
                str(row["target_criterion"]),
            )
        ].append(row)
    summary: list[dict[str, Any]] = []
    for (source, target), group in sorted(grouped.items()):
        misses = sum(int(_float(row.get("candidate_beats_target", 0))) for row in group)
        advantages = [_float(row.get("candidate_advantage")) for row in group]
        summary.append(
            {
                "source": source,
                "target_criterion": target,
                "num_comparisons": len(group),
                "candidate_beats_target": misses,
                "candidate_beats_target_fraction": misses / len(group) if group else float("nan"),
                "candidate_advantage_mean": _mean(advantages),
            }
        )
    return summary


def _lambda_slices(compact_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute lambda slices."""
    wanted = {0.10, 0.20, 0.50}
    return [row for row in compact_rows if any(math.isclose(float(row["lambda_org"]), value) for value in wanted)]


def _validate_ranges(
    per_graph_rows: list[dict[str, str]],
    *,
    num_partitions: int,
) -> dict[str, Any]:
    """Validate ranges against the configured codeword count."""
    if num_partitions < 1:
        raise ValueError("num_partitions must be at least one")
    tolerance = 5e-5
    log_k = math.log(float(num_partitions))
    distortion_fields = (
        "soft_D_E",
        "hard_D_E",
        "soft_D_F",
        "hard_D_F",
        "soft_D_C",
        "hard_D_C",
        "soft_D_H2",
        "hard_D_H2",
    )
    range_violations: list[dict[str, Any]] = []
    identity_errors: list[float] = []
    foster_errors: list[float] = []
    for row_index, row in enumerate(per_graph_rows):
        for field in distortion_fields:
            value = _float(row.get(field))
            if math.isfinite(value) and (value < -tolerance or value > 1.0 + tolerance):
                range_violations.append({"row": row_index, "field": field, "value": value})
        for field, lower, upper in (
            ("hard_h2", 0.0, log_k),
            ("soft_h2", 0.0, log_k),
            ("hard_k_eff", 1.0, float(num_partitions)),
            ("soft_k_eff", 1.0, float(num_partitions)),
            ("marginal_d2", 0.0, log_k),
        ):
            value = _float(row.get(field))
            if math.isfinite(value) and (value < lower - tolerance or value > upper + tolerance):
                range_violations.append({"row": row_index, "field": field, "value": value})
        soft_h2 = _float(row.get("soft_h2"))
        marginal_d2 = _float(row.get("marginal_d2"))
        if math.isfinite(soft_h2) and math.isfinite(marginal_d2):
            identity_errors.append(abs((soft_h2 + marginal_d2) - log_k))
        foster_sum = _float(row.get("source_foster_sum"))
        foster_expected = _float(row.get("source_foster_expected"))
        if math.isfinite(foster_sum) and math.isfinite(foster_expected):
            foster_errors.append(abs(foster_sum - foster_expected))
    if range_violations:
        raise ValueError(f"Found {len(range_violations)} numerical range violations; first={range_violations[0]}")
    return {
        "range_violation_count": 0,
        "soft_h2_plus_d2_minus_log_k_max_abs": max(identity_errors, default=float("nan")),
        "foster_max_abs_error": max(foster_errors, default=float("nan")),
    }


def analyze_experiment(experiment_dir: Path, *, output_dir: Path) -> dict[str, Any]:
    """Analyze transductive results from an experiment directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    effective_config = _read_json(experiment_dir / "effective_config.json")
    try:
        num_partitions = int(effective_config["num_partitions"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("effective_config.json must define an integer num_partitions") from error

    aggregate = _read_csv(experiment_dir / "aggregate_results.csv")
    per_graph = _read_csv(experiment_dir / "per_graph_results.csv")
    source_rho = _read_csv(experiment_dir / "source_rho_correlations.csv")
    random_partition = _read_csv(experiment_dir / "random_partition_distortion_correlations.csv")
    cross_objective = _read_csv(experiment_dir / "cross_objective_diagnostics.csv")
    exclusions = _read_csv(experiment_dir / "criterion_exclusions.csv")

    compact = _compact_aggregate(aggregate)
    source_rho_summary = _pair_summary(source_rho)
    random_partition_summary = _pair_summary(random_partition)
    cross_summary = _cross_objective_summary(cross_objective)
    validation = _validate_ranges(per_graph, num_partitions=num_partitions)
    validation["post_selection_cross_objective_misses"] = sum(
        int(_float(row.get("candidate_beats_target", 0))) for row in cross_objective
    )
    validation["collision_entropy_exclusion_count"] = sum(
        1 for row in exclusions if str(row.get("criterion")) == COLLISION_ENTROPY
    )
    if validation["post_selection_cross_objective_misses"]:
        raise ValueError("Post-selection cross-objective misses are nonzero")

    _write_csv(path=output_dir / "final_analysis_aggregate_compact.csv", rows=compact)
    _write_csv(path=output_dir / "final_analysis_transition_summary.csv", rows=_transition_summary(compact))
    _write_csv(path=output_dir / "final_analysis_source_rho_pair_summary.csv", rows=source_rho_summary)
    _write_csv(path=output_dir / "final_analysis_random_partition_pair_summary.csv", rows=random_partition_summary)
    _write_csv(path=output_dir / "final_analysis_cross_objective_summary.csv", rows=cross_summary)
    _write_csv(path=output_dir / "final_analysis_main_lambda_slices.csv", rows=_lambda_slices(compact))

    summary = {
        "experiment_dir": str(experiment_dir),
        "num_per_graph_rows": len(per_graph),
        "num_aggregate_rows": len(aggregate),
        "validation": validation,
    }
    _write_json(path=output_dir / "final_analysis_summary.json", value=summary)
    return summary


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="Analyze transductive relational-distortion outputs")
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    experiment_dir = args.experiment_dir.absolute()
    output_dir = args.output_dir.absolute() if args.output_dir is not None else experiment_dir / "analysis"
    summary = analyze_experiment(experiment_dir, output_dir=output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
