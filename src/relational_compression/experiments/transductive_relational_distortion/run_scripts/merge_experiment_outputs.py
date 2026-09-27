"""Command-line utilities for reproducible experiment workflows."""

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

from relational_compression.experiments.transductive_relational_distortion.evaluate import (
    aggregate_rows,
    write_csv,
    write_json,
)
from relational_compression.experiments.transductive_relational_distortion.run_scripts.run_experiment import (
    _cross_objective_diagnostics,
    _cross_objective_miss_counts,
)
from relational_compression.experiments.transductive_relational_distortion.source_geometry import COLLISION_ENTROPY


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read csv."""
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> Any:
    """Read json."""
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _float_or_nan(value: Any) -> float:
    """Compute float or nan."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _summarize_metric_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Summarize metric rows."""
    result: dict[str, float] = {}
    if not rows:
        return result
    skip = {
        "collection",
        "source_variant",
        "split",
        "criterion",
        "criterion_a",
        "criterion_b",
        "dataset_index",
        "graph_index",
        "original_graph_index",
    }
    keys = sorted({key for row in rows for key in row if key not in skip})
    for key in keys:
        values = [_float_or_nan(row.get(key)) for row in rows]
        finite = [value for value in values if math.isfinite(value)]
        if not finite:
            continue
        count = len(finite)
        mean = sum(finite) / count
        variance = sum((value - mean) ** 2 for value in finite) / (count - 1 if count > 1 else 1)
        result[f"{key}_mean"] = mean
        result[f"{key}_std"] = math.sqrt(variance)
    return result


def _copy_partition_artifacts(rows: list[dict[str, Any]], *, input_dir: Path, output_dir: Path) -> None:
    """Compute copy partition artifacts."""
    for row in rows:
        relative = str(row.get("partition_artifact", ""))
        if not relative:
            continue
        source = input_dir / relative
        destination = output_dir / relative
        if not source.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _merge_selected_graphs(input_dirs: list[Path]) -> list[dict[str, Any]]:
    """Merge selected graphs."""
    rows: list[dict[str, Any]] = []
    for input_dir in input_dirs:
        selected = _read_json(input_dir / "selected_graphs.json")
        if isinstance(selected, list):
            rows.extend(selected)
    return rows


def _merge_collection_metadata(input_dirs: list[Path]) -> dict[str, Any]:
    """Merge collection metadata."""
    merged: dict[str, Any] = {"source_experiment_dirs": [str(path) for path in input_dirs]}
    collections: dict[str, Any] = {}
    for input_dir in input_dirs:
        metadata = _read_json(input_dir / "collection_metadata.json")
        if isinstance(metadata, dict):
            collections.update(metadata)
    merged["collections"] = collections
    return merged


def _merge_effective_config(input_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    """Merge effective config."""
    config = _read_json(input_dirs[0] / "effective_config.json")
    if not isinstance(config, dict):
        return {"source_experiment_dirs": [str(path) for path in input_dirs]}
    config = dict(config)
    config["experiment_name"] = output_dir.name
    config["experiment_dir"] = str(output_dir)
    config["merged_source_experiment_dirs"] = [str(path) for path in input_dirs]
    collections: list[str] = []
    for input_dir in input_dirs:
        source_config = _read_json(input_dir / "effective_config.json")
        if isinstance(source_config, dict):
            collections.extend(str(value) for value in source_config.get("source_collections", []))
    if collections:
        config["source_collections"] = sorted(set(collections))
    return config


def merge_experiment_outputs(input_dirs: list[Path], *, output_dir: Path) -> dict[str, Any]:
    """Merge experiment outputs."""
    input_dirs = [path.absolute() for path in input_dirs]
    output_dir = output_dir.absolute()
    output_dir.mkdir(parents=True, exist_ok=True)

    per_graph_rows: list[dict[str, Any]] = []
    source_rho_rows: list[dict[str, Any]] = []
    random_partition_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    for input_dir in input_dirs:
        rows = _read_csv(input_dir / "per_graph_results.csv")
        _copy_partition_artifacts(rows, input_dir=input_dir, output_dir=output_dir)
        per_graph_rows.extend(rows)
        source_rho_rows.extend(_read_csv(input_dir / "source_rho_correlations.csv"))
        random_partition_rows.extend(_read_csv(input_dir / "random_partition_distortion_correlations.csv"))
        exclusion_rows.extend(_read_csv(input_dir / "criterion_exclusions.csv"))

    if not per_graph_rows:
        raise ValueError("No per-graph rows found in input experiment directories")

    aggregate = aggregate_rows(per_graph_rows)
    cross_objective_rows = _cross_objective_diagnostics(per_graph_rows)
    cross_objective_miss_counts = _cross_objective_miss_counts(cross_objective_rows)
    cross_objective_misses = int(sum(cross_objective_miss_counts.values()))
    diagnostic_summaries = {
        "source_rho": _summarize_metric_rows(source_rho_rows),
        "random_partition": _summarize_metric_rows(random_partition_rows),
        "cross_objective": _summarize_metric_rows(cross_objective_rows),
        "post_selection_cross_objective_misses": cross_objective_misses,
        "post_selection_cross_objective_misses_by_target": cross_objective_miss_counts,
        "exclusions": {
            "undefined_source_distortion_count": len(exclusion_rows),
            "undefined_collision_entropy_count": sum(
                1 for row in exclusion_rows if row.get("criterion") == COLLISION_ENTROPY
            ),
        },
    }

    write_csv(path=output_dir / "per_graph_results.csv", rows=per_graph_rows)
    write_csv(path=output_dir / "aggregate_results.csv", rows=aggregate)
    if source_rho_rows:
        write_csv(path=output_dir / "source_rho_correlations.csv", rows=source_rho_rows)
    if random_partition_rows:
        write_csv(path=output_dir / "random_partition_distortion_correlations.csv", rows=random_partition_rows)
    if cross_objective_rows:
        write_csv(path=output_dir / "cross_objective_diagnostics.csv", rows=cross_objective_rows)
    if exclusion_rows:
        write_csv(path=output_dir / "criterion_exclusions.csv", rows=exclusion_rows)
    write_json(path=output_dir / "aggregate_results.json", value=aggregate)
    write_json(path=output_dir / "diagnostic_summaries.json", value=diagnostic_summaries)
    write_json(path=output_dir / "collection_metadata.json", value=_merge_collection_metadata(input_dirs))
    selected_graphs = _merge_selected_graphs(input_dirs)
    if selected_graphs:
        write_json(path=output_dir / "selected_graphs.json", value=selected_graphs)
    write_json(
        path=output_dir / "effective_config.json",
        value=_merge_effective_config(input_dirs=input_dirs, output_dir=output_dir),
    )

    result = {
        "experiment_dir": str(output_dir),
        "source_experiment_dirs": [str(path) for path in input_dirs],
        "num_rows": len(per_graph_rows),
        "num_graphs": len(
            {(row["collection"], row["source_variant"], row["split"], row["dataset_index"]) for row in per_graph_rows}
        ),
        "aggregate": aggregate,
        "diagnostics": diagnostic_summaries,
    }
    write_json(path=output_dir / "result.json", value=result)
    if cross_objective_misses:
        raise RuntimeError(
            "Post-selection cross-objective invariant failed after merge: "
            f"{cross_objective_misses} selected candidate misses exceed {1e-6}"
        )
    return result


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="Merge transductive relational-distortion experiment outputs")
    parser.add_argument("input_dirs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = merge_experiment_outputs(args.input_dirs, output_dir=args.output_dir)
    print(json.dumps({"experiment_dir": result["experiment_dir"], "num_rows": result["num_rows"]}, indent=2))


if __name__ == "__main__":
    main()
