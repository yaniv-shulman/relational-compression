"""Command-line utilities for reproducible experiment workflows."""

# ruff: noqa: E402,I001
import argparse
import csv
import itertools
import json
import math
import os
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/relational_compression_mplconfig")

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import rankdata
from sklearn.metrics import adjusted_rand_score

from relational_compression.experiments.transductive_relational_distortion.source_geometry import (
    COLLISION,
    COLLISION_ENTROPY,
    EDGE,
    FOURIER,
)

CRITERIA = (EDGE, FOURIER, COLLISION, COLLISION_ENTROPY)
CRITERION_LABELS = {
    EDGE: r"$D_E$",
    FOURIER: r"$D_F$",
    COLLISION: r"$D_C$",
    COLLISION_ENTROPY: r"$D_{H_2}$",
}
CRITERION_SHORT = {
    EDGE: "D_E",
    FOURIER: "D_F",
    COLLISION: "D_C",
    COLLISION_ENTROPY: "D_H2",
}
SOURCE_ORDER = ("MalNet unweighted", "MalNet weighted", "COLLAB", "PROTEINS")
COMPLETE_CASE_SETS = {
    "four_way": CRITERIA,
    "edge_fourier_collision_three_way": (EDGE, FOURIER, COLLISION),
}
MAIN_RATE_MATCHED_PAIRS = (
    (FOURIER, COLLISION),
    (FOURIER, COLLISION_ENTROPY),
    (COLLISION, COLLISION_ENTROPY),
)
R_TARGETS = (0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00)
PRIMARY_TOLERANCE = 0.10
SENSITIVITY_TOLERANCE = 0.15
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 1337
MIN_FIGURE_GRAPHS = 5
EPS = 1e-6


@dataclass(frozen=True)
class Match:
    """Store Match values."""

    row: dict[str, str]
    distance: float

    @property
    def criterion(self) -> str:
        """Compute criterion."""
        return self.row["criterion"]

    @property
    def hard_h2(self) -> float:
        """Compute hard h2."""
        return _float(self.row["hard_h2"])

    @property
    def lambda_org(self) -> float:
        """Compute lambda org."""
        return _float(self.row["lambda_org"])


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read csv."""
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write csv."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    """Write json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True))


def _float(value: Any) -> float:
    """Compute float."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def _safe_mean(values: list[float]) -> float:
    """Compute safe mean."""
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    return float(finite.mean()) if finite.size else float("nan")


def _safe_std(values: list[float]) -> float:
    """Compute safe std."""
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if not finite.size:
        return float("nan")
    return float(finite.std(ddof=1 if finite.size > 1 else 0))


def _safe_quantile(values: list[float], q: float) -> float:
    """Compute safe quantile."""
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    return float(np.quantile(a=finite, q=q)) if finite.size else float("nan")


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


def _source_slug(source: str) -> str:
    """Compute source slug."""
    return source.lower().replace(" ", "_").replace("-", "_")


def _criterion_pair_label(first: str, second: str) -> str:
    """Compute criterion pair label."""
    return f"{CRITERION_LABELS[first]} vs. {CRITERION_LABELS[second]}"


def _h2_to_keff(values: np.ndarray) -> np.ndarray:
    """Compute h2 to keff."""
    return np.asarray(np.exp(values))


def _keff_to_h2(values: np.ndarray) -> np.ndarray:
    """Compute keff to h2."""
    return np.asarray(np.log(np.clip(values, 1e-12, None)))


def _graph_key(row: dict[str, str]) -> tuple[str, str, str, str, str, str]:
    """Compute graph key."""
    return (
        row["collection"],
        row["source_variant"],
        row["split"],
        row["graph_index"],
        row["dataset_index"],
        row.get("original_graph_index", ""),
    )


def _graph_identity(row: dict[str, str]) -> dict[str, Any]:
    """Compute graph identity."""
    return {
        "source": _source_label(collection=row["collection"], source_variant=row["source_variant"]),
        "collection": row["collection"],
        "source_variant": row["source_variant"],
        "split": row["split"],
        "graph_index": int(_float(row["graph_index"])),
        "dataset_index": int(_float(row["dataset_index"])),
        "original_graph_index": row.get("original_graph_index", ""),
        "num_nodes": int(_float(row["num_nodes"])),
        "num_edges": int(_float(row["num_edges"])),
    }


def _parse_json_list(value: str) -> list[float]:
    """Parse json list."""
    parsed = json.loads(value)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected list-valued JSON, got {type(parsed).__name__}")
    return [float(item) for item in parsed]


def _hard_distortion(row: dict[str, str], criterion: str) -> float:
    """Compute hard distortion."""
    return _float(row.get(f"hard_{CRITERION_SHORT[criterion]}"))


def _bootstrap_ci(values: list[float], *, rng: np.random.Generator) -> tuple[float, float]:
    """Compute bootstrap ci."""
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return float("nan"), float("nan")
    if finite.size == 1:
        return float(finite[0]), float(finite[0])
    indices = rng.integers(0, finite.size, size=(BOOTSTRAP_RESAMPLES, finite.size))
    means = finite[indices].mean(axis=1)
    return float(np.quantile(a=means, q=0.025)), float(np.quantile(a=means, q=0.975))


def average_ranks(values: dict[str, float]) -> dict[str, float]:
    """Compute average ranks."""
    criteria = [criterion for criterion, value in values.items() if math.isfinite(value)]
    if not criteria:
        return {}
    ranked = rankdata([values[criterion] for criterion in criteria], method="average")
    return {criterion: float(rank) for criterion, rank in zip(criteria, ranked, strict=True)}


def collision_jaccard(assignments_a: np.ndarray, assignments_b: np.ndarray) -> float:
    """Compute collision jaccard."""
    assignments_a = np.asarray(assignments_a)
    assignments_b = np.asarray(assignments_b)
    if assignments_a.ndim != 1 or assignments_b.ndim != 1:
        raise ValueError("assignments must be one-dimensional")
    if assignments_a.shape != assignments_b.shape:
        raise ValueError("assignments must have equal length")

    labels_a, inverse_a = np.unique(assignments_a, return_inverse=True)
    labels_b, inverse_b = np.unique(assignments_b, return_inverse=True)
    contingency = np.zeros((labels_a.size, labels_b.size), dtype=np.int64)
    np.add.at(contingency, (inverse_a, inverse_b), 1)

    counts_a = contingency.sum(axis=1)
    counts_b = contingency.sum(axis=0)
    intersection = _choose2(contingency).sum()
    pairs_a = _choose2(counts_a).sum()
    pairs_b = _choose2(counts_b).sum()
    union = pairs_a + pairs_b - intersection
    if union == 0:
        return float("nan")
    return float(intersection / union)


def _choose2(values: np.ndarray) -> np.ndarray:
    """Compute choose2."""
    values = np.asarray(values, dtype=np.int64)
    return np.asarray(values * (values - 1) // 2)


def select_rate_match(rows: list[dict[str, str]], r_target: float, tolerance: float) -> Match | None:
    """Select rate match."""
    candidates: list[tuple[float, float, dict[str, str]]] = []
    for row in rows:
        hard_h2 = _float(row["hard_h2"])
        if not math.isfinite(hard_h2):
            continue
        distance = abs(hard_h2 - float(r_target))
        if distance <= float(tolerance) + 1e-12:
            candidates.append((distance, _float(row["lambda_org"]), row))
    if not candidates:
        return None
    distance, _, row = min(candidates, key=lambda item: (item[0], item[1]))
    return Match(row=row, distance=distance)


def _load_assignments(experiment_dir: Path, row: dict[str, str], cache: dict[str, np.ndarray]) -> np.ndarray:
    """Load assignments."""
    relative = row["partition_artifact"]
    if relative not in cache:
        artifact = torch.load(experiment_dir / relative, map_location="cpu", weights_only=False)
        assignments = artifact["assignments"].detach().cpu().numpy()
        expected_nodes = int(_float(row["num_nodes"]))
        if assignments.shape != (expected_nodes,):
            raise ValueError(
                f"Artifact node-count mismatch for {relative}: got {assignments.shape}, expected {(expected_nodes,)}"
            )
        cache[relative] = assignments
    return cache[relative]


def _validate_artifact_and_h2(
    *,
    experiment_dir: Path,
    rows: list[dict[str, str]],
    artifact_cache: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Validate artifact and h2."""
    max_h2_error = 0.0
    checked = 0
    self_ari_errors = 0
    self_jaccard_errors = 0
    permutation_errors = 0
    for row in rows:
        assignments = _load_assignments(experiment_dir=experiment_dir, row=row, cache=artifact_cache)
        volume_fractions = np.asarray(_parse_json_list(row["hard_volume_fractions"]), dtype=np.float64)
        collision = float(np.square(volume_fractions).sum())
        recomputed_h2 = -math.log(max(collision, float(np.finfo(np.float64).tiny)))
        max_h2_error = max(max_h2_error, abs(recomputed_h2 - _float(row["hard_h2"])))

        if not math.isclose(adjusted_rand_score(assignments, assignments), 1.0):
            self_ari_errors += 1
        if not math.isclose(collision_jaccard(assignments_a=assignments, assignments_b=assignments), 1.0):
            self_jaccard_errors += 1
        permuted = assignments + 17
        if not math.isclose(adjusted_rand_score(assignments, permuted), 1.0):
            permutation_errors += 1
        if not math.isclose(collision_jaccard(assignments_a=assignments, assignments_b=permuted), 1.0):
            permutation_errors += 1
        checked += 1
    return {
        "validated_artifacts": checked,
        "hard_h2_from_volume_fractions_max_abs_error": max_h2_error,
        "self_ari_error_count": self_ari_errors,
        "self_collision_jaccard_error_count": self_jaccard_errors,
        "label_permutation_error_count": permutation_errors,
    }


def _build_matches(
    per_graph_rows: list[dict[str, str]], *, r_targets: tuple[float, ...], tolerance: float
) -> tuple[
    dict[tuple[tuple[str, str, str, str, str, str], str, float], Match], dict[tuple[str, str], list[dict[str, str]]]
]:
    """Build matches."""
    rows_by_graph_criterion: dict[tuple[tuple[str, str, str, str, str, str], str], list[dict[str, str]]] = defaultdict(
        list
    )
    rows_by_source_criterion: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in per_graph_rows:
        key = _graph_key(row)
        criterion = row["criterion"]
        source = _source_label(collection=row["collection"], source_variant=row["source_variant"])
        rows_by_graph_criterion[(key, criterion)].append(row)
        rows_by_source_criterion[(source, criterion)].append(row)

    matches: dict[tuple[tuple[str, str, str, str, str, str], str, float], Match] = {}
    for (key, criterion), rows in rows_by_graph_criterion.items():
        for target in r_targets:
            match = select_rate_match(rows=rows, r_target=target, tolerance=tolerance)
            if match is not None:
                matches[(key, criterion, target)] = match
    return matches, rows_by_source_criterion


def complete_rate_matched_case(
    matches: dict[tuple[tuple[str, str, str, str, str, str], str, float], Match],
    key: tuple[str, str, str, str, str, str],
    target: float,
    criteria: tuple[str, ...],
    tolerance: float,
) -> dict[str, Match] | None:
    """Compute complete rate matched case."""
    selected: dict[str, Match] = {}
    for criterion in criteria:
        match = matches.get((key, criterion, target))
        if match is None:
            return None
        if match.distance > tolerance + 1e-12:
            return None
        if not math.isfinite(match.hard_h2):
            return None
        selected[criterion] = match

    h2_values = [match.hard_h2 for match in selected.values()]
    if max(h2_values) - min(h2_values) > tolerance + 1e-12:
        return None
    return selected


def _complete_case_records(
    *,
    per_graph_rows: list[dict[str, str]],
    matches: dict[tuple[tuple[str, str, str, str, str, str], str, float], Match],
    r_targets: tuple[float, ...],
    tolerance: float,
    case_sets: dict[str, tuple[str, ...]] = COMPLETE_CASE_SETS,
) -> list[dict[str, Any]]:
    """Compute complete case records."""
    graph_rows = {_graph_key(row): row for row in per_graph_rows}
    records: list[dict[str, Any]] = []
    for key, reference_row in sorted(graph_rows.items()):
        for target in r_targets:
            for analysis_type, criteria in case_sets.items():
                selected = complete_rate_matched_case(
                    matches=matches, key=key, target=target, criteria=criteria, tolerance=tolerance
                )
                if selected is None:
                    continue
                if any(
                    not math.isfinite(_hard_distortion(row=match.row, criterion=evaluation))
                    for match in selected.values()
                    for evaluation in criteria
                ):
                    continue
                h2_values = [match.hard_h2 for match in selected.values()]
                records.append(
                    {
                        **_graph_identity(reference_row),
                        "analysis_type": analysis_type,
                        "criteria": criteria,
                        "criteria_set": ";".join(criteria),
                        "R_target": target,
                        "matches": selected,
                        "hard_h2_min": min(h2_values),
                        "hard_h2_max": max(h2_values),
                        "hard_h2_span": max(h2_values) - min(h2_values),
                        "lambda_org_values": ";".join(
                            f"{selected[criterion].lambda_org:.6g}" for criterion in criteria
                        ),
                        "h2_tolerance": tolerance,
                    }
                )
    return records


def _complete_case_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute complete case rows."""
    output: list[dict[str, Any]] = []
    for record in records:
        output.append({key: value for key, value in record.items() if key not in {"criteria", "matches"}})
    return output


def _complete_case_coverage_rows(
    *,
    per_graph_rows: list[dict[str, str]],
    matches: dict[tuple[tuple[str, str, str, str, str, str], str, float], Match],
    r_targets: tuple[float, ...],
    tolerance: float,
) -> list[dict[str, Any]]:
    """Compute complete case coverage rows."""
    available: dict[tuple[str, str], set[tuple[str, str, str, str, str, str]]] = defaultdict(set)
    for row in per_graph_rows:
        source = _source_label(collection=row["collection"], source_variant=row["source_variant"])
        available[(source, row["criterion"])].add(_graph_key(row))

    output: list[dict[str, Any]] = []
    for source in SOURCE_ORDER:
        for target in r_targets:
            for analysis_type, criteria in COMPLETE_CASE_SETS.items():
                eligible = set.intersection(*(available.get((source, criterion), set()) for criterion in criteria))
                individually_matched = [
                    key for key in eligible if all((key, criterion, target) in matches for criterion in criteria)
                ]
                spans = []
                complete_count = 0
                for key in eligible:
                    selected = complete_rate_matched_case(
                        matches=matches, key=key, target=target, criteria=criteria, tolerance=tolerance
                    )
                    if selected is None:
                        continue
                    if any(
                        not math.isfinite(_hard_distortion(row=match.row, criterion=evaluation))
                        for match in selected.values()
                        for evaluation in criteria
                    ):
                        continue
                    h2_values = [match.hard_h2 for match in selected.values()]
                    spans.append(max(h2_values) - min(h2_values))
                    complete_count += 1
                output.append(
                    {
                        "analysis_type": analysis_type,
                        "source": source,
                        "R_target": target,
                        "criteria_set": ";".join(criteria),
                        "eligible_graphs": len(eligible),
                        "individual_complete_matches": len(individually_matched),
                        "complete_rate_matched_cases": complete_count,
                        "complete_fraction": complete_count / len(eligible) if eligible else float("nan"),
                        "mean_complete_h2_span": _safe_mean(spans),
                        "max_complete_h2_span": max(spans, default=float("nan")),
                        "h2_tolerance": tolerance,
                    }
                )
    return output


def _coverage_rows(
    *,
    per_graph_rows: list[dict[str, str]],
    matches: dict[tuple[tuple[str, str, str, str, str, str], str, float], Match],
    r_targets: tuple[float, ...],
    tolerance: float,
) -> list[dict[str, Any]]:
    """Compute coverage rows."""
    graph_rows: dict[tuple[str, str, str, str, str, str], dict[str, str]] = {}
    available: dict[tuple[str, str], set[tuple[str, str, str, str, str, str]]] = defaultdict(set)
    for row in per_graph_rows:
        key = _graph_key(row)
        source = _source_label(collection=row["collection"], source_variant=row["source_variant"])
        graph_rows[key] = row
        available[(source, row["criterion"])].add(key)

    output: list[dict[str, Any]] = []
    for source in SOURCE_ORDER:
        for target in r_targets:
            for criterion in CRITERIA:
                keys = available.get((source, criterion), set())
                selected = [matches[(key, criterion, target)] for key in keys if (key, criterion, target) in matches]
                distances = [match.distance for match in selected]
                output.append(
                    {
                        "row_type": "criterion",
                        "source": source,
                        "R_target": target,
                        "criterion": criterion,
                        "eligible_graphs": len(keys),
                        "matched_graphs": len(selected),
                        "match_fraction": len(selected) / len(keys) if keys else float("nan"),
                        "mean_abs_h2_error": _safe_mean(distances),
                        "max_abs_h2_error": max(distances, default=float("nan")),
                        "h2_tolerance": tolerance,
                    }
                )

            for first, second in itertools.combinations(CRITERIA, 2):
                first_keys = available.get((source, first), set())
                second_keys = available.get((source, second), set())
                eligible = first_keys & second_keys
                both_individual = [
                    key for key in eligible if (key, first, target) in matches and (key, second, target) in matches
                ]
                pair_diffs = [
                    abs(matches[(key, first, target)].hard_h2 - matches[(key, second, target)].hard_h2)
                    for key in both_individual
                    if abs(matches[(key, first, target)].hard_h2 - matches[(key, second, target)].hard_h2)
                    <= tolerance + 1e-12
                ]
                output.append(
                    {
                        "row_type": "pair",
                        "source": source,
                        "R_target": target,
                        "criterion_a": first,
                        "criterion_b": second,
                        "eligible_graphs": len(eligible),
                        "both_individual_matches": len(both_individual),
                        "matched_graphs": len(pair_diffs),
                        "match_fraction": len(pair_diffs) / len(eligible) if eligible else float("nan"),
                        "mean_pair_abs_h2_difference": _safe_mean(pair_diffs),
                        "max_pair_abs_h2_difference": max(pair_diffs, default=float("nan")),
                        "h2_tolerance": tolerance,
                    }
                )
    return output


def _accepted_pair_matches(
    matches: dict[tuple[tuple[str, str, str, str, str, str], str, float], Match],
    key: tuple[str, str, str, str, str, str],
    first: str,
    second: str,
    target: float,
    tolerance: float,
) -> tuple[Match, Match] | None:
    """Compute accepted pair matches."""
    first_match = matches.get((key, first, target))
    second_match = matches.get((key, second, target))
    if first_match is None or second_match is None:
        return None
    if abs(first_match.hard_h2 - second_match.hard_h2) > tolerance + 1e-12:
        return None
    return first_match, second_match


def _partition_overlap_rows(
    *,
    experiment_dir: Path,
    per_graph_rows: list[dict[str, str]],
    matches: dict[tuple[tuple[str, str, str, str, str, str], str, float], Match],
    r_targets: tuple[float, ...],
    tolerance: float,
    artifact_cache: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Compute partition overlap rows."""
    graph_rows = {_graph_key(row): row for row in per_graph_rows}
    output: list[dict[str, Any]] = []
    for key, reference_row in sorted(graph_rows.items()):
        for target in r_targets:
            for first, second in itertools.combinations(CRITERIA, 2):
                accepted = _accepted_pair_matches(
                    matches=matches, key=key, first=first, second=second, target=target, tolerance=tolerance
                )
                if accepted is None:
                    continue
                first_match, second_match = accepted
                first_assignments = _load_assignments(
                    experiment_dir=experiment_dir, row=first_match.row, cache=artifact_cache
                )
                second_assignments = _load_assignments(
                    experiment_dir=experiment_dir, row=second_match.row, cache=artifact_cache
                )
                output.append(
                    {
                        **_graph_identity(reference_row),
                        "R_target": target,
                        "criterion_a": first,
                        "criterion_b": second,
                        "criterion_pair": _criterion_pair_label(first=first, second=second),
                        "hard_h2_a": first_match.hard_h2,
                        "hard_h2_b": second_match.hard_h2,
                        "abs_h2_difference": abs(first_match.hard_h2 - second_match.hard_h2),
                        "lambda_org_a": first_match.lambda_org,
                        "lambda_org_b": second_match.lambda_org,
                        "ari": float(adjusted_rand_score(first_assignments, second_assignments)),
                        "collision_jaccard": collision_jaccard(
                            assignments_a=first_assignments, assignments_b=second_assignments
                        ),
                        "h2_tolerance": tolerance,
                    }
                )
    return output


def _overlap_summary_rows(rows: list[dict[str, Any]], *, bootstrap_seed: int) -> list[dict[str, Any]]:
    """Compute overlap summary rows."""
    grouped: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["source"], row["criterion_a"], row["criterion_b"], float(row["R_target"]))].append(row)

    rng = np.random.default_rng(int(bootstrap_seed))
    output: list[dict[str, Any]] = []
    for (source, first, second, target), group in sorted(grouped.items()):
        item: dict[str, Any] = {
            "source": source,
            "criterion_a": first,
            "criterion_b": second,
            "criterion_pair": _criterion_pair_label(first=first, second=second),
            "R_target": target,
            "num_graphs": len(group),
        }
        for metric in ("ari", "collision_jaccard"):
            values = [_float(row[metric]) for row in group]
            low, high = _bootstrap_ci(values, rng=rng)
            item[f"{metric}_mean"] = _safe_mean(values)
            item[f"{metric}_std"] = _safe_std(values)
            item[f"{metric}_median"] = _safe_quantile(values=values, q=0.5)
            item[f"{metric}_q25"] = _safe_quantile(values=values, q=0.25)
            item[f"{metric}_q75"] = _safe_quantile(values=values, q=0.75)
            item[f"{metric}_bootstrap95_low"] = low
            item[f"{metric}_bootstrap95_high"] = high
        output.append(item)
    return output


def _cross_distortion_rows(
    *,
    per_graph_rows: list[dict[str, str]],
    matches: dict[tuple[tuple[str, str, str, str, str, str], str, float], Match],
    r_targets: tuple[float, ...],
    tolerance: float,
) -> list[dict[str, Any]]:
    """Compute cross distortion rows."""
    graph_rows = {_graph_key(row): row for row in per_graph_rows}
    regret_rows: list[dict[str, Any]] = []
    for key, reference_row in sorted(graph_rows.items()):
        for target in r_targets:
            matched = {
                criterion: matches[(key, criterion, target)]
                for criterion in CRITERIA
                if (key, criterion, target) in matches
            }
            for evaluation in CRITERIA:
                available_values: dict[str, float] = {}
                for training, match in matched.items():
                    if abs(match.hard_h2 - target) > tolerance + 1e-12:
                        continue
                    value = _hard_distortion(row=match.row, criterion=evaluation)
                    if math.isfinite(value):
                        available_values[training] = value
                if not available_values:
                    continue
                direct = matched.get(evaluation)
                if direct is None:
                    continue
                if abs(direct.hard_h2 - target) > tolerance + 1e-12:
                    continue
                direct_value = _hard_distortion(row=direct.row, criterion=evaluation)
                if not math.isfinite(direct_value):
                    continue
                for training, value in available_values.items():
                    if abs(matched[training].hard_h2 - direct.hard_h2) > tolerance + 1e-12:
                        continue
                    regret_rows.append(
                        {
                            **_graph_identity(reference_row),
                            "R_target": target,
                            "training_criterion": training,
                            "evaluation_criterion": evaluation,
                            "training_hard_h2": matched[training].hard_h2,
                            "direct_hard_h2": direct.hard_h2,
                            "abs_h2_difference": abs(matched[training].hard_h2 - direct.hard_h2),
                            "training_lambda_org": matched[training].lambda_org,
                            "direct_lambda_org": direct.lambda_org,
                            "distortion": value,
                            "direct_own_distortion": direct_value,
                            "own_target_excess_distortion": value - direct_value,
                            "h2_tolerance": tolerance,
                        }
                    )
    return regret_rows


def _best_available_rows(complete_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute best available rows."""
    output: list[dict[str, Any]] = []
    for record in complete_records:
        criteria: tuple[str, ...] = record["criteria"]
        matched: dict[str, Match] = record["matches"]
        for evaluation in criteria:
            values = {
                training: _hard_distortion(row=match.row, criterion=evaluation)
                for training, match in matched.items()
                if training in criteria
            }
            values = {training: value for training, value in values.items() if math.isfinite(value)}
            if len(values) != len(criteria):
                continue
            best_value = min(values.values())
            best_trainings = [training for training, value in values.items() if value <= best_value + EPS]
            for training, value in values.items():
                match = matched[training]
                output.append(
                    {
                        **{key: record[key] for key in _graph_identity(record)},
                        "analysis_type": record["analysis_type"],
                        "criteria_set": record["criteria_set"],
                        "R_target": record["R_target"],
                        "training_criterion": training,
                        "evaluation_criterion": evaluation,
                        "hard_h2": match.hard_h2,
                        "hard_h2_min": record["hard_h2_min"],
                        "hard_h2_max": record["hard_h2_max"],
                        "hard_h2_span": record["hard_h2_span"],
                        "lambda_org": match.lambda_org,
                        "distortion": value,
                        "best_available_distortion": best_value,
                        "best_available_excess_distortion": value - best_value,
                        "best_available_training_criteria": ";".join(best_trainings),
                        "h2_tolerance": record["h2_tolerance"],
                    }
                )
    return output


def _rank_rows(best_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute rank rows."""
    grouped: dict[tuple[str, str, str, str, int, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in best_rows:
        key = (
            str(row["analysis_type"]),
            str(row["collection"]),
            str(row["source_variant"]),
            str(row["split"]),
            int(row["graph_index"]),
            float(row["R_target"]),
            str(row["evaluation_criterion"]),
        )
        grouped[key].append(row)

    output: list[dict[str, Any]] = []
    for group in grouped.values():
        values = {row["training_criterion"]: _float(row["distortion"]) for row in group}
        ranks = average_ranks(values)
        finite_values = [value for value in values.values() if math.isfinite(value)]
        if not finite_values:
            continue
        best = min(finite_values)
        winners = [criterion for criterion, value in values.items() if math.isfinite(value) and value <= best + EPS]
        for row in group:
            training = row["training_criterion"]
            if training not in ranks:
                continue
            output.append(
                {
                    **{key: row[key] for key in _graph_identity(row)},
                    "analysis_type": row["analysis_type"],
                    "criteria_set": row["criteria_set"],
                    "R_target": row["R_target"],
                    "training_criterion": training,
                    "evaluation_criterion": row["evaluation_criterion"],
                    "rank": ranks[training],
                    "win_fraction": (1.0 / len(winners)) if training in winners else 0.0,
                    "num_available_trainings": len(ranks),
                    "distortion": row["distortion"],
                    "hard_h2_span": row["hard_h2_span"],
                }
            )
    return output


def _summarize_rows(
    rows: list[dict[str, Any]],
    *,
    keys: tuple[str, ...],
    metrics: tuple[str, ...],
    count_name: str = "num_graph_rate_values",
) -> list[dict[str, Any]]:
    """Summarize rows."""
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    output: list[dict[str, Any]] = []
    for key_values, group in sorted(grouped.items()):
        item = dict(zip(keys, key_values, strict=True))
        item[count_name] = len(group)
        for metric in metrics:
            values = [_float(row.get(metric)) for row in group]
            item[f"{metric}_mean"] = _safe_mean(values)
            item[f"{metric}_std"] = _safe_std(values)
            item[f"{metric}_median"] = _safe_quantile(values=values, q=0.5)
        output.append(item)
    return output


def _pareto_rows(
    best_rows: list[dict[str, Any]], *, criteria: tuple[str, ...], analysis_type: str
) -> list[dict[str, Any]]:
    """Compute pareto rows."""
    values_by_case: dict[tuple[str, str, str, int, float], dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    identity_by_case: dict[tuple[str, str, str, int, float], dict[str, Any]] = {}
    span_by_case: dict[tuple[str, str, str, int, float], float] = {}
    for row in best_rows:
        if row["analysis_type"] != analysis_type:
            continue
        if row["training_criterion"] not in criteria or row["evaluation_criterion"] not in criteria:
            continue
        case = (
            row["collection"],
            row["source_variant"],
            row["split"],
            int(row["graph_index"]),
            float(row["R_target"]),
        )
        identity_by_case[case] = _graph_identity(row)
        span_by_case[case] = _float(row["hard_h2_span"])
        values_by_case[case][row["training_criterion"]][row["evaluation_criterion"]] = _float(row["distortion"])

    output: list[dict[str, Any]] = []
    for case, values in sorted(values_by_case.items()):
        if any(training not in values for training in criteria):
            continue
        if any(any(evaluation not in values[training] for evaluation in criteria) for training in criteria):
            continue
        dominance: dict[str, set[str]] = {criterion: set() for criterion in criteria}
        dominated_by: dict[str, set[str]] = {criterion: set() for criterion in criteria}
        for first, second in itertools.permutations(criteria, 2):
            first_values = np.asarray([values[first][evaluation] for evaluation in criteria], dtype=np.float64)
            second_values = np.asarray([values[second][evaluation] for evaluation in criteria], dtype=np.float64)
            if np.all(first_values <= second_values + EPS) and np.any(first_values < second_values - EPS):
                dominance[first].add(second)
                dominated_by[second].add(first)
        no_strict_dominance = int(not any(dominance[criterion] for criterion in criteria))
        identity = identity_by_case[case]
        for criterion in criteria:
            output.append(
                {
                    **identity,
                    "R_target": case[-1],
                    "analysis_type": analysis_type,
                    "criteria_set": ";".join(criteria),
                    "training_criterion": criterion,
                    "pareto_front_member": int(not dominated_by[criterion]),
                    "strictly_dominated_by_count": len(dominated_by[criterion]),
                    "strict_dominates_count": len(dominance[criterion]),
                    "strict_dominates_any": int(bool(dominance[criterion])),
                    "no_strict_dominance_case": no_strict_dominance,
                    "hard_h2_span": span_by_case[case],
                }
            )
    return output


def _integrated_overlap_summary(overlap_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute integrated overlap summary."""
    rows = _summarize_rows(
        overlap_rows,
        keys=("source", "criterion_a", "criterion_b", "criterion_pair"),
        metrics=("ari", "collision_jaccard"),
    )
    for row in rows:
        group = [
            item
            for item in overlap_rows
            if item["source"] == row["source"]
            and item["criterion_a"] == row["criterion_a"]
            and item["criterion_b"] == row["criterion_b"]
        ]
        for target in (1.0, 1.5, 2.0):
            values = [_float(item["ari"]) for item in group if math.isclose(float(item["R_target"]), target)]
            row[f"ari_at_R_{target:.1f}_mean"] = _safe_mean(values)
            row[f"ari_at_R_{target:.1f}_num_graphs"] = len([value for value in values if math.isfinite(value)])
    return rows


def _fourier_collision_summary(
    overlap_rows: list[dict[str, Any]], regret_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compute fourier collision summary."""
    output: list[dict[str, Any]] = []
    for source in SOURCE_ORDER:
        for target in R_TARGETS:
            overlap = [
                row
                for row in overlap_rows
                if row["source"] == source
                and math.isclose(float(row["R_target"]), target)
                and {row["criterion_a"], row["criterion_b"]} == {FOURIER, COLLISION}
            ]
            f_to_c = [
                _float(row["own_target_excess_distortion"])
                for row in regret_rows
                if row["source"] == source
                and math.isclose(float(row["R_target"]), target)
                and row["training_criterion"] == FOURIER
                and row["evaluation_criterion"] == COLLISION
            ]
            c_to_f = [
                _float(row["own_target_excess_distortion"])
                for row in regret_rows
                if row["source"] == source
                and math.isclose(float(row["R_target"]), target)
                and row["training_criterion"] == COLLISION
                and row["evaluation_criterion"] == FOURIER
            ]
            output.append(
                {
                    "source": source,
                    "R_target": target,
                    "num_overlap_graphs": len(overlap),
                    "ari_mean": _safe_mean([_float(row["ari"]) for row in overlap]),
                    "collision_jaccard_mean": _safe_mean([_float(row["collision_jaccard"]) for row in overlap]),
                    "delta_F_to_C_mean": _safe_mean(f_to_c),
                    "delta_C_to_F_mean": _safe_mean(c_to_f),
                    "num_F_to_C_graphs": len([value for value in f_to_c if math.isfinite(value)]),
                    "num_C_to_F_graphs": len([value for value in c_to_f if math.isfinite(value)]),
                }
            )
    return output


def _h2_specific_summary(overlap_rows: list[dict[str, Any]], regret_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute h2 specific summary."""
    rows: list[dict[str, Any]] = []
    for source in SOURCE_ORDER:
        for target in R_TARGETS:
            for other in (EDGE, FOURIER, COLLISION):
                pair_overlap = [
                    row
                    for row in overlap_rows
                    if row["source"] == source
                    and math.isclose(float(row["R_target"]), target)
                    and {row["criterion_a"], row["criterion_b"]} == {other, COLLISION_ENTROPY}
                ]
                h2_to_other = [
                    _float(row["own_target_excess_distortion"])
                    for row in regret_rows
                    if row["source"] == source
                    and math.isclose(float(row["R_target"]), target)
                    and row["training_criterion"] == COLLISION_ENTROPY
                    and row["evaluation_criterion"] == other
                ]
                other_to_h2 = [
                    _float(row["own_target_excess_distortion"])
                    for row in regret_rows
                    if row["source"] == source
                    and math.isclose(float(row["R_target"]), target)
                    and row["training_criterion"] == other
                    and row["evaluation_criterion"] == COLLISION_ENTROPY
                ]
                rows.append(
                    {
                        "source": source,
                        "R_target": target,
                        "comparison": _criterion_pair_label(first=other, second=COLLISION_ENTROPY),
                        "num_overlap_graphs": len(pair_overlap),
                        "ari_mean": _safe_mean([_float(row["ari"]) for row in pair_overlap]),
                        "collision_jaccard_mean": _safe_mean(
                            [_float(row["collision_jaccard"]) for row in pair_overlap]
                        ),
                        "delta_H2_to_other_mean": _safe_mean(h2_to_other),
                        "delta_other_to_H2_mean": _safe_mean(other_to_h2),
                    }
                )
    return rows


def _main_rate2_pair_table(
    overlap_summary: list[dict[str, Any]], regret_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compute main rate2 pair table."""
    target = 2.0
    output: list[dict[str, Any]] = []
    for source in SOURCE_ORDER:
        for first, second in MAIN_RATE_MATCHED_PAIRS:
            overlap = [
                row
                for row in overlap_summary
                if row["source"] == source
                and row["criterion_a"] == first
                and row["criterion_b"] == second
                and math.isclose(float(row["R_target"]), target)
            ]
            a_to_b = [
                _float(row["own_target_excess_distortion"])
                for row in regret_rows
                if row["source"] == source
                and math.isclose(float(row["R_target"]), target)
                and row["training_criterion"] == first
                and row["evaluation_criterion"] == second
            ]
            b_to_a = [
                _float(row["own_target_excess_distortion"])
                for row in regret_rows
                if row["source"] == source
                and math.isclose(float(row["R_target"]), target)
                and row["training_criterion"] == second
                and row["evaluation_criterion"] == first
            ]
            output.append(
                {
                    "source": source,
                    "criterion_a": first,
                    "criterion_b": second,
                    "criterion_pair": _criterion_pair_label(first=first, second=second),
                    "R_target": target,
                    "num_graphs": 0 if not overlap else int(overlap[0]["num_graphs"]),
                    "ari_mean": float("nan") if not overlap else _float(overlap[0]["ari_mean"]),
                    "delta_a_to_b_mean": _safe_mean(a_to_b),
                    "delta_b_to_a_mean": _safe_mean(b_to_a),
                    "num_delta_a_to_b_graphs": len([value for value in a_to_b if math.isfinite(value)]),
                    "num_delta_b_to_a_graphs": len([value for value in b_to_a if math.isfinite(value)]),
                }
            )
    return output


def _broad_robustness_summary(
    rank_rows: list[dict[str, Any]], pareto_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Compute broad robustness summary."""
    rank_grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rank_rows:
        rank_grouped[(row["analysis_type"], row["source"], row["training_criterion"])].append(row)
    pareto_grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pareto_rows:
        pareto_grouped[(row["analysis_type"], row["source"], row["training_criterion"])].append(row)
    output: list[dict[str, Any]] = []
    for analysis_type, criteria in COMPLETE_CASE_SETS.items():
        for source in SOURCE_ORDER:
            for criterion in criteria:
                ranks = rank_grouped.get((analysis_type, source, criterion), [])
                pareto = pareto_grouped.get((analysis_type, source, criterion), [])
                output.append(
                    {
                        "analysis_type": analysis_type,
                        "source": source,
                        "training_criterion": criterion,
                        "mean_cross_distortion_rank": _safe_mean([_float(row["rank"]) for row in ranks]),
                        "median_cross_distortion_rank": _safe_quantile(
                            values=[_float(row["rank"]) for row in ranks], q=0.5
                        ),
                        "mean_win_fraction": _safe_mean([_float(row["win_fraction"]) for row in ranks]),
                        "pareto_front_membership_fraction": _safe_mean(
                            [_float(row["pareto_front_member"]) for row in pareto]
                        ),
                        "strict_dominance_frequency": _safe_mean(
                            [_float(row["strict_dominates_any"]) for row in pareto]
                        ),
                        "num_rank_graph_rate_eval_values": len(ranks),
                        "num_complete_graph_rate_cases": len(pareto),
                    }
                )
    return output


def _transfer_summary(regret_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute transfer summary."""
    return _summarize_rows(
        regret_rows,
        keys=("source", "training_criterion", "evaluation_criterion"),
        metrics=("own_target_excess_distortion",),
        count_name="num_graph_rate_comparisons",
    )


def _plot_overlap(summary_rows: list[dict[str, Any]], *, metric: str, output_prefix: Path) -> None:
    """Plot overlap."""
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8), sharex=True, sharey=metric == "ari")
    pairs = list(itertools.combinations(CRITERIA, 2))
    markers = ["o", "s", "^", "D", "P", "X"]
    linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
    colors = plt.cm.tab10(np.linspace(start=0.0, stop=1.0, num=len(pairs)))
    for ax, source in zip(axes.flat, SOURCE_ORDER, strict=True):
        for pair_index, (first, second) in enumerate(pairs):
            rows = [
                row
                for row in summary_rows
                if row["source"] == source
                and row["criterion_a"] == first
                and row["criterion_b"] == second
                and int(row["num_graphs"]) >= MIN_FIGURE_GRAPHS
            ]
            rows.sort(key=lambda row: float(row["R_target"]))
            if not rows:
                continue
            x = np.asarray([float(row["R_target"]) for row in rows])
            y = np.asarray([float(row[f"{metric}_mean"]) for row in rows])
            low = np.asarray([float(row[f"{metric}_q25"]) for row in rows])
            high = np.asarray([float(row[f"{metric}_q75"]) for row in rows])
            ax.plot(
                x,
                y,
                label=_criterion_pair_label(first=first, second=second),
                color=colors[pair_index],
                marker=markers[pair_index],
                linestyle=linestyles[pair_index],
                linewidth=1.6,
                markersize=4,
            )
            ax.fill_between(x, low, high, color=colors[pair_index], alpha=0.12, linewidth=0)
        ax.set_title(source)
        ax.grid(True, alpha=0.25)
        secax = ax.secondary_xaxis("top", functions=(_h2_to_keff, _keff_to_h2))
        secax.set_xlabel(r"$K_{\mathrm{eff}}$")
        secax.set_xticks([2, 4, 6, 8])
        secax.set_xticklabels(["2", "4", "6", "8"])
    for ax in axes[-1, :]:
        ax.set_xlabel(r"occupancy-matched hard $R_2=H_2(\bar q_G)$")
    for ax in axes[:, 0]:
        ax.set_ylabel("mean ARI" if metric == "ari" else "mean collision Jaccard")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def _plot_main_rate_matched_ari(summary_rows: list[dict[str, Any]], *, output_prefix: Path) -> None:
    """Plot main rate matched ari."""
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11.6, 7.6), sharex=True, sharey=True)
    markers = ["o", "s", "^"]
    colors = plt.cm.Dark2(np.linspace(start=0.0, stop=0.7, num=len(MAIN_RATE_MATCHED_PAIRS)))
    for ax, source in zip(axes.flat, SOURCE_ORDER, strict=True):
        for pair_index, (first, second) in enumerate(MAIN_RATE_MATCHED_PAIRS):
            rows = [
                row
                for row in summary_rows
                if row["source"] == source
                and row["criterion_a"] == first
                and row["criterion_b"] == second
                and int(row["num_graphs"]) >= MIN_FIGURE_GRAPHS
            ]
            rows.sort(key=lambda row: float(row["R_target"]))
            if not rows:
                continue
            x = np.asarray([float(row["R_target"]) for row in rows])
            y = np.asarray([float(row["ari_mean"]) for row in rows])
            low = np.asarray([float(row["ari_q25"]) for row in rows])
            high = np.asarray([float(row["ari_q75"]) for row in rows])
            ax.plot(
                x,
                y,
                label=_criterion_pair_label(first=first, second=second),
                color=colors[pair_index],
                marker=markers[pair_index],
                linewidth=1.8,
                markersize=4.5,
            )
            ax.fill_between(x, low, high, color=colors[pair_index], alpha=0.14, linewidth=0)
        ax.set_title(source)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(-0.05, 1.05)
        secax = ax.secondary_xaxis("top", functions=(_h2_to_keff, _keff_to_h2))
        secax.set_xlabel(r"$K_{\mathrm{eff}}$")
        secax.set_xticks([2, 4, 6, 8])
        secax.set_xticklabels(["2", "4", "6", "8"])
    for ax in axes[-1, :]:
        ax.set_xlabel(r"occupancy-matched hard $R_2=H_2(\bar q_G)$")
    for ax in axes[:, 0]:
        ax.set_ylabel("mean ARI")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def _plot_rank_sweep(
    rank_sweep_rows: list[dict[str, Any]], output_prefix: Path, *, analysis_type: str = "four_way"
) -> None:
    """Plot rank sweep."""
    criteria = COMPLETE_CASE_SETS[analysis_type]
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11, 7.5), sharex=True, sharey=True)
    markers = ["o", "s", "^", "D"]
    colors = plt.cm.tab10(np.linspace(start=0.0, stop=0.8, num=len(criteria)))
    for ax, source in zip(axes.flat, SOURCE_ORDER, strict=True):
        for index, criterion in enumerate(criteria):
            rows = [
                row
                for row in rank_sweep_rows
                if row["source"] == source
                and row["analysis_type"] == analysis_type
                and row["training_criterion"] == criterion
                and int(row["num_graph_rate_eval_values"]) >= MIN_FIGURE_GRAPHS
            ]
            rows.sort(key=lambda row: float(row["R_target"]))
            if not rows:
                continue
            ax.plot(
                [float(row["R_target"]) for row in rows],
                [float(row["rank_mean"]) for row in rows],
                label=CRITERION_LABELS[criterion],
                color=colors[index],
                marker=markers[index],
                linewidth=1.8,
                markersize=4,
            )
        ax.set_title(source)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(1.0, len(criteria) + 0.05)
        secax = ax.secondary_xaxis("top", functions=(_h2_to_keff, _keff_to_h2))
        secax.set_xlabel(r"$K_{\mathrm{eff}}$")
        secax.set_xticks([2, 4, 6, 8])
        secax.set_xticklabels(["2", "4", "6", "8"])
    for ax in axes[-1, :]:
        ax.set_xlabel(r"occupancy-matched hard $R_2=H_2(\bar q_G)$")
    for ax in axes[:, 0]:
        ax.set_ylabel("mean cross-distortion rank")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def _plot_regret_by_source(summary_rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Plot regret by source."""
    markers = ["o", "s", "^", "D"]
    colors = plt.cm.tab10(np.linspace(start=0.0, stop=0.8, num=len(CRITERIA)))
    for source in SOURCE_ORDER:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11, 7.5), sharex=True)
        for ax, evaluation in zip(axes.flat, CRITERIA, strict=True):
            for index, training in enumerate(CRITERIA):
                rows = [
                    row
                    for row in summary_rows
                    if row["source"] == source
                    and row["evaluation_criterion"] == evaluation
                    and row["training_criterion"] == training
                    and int(row["num_graph_rate_comparisons"]) >= MIN_FIGURE_GRAPHS
                ]
                rows.sort(key=lambda row: float(row["R_target"]))
                if not rows:
                    continue
                ax.plot(
                    [float(row["R_target"]) for row in rows],
                    [float(row["own_target_excess_distortion_mean"]) for row in rows],
                    label=CRITERION_LABELS[training],
                    color=colors[index],
                    marker=markers[index],
                    linewidth=1.6,
                    markersize=4,
                )
            ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
            ax.set_title(f"evaluated by {CRITERION_LABELS[evaluation]}")
            ax.grid(True, alpha=0.25)
            secax = ax.secondary_xaxis("top", functions=(_h2_to_keff, _keff_to_h2))
            secax.set_xlabel(r"$K_{\mathrm{eff}}$")
            secax.set_xticks([2, 4, 6, 8])
            secax.set_xticklabels(["2", "4", "6", "8"])
        fig.suptitle(f"{source}: own-target excess distortion")
        for ax in axes[-1, :]:
            ax.set_xlabel(r"occupancy-matched hard $R_2=H_2(\bar q_G)$")
        for ax in axes[:, 0]:
            ax.set_ylabel(r"mean $\Delta_{a\to b}$")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
        fig.tight_layout(rect=(0, 0.08, 1, 0.96))
        prefix = output_dir / f"{_source_slug(source)}_cross_distortion_regret_vs_h2"
        fig.savefig(prefix.with_suffix(".png"), dpi=220)
        fig.savefig(prefix.with_suffix(".pdf"))
        plt.close(fig)


def _plot_rank_heatmaps(
    heatmap_rows: list[dict[str, Any]], output_prefix: Path, *, analysis_type: str = "four_way"
) -> None:
    """Plot rank heatmaps."""
    criteria = COMPLETE_CASE_SETS[analysis_type]
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10.5, 8.0), constrained_layout=True)
    for ax, source in zip(axes.flat, SOURCE_ORDER, strict=True):
        matrix = np.full(shape=(len(criteria), len(criteria)), fill_value=np.nan)
        for row in heatmap_rows:
            if row["source"] != source or row["analysis_type"] != analysis_type:
                continue
            i = criteria.index(row["training_criterion"])
            j = criteria.index(row["evaluation_criterion"])
            matrix[i, j] = _float(row["rank_mean"])
        im = ax.imshow(matrix, vmin=1.0, vmax=float(len(criteria)), cmap="viridis_r")
        ax.set_title(source)
        ax.set_xticks(range(len(criteria)), [CRITERION_SHORT[c] for c in criteria], rotation=30, ha="right")
        ax.set_yticks(range(len(criteria)), [CRITERION_SHORT[c] for c in criteria])
        ax.set_xlabel("evaluation distortion")
        ax.set_ylabel("training criterion")
        for i in range(len(criteria)):
            for j in range(len(criteria)):
                value = matrix[i, j]
                text = "" if not math.isfinite(value) else f"{value:.2f}"
                ax.text(j, i, text, ha="center", va="center", color="white" if value > 2.4 else "black")
    fig.colorbar(im, ax=axes, shrink=0.8, label="mean rank (lower is better)")
    fig.savefig(output_prefix.with_suffix(".png"), dpi=220)
    fig.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(fig)


def _write_markdown_report(
    path: Path,
    *,
    integrated_overlap: list[dict[str, Any]],
    transfer: list[dict[str, Any]],
    broad: list[dict[str, Any]],
    fourier_collision: list[dict[str, Any]],
    h2_summary: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    """Write markdown report."""
    lines = [
        "# Relative Distortion Analysis",
        "",
        "This analysis uses hard-H2-matched stored partitions from the corrected Experiment 5 run.",
        f"Primary tolerance: `{manifest['primary_h2_tolerance']}` nats.",
        "",
        "## Integrated Partition Overlap",
        "",
        "| Source | Pair | n | mean ARI | median ARI | mean Jaccard | median Jaccard |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in integrated_overlap:
        lines.append(
            "| {source} | {pair} | {n} | {ari:.3f} | {ari_med:.3f} | {jac:.3f} | {jac_med:.3f} |".format(
                source=row["source"],
                pair=row["criterion_pair"],
                n=row["num_graph_rate_values"],
                ari=_float(row["ari_mean"]),
                ari_med=_float(row["ari_median"]),
                jac=_float(row["collision_jaccard_mean"]),
                jac_med=_float(row["collision_jaccard_median"]),
            )
        )
    lines.extend(["", "## Cross-Distortion Transfer", ""])
    for source in SOURCE_ORDER:
        lines.extend(
            [
                f"### {source}",
                "",
                "Each column uses one fixed evaluation distortion; compare rows within a column.",
                "",
                "| Train | D_E | D_F | D_C | D_H2 |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for training in CRITERIA:
            cells = []
            for evaluation in CRITERIA:
                rows = [
                    row
                    for row in transfer
                    if row["source"] == source
                    and row["training_criterion"] == training
                    and row["evaluation_criterion"] == evaluation
                ]
                cells.append("nan" if not rows else f"{_float(rows[0]['own_target_excess_distortion_mean']):.4f}")
            lines.append(f"| {CRITERION_SHORT[training]} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.extend(
        [
            "## Broad Robustness",
            "",
            "| Analysis | Source | Train | mean rank | win frac. | Pareto-front frac. | dominance freq. | complete cases |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in broad:
        lines.append(
            "| {analysis} | {source} | {criterion} | {rank:.3f} | {win:.3f} | {pareto:.3f} | {dom:.3f} | {n} |".format(
                analysis=row["analysis_type"],
                source=row["source"],
                criterion=CRITERION_SHORT[row["training_criterion"]],
                rank=_float(row["mean_cross_distortion_rank"]),
                win=_float(row["mean_win_fraction"]),
                pareto=_float(row["pareto_front_membership_fraction"]),
                dom=_float(row["strict_dominance_frequency"]),
                n=row["num_complete_graph_rate_cases"],
            )
        )

    lines.extend(
        [
            "",
            "## D_F / D_C Summary",
            "",
            "| Source | R | n | ARI | Jaccard | Delta F->C | Delta C->F |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in fourier_collision:
        if int(row["num_overlap_graphs"]) == 0:
            continue
        lines.append(
            "| {source} | {r:.2f} | {n} | {ari:.3f} | {jac:.3f} | {ftoc:.4f} | {ctof:.4f} |".format(
                source=row["source"],
                r=_float(row["R_target"]),
                n=row["num_overlap_graphs"],
                ari=_float(row["ari_mean"]),
                jac=_float(row["collision_jaccard_mean"]),
                ftoc=_float(row["delta_F_to_C_mean"]),
                ctof=_float(row["delta_C_to_F_mean"]),
            )
        )

    lines.extend(
        [
            "",
            "## D_H2 Pair Summary",
            "",
            "| Source | R | Pair | n | ARI | Jaccard |",
            "| --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in h2_summary:
        if int(row["num_overlap_graphs"]) == 0:
            continue
        lines.append(
            "| {source} | {r:.2f} | {pair} | {n} | {ari:.3f} | {jac:.3f} |".format(
                source=row["source"],
                r=_float(row["R_target"]),
                pair=row["comparison"],
                n=row["num_overlap_graphs"],
                ari=_float(row["ari_mean"]),
                jac=_float(row["collision_jaccard_mean"]),
            )
        )
    path.write_text("\n".join(lines) + "\n")


def _git_sha() -> str:
    """Compute git sha."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_relative_analysis(
    *,
    experiment_dir: Path,
    output_dir: Path,
    r_targets: tuple[float, ...] = R_TARGETS,
    tolerance: float = PRIMARY_TOLERANCE,
    sensitivity_tolerance: float = SENSITIVITY_TOLERANCE,
) -> dict[str, Any]:
    """Run relative analysis."""
    per_graph_rows = _read_csv(experiment_dir / "per_graph_results.csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_cache: dict[str, np.ndarray] = {}

    matches, _ = _build_matches(per_graph_rows, r_targets=r_targets, tolerance=tolerance)
    coverage = _coverage_rows(
        per_graph_rows=per_graph_rows,
        matches=matches,
        r_targets=r_targets,
        tolerance=tolerance,
    )
    overlap = _partition_overlap_rows(
        experiment_dir=experiment_dir,
        per_graph_rows=per_graph_rows,
        matches=matches,
        r_targets=r_targets,
        tolerance=tolerance,
        artifact_cache=artifact_cache,
    )
    overlap_summary = _overlap_summary_rows(overlap, bootstrap_seed=BOOTSTRAP_SEED)
    regret = _cross_distortion_rows(
        per_graph_rows=per_graph_rows,
        matches=matches,
        r_targets=r_targets,
        tolerance=tolerance,
    )
    complete_records = _complete_case_records(
        per_graph_rows=per_graph_rows,
        matches=matches,
        r_targets=r_targets,
        tolerance=tolerance,
    )
    complete_cases = _complete_case_rows(complete_records)
    complete_case_coverage = _complete_case_coverage_rows(
        per_graph_rows=per_graph_rows,
        matches=matches,
        r_targets=r_targets,
        tolerance=tolerance,
    )
    best = _best_available_rows(complete_records)
    rank = _rank_rows(best)
    rank_summary = _summarize_rows(
        rank,
        keys=("analysis_type", "source", "training_criterion", "evaluation_criterion", "R_target"),
        metrics=("rank", "win_fraction"),
        count_name="num_graph_rate_eval_values",
    )
    rank_sweep = _summarize_rows(
        rank,
        keys=("analysis_type", "source", "training_criterion", "R_target"),
        metrics=("rank", "win_fraction"),
        count_name="num_graph_rate_eval_values",
    )
    heatmap = _summarize_rows(
        rank,
        keys=("analysis_type", "source", "training_criterion", "evaluation_criterion"),
        metrics=("rank", "win_fraction"),
        count_name="num_graph_rate_eval_values",
    )
    regret_summary = _summarize_rows(
        regret,
        keys=("source", "training_criterion", "evaluation_criterion", "R_target"),
        metrics=("own_target_excess_distortion",),
        count_name="num_graph_rate_comparisons",
    )
    transfer = _transfer_summary(regret)
    best_summary = _summarize_rows(
        best,
        keys=("analysis_type", "source", "training_criterion", "evaluation_criterion", "R_target"),
        metrics=("best_available_excess_distortion",),
        count_name="num_graph_rate_comparisons",
    )
    pareto = _pareto_rows(best, criteria=CRITERIA, analysis_type="four_way")
    pareto.extend(
        _pareto_rows(
            best,
            criteria=COMPLETE_CASE_SETS["edge_fourier_collision_three_way"],
            analysis_type="edge_fourier_collision_three_way",
        )
    )
    pareto_summary = _summarize_rows(
        pareto,
        keys=("analysis_type", "source", "training_criterion", "R_target"),
        metrics=("pareto_front_member", "strict_dominates_any", "strict_dominates_count", "no_strict_dominance_case"),
        count_name="num_complete_graph_rate_cases",
    )
    broad = _broad_robustness_summary(rank_rows=rank, pareto_rows=pareto)
    integrated_overlap = _integrated_overlap_summary(overlap)
    fourier_collision = _fourier_collision_summary(overlap_rows=overlap, regret_rows=regret)
    h2_specific = _h2_specific_summary(overlap_rows=overlap, regret_rows=regret)
    main_rate2_table = _main_rate2_pair_table(overlap_summary=overlap_summary, regret_rows=regret)

    matched_artifact_rows = [match.row for match in matches.values()]
    validation = _validate_artifact_and_h2(
        experiment_dir=experiment_dir,
        rows=matched_artifact_rows,
        artifact_cache=artifact_cache,
    )

    sensitivity_matches, _ = _build_matches(
        per_graph_rows,
        r_targets=r_targets,
        tolerance=sensitivity_tolerance,
    )
    sensitivity_coverage = _coverage_rows(
        per_graph_rows=per_graph_rows,
        matches=sensitivity_matches,
        r_targets=r_targets,
        tolerance=sensitivity_tolerance,
    )
    sensitivity_overlap = _partition_overlap_rows(
        experiment_dir=experiment_dir,
        per_graph_rows=per_graph_rows,
        matches=sensitivity_matches,
        r_targets=r_targets,
        tolerance=sensitivity_tolerance,
        artifact_cache=artifact_cache,
    )
    sensitivity_overlap_summary = _overlap_summary_rows(sensitivity_overlap, bootstrap_seed=BOOTSTRAP_SEED)

    csv_outputs = {
        "relative_analysis_rate_coverage.csv": coverage,
        "relative_analysis_complete_case_coverage.csv": complete_case_coverage,
        "relative_analysis_complete_cases.csv": complete_cases,
        "relative_analysis_partition_overlap_per_graph.csv": overlap,
        "relative_analysis_partition_overlap_summary.csv": overlap_summary,
        "relative_analysis_cross_distortion_regret_per_graph.csv": regret,
        "relative_analysis_best_available_excess_per_graph.csv": best,
        "relative_analysis_cross_distortion_rank_per_graph.csv": rank,
        "relative_analysis_cross_distortion_rank_summary.csv": rank_summary,
        "relative_analysis_cross_distortion_rank_sweep.csv": rank_sweep,
        "relative_analysis_cross_distortion_rank_heatmap_summary.csv": heatmap,
        "relative_analysis_cross_distortion_regret_summary.csv": regret_summary,
        "relative_analysis_best_available_excess_summary.csv": best_summary,
        "relative_analysis_cross_distortion_transfer_summary.csv": transfer,
        "relative_analysis_pareto_per_graph.csv": pareto,
        "relative_analysis_pareto_summary.csv": pareto_summary,
        "relative_analysis_broad_robustness_summary.csv": broad,
        "relative_analysis_overlap_integrated_summary.csv": integrated_overlap,
        "relative_analysis_fourier_collision_summary.csv": fourier_collision,
        "relative_analysis_h2_specific_summary.csv": h2_specific,
        "relative_analysis_main_rate2_pair_table.csv": main_rate2_table,
        "relative_analysis_rate_coverage_tol0p15.csv": sensitivity_coverage,
        "relative_analysis_partition_overlap_summary_tol0p15.csv": sensitivity_overlap_summary,
    }
    for filename, rows in csv_outputs.items():
        _write_csv(path=output_dir / filename, rows=rows)

    _plot_overlap(overlap_summary, metric="ari", output_prefix=output_dir / "relative_partition_ari_vs_h2")
    _plot_overlap(
        overlap_summary,
        metric="collision_jaccard",
        output_prefix=output_dir / "relative_partition_collision_jaccard_vs_h2",
    )
    _plot_main_rate_matched_ari(
        overlap_summary,
        output_prefix=output_dir / "transductive_rate_matched_partition_agreement",
    )
    _plot_rank_sweep(rank_sweep, output_prefix=output_dir / "relative_cross_distortion_mean_rank_vs_h2")
    _plot_regret_by_source(regret_summary, output_dir=output_dir)
    _plot_rank_heatmaps(heatmap, output_prefix=output_dir / "relative_cross_distortion_rank_heatmaps")

    manifest = {
        "experiment_dir": str(experiment_dir),
        "output_dir": str(output_dir),
        "analysis_script_commit_sha": _git_sha(),
        "R_targets": list(r_targets),
        "primary_h2_tolerance": tolerance,
        "pairwise_h2_tolerance": tolerance,
        "sensitivity_h2_tolerance": sensitivity_tolerance,
        "criterion_names": list(CRITERIA),
        "source_collections": list(SOURCE_ORDER),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "minimum_graphs_drawn_in_figures": MIN_FIGURE_GRAPHS,
        "num_per_graph_rows": len(per_graph_rows),
        "num_rate_matches": len(matches),
        "num_overlap_rows": len(overlap),
        "num_complete_case_rows": len(complete_cases),
        "num_cross_regret_rows": len(regret),
        "num_rank_rows": len(rank),
        "complete_case_counts": {
            analysis_type: {
                source: sum(
                    1
                    for record in complete_records
                    if record["analysis_type"] == analysis_type and record["source"] == source
                )
                for source in SOURCE_ORDER
            }
            for analysis_type in COMPLETE_CASE_SETS
        },
        "complete_case_max_h2_span": {
            analysis_type: max(
                [
                    _float(record["hard_h2_span"])
                    for record in complete_records
                    if record["analysis_type"] == analysis_type
                ],
                default=float("nan"),
            )
            for analysis_type in COMPLETE_CASE_SETS
        },
        "validation": validation,
        "csv_outputs": sorted(csv_outputs),
        "figure_outputs": sorted(path.name for path in output_dir.glob("*.png"))
        + sorted(path.name for path in output_dir.glob("*.pdf")),
    }
    _write_json(path=output_dir / "relative_analysis_manifest.json", value=manifest)
    _write_json(
        path=output_dir / "relative_analysis_summary.json",
        value={
            "manifest": manifest,
            "artifact_validation": validation,
            "primary_coverage_rows": coverage,
            "complete_case_coverage_rows": complete_case_coverage,
            "integrated_overlap": integrated_overlap,
            "cross_distortion_transfer": transfer,
            "broad_robustness": broad,
            "main_rate2_pair_table": main_rate2_table,
        },
    )
    _write_markdown_report(
        output_dir / "relative_analysis_report.md",
        integrated_overlap=integrated_overlap,
        transfer=transfer,
        broad=broad,
        fourier_collision=fourier_collision,
        h2_summary=h2_specific,
        manifest=manifest,
    )
    return manifest


def main() -> None:
    """Run the command-line entry point."""
    parser = argparse.ArgumentParser(description="Analyze rate-matched relative relational distortions.")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("out/experiments/trd_corrected_lr0p05_s600_four_collections"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--h2-tolerance", type=float, default=PRIMARY_TOLERANCE)
    parser.add_argument("--sensitivity-h2-tolerance", type=float, default=SENSITIVITY_TOLERANCE)
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.absolute()
    output_dir = (
        args.output_dir.absolute() if args.output_dir is not None else experiment_dir / "relative_distortion_analysis"
    )
    manifest = run_relative_analysis(
        experiment_dir=experiment_dir,
        output_dir=output_dir,
        tolerance=float(args.h2_tolerance),
        sensitivity_tolerance=float(args.sensitivity_h2_tolerance),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
