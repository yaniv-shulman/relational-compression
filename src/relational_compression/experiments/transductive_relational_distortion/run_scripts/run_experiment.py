"""Command-line utilities for reproducible experiment workflows."""

import argparse
import importlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import rankdata

from relational_compression.experiments.inductive_normalized_cut.data import seed_everything
from relational_compression.experiments.transductive_relational_distortion.data_sources import (
    graph_record_row,
    load_source_graph_records,
)
from relational_compression.experiments.transductive_relational_distortion.evaluate import (
    aggregate_rows,
    evaluate_optimized_partition,
    write_csv,
    write_json,
)
from relational_compression.experiments.transductive_relational_distortion.objectives import (
    assignment_probabilities,
    hard_evaluation,
    soft_objective,
)
from relational_compression.experiments.transductive_relational_distortion.optimize import (
    InitialLogitSpec,
    OptimizedPartition,
    RestartSummary,
    optimize_partition,
)
from relational_compression.experiments.transductive_relational_distortion.source_geometry import (
    COLLISION,
    COLLISION_ENTROPY,
    EDGE,
    FOURIER,
    SOURCE_CRITERIA,
    SourceGeometry,
    load_or_compute_source_geometry,
)
from relational_compression.paths import get_experiment_name

OBJECTIVE_COMPARISON_TOLERANCE = 1e-6
CRITERION_FIELD_STEM = {
    EDGE: "D_E",
    FOURIER: "D_F",
    COLLISION: "D_C",
    COLLISION_ENTROPY: "D_H2",
}


def _json_default(value: Any) -> Any:
    """Serialize supported nonstandard values for JSON output."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _config_dict(config: Any) -> dict[str, Any]:
    """Collect serializable configuration fields."""
    result: dict[str, Any] = {}
    for name in dir(config):
        if name.startswith("_"):
            continue
        value = getattr(config, name)
        if callable(value) or getattr(value, "__name__", None) == "annotations":
            continue
        try:
            json.dumps(value, default=_json_default)
        except TypeError:
            continue
        result[name] = value
    return result


def _parse_float_list(value: str) -> tuple[float, ...]:
    """Parse float list."""
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _parse_str_list(value: str) -> tuple[str, ...]:
    """Parse str list."""
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_int_list(value: str) -> tuple[int, ...]:
    """Parse int list."""
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _resolve_device(requested: str) -> torch.device:
    """Resolve device."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _load_config(name: str) -> Any:
    """Load config."""
    return importlib.import_module(
        f"relational_compression.experiments.transductive_relational_distortion.configs.{name}"
    )


def _resolve_experiment_dir(config: Any) -> Path:
    """Resolve experiment dir."""
    if getattr(config, "experiment_name", None) is None:
        name = get_experiment_name(
            experiment_base_name=str(config.experiment_base_name),
            unique_postfix=getattr(config, "unique_postfix", None),
        )
    else:
        name = str(config.experiment_name)
    return Path(config.experiments_dir).absolute() / name


def _lambda_token(value: float) -> str:
    """Compute lambda token."""
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


def _save_partition_artifact(
    *,
    experiment_dir: Path,
    collection: str,
    source_variant: str,
    split: str,
    graph_index: int,
    optimized: OptimizedPartition,
) -> Path:
    """Save partition artifact."""
    artifact_path = (
        experiment_dir
        / "partition_artifacts"
        / str(collection)
        / str(source_variant)
        / str(split)
        / f"graph_{int(graph_index):06d}"
        / f"{optimized.criterion}_lambda{_lambda_token(optimized.lambda_org)}.pt"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        obj={
            "criterion": optimized.criterion,
            "lambda_org": float(optimized.lambda_org),
            "selected_restart": int(optimized.selected_restart),
            "selected_seed": int(optimized.selected_seed),
            "assignments": optimized.assignments.cpu(),
            "logits": optimized.logits.cpu(),
            "probabilities": optimized.probabilities.cpu(),
            "restart_summaries": [summary.__dict__ for summary in optimized.restart_summaries],
        },
        f=artifact_path,
    )
    return artifact_path.relative_to(experiment_dir)


def _finite_corr(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """Compute finite corr."""
    if x_values.size < 2 or y_values.size < 2:
        return float("nan")
    if np.isclose(x_values.std(), 0.0) or np.isclose(y_values.std(), 0.0):
        return float("nan")
    return float(np.corrcoef(x_values, y_values)[0, 1])


def _similarity_metrics(x_values: np.ndarray, y_values: np.ndarray) -> dict[str, float]:
    """Compute similarity metrics."""
    x_values = np.asarray(x_values, dtype=np.float64)
    y_values = np.asarray(y_values, dtype=np.float64)
    denominator = float(np.linalg.norm(x_values) * np.linalg.norm(y_values))
    return {
        "pearson": _finite_corr(x_values=x_values, y_values=y_values),
        "spearman": _finite_corr(x_values=rankdata(x_values), y_values=rankdata(y_values)),
        "cosine": float(np.dot(x_values, y_values) / denominator) if denominator > 0.0 else float("nan"),
        "l1_distance": float(np.abs(x_values - y_values).sum()),
        "l2_distance": float(np.linalg.norm(x_values - y_values)),
    }


def _balanced_random_assignments(num_nodes: int, num_partitions: int, rng: np.random.Generator) -> torch.Tensor:
    """Compute balanced random assignments."""
    labels = np.arange(num_nodes, dtype=np.int64) % int(num_partitions)
    rng.shuffle(labels)
    return torch.as_tensor(labels, dtype=torch.long)


def _distortion_value(hard: Any, criterion: str) -> float:
    """Compute distortion value."""
    if criterion == EDGE:
        return float(hard.edge_distortion)
    if criterion == FOURIER:
        return float(hard.fourier_distortion)
    if criterion == COLLISION:
        return float(hard.collision_distortion)
    if criterion == COLLISION_ENTROPY:
        return float(hard.collision_entropy_distortion)
    raise ValueError(f"Unsupported source distortion criterion: {criterion}")


def _source_rho_correlation_rows(
    *,
    graph_identity: dict[str, Any],
    geometry: SourceGeometry,
) -> list[dict[str, Any]]:
    """Compute source rho correlation rows."""
    rows: list[dict[str, Any]] = []
    criteria = geometry.defined_criteria
    for first, second in itertools.combinations(criteria, 2):
        rho_first = geometry.rho_for(first).detach().cpu().numpy().astype(np.float64, copy=False)
        rho_second = geometry.rho_for(second).detach().cpu().numpy().astype(np.float64, copy=False)
        rows.append(
            {
                **graph_identity,
                "criterion_a": first,
                "criterion_b": second,
                **_similarity_metrics(x_values=rho_first, y_values=rho_second),
            }
        )
    return rows


def _random_partition_correlation_rows(
    geometry: SourceGeometry,
    *,
    graph_identity: dict[str, Any],
    num_partitions: int,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Compute random partition correlation rows."""
    if count <= 1:
        return []
    criteria = geometry.defined_criteria
    values: dict[str, list[float]] = {criterion: [] for criterion in criteria}
    rng = np.random.default_rng(int(seed))
    for _ in range(int(count)):
        assignments = _balanced_random_assignments(
            num_nodes=geometry.num_nodes, num_partitions=int(num_partitions), rng=rng
        )
        hard = hard_evaluation(assignments=assignments, geometry=geometry, num_partitions=int(num_partitions))
        for criterion in criteria:
            values[criterion].append(_distortion_value(hard=hard, criterion=criterion))

    rows: list[dict[str, Any]] = []
    for first, second in itertools.combinations(criteria, 2):
        first_values = np.asarray(values[first], dtype=np.float64)
        second_values = np.asarray(values[second], dtype=np.float64)
        rows.append(
            {
                **graph_identity,
                "criterion_a": first,
                "criterion_b": second,
                "random_partition_count": int(count),
                "pearson": _finite_corr(x_values=first_values, y_values=second_values),
                "spearman": _finite_corr(x_values=rankdata(first_values), y_values=rankdata(second_values)),
                f"hard_{CRITERION_FIELD_STEM[first]}_mean": float(first_values.mean()),
                f"hard_{CRITERION_FIELD_STEM[second]}_mean": float(second_values.mean()),
            }
        )
    return rows


def _summarize_metric_rows(rows: list[dict[str, Any]], prefix: str = "") -> dict[str, float]:
    """Summarize metric rows."""
    result: dict[str, float] = {}
    if len(rows) == 0:
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
    keys = sorted(key for key in rows[0] if key not in skip)
    for key in keys:
        values = []
        for row in rows:
            value = row.get(key)
            if isinstance(value, int | float) and math.isfinite(float(value)):
                values.append(float(value))
        if len(values) == 0:
            continue
        array = np.asarray(values, dtype=np.float64)
        result[f"{prefix}{key}_mean"] = float(array.mean())
        result[f"{prefix}{key}_std"] = float(array.std(ddof=1 if array.size > 1 else 0))
    return result


def _objective_column(criterion: str) -> str:
    """Compute objective column."""
    return f"soft_total_loss_{criterion}_objective"


def _cross_objective_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compute cross objective diagnostics."""
    by_key: dict[tuple[str, str, str, int, float], dict[str, dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["collection"]),
            str(row["source_variant"]),
            str(row["split"]),
            int(row["dataset_index"]),
            float(row["lambda_org"]),
        )
        by_key.setdefault(key, {})[str(row["criterion"])] = row

    diagnostics: list[dict[str, Any]] = []
    for (collection, source_variant, split, dataset_index, lambda_org), grouped in sorted(by_key.items()):
        for target, target_row in grouped.items():
            target_column = _objective_column(target)
            target_loss = float(target_row.get(target_column, float("nan")))
            if not math.isfinite(target_loss):
                continue
            for candidate, candidate_row in grouped.items():
                candidate_loss = float(candidate_row.get(target_column, float("nan")))
                if not math.isfinite(candidate_loss):
                    continue
                diagnostics.append(
                    {
                        "collection": collection,
                        "source_variant": source_variant,
                        "split": split,
                        "dataset_index": dataset_index,
                        "original_graph_index": int(target_row["original_graph_index"]),
                        "lambda_org": lambda_org,
                        "target_criterion": target,
                        "candidate_criterion": candidate,
                        "L_target_q_target": target_loss,
                        "L_target_q_candidate": candidate_loss,
                        "candidate_advantage": target_loss - candidate_loss,
                        "objective_comparison_tolerance": OBJECTIVE_COMPARISON_TOLERANCE,
                        "candidate_beats_target": int(
                            candidate != target and candidate_loss < target_loss - OBJECTIVE_COMPARISON_TOLERANCE
                        ),
                        "target_warm_start_improved": int(target_row.get("warm_start_improved", 0)),
                        "target_warm_start_gain": float(target_row.get("warm_start_gain", 0.0)),
                    }
                )
    return diagnostics


def _cross_objective_miss_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Compute cross objective miss counts."""
    counts = {criterion: 0 for criterion in SOURCE_CRITERIA}
    for row in rows:
        if int(row.get("candidate_beats_target", 0)):
            counts[str(row["target_criterion"])] = counts.get(str(row["target_criterion"]), 0) + 1
    return counts


def _write_plots(experiment_dir: Path, aggregate: list[dict[str, Any]]) -> None:
    """Write plots."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plot_dir = experiment_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    groups = sorted({(str(row["collection"]), str(row["source_variant"]), str(row["split"])) for row in aggregate})
    colors = {
        EDGE: "#333333",
        FOURIER: "#0072B2",
        COLLISION: "#009E73",
        COLLISION_ENTROPY: "#D55E00",
    }
    for collection, source_variant, split in groups:
        rows = [
            row
            for row in aggregate
            if str(row["collection"]) == collection
            and str(row["source_variant"]) == source_variant
            and str(row["split"]) == split
        ]
        prefix = f"{collection}_{source_variant}_{split}"
        for mode, x_metric, y_metric_template, filename in (
            ("hard", "hard_h2_mean", "hard_{stem}_mean", "hard_own_complexity_distortion.png"),
            ("soft", "soft_h2_mean", "soft_{stem}_mean", "soft_own_complexity_distortion.png"),
        ):
            fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(4.8, 3.4), constrained_layout=True)
            for criterion in SOURCE_CRITERIA:
                criterion_rows = [row for row in rows if str(row["criterion"]) == criterion]
                if len(criterion_rows) == 0:
                    continue
                criterion_rows.sort(key=lambda row: float(row["lambda_org"]))
                stem = CRITERION_FIELD_STEM[criterion]
                y_metric = y_metric_template.format(stem=stem)
                axis.plot(
                    [float(row[x_metric]) for row in criterion_rows],
                    [float(row[y_metric]) for row in criterion_rows],
                    marker="o",
                    label=criterion,
                    color=colors.get(criterion, "black"),
                )
            axis.set_xlabel(f"{mode} H2")
            axis.set_ylabel(f"{mode} own distortion")
            axis.legend(fontsize=8)
            fig.savefig(plot_dir / f"{prefix}_{filename}", dpi=150)
            plt.close(fig)

        for eval_criterion in SOURCE_CRITERIA:
            stem = CRITERION_FIELD_STEM[eval_criterion]
            metric = f"hard_{stem}_mean"
            if not any(metric in row and math.isfinite(float(row[metric])) for row in rows):
                continue
            fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(4.8, 3.4), constrained_layout=True)
            for train_criterion in SOURCE_CRITERIA:
                criterion_rows = [row for row in rows if str(row["criterion"]) == train_criterion]
                if len(criterion_rows) == 0:
                    continue
                criterion_rows.sort(key=lambda row: float(row["lambda_org"]))
                axis.plot(
                    [float(row["hard_h2_mean"]) for row in criterion_rows],
                    [float(row[metric]) for row in criterion_rows],
                    marker="o",
                    label=train_criterion,
                    color=colors.get(train_criterion, "black"),
                )
            axis.set_xlabel("hard H2")
            axis.set_ylabel(f"hard {stem}")
            axis.legend(fontsize=8)
            fig.savefig(plot_dir / f"{prefix}_hard_eval_{eval_criterion}.png", dpi=150)
            plt.close(fig)


def _optimize_with_settings(
    geometry: SourceGeometry,
    *,
    criterion: str,
    lambda_org: float,
    config: Any,
    device: torch.device,
    restart_seeds: tuple[int, ...],
    include_collapse_restart: bool,
    initial_logit_specs: tuple[InitialLogitSpec, ...] = (),
) -> OptimizedPartition:
    """Compute optimize with settings."""
    return optimize_partition(
        geometry,
        criterion=criterion,
        lambda_org=float(lambda_org),
        num_partitions=int(config.num_partitions),
        temperature=float(config.assignment_temperature),
        learning_rate=float(config.learning_rate),
        optimization_steps=int(config.optimization_steps),
        init_scale=float(config.init_scale),
        restart_seeds=restart_seeds,
        include_collapse_restart=include_collapse_restart,
        collapse_init_bias=float(getattr(config, "collapse_init_bias", 4.0)),
        initial_logit_specs=initial_logit_specs,
        device=device,
    )


def _best_solution(
    candidates: list[tuple[str, OptimizedPartition]],
) -> tuple[str, OptimizedPartition, float, float]:
    """Compute best solution."""
    base_loss = float(candidates[0][1].soft_result.loss)
    best_kind, best_solution = min(candidates, key=lambda item: float(item[1].soft_result.loss))
    best_warm_loss = min(
        (float(solution.soft_result.loss) for kind, solution in candidates if kind != "ordinary_restarts"),
        default=float("nan"),
    )
    return best_kind, best_solution, base_loss, best_warm_loss


def _with_target_objective(
    solution: OptimizedPartition,
    *,
    geometry: SourceGeometry,
    target: str,
    lambda_org: float,
    temperature: float,
    device: torch.device,
) -> OptimizedPartition:
    """Compute with target objective."""
    logits = solution.logits.detach().cpu()
    with torch.no_grad():
        device_logits = logits.to(device=device)
        soft = soft_objective(
            logits=device_logits,
            geometry=geometry.to(device),
            criterion=target,
            lambda_org=float(lambda_org),
            temperature=float(temperature),
        )
        probabilities = assignment_probabilities(device_logits, temperature=float(temperature))
        assignments = probabilities.argmax(dim=-1)
    loss = float(soft.loss.detach().cpu())
    summary = RestartSummary(
        restart_index=0,
        init_kind=f"final_candidate_from_{solution.criterion}",
        seed=int(solution.selected_seed),
        best_loss=loss,
        best_step=0,
        best_own_distortion=float(soft.own_distortion.detach().cpu()),
        best_marginal_d2=float(soft.marginal_d2.detach().cpu()),
        best_soft_k_eff=float(soft.soft_k_eff.detach().cpu()),
        best_assignment_confidence=float(soft.assignment_confidence.detach().cpu()),
        best_assignment_entropy=float(soft.assignment_entropy.detach().cpu()),
        final_loss=loss,
        final_own_distortion=float(soft.own_distortion.detach().cpu()),
        final_marginal_d2=float(soft.marginal_d2.detach().cpu()),
        final_soft_k_eff=float(soft.soft_k_eff.detach().cpu()),
        assignment_confidence=float(soft.assignment_confidence.detach().cpu()),
        assignment_entropy=float(soft.assignment_entropy.detach().cpu()),
        finite=math.isfinite(loss),
        encountered_nonfinite=False,
    )
    return OptimizedPartition(
        criterion=target,
        lambda_org=float(lambda_org),
        selected_restart=0,
        selected_seed=int(solution.selected_seed),
        logits=logits,
        probabilities=probabilities.detach().cpu(),
        assignments=assignments.detach().cpu(),
        soft_result=soft,
        restart_summaries=[summary],
    )


def _final_target_candidate_closure(
    geometry: SourceGeometry,
    *,
    criteria: tuple[str, ...],
    lambda_org: float,
    candidates: dict[str, list[tuple[str, OptimizedPartition]]],
    config: Any,
    device: torch.device,
) -> dict[str, tuple[str, OptimizedPartition, float, float]]:
    """Compute final target candidate closure."""
    universe = [
        (source_target, kind, solution)
        for source_target, source_candidates in candidates.items()
        for kind, solution in source_candidates
    ]
    closed: dict[str, tuple[str, OptimizedPartition, float, float]] = {}
    for target in criteria:
        target_candidates = list(candidates[target])
        for source_target, kind, solution in universe:
            target_candidates.append(
                (
                    f"final_candidate_from_{source_target}_{kind}",
                    _with_target_objective(
                        solution,
                        geometry=geometry,
                        target=target,
                        lambda_org=float(lambda_org),
                        temperature=float(config.assignment_temperature),
                        device=device,
                    ),
                )
            )
        closed[target] = _best_solution(target_candidates)
    return closed


def _run_cross_warm_starts(
    geometry: SourceGeometry,
    *,
    lambda_org: float,
    criteria: tuple[str, ...],
    source_solutions: dict[str, OptimizedPartition],
    config: Any,
    device: torch.device,
    round_name: str,
) -> dict[str, OptimizedPartition]:
    """Run cross warm starts."""
    warm: dict[str, OptimizedPartition] = {}
    for target in criteria:
        initial_specs = tuple(
            InitialLogitSpec(
                init_kind=f"{round_name}_from_{source}",
                seed=int(source_solutions[source].selected_seed),
                logits=source_solutions[source].logits,
            )
            for source in criteria
            if source != target
        )
        warm[target] = _optimize_with_settings(
            geometry,
            criterion=target,
            lambda_org=float(lambda_org),
            config=config,
            device=device,
            restart_seeds=(),
            include_collapse_restart=False,
            initial_logit_specs=initial_specs,
        )
    return warm


def _solve_graph_lambda(
    geometry: SourceGeometry,
    *,
    criteria: tuple[str, ...],
    lambda_org: float,
    config: Any,
    device: torch.device,
) -> dict[str, tuple[str, OptimizedPartition, float, float]]:
    """Compute solve graph lambda."""
    base_solutions = {
        criterion: _optimize_with_settings(
            geometry,
            criterion=criterion,
            lambda_org=float(lambda_org),
            config=config,
            device=device,
            restart_seeds=tuple(int(seed) for seed in config.restart_seeds),
            include_collapse_restart=bool(getattr(config, "include_collapse_restart", False)),
        )
        for criterion in criteria
    }
    candidates: dict[str, list[tuple[str, OptimizedPartition]]] = {
        criterion: [("ordinary_restarts", solution)] for criterion, solution in base_solutions.items()
    }

    if bool(getattr(config, "cross_warm_start", True)) and len(criteria) > 1:
        first_warm = _run_cross_warm_starts(
            geometry,
            lambda_org=float(lambda_org),
            criteria=criteria,
            source_solutions=base_solutions,
            config=config,
            device=device,
            round_name="warm",
        )
        for criterion, solution in first_warm.items():
            candidates[criterion].append(("cross_warm_start", solution))

        best_after_first = {criterion: _best_solution(items)[1] for criterion, items in candidates.items()}
        improved = any(best_after_first[criterion] is not base_solutions[criterion] for criterion in criteria)
        if improved and int(getattr(config, "cross_warm_start_closure_rounds", 1)) > 0:
            closure_warm = _run_cross_warm_starts(
                geometry,
                lambda_org=float(lambda_org),
                criteria=criteria,
                source_solutions=best_after_first,
                config=config,
                device=device,
                round_name="closure_warm",
            )
            for criterion, solution in closure_warm.items():
                candidates[criterion].append(("closure_warm_start", solution))

    return _final_target_candidate_closure(
        geometry,
        criteria=criteria,
        lambda_org=float(lambda_org),
        candidates=candidates,
        config=config,
        device=device,
    )


def run_experiment(config: Any) -> dict[str, Any]:
    """Run experiment."""
    seed_everything(int(config.seed))
    device = _resolve_device(str(config.device))
    records, collection_metadata = load_source_graph_records(config)
    experiment_dir = _resolve_experiment_dir(config)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    effective_config = _config_dict(config)
    effective_config["resolved_device"] = str(device)
    (experiment_dir / "effective_config.json").write_text(json.dumps(effective_config, indent=2, default=_json_default))
    write_json(path=experiment_dir / "collection_metadata.json", value=collection_metadata)

    graph_rows = [graph_record_row(record) for record in records]
    write_csv(path=experiment_dir / "selected_graphs.csv", rows=graph_rows)
    write_json(path=experiment_dir / "selected_graphs.json", value=graph_rows)

    configured_geometry_cache_dir = getattr(config, "source_geometry_cache_dir", None)
    geometry_cache_dir = (
        Path(configured_geometry_cache_dir).absolute()
        if configured_geometry_cache_dir is not None
        else experiment_dir / "source_geometry_cache"
    )
    rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    source_rho_rows: list[dict[str, Any]] = []
    random_partition_rows: list[dict[str, Any]] = []

    for graph_index, record in enumerate(records):
        data = record.data
        graph_identity = {
            "collection": record.collection,
            "source_variant": record.source_variant,
            "split": record.split,
            "dataset_index": int(record.dataset_index),
            "graph_index": int(graph_index),
            "original_graph_index": int(record.original_graph_index),
        }
        geometry = load_or_compute_source_geometry(
            data,
            cache_dir=geometry_cache_dir,
            cache_version=str(config.source_geometry_cache_version),
            solve_batch_size=int(config.source_geometry_solve_batch_size),
        )
        source_rho_rows.extend(_source_rho_correlation_rows(graph_identity=graph_identity, geometry=geometry))
        random_partition_rows.extend(
            _random_partition_correlation_rows(
                geometry,
                graph_identity=graph_identity,
                num_partitions=int(config.num_partitions),
                count=int(getattr(config, "random_partition_count", 0)),
                seed=int(getattr(config, "random_partition_seed", config.seed)) + int(graph_index),
            )
        )

        requested_criteria = tuple(str(criterion) for criterion in tuple(config.criteria))
        criteria = tuple(criterion for criterion in requested_criteria if criterion in geometry.defined_criteria)
        for criterion in requested_criteria:
            if criterion not in geometry.defined_criteria:
                exclusion_rows.append(
                    {
                        **graph_identity,
                        "criterion": criterion,
                        "reason": "undefined_source_distortion",
                        "source_collision_entropy_denominator": float(geometry.collision_entropy_denominator),
                    }
                )
        if len(criteria) == 0:
            continue

        for lambda_org in tuple(config.lambda_org_values):
            solved = _solve_graph_lambda(
                geometry,
                criteria=criteria,
                lambda_org=float(lambda_org),
                config=config,
                device=device,
            )
            for criterion, (selected_candidate_kind, optimized, base_loss, best_warm_loss) in solved.items():
                artifact_path = _save_partition_artifact(
                    experiment_dir=experiment_dir,
                    collection=record.collection,
                    source_variant=record.source_variant,
                    split=record.split,
                    graph_index=graph_index,
                    optimized=optimized,
                )
                row = evaluate_optimized_partition(
                    data=data,
                    geometry=geometry,
                    optimized=optimized,
                    split=record.split,
                    graph_index=graph_index,
                    num_partitions=int(config.num_partitions),
                    optimization_steps=int(config.optimization_steps),
                    temperature=float(config.assignment_temperature),
                    artifact_path=artifact_path,
                    extra_fields={
                        "selected_candidate_kind": selected_candidate_kind,
                        "selected_init_kind": optimized.restart_summaries[optimized.selected_restart].init_kind,
                        "base_target_soft_total_loss": base_loss,
                        "warm_start_target_soft_total_loss": best_warm_loss,
                        "objective_comparison_tolerance": OBJECTIVE_COMPARISON_TOLERANCE,
                        "warm_start_improved": int(
                            math.isfinite(best_warm_loss)
                            and best_warm_loss < base_loss - OBJECTIVE_COMPARISON_TOLERANCE
                        ),
                        "warm_start_gain": base_loss - best_warm_loss if math.isfinite(best_warm_loss) else 0.0,
                    },
                )
                rows.append(row)
                print(
                    "[transductive-rel-dist "
                    f"collection={record.collection} graph={graph_index + 1}/{len(records)} "
                    f"criterion={criterion} lambda={float(lambda_org):.3g}] "
                    f"loss={row['soft_total_loss']:.4f} hard_D_E={row['hard_D_E']:.4f} "
                    f"hard_D_F={row['hard_D_F']:.4f} hard_D_C={row['hard_D_C']:.4f} "
                    f"hard_D_H2={row['hard_D_H2']:.4f} hard_Keff={row['hard_k_eff']:.3f} "
                    f"conf={row['assignment_confidence']:.3f}",
                    flush=True,
                )

    aggregate = aggregate_rows(rows)
    cross_objective_rows = _cross_objective_diagnostics(rows)
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
                1 for row in exclusion_rows if row["criterion"] == COLLISION_ENTROPY
            ),
        },
    }

    write_csv(path=experiment_dir / "per_graph_results.csv", rows=rows)
    write_csv(path=experiment_dir / "aggregate_results.csv", rows=aggregate)
    if len(source_rho_rows) > 0:
        write_csv(path=experiment_dir / "source_rho_correlations.csv", rows=source_rho_rows)
    if len(random_partition_rows) > 0:
        write_csv(path=experiment_dir / "random_partition_distortion_correlations.csv", rows=random_partition_rows)
    if len(cross_objective_rows) > 0:
        write_csv(path=experiment_dir / "cross_objective_diagnostics.csv", rows=cross_objective_rows)
    if cross_objective_misses > 0:
        write_json(path=experiment_dir / "diagnostic_summaries.json", value=diagnostic_summaries)
        raise RuntimeError(
            "Post-selection cross-objective invariant failed: "
            f"{cross_objective_misses} selected candidate misses exceed {OBJECTIVE_COMPARISON_TOLERANCE}"
        )
    if len(exclusion_rows) > 0:
        write_csv(path=experiment_dir / "criterion_exclusions.csv", rows=exclusion_rows)
    write_json(path=experiment_dir / "aggregate_results.json", value=aggregate)
    write_json(path=experiment_dir / "diagnostic_summaries.json", value=diagnostic_summaries)
    result = {
        "experiment_dir": str(experiment_dir),
        "num_rows": len(rows),
        "num_graphs": len(records),
        "aggregate": aggregate,
        "diagnostics": diagnostic_summaries,
    }
    write_json(path=experiment_dir / "result.json", value=result)
    if bool(config.write_plots):
        _write_plots(experiment_dir=experiment_dir, aggregate=aggregate)
    return result


def get_experiment_config() -> Any:
    """Return get experiment config."""
    parser = argparse.ArgumentParser(description="Transductive relational-distortion graph experiment")
    parser.add_argument("--config", type=str, default="default")
    parser.add_argument("--dataset-root-dir", type=Path, default=None)
    parser.add_argument("--tu-dataset-root-dir", type=Path, default=None)
    parser.add_argument("--dataset-backend", choices=("malnet_tiny", "synthetic"), default=None)
    parser.add_argument("--source-collections", type=str, default=None)
    parser.add_argument("--graphs-per-collection", type=int, default=None)
    parser.add_argument("--graph-seed", type=int, default=None)
    parser.add_argument("--malnet-split", type=str, default=None)
    parser.add_argument("--weight-seed", type=int, default=None)
    parser.add_argument("--weight-sigma", type=float, default=None)
    parser.add_argument("--experiments-dir", type=Path, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--unique-postfix", type=str, default=None)
    parser.add_argument("--splits", type=str, default=None, help="Deprecated alias for --malnet-split.")
    parser.add_argument("--max-graphs-per-split", type=int, default=None, help="Deprecated graph-count alias.")
    parser.add_argument(
        "--criteria",
        type=str,
        default=None,
        help="Comma-separated criteria: edge,fourier,collision,collision_entropy.",
    )
    parser.add_argument("--lambda-org-values", type=str, default=None, help="Comma-separated lambda_org values.")
    parser.add_argument("--num-partitions", type=int, default=None)
    parser.add_argument("--assignment-temperature", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--optimization-steps", type=int, default=None)
    parser.add_argument("--init-scale", type=float, default=None)
    parser.add_argument("--restart-seeds", type=str, default=None, help="Comma-separated restart seeds.")
    parser.add_argument("--include-collapse-restart", action="store_true")
    parser.add_argument("--no-collapse-restart", action="store_true")
    parser.add_argument("--collapse-init-bias", type=float, default=None)
    parser.add_argument("--cross-warm-start", action="store_true")
    parser.add_argument("--no-cross-warm-start", action="store_true")
    parser.add_argument("--cross-warm-start-closure-rounds", type=int, default=None)
    parser.add_argument("--random-partition-count", type=int, default=None)
    parser.add_argument("--random-partition-seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--source-geometry-cache-dir", type=Path, default=None)
    parser.add_argument("--source-geometry-solve-batch-size", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    config = _load_config(args.config)
    if args.dataset_root_dir is not None:
        config.dataset_root_dir = args.dataset_root_dir.absolute()
    if args.tu_dataset_root_dir is not None:
        config.tu_dataset_root_dir = args.tu_dataset_root_dir.absolute()
    if args.dataset_backend is not None:
        config.dataset_backend = args.dataset_backend
        if args.dataset_backend == "synthetic":
            config.source_collections = ("synthetic",)
    if args.source_collections is not None:
        config.source_collections = _parse_str_list(args.source_collections)
    if args.graphs_per_collection is not None:
        config.graphs_per_collection = args.graphs_per_collection
    if args.graph_seed is not None:
        config.graph_seed = args.graph_seed
    if args.malnet_split is not None:
        config.malnet_split = args.malnet_split
    if args.splits is not None:
        parsed_splits = _parse_str_list(args.splits)
        if len(parsed_splits) > 0:
            config.malnet_split = parsed_splits[0]
    if args.max_graphs_per_split is not None:
        config.graphs_per_collection = args.max_graphs_per_split
    if args.weight_seed is not None:
        config.weight_seed = args.weight_seed
    if args.weight_sigma is not None:
        config.weight_sigma = args.weight_sigma
    if args.experiments_dir is not None:
        config.experiments_dir = args.experiments_dir.absolute()
    if args.experiment_name is not None:
        config.experiment_name = args.experiment_name
    if args.unique_postfix is not None:
        config.unique_postfix = args.unique_postfix
    if args.criteria is not None:
        config.criteria = _parse_str_list(args.criteria)
    if args.lambda_org_values is not None:
        config.lambda_org_values = _parse_float_list(args.lambda_org_values)
    if args.num_partitions is not None:
        config.num_partitions = args.num_partitions
    if args.assignment_temperature is not None:
        config.assignment_temperature = args.assignment_temperature
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.optimization_steps is not None:
        config.optimization_steps = args.optimization_steps
    if args.init_scale is not None:
        config.init_scale = args.init_scale
    if args.restart_seeds is not None:
        config.restart_seeds = _parse_int_list(args.restart_seeds)
    if args.include_collapse_restart:
        config.include_collapse_restart = True
    if args.no_collapse_restart:
        config.include_collapse_restart = False
    if args.collapse_init_bias is not None:
        config.collapse_init_bias = args.collapse_init_bias
    if args.cross_warm_start:
        config.cross_warm_start = True
    if args.no_cross_warm_start:
        config.cross_warm_start = False
    if args.cross_warm_start_closure_rounds is not None:
        config.cross_warm_start_closure_rounds = args.cross_warm_start_closure_rounds
    if args.random_partition_count is not None:
        config.random_partition_count = args.random_partition_count
    if args.random_partition_seed is not None:
        config.random_partition_seed = args.random_partition_seed
    if args.device is not None:
        config.device = args.device
    if args.source_geometry_cache_dir is not None:
        config.source_geometry_cache_dir = args.source_geometry_cache_dir.absolute()
    if args.source_geometry_solve_batch_size is not None:
        config.source_geometry_solve_batch_size = args.source_geometry_solve_batch_size
    if args.no_plots:
        config.write_plots = False
    return config


def main() -> None:
    """Run the command-line entry point."""
    result = run_experiment(get_experiment_config())
    print(json.dumps({"experiment_dir": result["experiment_dir"], "num_rows": result["num_rows"]}, indent=2))


if __name__ == "__main__":
    main()
