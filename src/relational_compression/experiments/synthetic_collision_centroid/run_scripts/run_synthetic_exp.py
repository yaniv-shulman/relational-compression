"""Command-line utilities for reproducible experiment workflows."""

import argparse
import copy
import csv
import importlib
import json
import math
import shutil
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import SGD

from relational_compression.experiments.synthetic_collision_centroid.data import generate_imbalanced_gaussian_mixture
from relational_compression.experiments.synthetic_collision_centroid.metrics import (
    centroid_distortion,
    centroid_statistics,
    decoder_decomposition,
    decoder_distortion,
    normalized_collision_pair_distortion,
    pairwise_squared_distances,
    raw_collision_pair_distortion,
)
from relational_compression.experiments.synthetic_collision_centroid.models import (
    BinaryCodeEncoder,
    CodebookDecoder,
    factorized_code_probabilities,
    hard_code_ids,
)
from relational_compression.paths import get_experiment_dir, get_experiment_name

_CONFIG_FIELD_NAMES = (
    "task_model_name",
    "dataset_name",
    "dataset_version",
    "num_experiments",
    "experiments_dir",
    "experiment_base_name",
    "experiment_name",
    "unique_postfix",
    "num_points",
    "component_weights",
    "component_means",
    "component_covariances",
    "bits",
    "hidden_dim",
    "code_temperature",
    "num_steps",
    "learning_rate",
    "momentum",
    "random_state_samples",
    "random_state_scale_min",
    "random_state_scale_max",
    "decoder_steps",
    "decoder_learning_rate",
    "decoder_initial_scale",
    "seed",
    "log_to_tensorboard_global",
    "tensorboard_log_steps",
    "figure_dpi",
    "device",
)


def _json_default(value: Any) -> Any:
    """Serialize supported nonstandard values for JSON output."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _config_dict(config: Any) -> dict[str, Any]:
    """Collect serializable configuration fields."""
    return {name: getattr(config, name) for name in _CONFIG_FIELD_NAMES if hasattr(config, name)}


def _tensorboard_enabled(config: Any) -> bool:
    """Return whether TensorBoard logging is enabled."""
    return bool(config.log_to_tensorboard_global) and int(config.tensorboard_log_steps) > 0


def _resolve_experiment_name(config: Any) -> str:
    """Resolve experiment name."""
    if getattr(config, "experiment_name", None) is not None:
        return str(config.experiment_name)
    return get_experiment_name(experiment_base_name=config.experiment_base_name, unique_postfix=config.unique_postfix)


def _seed_everything(seed: int) -> None:
    """Seed everything."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _flatten_gradients(gradients: tuple[Tensor, ...]) -> Tensor:
    """Flatten gradients."""
    return torch.cat([gradient.reshape(-1) for gradient in gradients])


def _gradient_comparison(
    model: nn.Module,
    points: Tensor,
    squared_distances: Tensor,
    *,
    code_temperature: float,
) -> dict[str, float]:
    """Compute gradient comparison."""
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    logits = model(points)
    probabilities = factorized_code_probabilities(logits, temperature=code_temperature)
    centroid_value = centroid_distortion(points=points, code_probabilities=probabilities)
    pair_value = normalized_collision_pair_distortion(
        points=points,
        code_probabilities=probabilities,
        squared_distances=squared_distances,
    )
    centroid_gradients = torch.autograd.grad(centroid_value, parameters, retain_graph=True)
    pair_gradients = torch.autograd.grad(pair_value, parameters)
    centroid_gradient = _flatten_gradients(centroid_gradients)
    pair_gradient = _flatten_gradients(pair_gradients)
    tiny = torch.finfo(centroid_gradient.dtype).tiny
    cosine = torch.nn.functional.cosine_similarity(centroid_gradient, pair_gradient, dim=0)
    relative_error = (centroid_gradient - pair_gradient).norm() / centroid_gradient.norm().clamp_min(tiny)
    return {
        "centroid_distortion": float(centroid_value.detach().cpu()),
        "normalized_pair_distortion": float(pair_value.detach().cpu()),
        "gradient_cosine_similarity": float(cosine.detach().cpu()),
        "gradient_relative_error": float(relative_error.detach().cpu()),
    }


def _encoder_metrics(
    model: nn.Module,
    points: Tensor,
    squared_distances: Tensor,
    *,
    code_temperature: float,
) -> dict[str, float]:
    """Compute encoder metrics."""
    with torch.no_grad():
        logits = model(points)
        probabilities = factorized_code_probabilities(logits, temperature=code_temperature)
        centroid_value = centroid_distortion(points=points, code_probabilities=probabilities)
        pair_value = normalized_collision_pair_distortion(
            points=points,
            code_probabilities=probabilities,
            squared_distances=squared_distances,
        )
        raw_value = raw_collision_pair_distortion(
            points=points,
            code_probabilities=probabilities,
            squared_distances=squared_distances,
        )
        hard_ids = hard_code_ids(logits)
        active_codes = torch.unique(hard_ids).numel()
        return {
            "centroid_distortion": float(centroid_value.cpu()),
            "normalized_pair_distortion": float(pair_value.cpu()),
            "raw_collision_pair_distortion": float(raw_value.cpu()),
            "objective_absolute_difference": float((centroid_value - pair_value).abs().cpu()),
            "active_hard_codes": int(active_codes),
        }


def _train_encoder(
    *,
    model: BinaryCodeEncoder,
    points: Tensor,
    squared_distances: Tensor,
    objective: str,
    num_steps: int,
    learning_rate: float,
    momentum: float,
    code_temperature: float,
    writer: Any | None,
    tensorboard_log_steps: int,
) -> list[dict[str, float]]:
    """Train encoder."""
    if objective not in {"centroid", "pairwise"}:
        raise ValueError("objective must be 'centroid' or 'pairwise'")

    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    should_log = writer is not None and tensorboard_log_steps > 0
    history: list[dict[str, float]] = []

    for step in range(num_steps + 1):
        if step > 0:
            optimizer.zero_grad(set_to_none=True)
            probabilities = factorized_code_probabilities(model(points), temperature=code_temperature)
            if objective == "centroid":
                loss = centroid_distortion(points=points, code_probabilities=probabilities)
            else:
                loss = normalized_collision_pair_distortion(
                    points=points,
                    code_probabilities=probabilities,
                    squared_distances=squared_distances,
                )
            loss.backward()
            optimizer.step()

        metrics = _encoder_metrics(
            model=model,
            points=points,
            squared_distances=squared_distances,
            code_temperature=code_temperature,
        )
        metrics["step"] = step
        history.append(metrics)

        if should_log and writer is not None and step % tensorboard_log_steps == 0:
            prefix = f"encoder/{objective}"
            for name, value in metrics.items():
                if name != "step":
                    writer.add_scalar(tag=f"{prefix}/{name}", scalar_value=value, global_step=step)

    return history


def _random_state_diagnostics(
    *,
    points: Tensor,
    squared_distances: Tensor,
    bits: int,
    code_temperature: float,
    sample_count: int,
    scale_min: float,
    scale_max: float,
    seed: int,
) -> list[dict[str, float]]:
    """Compute random state diagnostics."""
    if sample_count < 1:
        raise ValueError("random_state_samples must be positive")
    if not 0.0 < scale_min <= scale_max:
        raise ValueError("random state scales must satisfy 0 < min <= max")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    log_min = math.log(scale_min)
    log_max = math.log(scale_max)
    rows: list[dict[str, float]] = []
    num_codes = 2**bits

    with torch.no_grad():
        for sample_index in range(sample_count):
            interpolation = float(torch.rand((), generator=generator))
            scale = math.exp(log_min + interpolation * (log_max - log_min))
            weights = torch.randn(points.shape[1], bits, generator=generator) * scale
            bias = torch.randn(bits, generator=generator) * scale
            logits = points @ weights.to(points.device) + bias.to(points.device)
            probabilities = factorized_code_probabilities(logits, temperature=code_temperature)
            centroid_value = centroid_distortion(points=points, code_probabilities=probabilities)
            pair_value = normalized_collision_pair_distortion(
                points=points,
                code_probabilities=probabilities,
                squared_distances=squared_distances,
            )
            raw_value = raw_collision_pair_distortion(
                points=points,
                code_probabilities=probabilities,
                squared_distances=squared_distances,
            )
            rows.append(
                {
                    "sample": sample_index,
                    "logit_scale": scale,
                    "centroid_distortion": float(centroid_value.cpu()),
                    "normalized_pair_distortion": float(pair_value.cpu()),
                    "raw_collision_pair_distortion": float(raw_value.cpu()),
                    "num_codes_times_raw_distortion": float((num_codes * raw_value).cpu()),
                }
            )
    return rows


def _train_decoder(
    *,
    points: Tensor,
    code_probabilities: Tensor,
    num_codes: int,
    num_steps: int,
    learning_rate: float,
    initial_scale: float,
    seed: int,
    writer: Any | None,
    tensorboard_log_steps: int,
) -> tuple[CodebookDecoder, list[dict[str, float]]]:
    """Train decoder."""
    decoder = CodebookDecoder(
        num_codes=num_codes,
        output_dim=points.shape[1],
        initial_scale=initial_scale,
        seed=seed,
    ).to(points.device)
    optimizer = SGD(decoder.parameters(), lr=learning_rate)
    should_log = writer is not None and tensorboard_log_steps > 0
    history: list[dict[str, float]] = []

    for step in range(num_steps + 1):
        if step > 0:
            optimizer.zero_grad(set_to_none=True)
            loss = decoder_distortion(
                points=points, code_probabilities=code_probabilities, decoder_vectors=decoder.code_vectors
            )
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            decomposition = decoder_decomposition(
                points=points, code_probabilities=code_probabilities, decoder_vectors=decoder.code_vectors
            )
            row = {
                "step": step,
                "decoder_distortion": float(decomposition.decoder_distortion.cpu()),
                "centroid_distortion": float(decomposition.centroid_distortion.cpu()),
                "decoder_centroid_gap": float(decomposition.decoder_centroid_gap.cpu()),
                "decomposition_residual": float(decomposition.residual.cpu()),
            }
            history.append(row)
            if should_log and writer is not None and step % tensorboard_log_steps == 0:
                for name, value in row.items():
                    if name != "step":
                        writer.add_scalar(tag=f"decoder/{name}", scalar_value=value, global_step=step)

    return decoder, history


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    """Write csv."""
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_results(
    *,
    experiment_dir: Path,
    points: Tensor,
    component_means: Tensor,
    final_model: BinaryCodeEncoder,
    code_temperature: float,
    centroid_history: list[dict[str, float]],
    pairwise_history: list[dict[str, float]],
    random_rows: list[dict[str, float]],
    decoder_history: list[dict[str, float]],
    figure_dpi: int,
) -> list[str]:
    """Plot results."""
    import matplotlib.pyplot as plt

    figures_dir = experiment_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    points_cpu = points.detach().cpu()
    component_means_cpu = component_means.detach().cpu()
    with torch.no_grad():
        logits = final_model(points)
        probabilities = factorized_code_probabilities(logits, temperature=code_temperature)
        hard_ids = hard_code_ids(logits).cpu()
        centroids = centroid_statistics(points=points, code_probabilities=probabilities).centroids.cpu()

    figure, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 9))
    axis = axes[0, 0]
    axis.scatter(points_cpu[:, 0], points_cpu[:, 1], c=hard_ids, s=9, alpha=0.65, cmap="tab10")
    axis.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="X",
        s=90,
        edgecolors="black",
        linewidths=0.8,
        label="soft code centroid",
    )
    axis.scatter(
        component_means_cpu[:, 0],
        component_means_cpu[:, 1],
        marker="+",
        s=100,
        linewidths=1.5,
        label="mixture mean",
    )
    axis.set_title("Learned hard partition and soft centroids")
    axis.set_xlabel("x1")
    axis.set_ylabel("x2")
    axis.legend(fontsize=8)

    axis = axes[0, 1]
    centroid_steps = [row["step"] for row in centroid_history]
    pairwise_steps = [row["step"] for row in pairwise_history]
    axis.plot(
        centroid_steps,
        [row["centroid_distortion"] for row in centroid_history],
        label="centroid-trained: centroid",
    )
    axis.plot(
        centroid_steps,
        [row["normalized_pair_distortion"] for row in centroid_history],
        linestyle="--",
        label="centroid-trained: pair",
    )
    axis.plot(
        pairwise_steps,
        [row["centroid_distortion"] for row in pairwise_history],
        label="pair-trained: centroid",
    )
    axis.plot(
        pairwise_steps,
        [row["normalized_pair_distortion"] for row in pairwise_history],
        linestyle="--",
        label="pair-trained: pair",
    )
    axis.set_title("Equivalent objectives during optimization")
    axis.set_xlabel("training step")
    axis.set_ylabel("distortion")
    axis.legend(fontsize=8)

    axis = axes[1, 0]
    centroid_values = [row["centroid_distortion"] for row in random_rows]
    normalized_values = [row["normalized_pair_distortion"] for row in random_rows]
    raw_values = [row["num_codes_times_raw_distortion"] for row in random_rows]
    axis.scatter(centroid_values, normalized_values, s=18, alpha=0.7, label="qZ-normalized pair")
    axis.scatter(centroid_values, raw_values, s=18, alpha=0.5, label="K x raw collision pair")
    lower = min(centroid_values + normalized_values + raw_values)
    upper = max(centroid_values + normalized_values + raw_values)
    axis.plot([lower, upper], [lower, upper], linestyle=":", linewidth=1.0, label="identity")
    axis.set_title("Normalization is what recovers centroid distortion")
    axis.set_xlabel("centroid distortion")
    axis.set_ylabel("pairwise distortion")
    axis.legend(fontsize=8)

    axis = axes[1, 1]
    decoder_steps = [row["step"] for row in decoder_history]
    axis.plot(decoder_steps, [row["decoder_distortion"] for row in decoder_history], label="decoder distortion")
    axis.plot(decoder_steps, [row["centroid_distortion"] for row in decoder_history], label="centroid distortion")
    axis.plot(decoder_steps, [row["decoder_centroid_gap"] for row in decoder_history], label="decoder-centroid gap")
    axis.set_title("Decoder distortion = centroid distortion + decoder gap")
    axis.set_xlabel("decoder step")
    axis.set_ylabel("distortion")
    axis.legend(fontsize=8)

    figure.tight_layout()
    overview_path = figures_dir / "synthetic_collision_centroid_equivalence.png"
    figure.savefig(overview_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4.5))
    decoder_total = [row["decoder_distortion"] for row in decoder_history]
    centroid_total = [row["centroid_distortion"] for row in decoder_history]
    gap = [row["decoder_centroid_gap"] for row in decoder_history]
    axis.plot(decoder_steps, decoder_total, label="decoder distortion")
    axis.plot(
        decoder_steps,
        [a + b for a, b in zip(centroid_total, gap, strict=True)],
        linestyle="--",
        label="centroid + gap",
    )
    axis.plot(decoder_steps, centroid_total, linestyle=":", label="centroid floor")
    axis.set_xlabel("decoder step")
    axis.set_ylabel("distortion")
    axis.set_title("Exact decoder-centroid decomposition")
    axis.legend()
    figure.tight_layout()
    decoder_path = figures_dir / "decoder_centroid_decomposition.png"
    figure.savefig(decoder_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(figure)

    return [str(overview_path), str(decoder_path)]


def _run_single_experiment(
    *,
    run_dir: Path,
    config: Any,
    run_index: int,
    device: torch.device,
) -> dict[str, Any]:
    """Run single experiment."""
    seed = int(config.seed) + run_index
    _seed_everything(seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    component_weights = torch.tensor(config.component_weights, dtype=torch.float32)
    component_means = torch.tensor(config.component_means, dtype=torch.float32)
    component_covariances = torch.tensor(config.component_covariances, dtype=torch.float32)
    mixture = generate_imbalanced_gaussian_mixture(
        num_points=int(config.num_points),
        component_weights=component_weights,
        component_means=component_means,
        component_covariances=component_covariances,
        seed=seed,
    )
    points = mixture.points.to(device)
    squared_distances = pairwise_squared_distances(points).detach()

    base_model = BinaryCodeEncoder(
        input_dim=points.shape[1],
        hidden_dim=int(config.hidden_dim),
        bits=int(config.bits),
    ).to(device)
    centroid_model = copy.deepcopy(base_model)
    pairwise_model = copy.deepcopy(base_model)

    tensorboard_dir = run_dir / "tensorboard_logs"
    writer = None
    if _tensorboard_enabled(config):
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(tensorboard_dir)

    initial_gradient = _gradient_comparison(
        model=base_model,
        points=points,
        squared_distances=squared_distances,
        code_temperature=float(config.code_temperature),
    )
    centroid_history = _train_encoder(
        model=centroid_model,
        points=points,
        squared_distances=squared_distances,
        objective="centroid",
        num_steps=int(config.num_steps),
        learning_rate=float(config.learning_rate),
        momentum=float(config.momentum),
        code_temperature=float(config.code_temperature),
        writer=writer,
        tensorboard_log_steps=int(config.tensorboard_log_steps),
    )
    pairwise_history = _train_encoder(
        model=pairwise_model,
        points=points,
        squared_distances=squared_distances,
        objective="pairwise",
        num_steps=int(config.num_steps),
        learning_rate=float(config.learning_rate),
        momentum=float(config.momentum),
        code_temperature=float(config.code_temperature),
        writer=writer,
        tensorboard_log_steps=int(config.tensorboard_log_steps),
    )

    final_centroid_gradient = _gradient_comparison(
        model=centroid_model,
        points=points,
        squared_distances=squared_distances,
        code_temperature=float(config.code_temperature),
    )
    final_pairwise_gradient = _gradient_comparison(
        model=pairwise_model,
        points=points,
        squared_distances=squared_distances,
        code_temperature=float(config.code_temperature),
    )

    with torch.no_grad():
        parameter_difference = torch.cat(
            [
                (centroid_parameter - pairwise_parameter).reshape(-1)
                for centroid_parameter, pairwise_parameter in zip(
                    centroid_model.parameters(),
                    pairwise_model.parameters(),
                    strict=True,
                )
            ]
        )
        parameter_rms_difference = float(parameter_difference.square().mean().sqrt().cpu())
        final_probabilities = factorized_code_probabilities(
            centroid_model(points),
            temperature=float(config.code_temperature),
        ).detach()

    random_rows = _random_state_diagnostics(
        points=points,
        squared_distances=squared_distances,
        bits=int(config.bits),
        code_temperature=float(config.code_temperature),
        sample_count=int(config.random_state_samples),
        scale_min=float(config.random_state_scale_min),
        scale_max=float(config.random_state_scale_max),
        seed=seed + 10_000,
    )
    _, decoder_history = _train_decoder(
        points=points,
        code_probabilities=final_probabilities,
        num_codes=2 ** int(config.bits),
        num_steps=int(config.decoder_steps),
        learning_rate=float(config.decoder_learning_rate),
        initial_scale=float(config.decoder_initial_scale),
        seed=seed + 20_000,
        writer=writer,
        tensorboard_log_steps=int(config.tensorboard_log_steps),
    )

    _write_csv(path=run_dir / "encoder_centroid_history.csv", rows=centroid_history)
    _write_csv(path=run_dir / "encoder_pairwise_history.csv", rows=pairwise_history)
    _write_csv(path=run_dir / "random_state_equivalence.csv", rows=random_rows)
    _write_csv(path=run_dir / "decoder_history.csv", rows=decoder_history)

    figure_paths = _plot_results(
        experiment_dir=run_dir,
        points=points,
        component_means=component_means,
        final_model=centroid_model,
        code_temperature=float(config.code_temperature),
        centroid_history=centroid_history,
        pairwise_history=pairwise_history,
        random_rows=random_rows,
        decoder_history=decoder_history,
        figure_dpi=int(config.figure_dpi),
    )

    if writer is not None:
        writer.flush()
        writer.close()

    random_normalized_errors = [
        abs(row["centroid_distortion"] - row["normalized_pair_distortion"]) for row in random_rows
    ]
    random_raw_errors = [abs(row["centroid_distortion"] - row["num_codes_times_raw_distortion"]) for row in random_rows]
    decoder_residuals = [abs(row["decomposition_residual"]) for row in decoder_history]
    result = {
        "run_index": run_index,
        "seed": seed,
        "component_counts": mixture.component_counts.tolist(),
        "initial_gradient_equivalence": initial_gradient,
        "final_centroid_model_gradient_equivalence": final_centroid_gradient,
        "final_pairwise_model_gradient_equivalence": final_pairwise_gradient,
        "final_centroid_model": centroid_history[-1],
        "final_pairwise_model": pairwise_history[-1],
        "parameter_rms_difference_between_training_views": parameter_rms_difference,
        "random_state_equivalence": {
            "samples": len(random_rows),
            "max_absolute_normalized_pair_error": max(random_normalized_errors),
            "mean_absolute_num_codes_times_raw_error": sum(random_raw_errors) / len(random_raw_errors),
        },
        "decoder_decomposition": {
            "initial": decoder_history[0],
            "final": decoder_history[-1],
            "max_absolute_residual": max(decoder_residuals),
        },
        "figures": figure_paths,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2, default=_json_default))
    return result


def run_experiment(config: Any) -> list[dict[str, Any]]:
    """Run experiment."""
    if int(config.num_experiments) < 1:
        raise ValueError("num_experiments must be positive")

    device = torch.device(config.device)
    experiment_name = _resolve_experiment_name(config)
    experiment_dir = get_experiment_dir(experiment_name, experiments_dir=Path(config.experiments_dir))
    experiment_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config.config_file, experiment_dir / Path(config.config_file).name)
    (experiment_dir / "effective_config.json").write_text(
        json.dumps(_config_dict(config), indent=2, default=_json_default)
    )

    results: list[dict[str, Any]] = []
    for run_index in range(int(config.num_experiments)):
        run_name = f"run_{run_index:02d}_deterministic"
        results.append(
            _run_single_experiment(
                run_dir=experiment_dir / run_name,
                config=config,
                run_index=run_index,
                device=device,
            )
        )

    (experiment_dir / "all_run_results.json").write_text(json.dumps(results, indent=2, default=_json_default))
    print(json.dumps(results, indent=2, default=_json_default))
    print(f"Wrote synthetic collision-centroid experiment to {experiment_dir}")
    return results


def get_experiment_config() -> Any:
    """Return get experiment config."""
    parser = argparse.ArgumentParser(description="Synthetic collision-centroid equivalence experiment")
    parser.add_argument("--config", type=str, default="default")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--unique-postfix", type=str, default=None)
    parser.add_argument("--experiments-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-experiments", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-points", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--random-state-samples", type=int, default=None)
    parser.add_argument("--decoder-steps", type=int, default=None)
    parser.add_argument("--disable-tensorboard", action="store_true")
    args: Namespace = parser.parse_args()

    config: Any = importlib.import_module(
        f"relational_compression.experiments.synthetic_collision_centroid.configs.{args.config}"
    )
    for arg_name, config_name in (
        ("num_experiments", "num_experiments"),
        ("seed", "seed"),
        ("num_points", "num_points"),
        ("num_steps", "num_steps"),
        ("learning_rate", "learning_rate"),
        ("random_state_samples", "random_state_samples"),
        ("decoder_steps", "decoder_steps"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            setattr(config, config_name, value)
    if args.experiment_name is not None:
        config.experiment_name = args.experiment_name
    if args.unique_postfix is not None:
        config.unique_postfix = args.unique_postfix
    if args.experiments_dir is not None:
        config.experiments_dir = args.experiments_dir.absolute()
    if args.disable_tensorboard:
        config.log_to_tensorboard_global = False
    config.device = args.device
    return config


def main() -> None:
    """Run the command-line entry point."""
    config = get_experiment_config()
    run_experiment(config)


if __name__ == "__main__":
    main()
