"""Command-line utilities for reproducible experiment workflows."""

import argparse
import copy
import csv
import importlib
import json
import math
import shutil
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Sequence, cast

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from relational_compression.experiments.inductive_normalized_cut.baselines import (
    SpectralNormalizedCutConfig,
    evaluate_spectral_normalized_cut,
)
from relational_compression.experiments.inductive_normalized_cut.data import (
    PreprocessConfig,
    load_cached_splits,
    make_synthetic_splits,
    seed_everything,
)
from relational_compression.experiments.inductive_normalized_cut.metrics import (
    hard_normalized_cut,
    soft_normalized_cut,
    summarize_hard_result,
    summarize_soft_result,
    summarize_tensor,
)
from relational_compression.experiments.inductive_normalized_cut.models import make_model
from relational_compression.paths import get_experiment_dir, get_experiment_name
from relational_compression.randomness import capture_rng_state, restore_rng_state

_CONFIG_FIELD_NAMES = (
    "task_model_name",
    "dataset_name",
    "dataset_version",
    "num_experiments",
    "dataset_root_dir",
    "preprocessing_version",
    "min_nodes_per_graph",
    "max_nodes_per_graph",
    "random_walk_steps",
    "random_walk_num_probes",
    "positional_encoding_seed",
    "include_directed_features",
    "pagerank_alpha",
    "pagerank_max_iter",
    "pagerank_tolerance",
    "dataset_backend",
    "experiments_dir",
    "experiment_base_name",
    "experiment_name",
    "unique_postfix",
    "num_epochs",
    "batch_size",
    "num_workers",
    "learning_rate",
    "min_learning_rate",
    "learning_rate_warmup_steps",
    "learning_rate_warmup_start_factor",
    "weight_decay",
    "gradient_clip_norm",
    "num_partitions",
    "model_name",
    "hidden_dim",
    "num_layers",
    "assignment_temperature",
    "use_graph_context",
    "activation_name",
    "output_head_hidden_multiplier",
    "gps_heads",
    "gps_dropout",
    "gps_ffn_multiplier",
    "q_z_floor",
    "separation_weight",
    "run_spectral_baseline",
    "spectral_seed",
    "spectral_n_init",
    "spectral_max_iter",
    "spectral_tolerance",
    "spectral_cache_version",
    "spectral_absolute_margin",
    "spectral_relative_margin",
    "seed",
    "log_to_tensorboard_global",
    "tensorboard_log_steps",
    "max_train_batches",
    "max_val_batches",
    "device",
)

_CHECKPOINT_CONFIG_FIELDS = (
    "task_model_name",
    "dataset_name",
    "dataset_version",
    "preprocessing_version",
    "min_nodes_per_graph",
    "max_nodes_per_graph",
    "random_walk_steps",
    "random_walk_num_probes",
    "positional_encoding_seed",
    "include_directed_features",
    "pagerank_alpha",
    "pagerank_max_iter",
    "pagerank_tolerance",
    "dataset_backend",
    "num_epochs",
    "batch_size",
    "learning_rate",
    "min_learning_rate",
    "learning_rate_warmup_steps",
    "learning_rate_warmup_start_factor",
    "weight_decay",
    "gradient_clip_norm",
    "num_partitions",
    "model_name",
    "hidden_dim",
    "num_layers",
    "assignment_temperature",
    "use_graph_context",
    "activation_name",
    "output_head_hidden_multiplier",
    "gps_heads",
    "gps_dropout",
    "gps_ffn_multiplier",
    "q_z_floor",
    "separation_weight",
    "seed",
    "max_train_batches",
    "max_val_batches",
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
    result: dict[str, Any] = {}
    for name in _CONFIG_FIELD_NAMES:
        if not hasattr(config, name):
            continue
        value = getattr(config, name)
        if callable(value):
            continue
        try:
            json.dumps(value, default=_json_default)
        except TypeError:
            continue
        result[name] = json.loads(json.dumps(value, default=_json_default))
    return result


def _checkpoint_config_dict(config: Any) -> dict[str, Any]:
    """Collect checkpoint-relevant configuration fields."""
    return {name: getattr(config, name) for name in _CHECKPOINT_CONFIG_FIELDS if hasattr(config, name)}


def _tensorboard_enabled(config: Any) -> bool:
    """Return whether TensorBoard logging is enabled."""
    return bool(config.log_to_tensorboard_global) and int(config.tensorboard_log_steps) > 0


def _resolve_experiment_name(config: Any) -> str:
    """Resolve experiment name."""
    if getattr(config, "experiment_name", None) is not None:
        return str(config.experiment_name)
    return get_experiment_name(experiment_base_name=config.experiment_base_name, unique_postfix=config.unique_postfix)


def _run_checkpoint_dir(*, experiment_root: Path, run_name: str) -> Path:
    """Run checkpoint dir."""
    return experiment_root / "checkpoints" / run_name


def _atomic_torch_save(value: Any, path: Path) -> None:
    """Atomically save atomic torch save."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    torch.save(obj=value, f=temporary_path)
    temporary_path.replace(path)


def _clone_state_dict_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Clone state dict to cpu."""
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def make_cosine_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    min_learning_rate: float,
    warmup_steps: int,
    warmup_start_factor: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create make cosine lr scheduler."""
    if min_learning_rate < 0.0:
        raise ValueError("min_learning_rate must be non-negative")
    if warmup_steps < 0:
        raise ValueError("learning_rate_warmup_steps must be non-negative")
    if not 0.0 < warmup_start_factor <= 1.0:
        raise ValueError("learning_rate_warmup_start_factor must be in (0, 1]")
    base_learning_rate = float(optimizer.param_groups[0]["lr"])
    if min_learning_rate > base_learning_rate:
        raise ValueError("min_learning_rate must be less than or equal to learning_rate")

    total_steps = max(1, int(total_steps))
    warmup_steps = min(int(warmup_steps), total_steps)
    cosine_steps = max(1, total_steps - warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=float(min_learning_rate))
    if warmup_steps == 0:
        return cosine
    warmup = LinearLR(
        optimizer,
        start_factor=float(warmup_start_factor),
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


def _batch_metrics(batch: Batch, model: torch.nn.Module, config: Any) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute batch metrics."""
    output = model(batch)
    soft = soft_normalized_cut(
        probabilities=output.probabilities,
        edge_index=batch.edge_index,
        batch=batch.batch,
        num_graphs=batch.num_graphs,
        q_z_floor=float(config.q_z_floor),
        separation_weight=float(config.separation_weight),
    )
    hard = hard_normalized_cut(
        assignments=output.hard_ids,
        edge_index=batch.edge_index,
        num_partitions=int(config.num_partitions),
        batch=batch.batch,
        num_graphs=batch.num_graphs,
    )
    metrics: dict[str, float] = {}
    metrics.update(summarize_soft_result(soft))
    metrics.update(summarize_hard_result(hard))
    metrics["loss"] = float(soft.loss.detach().cpu())
    return soft.loss, metrics


def _batch_loss(batch: Batch, model: torch.nn.Module, config: Any) -> torch.Tensor:
    """Compute batch loss."""
    output = model(batch)
    return soft_normalized_cut(
        probabilities=output.probabilities,
        edge_index=batch.edge_index,
        batch=batch.batch,
        num_graphs=batch.num_graphs,
        q_z_floor=float(config.q_z_floor),
        separation_weight=float(config.separation_weight),
    ).loss


def _prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    """Prefix metrics."""
    return {f"{prefix}_{name}": value for name, value in metrics.items()}


def _write_history_csv(path: Path, history: list[dict[str, float]]) -> None:
    """Write history csv."""
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in history for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def evaluate_model(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: Any,
    max_batches: int | None,
) -> dict[str, float]:
    """Evaluate model."""
    model.eval()
    soft_ncuts: list[torch.Tensor] = []
    soft_nassocs: list[torch.Tensor] = []
    separation_d2s: list[torch.Tensor] = []
    effective_partitions: list[torch.Tensor] = []
    assignment_entropies: list[torch.Tensor] = []
    assignment_confidences: list[torch.Tensor] = []
    hard_ncuts: list[torch.Tensor] = []
    hard_nassocs: list[torch.Tensor] = []
    hard_active: list[torch.Tensor] = []
    min_volume: list[torch.Tensor] = []
    max_volume: list[torch.Tensor] = []
    within_edge: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = batch.to(device)
            output = model(batch)
            soft = soft_normalized_cut(
                probabilities=output.probabilities,
                edge_index=batch.edge_index,
                batch=batch.batch,
                num_graphs=batch.num_graphs,
                q_z_floor=float(config.q_z_floor),
                separation_weight=float(config.separation_weight),
            )
            hard = hard_normalized_cut(
                assignments=output.hard_ids,
                edge_index=batch.edge_index,
                num_partitions=int(config.num_partitions),
                batch=batch.batch,
                num_graphs=batch.num_graphs,
            )
            soft_ncuts.append(soft.ncut_per_graph.detach().cpu())
            soft_nassocs.append(soft.nassoc_per_graph.detach().cpu())
            separation_d2s.append(soft.separation_d2_per_graph.detach().cpu())
            effective_partitions.append(soft.effective_partitions_per_graph.detach().cpu())
            assignment_entropies.append(soft.assignment_entropy_per_graph.detach().cpu())
            assignment_confidences.append(soft.assignment_confidence_per_graph.detach().cpu())
            hard_ncuts.append(hard.ncut_per_graph.detach().cpu())
            hard_nassocs.append(hard.nassoc_per_graph.detach().cpu())
            hard_active.append(hard.active_partitions_per_graph.detach().cpu())
            min_volume.append(hard.min_partition_volume_fraction_per_graph.detach().cpu())
            max_volume.append(hard.max_partition_volume_fraction_per_graph.detach().cpu())
            within_edge.append(hard.within_edge_fraction_per_graph.detach().cpu())

    if not hard_ncuts:
        raise RuntimeError("evaluation loader produced no batches")

    metrics: dict[str, float] = {}
    metrics.update(summarize_tensor(values=torch.cat(soft_ncuts), prefix="soft_ncut"))
    metrics.update(summarize_tensor(values=torch.cat(soft_nassocs), prefix="soft_nassoc"))
    metrics.update(summarize_tensor(values=torch.cat(separation_d2s), prefix="separation_d2"))
    metrics.update(summarize_tensor(values=torch.cat(effective_partitions), prefix="effective_partitions"))
    metrics.update(summarize_tensor(values=torch.cat(assignment_entropies), prefix="assignment_entropy"))
    metrics.update(summarize_tensor(values=torch.cat(assignment_confidences), prefix="assignment_confidence"))
    metrics.update(summarize_tensor(values=torch.cat(hard_ncuts), prefix="hard_ncut"))
    metrics.update(summarize_tensor(values=torch.cat(hard_nassocs), prefix="hard_nassoc"))
    metrics.update(summarize_tensor(values=torch.cat(hard_active), prefix="hard_active_partitions"))
    metrics.update(summarize_tensor(values=torch.cat(min_volume), prefix="hard_min_volume_fraction"))
    metrics.update(summarize_tensor(values=torch.cat(max_volume), prefix="hard_max_volume_fraction"))
    metrics.update(summarize_tensor(values=torch.cat(within_edge), prefix="hard_within_edge_fraction"))
    return metrics


def _evaluate_spectral_split(
    *,
    dataset: Sequence[Data],
    split: str,
    config: Any,
    cache_dir: Path | None,
    max_graphs: int | None,
) -> dict[str, Any]:
    """Evaluate spectral split."""
    spectral_config = SpectralNormalizedCutConfig(
        num_partitions=int(config.num_partitions),
        seed=int(config.spectral_seed),
        n_init=int(config.spectral_n_init),
        max_iter=int(config.spectral_max_iter),
        tolerance=float(config.spectral_tolerance),
        cache_version=str(config.spectral_cache_version),
    )
    rows: list[dict[str, float]] = []
    for graph_index, data in enumerate(dataset):
        if max_graphs is not None and graph_index >= max_graphs:
            break
        result = evaluate_spectral_normalized_cut(data, cache_dir=cache_dir, config=spectral_config)
        rows.append(
            {
                "hard_ncut": float(result["hard_ncut"]),
                "hard_nassoc": float(result["hard_nassoc"]),
                "active_partitions": float(result["active_partitions"]),
                "hard_min_volume_fraction": float(result["hard_min_volume_fraction"]),
                "hard_max_volume_fraction": float(result["hard_max_volume_fraction"]),
                "hard_within_edge_fraction": float(result["hard_within_edge_fraction"]),
            }
        )

    summary: dict[str, Any] = {"split": split, "graph_count": len(rows)}
    if rows:
        for key in rows[0]:
            summary.update(summarize_tensor(values=torch.tensor([row[key] for row in rows]), prefix=key))
        summary["hard_ncut_values"] = [row["hard_ncut"] for row in rows]
    return summary


def _learned_hard_ncut_values(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: Any,
    max_batches: int | None,
) -> list[float]:
    """Compute learned hard ncut values."""
    model.eval()
    values: list[float] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = batch.to(device)
            output = model(batch)
            hard = hard_normalized_cut(
                assignments=output.hard_ids,
                edge_index=batch.edge_index,
                num_partitions=int(config.num_partitions),
                batch=batch.batch,
                num_graphs=batch.num_graphs,
            )
            values.extend(float(value) for value in hard.ncut_per_graph.detach().cpu())
    return values


def _spectral_gap_summary(
    learned_values: list[float],
    spectral_values: list[float],
    *,
    absolute_margin: float,
    relative_margin: float,
) -> dict[str, float]:
    """Compute spectral gap summary."""
    if len(learned_values) != len(spectral_values):
        raise ValueError("learned and spectral hard-Ncut value counts must match")
    learned_tensor = torch.tensor(learned_values, dtype=torch.float32)
    spectral_tensor = torch.tensor(spectral_values, dtype=torch.float32)
    gaps = learned_tensor - spectral_tensor
    relative_gaps = gaps / spectral_tensor.clamp_min(torch.finfo(spectral_tensor.dtype).tiny)
    return {
        "learned_minus_spectral_hard_ncut_gap_mean": float(gaps.mean()) if gaps.numel() else float("nan"),
        "learned_minus_spectral_hard_ncut_gap_median": (
            float(torch.quantile(input=gaps, q=0.5)) if gaps.numel() else float("nan")
        ),
        "learned_minus_spectral_hard_ncut_gap_min": float(gaps.min()) if gaps.numel() else float("nan"),
        "learned_minus_spectral_hard_ncut_gap_max": float(gaps.max()) if gaps.numel() else float("nan"),
        "learned_to_spectral_hard_ncut_mean_ratio": (
            float(learned_tensor.mean() / spectral_tensor.mean())
            if gaps.numel() and float(spectral_tensor.mean()) > 0
            else float("nan")
        ),
        "fraction_within_absolute_margin": (
            float((gaps.abs() <= float(absolute_margin)).float().mean()) if gaps.numel() else float("nan")
        ),
        "fraction_within_relative_margin": (
            float((relative_gaps.abs() <= float(relative_margin)).float().mean()) if gaps.numel() else float("nan")
        ),
        "absolute_margin": float(absolute_margin),
        "relative_margin": float(relative_margin),
    }


def _load_datasets(config: Any) -> tuple[dict[str, Sequence[Data]], Path | None]:
    """Load datasets."""
    if str(config.dataset_backend) == "synthetic":
        return (
            cast(
                dict[str, Sequence[Data]],
                make_synthetic_splits(
                    random_walk_steps=int(config.random_walk_steps),
                    random_walk_num_probes=int(config.random_walk_num_probes),
                    positional_encoding_seed=int(config.positional_encoding_seed),
                    seed=int(config.seed),
                ),
            ),
            None,
        )
    if str(config.dataset_backend) != "malnet_tiny":
        raise ValueError(f"Unsupported dataset_backend: {config.dataset_backend}")
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
    cache_dir = preprocess_config.cache_dir
    try:
        datasets = load_cached_splits(cache_dir)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc}\nPrepare the independent Relational Compression MalNet-Tiny cache first:\n"
            "  poetry run python -m relational_compression.experiments.inductive_normalized_cut.run_scripts.prepare_malnet_tiny"
        ) from exc
    return cast(dict[str, Sequence[Data]], datasets), cache_dir


def run_single_malnet_tiny_experiment(
    *,
    experiment_dir: Path,
    checkpoint_dir: Path,
    experiment_config: Any,
    run_index: int,
    device: torch.device,
) -> dict[str, Any]:
    """Run single malnet tiny experiment."""
    seed = int(experiment_config.seed) + run_index
    seed_everything(seed)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    datasets, _ = _load_datasets(experiment_config)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        datasets["train"],
        batch_size=int(experiment_config.batch_size),
        shuffle=True,
        num_workers=int(experiment_config.num_workers),
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        datasets["val"],
        batch_size=int(experiment_config.batch_size),
        shuffle=False,
        num_workers=int(experiment_config.num_workers),
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        datasets["test"],
        batch_size=int(experiment_config.batch_size),
        shuffle=False,
        num_workers=int(experiment_config.num_workers),
        pin_memory=pin_memory,
    )
    first_graph = datasets["train"][0]
    model = make_model(experiment_config, input_dim=int(first_graph.x.shape[-1])).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(experiment_config.learning_rate),
        weight_decay=float(experiment_config.weight_decay),
    )
    steps_per_epoch = len(train_loader)
    if experiment_config.max_train_batches is not None:
        steps_per_epoch = min(steps_per_epoch, int(experiment_config.max_train_batches))
    scheduler = make_cosine_lr_scheduler(
        optimizer,
        total_steps=max(1, int(experiment_config.num_epochs) * max(1, steps_per_epoch)),
        min_learning_rate=float(experiment_config.min_learning_rate),
        warmup_steps=int(experiment_config.learning_rate_warmup_steps),
        warmup_start_factor=float(experiment_config.learning_rate_warmup_start_factor),
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir / "latest.pt"
    tensorboard_log_dir = experiment_dir / "tensorboard_logs"
    writer = SummaryWriter(tensorboard_log_dir) if _tensorboard_enabled(experiment_config) else None
    global_step = 0
    start_epoch = 1
    best_score = float("inf")
    best_validation: dict[str, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch: int | None = None
    elapsed_before_restart = 0.0
    history: list[dict[str, float]] = []

    if latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=True)
        if checkpoint.get("config") != _checkpoint_config_dict(experiment_config):
            raise ValueError(f"Checkpoint config does not match this run: {latest_path}")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "rng_state" not in checkpoint:
            raise ValueError(f"Checkpoint does not contain RNG state and cannot be resumed exactly: {latest_path}")
        restore_rng_state(state=checkpoint["rng_state"], restore_cuda=device.type == "cuda")
        global_step = int(checkpoint["global_step"])
        start_epoch = int(checkpoint["completed_epoch"]) + 1
        best_score = float(checkpoint["best_score"])
        best_validation = checkpoint.get("best_validation")
        best_state = checkpoint.get("best_model_state_dict")
        best_epoch = checkpoint.get("best_epoch")
        elapsed_before_restart = float(checkpoint.get("wall_clock_time_seconds", 0.0))
        history = checkpoint.get("history", [])
        if start_epoch <= int(experiment_config.num_epochs):
            print(f"Resuming inductive normalized-cut run={run_index} from epoch {start_epoch} using {latest_path}")
        else:
            print(
                f"Inductive normalized-cut run={run_index} already completed through epoch "
                f"{start_epoch - 1}; using {latest_path} and skipping training"
            )

    start_time = time.time()
    for epoch in range(start_epoch, int(experiment_config.num_epochs) + 1):
        model.train()
        train_losses: list[torch.Tensor] = []
        for batch_idx, batch in enumerate(train_loader):
            if experiment_config.max_train_batches is not None and batch_idx >= int(
                experiment_config.max_train_batches
            ):
                break
            batch = batch.to(device)
            should_log_batch = writer is not None and global_step % int(experiment_config.tensorboard_log_steps) == 0
            if should_log_batch and writer is not None:
                loss, train_batch_metrics = _batch_metrics(batch=batch, model=model, config=experiment_config)
            else:
                loss = _batch_loss(batch=batch, model=model, config=experiment_config)
                train_batch_metrics = {}
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = None
            gradient_clip_norm = getattr(experiment_config, "gradient_clip_norm", None)
            if gradient_clip_norm is not None:
                if float(gradient_clip_norm) <= 0.0:
                    raise ValueError("gradient_clip_norm must be positive when set")
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters=model.parameters(), max_norm=float(gradient_clip_norm)
                )
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.detach())

            if should_log_batch and writer is not None:
                writer.add_scalar(tag="train/loss", scalar_value=loss, global_step=global_step)
                if grad_norm is not None:
                    writer.add_scalar(tag="train/gradient_norm", scalar_value=grad_norm, global_step=global_step)
                writer.add_scalar(
                    tag="train/soft_ncut", scalar_value=train_batch_metrics["soft_ncut_mean"], global_step=global_step
                )
                writer.add_scalar(
                    tag="train/separation_d2",
                    scalar_value=train_batch_metrics["separation_d2_mean"],
                    global_step=global_step,
                )
                writer.add_scalar(
                    tag="train/separation_loss",
                    scalar_value=train_batch_metrics["separation_loss"],
                    global_step=global_step,
                )
                writer.add_scalar(
                    tag="train/effective_partitions",
                    scalar_value=train_batch_metrics["effective_partitions_mean"],
                    global_step=global_step,
                )
                writer.add_scalar(
                    tag="train/hard_ncut", scalar_value=train_batch_metrics["hard_ncut_mean"], global_step=global_step
                )
                writer.add_scalar(
                    tag="train/hard_active_partitions",
                    scalar_value=train_batch_metrics["hard_active_partitions_mean"],
                    global_step=global_step,
                )
                writer.add_scalar(
                    tag="train/assignment_entropy",
                    scalar_value=train_batch_metrics["assignment_entropy_mean"],
                    global_step=global_step,
                )
                writer.add_scalar(
                    tag="train/assignment_confidence",
                    scalar_value=train_batch_metrics["assignment_confidence_mean"],
                    global_step=global_step,
                )
                writer.add_scalar(
                    tag="train/learning_rate", scalar_value=optimizer.param_groups[0]["lr"], global_step=global_step
                )
            global_step += 1

        validation = evaluate_model(
            model=model,
            loader=valid_loader,
            device=device,
            config=experiment_config,
            max_batches=experiment_config.max_val_batches,
        )
        train_loss_mean = float(torch.stack(train_losses).mean().detach().cpu()) if train_losses else float("nan")
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss_mean,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        row.update(_prefix_metrics(prefix="validate", metrics=validation))
        history.append(row)

        if writer is not None:
            for name, value in validation.items():
                if math.isfinite(value):
                    writer.add_scalar(tag=f"validate/{name}", scalar_value=value, global_step=epoch)
            writer.add_scalar(tag="train/epoch_loss", scalar_value=train_loss_mean, global_step=epoch)
            writer.flush()

        print(
            f"[inductive-ncut run={run_index} epoch={epoch}/{experiment_config.num_epochs}] "
            f"train_loss={train_loss_mean:.5g} "
            f"valid_hard_ncut={validation['hard_ncut_mean']:.5g} "
            f"valid_soft_ncut={validation['soft_ncut_mean']:.5g} "
            f"active={validation['hard_active_partitions_mean']:.5g}"
        )

        score = float(validation["hard_ncut_mean"])
        if score < best_score:
            best_score = score
            best_validation = copy.deepcopy(validation)
            best_state = _clone_state_dict_to_cpu(model)
            best_epoch = epoch
            _atomic_torch_save(
                value={
                    "model_state_dict": best_state,
                    "config": _checkpoint_config_dict(experiment_config),
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_validation": best_validation,
                    "checkpoint_metric": "hard_ncut_mean",
                    "checkpoint_metric_mode": "min",
                },
                path=checkpoint_dir / "best_model_checkpoint.pt",
            )

        _atomic_torch_save(
            value={
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "rng_state": capture_rng_state(include_cuda=device.type == "cuda"),
                "config": _checkpoint_config_dict(experiment_config),
                "run_index": run_index,
                "seed": seed,
                "completed_epoch": epoch,
                "global_step": global_step,
                "best_score": best_score,
                "best_validation": best_validation,
                "best_model_state_dict": best_state,
                "best_epoch": best_epoch,
                "wall_clock_time_seconds": elapsed_before_restart + time.time() - start_time,
                "history": history,
            },
            path=latest_path,
        )

    if writer is not None:
        writer.flush()

    if best_state is None or best_validation is None or best_epoch is None:
        raise RuntimeError("training did not complete any validation epochs")

    model.load_state_dict(best_state)
    test = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
        config=experiment_config,
        max_batches=experiment_config.max_val_batches,
    )

    spectral: dict[str, Any] = {}
    if bool(experiment_config.run_spectral_baseline):
        spectral_cache_dir = experiment_dir / "spectral_cache"
        max_eval_graphs = (
            None
            if experiment_config.max_val_batches is None
            else int(experiment_config.max_val_batches) * int(experiment_config.batch_size)
        )
        validation_learned_ncut_values = _learned_hard_ncut_values(
            model=model,
            loader=valid_loader,
            device=device,
            config=experiment_config,
            max_batches=experiment_config.max_val_batches,
        )
        test_learned_ncut_values = _learned_hard_ncut_values(
            model=model,
            loader=test_loader,
            device=device,
            config=experiment_config,
            max_batches=experiment_config.max_val_batches,
        )
        spectral["validation"] = _evaluate_spectral_split(
            dataset=datasets["val"],
            split="val",
            config=experiment_config,
            cache_dir=spectral_cache_dir,
            max_graphs=max_eval_graphs,
        )
        spectral["test"] = _evaluate_spectral_split(
            dataset=datasets["test"],
            split="test",
            config=experiment_config,
            cache_dir=spectral_cache_dir,
            max_graphs=max_eval_graphs,
        )
        spectral["validation_gap_summary"] = _spectral_gap_summary(
            learned_values=validation_learned_ncut_values,
            spectral_values=spectral["validation"].get("hard_ncut_values", []),
            absolute_margin=float(experiment_config.spectral_absolute_margin),
            relative_margin=float(experiment_config.spectral_relative_margin),
        )
        spectral["test_gap_summary"] = _spectral_gap_summary(
            learned_values=test_learned_ncut_values,
            spectral_values=spectral["test"].get("hard_ncut_values", []),
            absolute_margin=float(experiment_config.spectral_absolute_margin),
            relative_margin=float(experiment_config.spectral_relative_margin),
        )

    result: dict[str, Any] = {
        "seed": seed,
        "run_index": run_index,
        "best_epoch": best_epoch,
        "checkpoint_metric": "hard_ncut_mean",
        "checkpoint_metric_mode": "min",
        "split_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "best_validation": best_validation,
        "test": test,
        "spectral": spectral,
    }
    (experiment_dir / "result.json").write_text(json.dumps(result, indent=2, default=_json_default))
    _write_history_csv(path=experiment_dir / "history.csv", history=history)
    if writer is not None:
        writer.close()
    return result


def get_experiment_config() -> Any:
    """Return get experiment config."""
    parser = argparse.ArgumentParser(description="MalNet-Tiny inductive normalized-cut experiment runner")
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Config module below relational_compression.experiments.inductive_normalized_cut.configs.",
    )
    parser.add_argument("--num-experiments", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dataset-root-dir", type=Path, default=None)
    parser.add_argument("--dataset-backend", choices=("malnet_tiny", "synthetic"), default=None)
    parser.add_argument("--experiments-dir", type=Path, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--unique-postfix", type=str, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--random-walk-steps", type=int, default=None)
    parser.add_argument("--random-walk-num-probes", type=int, default=None)
    parser.add_argument("--positional-encoding-seed", type=int, default=None)
    parser.add_argument("--include-directed-features", action="store_true")
    parser.add_argument("--pagerank-alpha", type=float, default=None)
    parser.add_argument("--pagerank-max-iter", type=int, default=None)
    parser.add_argument("--pagerank-tolerance", type=float, default=None)
    parser.add_argument("--num-partitions", type=int, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--assignment-temperature", type=float, default=None)
    parser.add_argument("--activation-name", type=str, default=None)
    parser.add_argument("--output-head-hidden-multiplier", type=int, default=None)
    parser.add_argument("--gps-heads", type=int, default=None)
    parser.add_argument("--gps-dropout", type=float, default=None)
    parser.add_argument("--gps-ffn-multiplier", type=int, default=None)
    parser.add_argument("--separation-weight", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--min-learning-rate", type=float, default=None)
    parser.add_argument("--learning-rate-warmup-steps", type=int, default=None)
    parser.add_argument("--gradient-clip-norm", type=float, default=None)
    parser.add_argument("--spectral-seed", type=int, default=None)
    parser.add_argument("--spectral-n-init", type=int, default=None)
    parser.add_argument("--spectral-max-iter", type=int, default=None)
    parser.add_argument("--spectral-tolerance", type=float, default=None)
    parser.add_argument("--spectral-absolute-margin", type=float, default=None)
    parser.add_argument("--spectral-relative-margin", type=float, default=None)
    parser.add_argument("--disable-tensorboard", action="store_true")
    parser.add_argument("--skip-spectral", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    args: Namespace = parser.parse_args()

    cfg: Any = importlib.import_module(
        f"relational_compression.experiments.inductive_normalized_cut.configs.{args.config}"
    )
    for arg_name, config_name in (
        ("num_experiments", "num_experiments"),
        ("num_epochs", "num_epochs"),
        ("seed", "seed"),
        ("batch_size", "batch_size"),
        ("num_workers", "num_workers"),
        ("random_walk_steps", "random_walk_steps"),
        ("random_walk_num_probes", "random_walk_num_probes"),
        ("positional_encoding_seed", "positional_encoding_seed"),
        ("pagerank_alpha", "pagerank_alpha"),
        ("pagerank_max_iter", "pagerank_max_iter"),
        ("pagerank_tolerance", "pagerank_tolerance"),
        ("num_partitions", "num_partitions"),
        ("model_name", "model_name"),
        ("hidden_dim", "hidden_dim"),
        ("num_layers", "num_layers"),
        ("assignment_temperature", "assignment_temperature"),
        ("activation_name", "activation_name"),
        ("output_head_hidden_multiplier", "output_head_hidden_multiplier"),
        ("gps_heads", "gps_heads"),
        ("gps_dropout", "gps_dropout"),
        ("gps_ffn_multiplier", "gps_ffn_multiplier"),
        ("separation_weight", "separation_weight"),
        ("learning_rate", "learning_rate"),
        ("min_learning_rate", "min_learning_rate"),
        ("learning_rate_warmup_steps", "learning_rate_warmup_steps"),
        ("gradient_clip_norm", "gradient_clip_norm"),
        ("spectral_seed", "spectral_seed"),
        ("spectral_n_init", "spectral_n_init"),
        ("spectral_max_iter", "spectral_max_iter"),
        ("spectral_tolerance", "spectral_tolerance"),
        ("spectral_absolute_margin", "spectral_absolute_margin"),
        ("spectral_relative_margin", "spectral_relative_margin"),
        ("max_train_batches", "max_train_batches"),
        ("max_val_batches", "max_val_batches"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            setattr(cfg, config_name, value)
    if args.include_directed_features:
        cfg.include_directed_features = True
    if args.dataset_root_dir is not None:
        cfg.dataset_root_dir = args.dataset_root_dir.absolute()
    if args.dataset_backend is not None:
        cfg.dataset_backend = args.dataset_backend
    if args.experiments_dir is not None:
        cfg.experiments_dir = args.experiments_dir.absolute()
    if args.experiment_name is not None:
        cfg.experiment_name = args.experiment_name
    if args.unique_postfix is not None:
        cfg.unique_postfix = args.unique_postfix
    if args.disable_tensorboard:
        cfg.log_to_tensorboard_global = False
    if args.skip_spectral:
        cfg.run_spectral_baseline = False
    cfg.device = args.device
    return cfg


def run_experiment(config: Any) -> list[dict[str, Any]]:
    """Run experiment."""
    if int(config.num_experiments) < 1 or int(config.num_epochs) < 1:
        raise ValueError("num_experiments and num_epochs must be positive")

    device = torch.device(config.device)
    experiment_name = _resolve_experiment_name(config)
    experiment_dir = get_experiment_dir(experiment_name, experiments_dir=config.experiments_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(config, "config_file"):
        shutil.copyfile(config.config_file, experiment_dir / Path(config.config_file).name)
    (experiment_dir / "effective_config.json").write_text(
        json.dumps(_config_dict(config), indent=2, default=_json_default)
    )

    results: list[dict[str, Any]] = []
    for run_index in range(int(config.num_experiments)):
        run_name = f"run_{run_index:02d}_categorical"
        run_dir = experiment_dir / run_name
        results.append(
            run_single_malnet_tiny_experiment(
                experiment_dir=run_dir,
                checkpoint_dir=_run_checkpoint_dir(experiment_root=experiment_dir, run_name=run_name),
                experiment_config=config,
                run_index=run_index,
                device=device,
            )
        )
    (experiment_dir / "all_run_results.json").write_text(json.dumps(results, indent=2, default=_json_default))
    print(f"Wrote inductive normalized-cut experiment results to {experiment_dir}")
    return results


def main() -> None:
    """Run the command-line entry point."""
    run_experiment(get_experiment_config())


if __name__ == "__main__":
    main()
