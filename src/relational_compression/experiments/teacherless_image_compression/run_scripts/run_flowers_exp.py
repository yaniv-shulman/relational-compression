"""Command-line utilities for reproducible experiment workflows."""

import argparse
import copy
import importlib
import json
import math
import shutil
import time
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sized, cast

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from relational_compression.experiments.teacherless_image_compression.config import ExperimentConfig
from relational_compression.experiments.teacherless_image_compression.data import (
    build_flowers102_datasets,
    seed_everything,
)
from relational_compression.experiments.teacherless_image_compression.run import _make_model, _validate
from relational_compression.models.image_autoencoder import BinaryImageAutoencoder
from relational_compression.paths import get_experiment_dir, get_experiment_name
from relational_compression.randomness import capture_rng_state, restore_rng_state

_CONFIG_FIELD_NAMES = (
    "task_model_name",
    "dataset_name",
    "dataset_version",
    "num_experiments",
    "dataset_root_dir",
    "experiments_dir",
    "experiment_base_name",
    "experiment_name",
    "unique_postfix",
    "num_epochs",
    "image_size",
    "batch_size",
    "num_workers",
    "learning_rate",
    "min_learning_rate",
    "learning_rate_warmup_steps",
    "learning_rate_warmup_start_factor",
    "weight_decay",
    "bits",
    "hidden",
    "downsample_layers",
    "residual_layers",
    "residual_hidden",
    "grouping",
    "concentration_weight",
    "separation_weight",
    "separation_temperature",
    "separation_margin_gain",
    "threshold_scale_ema_decay",
    "threshold_scale_min",
    "download",
    "modes",
    "seed",
    "diagnostic_pair_samples",
    "log_to_tensorboard_global",
    "tensorboard_log_steps",
    "max_train_batches",
    "max_val_batches",
    "device",
)


def _json_default(value: Any) -> Any:
    """Serialize supported nonstandard values for JSON output."""
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, tuple):
        return list(value)

    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _config_dict(experiment_config: Any) -> dict[str, Any]:
    """Collect serializable configuration fields."""
    result: dict[str, Any] = {}

    for name in _CONFIG_FIELD_NAMES:
        if not hasattr(experiment_config, name):
            continue

        value = getattr(experiment_config, name)

        if callable(value):
            continue

        try:
            json.dumps(value, default=_json_default)
        except TypeError:
            continue

        result[name] = value

    return result


def _tensorboard_enabled(experiment_config: Any) -> bool:
    """Return whether TensorBoard logging is enabled."""
    return bool(experiment_config.log_to_tensorboard_global) and int(experiment_config.tensorboard_log_steps) > 0


def _as_experiment_config(experiment_config: Any) -> ExperimentConfig:
    """Compute as experiment config."""
    return ExperimentConfig(
        dataset_root=experiment_config.dataset_root_dir,
        output_root=experiment_config.experiments_dir,
        image_size=experiment_config.image_size,
        batch_size=experiment_config.batch_size,
        num_workers=experiment_config.num_workers,
        learning_rate=experiment_config.learning_rate,
        weight_decay=experiment_config.weight_decay,
        bits=experiment_config.bits,
        hidden=experiment_config.hidden,
        downsample_layers=experiment_config.downsample_layers,
        residual_layers=experiment_config.residual_layers,
        residual_hidden=experiment_config.residual_hidden,
        grouping=experiment_config.grouping,
        concentration_weight=getattr(experiment_config, "concentration_weight", ExperimentConfig.concentration_weight),
        seed=experiment_config.seed,
        diagnostic_pair_samples=experiment_config.diagnostic_pair_samples,
    )


def _checkpoint_config_dict(
    experiment_config: Any,
    *,
    model_config: ExperimentConfig,
) -> dict[str, Any]:
    """Collect configuration that affects resumed training."""
    checkpoint_config = asdict(model_config)
    checkpoint_config.update(
        {
            "num_epochs": int(experiment_config.num_epochs),
            "max_train_batches": experiment_config.max_train_batches,
            "max_val_batches": experiment_config.max_val_batches,
            "min_learning_rate": float(getattr(experiment_config, "min_learning_rate", 0.0)),
            "learning_rate_warmup_steps": int(getattr(experiment_config, "learning_rate_warmup_steps", 0)),
            "learning_rate_warmup_start_factor": float(
                getattr(experiment_config, "learning_rate_warmup_start_factor", 0.1)
            ),
            "separation_weight": float(getattr(experiment_config, "separation_weight", 0.0)),
            "separation_temperature": float(getattr(experiment_config, "separation_temperature", 1.0)),
            "separation_margin_gain": float(getattr(experiment_config, "separation_margin_gain", 3.0)),
            "threshold_scale_ema_decay": float(getattr(experiment_config, "threshold_scale_ema_decay", 0.99)),
            "threshold_scale_min": float(getattr(experiment_config, "threshold_scale_min", 1e-3)),
        }
    )
    return checkpoint_config


def get_experiment_config() -> Any:
    """Return get experiment config."""
    parser = argparse.ArgumentParser(description="Flowers102 teacherless image-compression experiment runner")
    parser.add_argument(
        "--config",
        type=str,
        default="flowers102.flowers102_mean_group",
        help="Config module below relational_compression.experiments.teacherless_image_compression.configs.",
    )
    parser.add_argument("--num-experiments", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dataset-root-dir", type=Path, default=None)
    parser.add_argument("--experiments-dir", type=Path, default=None)
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Exact experiment directory name to create or resume below the configured experiments directory.",
    )
    parser.add_argument("--unique-postfix", type=str, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mode", choices=("deterministic", "stochastic", "both"), default=None)
    parser.add_argument("--min-learning-rate", type=float, default=None)
    parser.add_argument("--learning-rate-warmup-steps", type=int, default=None)
    parser.add_argument("--learning-rate-warmup-start-factor", type=float, default=None)
    parser.add_argument("--separation-weight", type=float, default=None)
    parser.add_argument("--separation-temperature", type=float, default=None)
    parser.add_argument("--separation-margin-gain", type=float, default=None)
    parser.add_argument("--threshold-scale-ema-decay", type=float, default=None)
    parser.add_argument("--threshold-scale-min", type=float, default=None)
    parser.add_argument("--disable-tensorboard", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--max-train-batches", type=int, default=None, help="Convenience/smoke-test limit; omit for full epochs"
    )
    parser.add_argument(
        "--max-val-batches", type=int, default=None, help="Convenience/smoke-test limit; omit for full validation"
    )

    args: Namespace = parser.parse_args()
    config_module_path = f"relational_compression.experiments.teacherless_image_compression.configs.{args.config}"
    cfg: Any = importlib.import_module(config_module_path)

    if args.num_experiments is not None:
        cfg.num_experiments = args.num_experiments

    if args.num_epochs is not None:
        cfg.num_epochs = args.num_epochs

    if args.seed is not None:
        cfg.seed = args.seed

    if args.dataset_root_dir is not None:
        cfg.dataset_root_dir = args.dataset_root_dir.absolute()

    if args.experiments_dir is not None:
        cfg.experiments_dir = args.experiments_dir.absolute()

    cfg.experiment_name = args.experiment_name

    if args.unique_postfix is not None:
        cfg.unique_postfix = args.unique_postfix

    if args.mode is not None:
        cfg.modes = ("deterministic", "stochastic") if args.mode == "both" else (args.mode,)

    if args.min_learning_rate is not None:
        cfg.min_learning_rate = args.min_learning_rate

    if args.learning_rate_warmup_steps is not None:
        cfg.learning_rate_warmup_steps = args.learning_rate_warmup_steps

    if args.learning_rate_warmup_start_factor is not None:
        cfg.learning_rate_warmup_start_factor = args.learning_rate_warmup_start_factor

    if args.separation_weight is not None:
        cfg.separation_weight = args.separation_weight

    if args.separation_temperature is not None:
        cfg.separation_temperature = args.separation_temperature

    if args.separation_margin_gain is not None:
        cfg.separation_margin_gain = args.separation_margin_gain

    if args.threshold_scale_ema_decay is not None:
        cfg.threshold_scale_ema_decay = args.threshold_scale_ema_decay

    if args.threshold_scale_min is not None:
        cfg.threshold_scale_min = args.threshold_scale_min

    if args.disable_tensorboard:
        cfg.log_to_tensorboard_global = False

    if args.no_download:
        cfg.download = False

    if args.max_train_batches is not None:
        cfg.max_train_batches = args.max_train_batches

    if args.max_val_batches is not None:
        cfg.max_val_batches = args.max_val_batches

    cfg.device = args.device
    return cfg


def _clone_state_dict_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Clone state dict to cpu."""
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _atomic_torch_save(value: Any, path: Path) -> None:
    """Atomically save atomic torch save."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    torch.save(obj=value, f=temporary_path)
    temporary_path.replace(path)


def _sigmoid_soft_code_probabilities_from_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Compute sigmoid soft code probabilities from logits."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    return torch.sigmoid(logits / temperature)


def _threshold_normalized_margins_from_logits(logits: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Compute threshold normalized margins from logits."""
    if torch.any(scales <= 0.0):
        raise ValueError("threshold scales must be positive")

    return logits / scales


def _threshold_soft_code_probabilities_from_logits(
    logits: torch.Tensor,
    scales: torch.Tensor,
    margin_gain: float,
) -> torch.Tensor:
    """Compute threshold soft code probabilities from logits."""
    if margin_gain <= 0.0:
        raise ValueError("separation_margin_gain must be positive")

    return torch.sigmoid(margin_gain * _threshold_normalized_margins_from_logits(logits=logits, scales=scales))


def _mean_soft_pair_collision_log(
    soft_code_probabilities: torch.Tensor, pair_samples: int | None = None
) -> torch.Tensor:
    """Compute mean soft pair collision log."""
    flat = soft_code_probabilities.permute(0, 2, 3, 1).reshape(-1, soft_code_probabilities.shape[1])

    if pair_samples is None:
        left = flat[:, None, :]
        right = flat[None, :, :]
        sample_count = flat.shape[0] * flat.shape[0]
    else:
        sample_count = max(1, min(pair_samples, flat.shape[0] * flat.shape[0]))
        left_indices = torch.randint(high=flat.shape[0], size=(sample_count,), device=flat.device)
        right_indices = torch.randint(high=flat.shape[0], size=(sample_count,), device=flat.device)
        left = flat[left_indices]
        right = flat[right_indices]

    agreement = left * right + (1.0 - left) * (1.0 - right)
    log_collision = agreement.clamp_min(torch.finfo(agreement.dtype).tiny).log().sum(dim=-1)
    return torch.logsumexp(log_collision.reshape(-1), dim=0) - math.log(sample_count)


def _sigmoid_separation_d2_from_logits(
    logits: torch.Tensor,
    temperature: float,
    pair_samples: int | None = None,
) -> torch.Tensor:
    """Compute sigmoid separation d2 from logits."""
    soft_code_probabilities = _sigmoid_soft_code_probabilities_from_logits(logits=logits, temperature=temperature)
    return logits.shape[1] * math.log(2.0) + _mean_soft_pair_collision_log(
        soft_code_probabilities=soft_code_probabilities, pair_samples=pair_samples
    )


def _threshold_separation_d2_from_logits(
    logits: torch.Tensor,
    scales: torch.Tensor,
    margin_gain: float,
    pair_samples: int | None = None,
) -> torch.Tensor:
    """Compute threshold separation d2 from logits."""
    soft_code_probabilities = _threshold_soft_code_probabilities_from_logits(
        logits=logits, scales=scales, margin_gain=margin_gain
    )
    return logits.shape[1] * math.log(2.0) + _mean_soft_pair_collision_log(
        soft_code_probabilities=soft_code_probabilities, pair_samples=pair_samples
    )


def _make_cosine_lr_scheduler(
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

    total_steps = max(1, total_steps)
    warmup_steps = min(warmup_steps, total_steps)

    base_learning_rate = float(optimizer.param_groups[0]["lr"])
    if min_learning_rate > base_learning_rate:
        raise ValueError("min_learning_rate must be less than or equal to learning_rate")

    cosine_steps = max(1, total_steps - warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=min_learning_rate)
    if warmup_steps == 0:
        return cosine

    warmup = LinearLR(optimizer, start_factor=warmup_start_factor, end_factor=1.0, total_iters=warmup_steps)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


def _resolve_experiment_name(experiment_config: Any) -> str:
    """Resolve experiment name."""
    experiment_name = getattr(experiment_config, "experiment_name", None)
    if experiment_name is not None:
        return str(experiment_name)

    return get_experiment_name(
        experiment_base_name=experiment_config.experiment_base_name, unique_postfix=experiment_config.unique_postfix
    )


def _get_run_checkpoint_dir(experiment_dir: Path, run_name: str) -> Path:
    """Return get run checkpoint dir."""
    return experiment_dir / "checkpoints" / run_name


def _make_input_reconstruction_grid(images: torch.Tensor, reconstructions: torch.Tensor) -> torch.Tensor:
    """Create make input reconstruction grid."""
    pair = torch.stack((images[0], reconstructions[0])).detach().cpu().clamp(0.0, 1.0)
    return cast(torch.Tensor, make_grid(pair, nrow=2))


def _log_tensor_distribution(writer: SummaryWriter, tag_prefix: str, tensor: torch.Tensor, step: int) -> None:
    """Compute log tensor distribution."""
    values = tensor.detach()
    float_values = values.float()
    writer.add_histogram(tag=f"{tag_prefix}/histogram", values=values, global_step=step)
    writer.add_scalar(tag=f"{tag_prefix}/mean", scalar_value=float_values.mean(), global_step=step)
    writer.add_scalar(tag=f"{tag_prefix}/std", scalar_value=float_values.std(unbiased=False), global_step=step)
    writer.add_scalar(tag=f"{tag_prefix}/min", scalar_value=float_values.min(), global_step=step)
    writer.add_scalar(tag=f"{tag_prefix}/max", scalar_value=float_values.max(), global_step=step)
    writer.add_scalar(tag=f"{tag_prefix}/abs_mean", scalar_value=float_values.abs().mean(), global_step=step)
    writer.add_scalar(tag=f"{tag_prefix}/rms", scalar_value=float_values.square().mean().sqrt(), global_step=step)


def _log_threshold_probability_diagnostics(
    writer: SummaryWriter,
    tag_prefix: str,
    logits: torch.Tensor,
    threshold_scales: torch.Tensor,
    margin_gain: float,
    step: int,
) -> None:
    """Compute log threshold probability diagnostics."""
    margins = _threshold_normalized_margins_from_logits(logits=logits, scales=threshold_scales)
    probabilities = _threshold_soft_code_probabilities_from_logits(
        logits=logits, scales=threshold_scales, margin_gain=margin_gain
    )
    signed_confidence = 2.0 * probabilities - 1.0
    saturation = ((probabilities < 0.01) | (probabilities > 0.99)).float().mean()

    _log_tensor_distribution(writer=writer, tag_prefix=f"{tag_prefix}/normalized_margin", tensor=margins, step=step)
    _log_tensor_distribution(
        writer=writer, tag_prefix=f"{tag_prefix}/threshold_probability", tensor=probabilities, step=step
    )
    _log_tensor_distribution(
        writer=writer, tag_prefix=f"{tag_prefix}/signed_confidence", tensor=signed_confidence, step=step
    )
    writer.add_scalar(tag=f"{tag_prefix}/probability_saturation_fraction", scalar_value=saturation, global_step=step)


def _log_validation_reconstruction_images(
    *,
    writer: SummaryWriter,
    model: BinaryImageAutoencoder,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    log_separation: bool = False,
    separation_temperature: float = 1.0,
    separation_margin_gain: float = 3.0,
    threshold_scale_ema_decay: float = 0.99,
    threshold_scale_min: float = 1e-3,
    pair_samples: int | None = None,
) -> None:
    """Compute log validation reconstruction images."""
    was_training = model.training
    model.eval()

    try:
        images = next(iter(loader)).to(device)

        with torch.no_grad():
            hard_output = model(images, force_hard=True)

            model.quantizer.train()
            relaxed_output = model(images)
            model.quantizer.eval()

        writer.add_image(
            tag="validate/input_hard_reconstruction",
            img_tensor=_make_input_reconstruction_grid(images=images, reconstructions=hard_output.reconstruction),
            global_step=epoch,
        )
        writer.add_image(
            tag="validate/input_relaxed_reconstruction",
            img_tensor=_make_input_reconstruction_grid(images=images, reconstructions=relaxed_output.reconstruction),
            global_step=epoch,
        )
        _log_tensor_distribution(
            writer=writer,
            tag_prefix="validate/transformed_group_mean_logits",
            tensor=relaxed_output.quantizer.relaxed,
            step=epoch,
        )
        _log_tensor_distribution(
            writer=writer, tag_prefix="validate/raw_logits", tensor=relaxed_output.logits, step=epoch
        )
        if log_separation:
            threshold_scales = model.threshold_scales(
                relaxed_output.logits,
                ema_decay=threshold_scale_ema_decay,
                scale_min=threshold_scale_min,
                update=False,
            )
            threshold_separation_d2 = _threshold_separation_d2_from_logits(
                logits=relaxed_output.logits,
                scales=threshold_scales,
                margin_gain=separation_margin_gain,
                pair_samples=pair_samples,
            )
            sigmoid_separation_d2 = _sigmoid_separation_d2_from_logits(
                logits=relaxed_output.logits,
                temperature=separation_temperature,
                pair_samples=pair_samples,
            )
            writer.add_scalar(
                tag="validate/threshold_separation_d2", scalar_value=threshold_separation_d2, global_step=epoch
            )
            writer.add_scalar(
                tag="validate/sigmoid_separation_d2", scalar_value=sigmoid_separation_d2, global_step=epoch
            )
            writer.add_scalar(
                tag="validate/separation_margin_gain", scalar_value=separation_margin_gain, global_step=epoch
            )
            _log_tensor_distribution(
                writer=writer, tag_prefix="validate/threshold_scale", tensor=threshold_scales.flatten(), step=epoch
            )
            _log_threshold_probability_diagnostics(
                writer=writer,
                tag_prefix="validate",
                logits=relaxed_output.logits,
                threshold_scales=threshold_scales,
                margin_gain=separation_margin_gain,
                step=epoch,
            )
    finally:
        model.train(was_training)


def run_single_flowers102_experiment(
    *,
    experiment_dir: Path,
    checkpoint_dir: Path,
    experiment_config: Any,
    mode: str,
    run_index: int,
    device: torch.device,
) -> dict[str, Any]:
    """Run single flowers102 experiment."""
    stochastic = mode == "stochastic"
    seed = experiment_config.seed + run_index
    seed_everything(seed)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resume_checkpoint_path = checkpoint_dir / "latest.pt"

    datasets = build_flowers102_datasets(
        root=experiment_config.dataset_root_dir,
        image_size=experiment_config.image_size,
        download=experiment_config.download,
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        datasets["train"],
        batch_size=experiment_config.batch_size,
        shuffle=True,
        num_workers=experiment_config.num_workers,
        pin_memory=pin_memory,
    )

    valid_loader = DataLoader(
        datasets["valid"],
        batch_size=experiment_config.batch_size,
        shuffle=False,
        num_workers=experiment_config.num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        datasets["test"],
        batch_size=experiment_config.batch_size,
        shuffle=False,
        num_workers=experiment_config.num_workers,
        pin_memory=pin_memory,
    )

    model_config = _as_experiment_config(experiment_config)
    checkpoint_config = _checkpoint_config_dict(experiment_config, model_config=model_config)
    model = _make_model(model_config, stochastic=stochastic).to(device)
    concentration_weight = float(getattr(experiment_config, "concentration_weight", model_config.concentration_weight))
    separation_weight = float(getattr(experiment_config, "separation_weight", 0.0))
    separation_temperature = float(getattr(experiment_config, "separation_temperature", 1.0))
    if separation_temperature <= 0.0:
        raise ValueError("separation_temperature must be positive")
    separation_margin_gain = float(getattr(experiment_config, "separation_margin_gain", 3.0))
    if separation_margin_gain <= 0.0:
        raise ValueError("separation_margin_gain must be positive")
    log_separation_diagnostics = not stochastic
    use_separation = log_separation_diagnostics and separation_weight != 0.0
    threshold_scale_ema_decay = float(getattr(experiment_config, "threshold_scale_ema_decay", 0.99))
    threshold_scale_min = float(getattr(experiment_config, "threshold_scale_min", 1e-3))

    optimizer = AdamW(
        model.parameters(),
        lr=experiment_config.learning_rate,
        weight_decay=experiment_config.weight_decay,
    )
    mse = nn.MSELoss()
    writer = None
    tensorboard_log_dir = experiment_dir / "tensorboard_logs"

    if _tensorboard_enabled(experiment_config):
        writer = SummaryWriter(tensorboard_log_dir)

    global_step = 0
    steps_per_epoch = len(train_loader)
    if experiment_config.max_train_batches is not None:
        steps_per_epoch = min(steps_per_epoch, experiment_config.max_train_batches)
    total_steps = max(1, experiment_config.num_epochs * max(1, steps_per_epoch))
    scheduler = _make_cosine_lr_scheduler(
        optimizer,
        total_steps=total_steps,
        min_learning_rate=float(getattr(experiment_config, "min_learning_rate", 0.0)),
        warmup_steps=int(getattr(experiment_config, "learning_rate_warmup_steps", 0)),
        warmup_start_factor=float(getattr(experiment_config, "learning_rate_warmup_start_factor", 0.1)),
    )
    best_val_hard_mse = float("inf")
    best_metrics: dict[str, float] | None = None
    best_model_state: dict[str, torch.Tensor] | None = None
    best_epoch: int | None = None
    start_epoch = 1
    elapsed_before_restart = 0.0

    if resume_checkpoint_path.exists():
        checkpoint = torch.load(resume_checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("config") != checkpoint_config:
            raise ValueError(f"Checkpoint config does not match this run: {resume_checkpoint_path}")
        if checkpoint.get("mode") != mode or checkpoint.get("run_index") != run_index or checkpoint.get("seed") != seed:
            raise ValueError(f"Checkpoint identity does not match this run: {resume_checkpoint_path}")

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "rng_state" not in checkpoint:
            raise ValueError(
                f"Checkpoint does not contain RNG state and cannot be resumed exactly: {resume_checkpoint_path}"
            )
        restore_rng_state(state=checkpoint["rng_state"])
        global_step = int(checkpoint["global_step"])
        best_val_hard_mse = float(checkpoint["best_val_hard_mse"])
        best_metrics = checkpoint.get("best_metrics")
        best_model_state = checkpoint.get("best_model_state")
        best_epoch = checkpoint.get("best_epoch")
        start_epoch = int(checkpoint["completed_epoch"]) + 1
        elapsed_before_restart = float(checkpoint.get("wall_clock_time_seconds", 0.0))
        print(f"Resuming flowers102 {mode} run={run_index} from epoch {start_epoch} using {resume_checkpoint_path}")

    start_time = time.time()

    for epoch in range(start_epoch, experiment_config.num_epochs + 1):
        model.train()

        for batch_idx, images in enumerate(train_loader):
            if experiment_config.max_train_batches is not None and batch_idx >= experiment_config.max_train_batches:
                break

            images = images.to(device)
            output = model(images)
            distortion = mse(output.reconstruction, images)
            concentration = output.quantizer.concentration_loss
            separation_d2 = None
            threshold_separation_d2 = None
            sigmoid_separation_d2 = None
            threshold_scales = None
            separation_contribution = output.logits.new_zeros(())

            if log_separation_diagnostics:
                threshold_scales = model.threshold_scales(
                    output.logits,
                    ema_decay=threshold_scale_ema_decay,
                    scale_min=threshold_scale_min,
                    update=True,
                )
                threshold_separation_d2 = _threshold_separation_d2_from_logits(
                    logits=output.logits,
                    scales=threshold_scales,
                    margin_gain=separation_margin_gain,
                    pair_samples=experiment_config.diagnostic_pair_samples,
                )
                sigmoid_separation_d2 = _sigmoid_separation_d2_from_logits(
                    logits=output.logits,
                    temperature=separation_temperature,
                    pair_samples=experiment_config.diagnostic_pair_samples,
                )
                separation_d2 = threshold_separation_d2
                if use_separation:
                    separation_contribution = separation_weight * separation_d2

            loss = distortion + concentration_weight * concentration + separation_contribution

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if writer is not None and global_step % experiment_config.tensorboard_log_steps == 0:
                writer.add_scalar(tag="train/loss", scalar_value=loss, global_step=global_step)
                writer.add_scalar(tag="train/l2_distortion", scalar_value=distortion, global_step=global_step)
                writer.add_scalar(tag="train/concentration", scalar_value=concentration, global_step=global_step)
                writer.add_scalar(
                    tag="train/relaxed_hard_rms", scalar_value=concentration.sqrt(), global_step=global_step
                )
                writer.add_scalar(
                    tag="train/learning_rate", scalar_value=optimizer.param_groups[0]["lr"], global_step=global_step
                )
                _log_tensor_distribution(
                    writer=writer,
                    tag_prefix="train/transformed_group_mean_logits",
                    tensor=output.quantizer.relaxed,
                    step=global_step,
                )
                _log_tensor_distribution(
                    writer=writer, tag_prefix="train/raw_logits", tensor=output.logits, step=global_step
                )
                if log_separation_diagnostics and separation_d2 is not None:
                    writer.add_scalar(tag="train/separation_d2", scalar_value=separation_d2, global_step=global_step)
                    writer.add_scalar(
                        tag="train/separation_weighted", scalar_value=separation_contribution, global_step=global_step
                    )
                    writer.add_scalar(
                        tag="train/separation_temperature", scalar_value=separation_temperature, global_step=global_step
                    )
                    writer.add_scalar(
                        tag="train/separation_margin_gain", scalar_value=separation_margin_gain, global_step=global_step
                    )
                    if threshold_separation_d2 is not None:
                        writer.add_scalar(
                            tag="train/threshold_separation_d2",
                            scalar_value=threshold_separation_d2,
                            global_step=global_step,
                        )
                    if sigmoid_separation_d2 is not None:
                        writer.add_scalar(
                            tag="train/sigmoid_separation_d2",
                            scalar_value=sigmoid_separation_d2,
                            global_step=global_step,
                        )
                    if threshold_scales is not None:
                        _log_tensor_distribution(
                            writer=writer,
                            tag_prefix="train/threshold_scale",
                            tensor=threshold_scales.flatten(),
                            step=global_step,
                        )
                        _log_threshold_probability_diagnostics(
                            writer=writer,
                            tag_prefix="train",
                            logits=output.logits,
                            threshold_scales=threshold_scales,
                            margin_gain=separation_margin_gain,
                            step=global_step,
                        )
                writer.add_image(
                    tag="train/input_reconstruction",
                    img_tensor=_make_input_reconstruction_grid(images=images, reconstructions=output.reconstruction),
                    global_step=global_step,
                )
                writer.add_image(
                    tag="train/input_relaxed_reconstruction",
                    img_tensor=_make_input_reconstruction_grid(images=images, reconstructions=output.reconstruction),
                    global_step=global_step,
                )
                with torch.no_grad():
                    hard_reconstruction = model.reconstruct_hard(images)
                writer.add_image(
                    tag="train/input_hard_reconstruction",
                    img_tensor=_make_input_reconstruction_grid(images=images, reconstructions=hard_reconstruction),
                    global_step=global_step,
                )

            scheduler.step()
            global_step += 1

        metrics = _validate(
            model=model,
            loader=valid_loader,
            device=device,
            config=model_config,
            max_batches=experiment_config.max_val_batches,
        )

        if writer is not None:
            for name, value in metrics.items():
                if math.isfinite(value):
                    writer.add_scalar(tag=f"validate/{name}", scalar_value=value, global_step=epoch)
            _log_validation_reconstruction_images(
                writer=writer,
                model=model,
                loader=valid_loader,
                device=device,
                epoch=epoch,
                log_separation=log_separation_diagnostics,
                separation_temperature=separation_temperature,
                separation_margin_gain=separation_margin_gain,
                threshold_scale_ema_decay=threshold_scale_ema_decay,
                threshold_scale_min=threshold_scale_min,
                pair_samples=experiment_config.diagnostic_pair_samples,
            )

        print(
            f"[flowers102 {mode} run={run_index} epoch={epoch}/{experiment_config.num_epochs}] "
            + " ".join(
                f"{key}={value:.5g}"
                for key, value in metrics.items()
                if key in {"hard_mse", "hard_psnr", "relaxed_mse", "effective_codes", "self_collision"}
            )
        )

        if metrics["hard_mse"] < best_val_hard_mse:
            best_val_hard_mse = metrics["hard_mse"]
            best_metrics = copy.deepcopy(metrics)
            best_model_state = _clone_state_dict_to_cpu(model)
            best_epoch = epoch

            _atomic_torch_save(
                value={
                    "model_state_dict": best_model_state,
                    "config": checkpoint_config,
                    "mode": mode,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_val_metrics": best_metrics,
                },
                path=experiment_dir / "best_model_checkpoint.pt",
            )

        _atomic_torch_save(
            value={
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "rng_state": capture_rng_state(),
                "config": checkpoint_config,
                "mode": mode,
                "run_index": run_index,
                "seed": seed,
                "completed_epoch": epoch,
                "global_step": global_step,
                "best_val_hard_mse": best_val_hard_mse,
                "best_metrics": best_metrics,
                "best_model_state": best_model_state,
                "best_epoch": best_epoch,
                "wall_clock_time_seconds": elapsed_before_restart + time.time() - start_time,
            },
            path=resume_checkpoint_path,
        )

        if writer is not None:
            writer.flush()

    if writer is not None:
        writer.flush()
        writer.close()

    if best_model_state is None or best_metrics is None or best_epoch is None:
        raise RuntimeError("Training did not complete any validation epochs")

    model.load_state_dict(best_model_state)
    test_metrics = _validate(
        model=model,
        loader=test_loader,
        device=device,
        config=model_config,
        max_batches=experiment_config.max_val_batches,
    )
    result = {
        "mode": mode,
        "seed": seed,
        "run_index": run_index,
        "best_epoch": best_epoch,
        "wall_clock_time_seconds": elapsed_before_restart + time.time() - start_time,
        "split_sizes": {split: len(cast(Sized, dataset)) for split, dataset in datasets.items()},
        "best_validation": best_metrics,
        "test": test_metrics,
    }

    experiment_dir.joinpath("run_result.json").write_text(json.dumps(result, indent=2, default=_json_default))
    return result


def main() -> None:
    """Run the command-line entry point."""
    cfg = get_experiment_config()

    if cfg.num_experiments < 1 or cfg.num_epochs < 1:
        raise ValueError("num_experiments and num_epochs must be positive")

    invalid_modes = set(cfg.modes) - {"deterministic", "stochastic"}

    if invalid_modes:
        raise ValueError(f"Unsupported modes: {sorted(invalid_modes)}")

    device = torch.device(cfg.device)
    experiment_name = _resolve_experiment_name(cfg)
    experiment_dir = get_experiment_dir(experiment_name, experiments_dir=cfg.experiments_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(cfg.config_file, experiment_dir / cfg.config_file.name)
    experiment_dir.joinpath("effective_config.json").write_text(
        json.dumps(_config_dict(cfg), indent=2, default=_json_default)
    )

    results = []

    for run_index in range(cfg.num_experiments):
        for mode in cfg.modes:
            run_name = f"run_{run_index:02d}_{mode}"
            run_dir = experiment_dir / run_name
            checkpoint_dir = _get_run_checkpoint_dir(experiment_dir=experiment_dir, run_name=run_name)
            result = run_single_flowers102_experiment(
                experiment_dir=run_dir,
                checkpoint_dir=checkpoint_dir,
                experiment_config=cfg,
                mode=mode,
                run_index=run_index,
                device=device,
            )
            results.append(result)

    experiment_dir.joinpath("all_run_results.json").write_text(json.dumps(results, indent=2, default=_json_default))
    print(f"Wrote Flowers102 experiment results to {experiment_dir}")


if __name__ == "__main__":
    main()
