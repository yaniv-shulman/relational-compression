"""Command-line utilities for reproducible experiment workflows."""

import argparse
import copy
import importlib
import json
import math
import shutil
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Sized, cast

import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from relational_compression.experiments.teacher_image_compression.collisions import (
    signed_partition_loss,
    signed_teacher_graph,
    soft_bit_probabilities,
)
from relational_compression.experiments.teacher_image_compression.decoder_stage import (
    _config_dict_for_representation_checkpoint,
    make_cosine_lr_scheduler,
    train_decoder_stage,
)
from relational_compression.experiments.teacher_image_compression.metrics import evaluate_representation
from relational_compression.experiments.teacher_image_compression.models import (
    DINO_VITS8_CHECKPOINT_URL,
    DINO_VITS8_MODEL,
    DINO_VITS8_REPO,
    DINO_VITS8_SOURCE_REVISION,
    DinoVitS8PatchTeacher,
    FixedRandomPatchTeacher,
    make_student,
)
from relational_compression.experiments.teacher_image_compression.sampling import sample_tokens
from relational_compression.experiments.teacherless_image_compression.data import (
    build_flowers102_datasets,
    seed_everything,
)
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
    "alignment_weight",
    "code_temperature",
    "teacher_temperature",
    "sampled_tokens",
    "prefer_cross_image_neighbors",
    "teacher_backend",
    "teacher_repo",
    "teacher_model_name",
    "teacher_cache_dir",
    "teacher_download",
    "fixed_random_teacher_embedding_dim",
    "download",
    "seed",
    "validation_sampling_seed",
    "test_sampling_seed",
    "eval_sampled_tokens",
    "log_to_tensorboard_global",
    "tensorboard_log_steps",
    "similarity_diagnostic_epochs",
    "similarity_diagnostic_heatmap_size",
    "max_train_batches",
    "max_val_batches",
    "decoder_num_epochs",
    "decoder_learning_rate",
    "decoder_weight_decay",
    "decoder_reconstruction_log_epochs",
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


def _tensorboard_enabled(config: Any) -> bool:
    """Return whether TensorBoard logging is enabled."""
    return bool(config.log_to_tensorboard_global) and int(config.tensorboard_log_steps) > 0


def _resolve_experiment_name(config: Any) -> str:
    """Resolve experiment name."""
    experiment_name = getattr(config, "experiment_name", None)
    if experiment_name is not None:
        return str(experiment_name)
    return get_experiment_name(experiment_base_name=config.experiment_base_name, unique_postfix=config.unique_postfix)


def _run_checkpoint_dir(*, experiment_root: Path, run_name: str) -> Path:
    """Run checkpoint dir."""
    return experiment_root / "checkpoints" / run_name


def _resolved_sampling_seed(configured_seed: int | None, *, default: int) -> int:
    """Compute resolved sampling seed."""
    return default if configured_seed is None else int(configured_seed)


def _atomic_torch_save(value: Any, path: Path) -> None:
    """Atomically save atomic torch save."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    torch.save(obj=value, f=temporary_path)
    temporary_path.replace(path)


def _clone_state_dict_to_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Clone state dict to cpu."""
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


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


def _colorize_heatmap(values: torch.Tensor) -> torch.Tensor:
    """Compute colorize heatmap."""
    normalized = values.detach().float().clamp(-1.0, 1.0).add(1.0).mul(0.5)
    red = (1.5 * normalized).clamp(0.0, 1.0)
    green = (1.5 - 3.0 * (normalized - 0.5).abs()).clamp(0.0, 1.0)
    blue = (1.5 * (1.0 - normalized)).clamp(0.0, 1.0)
    return torch.stack((red, green, blue), dim=0)


def _teacher_similarity_diagnostic_panel(
    *,
    image: torch.Tensor,
    teacher_grid: torch.Tensor,
    heatmap_size: int,
) -> torch.Tensor:
    """Compute teacher similarity diagnostic panel."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must have shape [3, H, W]")
    if teacher_grid.ndim != 3:
        raise ValueError("teacher_grid must have shape [H, W, D]")
    if heatmap_size < 16:
        raise ValueError("heatmap_size must be at least 16")

    image_panel = F.interpolate(
        image.detach().float().clamp(0.0, 1.0).unsqueeze(0).cpu(),
        size=(heatmap_size, heatmap_size),
        mode="bilinear",
        align_corners=False,
    )[0]
    tokens = F.normalize(teacher_grid.detach().float().reshape(-1, teacher_grid.shape[-1]), dim=-1)
    similarity = tokens @ tokens.T
    heatmap_panel = F.interpolate(
        _colorize_heatmap(similarity).unsqueeze(0).cpu(),
        size=(heatmap_size, heatmap_size),
        mode="bilinear",
        align_corners=False,
    )[0].clamp(0.0, 1.0)
    separator = torch.full(size=(3, heatmap_size, max(2, heatmap_size // 64)), fill_value=0.85)
    return torch.cat((image_panel, separator, heatmap_panel), dim=2)


def _log_teacher_similarity_diagnostic(
    *,
    writer: SummaryWriter,
    image: torch.Tensor,
    teacher_grid: torch.Tensor,
    epoch: int,
    heatmap_size: int,
) -> None:
    """Compute log teacher similarity diagnostic."""
    panel = _teacher_similarity_diagnostic_panel(
        image=image,
        teacher_grid=teacher_grid,
        heatmap_size=heatmap_size,
    )
    writer.add_image(tag="train/image_and_teacher_similarity", img_tensor=panel, global_step=epoch)


def _make_teacher(config: Any, device: torch.device) -> torch.nn.Module:
    """Create make teacher."""
    backend = str(config.teacher_backend)
    if backend == "dino_vits8":
        teacher: torch.nn.Module = DinoVitS8PatchTeacher.from_torch_hub(
            cache_dir=Path(config.teacher_cache_dir),
            allow_download=bool(config.teacher_download),
            repo=str(config.teacher_repo),
            model_name=str(config.teacher_model_name),
        )
    elif backend == "fixed_random_patch":
        teacher = FixedRandomPatchTeacher(
            embedding_dim=int(config.fixed_random_teacher_embedding_dim),
            seed=int(config.seed),
        )
    else:
        raise ValueError(f"Unsupported teacher_backend: {backend}")

    teacher.to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def _teacher_metadata(config: Any) -> dict[str, Any]:
    """Compute teacher metadata."""
    backend = str(config.teacher_backend)
    if backend == "dino_vits8":
        return {
            "backend": backend,
            "repo": getattr(config, "teacher_repo", DINO_VITS8_REPO),
            "source_revision": DINO_VITS8_SOURCE_REVISION,
            "model": getattr(config, "teacher_model_name", DINO_VITS8_MODEL),
            "checkpoint_url": DINO_VITS8_CHECKPOINT_URL,
            "cache_dir": str(config.teacher_cache_dir),
        }

    if backend == "fixed_random_patch":
        return {
            "backend": backend,
            "embedding_dim": int(config.fixed_random_teacher_embedding_dim),
        }

    raise ValueError(f"Unsupported teacher_backend: {backend}")


def _representation_batch_loss(
    *,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    images: torch.Tensor,
    config: Any,
    return_diagnostics: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute representation batch loss."""
    with torch.no_grad():
        teacher_grid = teacher(images)

    output = student(images)
    if output.logits.shape[2:] != teacher_grid.shape[1:3]:
        raise ValueError(
            f"student latent grid {tuple(output.logits.shape[2:])} does not match "
            f"teacher patch grid {tuple(teacher_grid.shape[1:3])}"
        )

    sample = sample_tokens(
        teacher_grid=teacher_grid,
        logits=output.logits,
        hard=output.hard,
        max_tokens=int(config.sampled_tokens),
    )
    graph = signed_teacher_graph(
        teacher_tokens=sample.teacher_tokens,
        image_indices=sample.image_indices,
        teacher_temperature=float(config.teacher_temperature),
        prefer_cross_image=bool(config.prefer_cross_image_neighbors),
    )
    sampled_soft_probabilities = soft_bit_probabilities(
        logits=sample.logits, code_temperature=float(config.code_temperature)
    )
    teacher_loss = signed_partition_loss(
        soft_bit_probabilities=sampled_soft_probabilities, graph=graph, bits=int(config.bits)
    )
    loss = float(config.alignment_weight) * teacher_loss.loss
    result = {
        "loss": loss,
        "teacher_loss": teacher_loss.loss,
        "teacher_positive_loss": teacher_loss.positive_loss,
        "teacher_negative_loss": teacher_loss.negative_loss,
        "teacher_positive_weight_mass": teacher_loss.positive_weight_mass,
        "teacher_negative_weight_mass": teacher_loss.negative_weight_mass,
        "sampled_logits": sample.logits.detach(),
        "sampled_soft_probabilities": sampled_soft_probabilities.detach(),
        "teacher_positive_similarity": graph.similarities[graph.positive_weights > 0.0].detach(),
        "teacher_negative_similarity": graph.similarities[graph.negative_weights > 0.0].detach(),
        "teacher_positive_same_probability": teacher_loss.positive_same_probability.detach(),
        "teacher_negative_same_probability": teacher_loss.negative_same_probability.detach(),
    }
    if return_diagnostics:
        result["diagnostic_image"] = images[0].detach()
        result["diagnostic_teacher_grid"] = teacher_grid[0].detach()
    return result


def _checkpoint_score(metrics: dict[str, float]) -> float:
    """Compute checkpoint score."""
    score = float(metrics["teacher_loss"])
    return score if math.isfinite(score) else float("inf")


def run_single_flowers102_experiment(
    *,
    experiment_dir: Path,
    checkpoint_dir: Path,
    experiment_config: Any,
    run_index: int,
    device: torch.device,
) -> dict[str, Any]:
    """Run single flowers102 experiment."""
    seed = int(experiment_config.seed) + run_index
    seed_everything(seed)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    datasets = build_flowers102_datasets(
        root=Path(experiment_config.dataset_root_dir),
        image_size=int(experiment_config.image_size),
        download=bool(experiment_config.download),
    )
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        datasets["train"],
        batch_size=int(experiment_config.batch_size),
        shuffle=True,
        num_workers=int(experiment_config.num_workers),
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        datasets["valid"],
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

    teacher = _make_teacher(config=experiment_config, device=device)
    student = make_student(experiment_config).to(device)
    optimizer = AdamW(
        student.parameters(), lr=experiment_config.learning_rate, weight_decay=experiment_config.weight_decay
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

    representation_dir = experiment_dir / "representation"
    representation_dir.mkdir(parents=True, exist_ok=True)
    representation_checkpoint_dir = checkpoint_dir / "representation"
    representation_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = representation_checkpoint_dir / "latest.pt"
    tensorboard_log_dir = experiment_dir / "tensorboard_logs"
    writer = SummaryWriter(tensorboard_log_dir) if _tensorboard_enabled(experiment_config) else None
    global_step = 0
    start_epoch = 1
    best_score = float("inf")
    best_validation: dict[str, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch: int | None = None
    elapsed_before_restart = 0.0
    validation_sampling_seed = _resolved_sampling_seed(
        getattr(experiment_config, "validation_sampling_seed", None),
        default=seed + 100_000,
    )
    test_sampling_seed = _resolved_sampling_seed(
        getattr(experiment_config, "test_sampling_seed", None),
        default=seed + 200_000,
    )

    if latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=True)
        if checkpoint.get("config") != _config_dict_for_representation_checkpoint(experiment_config):
            raise ValueError(f"Checkpoint config does not match this run: {latest_path}")
        student.load_state_dict(checkpoint["student_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "rng_state" not in checkpoint:
            raise ValueError(f"Checkpoint does not contain RNG state and cannot be resumed exactly: {latest_path}")
        restore_rng_state(state=checkpoint["rng_state"], restore_cuda=device.type == "cuda")
        global_step = int(checkpoint["global_step"])
        completed_epoch = int(checkpoint["completed_epoch"])
        start_epoch = completed_epoch + 1
        best_score = float(checkpoint["best_score"])
        best_validation = checkpoint.get("best_validation")
        best_state = checkpoint.get("best_student_state_dict")
        best_epoch = checkpoint.get("best_epoch")
        elapsed_before_restart = float(checkpoint.get("wall_clock_time_seconds", 0.0))
        if start_epoch <= int(experiment_config.num_epochs):
            print(f"Resuming teacher-image run={run_index} from epoch {start_epoch} using {latest_path}")
        else:
            print(
                f"Teacher-image representation run={run_index} already completed through epoch "
                f"{completed_epoch}; using {latest_path} and skipping representation training"
            )

    start_time = time.time()
    for epoch in range(start_epoch, int(experiment_config.num_epochs) + 1):
        student.train()
        teacher.eval()

        for batch_idx, images in enumerate(train_loader):
            if experiment_config.max_train_batches is not None and batch_idx >= experiment_config.max_train_batches:
                break

            images = images.to(device)
            log_similarity_diagnostic = (
                writer is not None
                and batch_idx == 0
                and int(experiment_config.similarity_diagnostic_epochs) > 0
                and epoch % int(experiment_config.similarity_diagnostic_epochs) == 0
            )
            values = _representation_batch_loss(
                student=student,
                teacher=teacher,
                images=images,
                config=experiment_config,
                return_diagnostics=log_similarity_diagnostic,
            )
            optimizer.zero_grad(set_to_none=True)
            values["loss"].backward()
            optimizer.step()
            scheduler.step()

            if writer is not None and global_step % int(experiment_config.tensorboard_log_steps) == 0:
                writer.add_scalar(tag="train/loss", scalar_value=values["loss"], global_step=global_step)
                writer.add_scalar(
                    tag="train/teacher_loss", scalar_value=values["teacher_loss"], global_step=global_step
                )
                writer.add_scalar(
                    tag="train/teacher_positive_loss",
                    scalar_value=values["teacher_positive_loss"],
                    global_step=global_step,
                )
                writer.add_scalar(
                    tag="train/teacher_negative_loss",
                    scalar_value=values["teacher_negative_loss"],
                    global_step=global_step,
                )
                writer.add_scalar(
                    tag="train/teacher_positive_weight_mass",
                    scalar_value=values["teacher_positive_weight_mass"],
                    global_step=global_step,
                )
                writer.add_scalar(
                    tag="train/teacher_negative_weight_mass",
                    scalar_value=values["teacher_negative_weight_mass"],
                    global_step=global_step,
                )
                writer.add_scalar(
                    tag="train/weighted_teacher_loss",
                    scalar_value=float(experiment_config.alignment_weight) * values["teacher_loss"],
                    global_step=global_step,
                )
                writer.add_scalar(
                    tag="train/learning_rate", scalar_value=optimizer.param_groups[0]["lr"], global_step=global_step
                )
                writer.add_scalar(
                    tag="train/code_temperature",
                    scalar_value=float(experiment_config.code_temperature),
                    global_step=global_step,
                )
                writer.add_scalar(
                    tag="train/teacher_temperature",
                    scalar_value=float(experiment_config.teacher_temperature),
                    global_step=global_step,
                )
                _log_tensor_distribution(
                    writer=writer, tag_prefix="train/sampled_logits", tensor=values["sampled_logits"], step=global_step
                )
                _log_tensor_distribution(
                    writer=writer,
                    tag_prefix="train/sampled_soft_bit_probability",
                    tensor=values["sampled_soft_probabilities"],
                    step=global_step,
                )
                if values["teacher_positive_similarity"].numel() > 0:
                    _log_tensor_distribution(
                        writer=writer,
                        tag_prefix="train/teacher_positive_cosine",
                        tensor=values["teacher_positive_similarity"],
                        step=global_step,
                    )
                if values["teacher_negative_similarity"].numel() > 0:
                    _log_tensor_distribution(
                        writer=writer,
                        tag_prefix="train/teacher_negative_cosine",
                        tensor=values["teacher_negative_similarity"],
                        step=global_step,
                    )
                if values["teacher_positive_same_probability"].numel() > 0:
                    _log_tensor_distribution(
                        writer=writer,
                        tag_prefix="train/teacher_positive_soft_same_bucket_probability",
                        tensor=values["teacher_positive_same_probability"],
                        step=global_step,
                    )
                if values["teacher_negative_same_probability"].numel() > 0:
                    _log_tensor_distribution(
                        writer=writer,
                        tag_prefix="train/teacher_negative_soft_same_bucket_probability",
                        tensor=values["teacher_negative_same_probability"],
                        step=global_step,
                    )
            if writer is not None and log_similarity_diagnostic:
                _log_teacher_similarity_diagnostic(
                    writer=writer,
                    image=values["diagnostic_image"],
                    teacher_grid=values["diagnostic_teacher_grid"],
                    epoch=epoch,
                    heatmap_size=int(experiment_config.similarity_diagnostic_heatmap_size),
                )
            global_step += 1

        validation = evaluate_representation(
            student=student,
            teacher=teacher,
            loader=valid_loader,
            device=device,
            bits=int(experiment_config.bits),
            sampled_tokens=int(experiment_config.eval_sampled_tokens),
            code_temperature=float(experiment_config.code_temperature),
            teacher_temperature=float(experiment_config.teacher_temperature),
            prefer_cross_image=bool(experiment_config.prefer_cross_image_neighbors),
            max_batches=experiment_config.max_val_batches,
            seed=validation_sampling_seed,
        )
        if writer is not None:
            for name, value in validation.items():
                if math.isfinite(value):
                    writer.add_scalar(tag=f"validate/{name}", scalar_value=value, global_step=epoch)

        print(
            f"[teacher-image run={run_index} epoch={epoch}/{experiment_config.num_epochs}] "
            + " ".join(
                f"{key}={value:.5g}"
                for key, value in validation.items()
                if key
                in {
                    "teacher_loss",
                    "teacher_positive_hard_collision_rate",
                    "teacher_negative_hard_collision_rate",
                    "teacher_partition_gap",
                    "teacher_positive_collision_enrichment",
                    "effective_codes",
                }
            )
        )

        score = _checkpoint_score(validation)
        if score < best_score:
            best_score = score
            best_validation = copy.deepcopy(validation)
            best_state = _clone_state_dict_to_cpu(student)
            best_epoch = epoch
            _atomic_torch_save(
                value={
                    "student_state_dict": best_state,
                    "config": _config_dict_for_representation_checkpoint(experiment_config),
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "best_validation": best_validation,
                    "checkpoint_metric": "teacher_loss",
                    "checkpoint_metric_mode": "min",
                },
                path=representation_checkpoint_dir / "best_model_checkpoint.pt",
            )

        _atomic_torch_save(
            value={
                "student_state_dict": student.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "rng_state": capture_rng_state(include_cuda=device.type == "cuda"),
                "config": _config_dict_for_representation_checkpoint(experiment_config),
                "run_index": run_index,
                "seed": seed,
                "completed_epoch": epoch,
                "global_step": global_step,
                "best_score": best_score,
                "best_validation": best_validation,
                "best_student_state_dict": best_state,
                "best_epoch": best_epoch,
                "wall_clock_time_seconds": elapsed_before_restart + time.time() - start_time,
            },
            path=latest_path,
        )
        if writer is not None:
            writer.flush()

    if writer is not None:
        writer.flush()

    if best_state is None or best_validation is None or best_epoch is None:
        raise RuntimeError("representation stage did not complete any validation epochs")

    student.load_state_dict(best_state)
    test = evaluate_representation(
        student=student,
        teacher=teacher,
        loader=test_loader,
        device=device,
        bits=int(experiment_config.bits),
        sampled_tokens=int(experiment_config.eval_sampled_tokens),
        code_temperature=float(experiment_config.code_temperature),
        teacher_temperature=float(experiment_config.teacher_temperature),
        prefer_cross_image=bool(experiment_config.prefer_cross_image_neighbors),
        max_batches=experiment_config.max_val_batches,
        seed=test_sampling_seed,
    )
    result: dict[str, Any] = {
        "seed": seed,
        "validation_sampling_seed": validation_sampling_seed,
        "test_sampling_seed": test_sampling_seed,
        "run_index": run_index,
        "best_epoch": best_epoch,
        "checkpoint_metric": "teacher_loss",
        "split_sizes": {split: len(cast(Sized, dataset)) for split, dataset in datasets.items()},
        "teacher": _teacher_metadata(experiment_config),
        "best_validation": best_validation,
        "test": test,
    }
    (representation_dir / "representation_result.json").write_text(json.dumps(result, indent=2, default=_json_default))

    decoder_epoch_end = None
    if writer is not None:

        def decoder_epoch_end() -> None:
            """Compute decoder epoch end."""
            writer.flush()

    decoder_result = train_decoder_stage(
        student=student,
        config=experiment_config,
        run_dir=experiment_dir,
        checkpoint_dir=checkpoint_dir / "decoder",
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        device=device,
        writer=writer,
        on_epoch_end=decoder_epoch_end,
    )
    if len(decoder_result) > 0:
        result["decoder"] = decoder_result

    if writer is not None:
        writer.flush()
        writer.close()

    return result


def get_experiment_config() -> Any:
    """Return get experiment config."""
    parser = argparse.ArgumentParser(description="Flowers102 DINO-teacher discrete image-compression experiment runner")
    parser.add_argument(
        "--config",
        type=str,
        default="flowers102_dino_vits8",
        help="Config module below relational_compression.experiments.teacher_image_compression.configs.",
    )
    parser.add_argument("--num-experiments", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dataset-root-dir", type=Path, default=None)
    parser.add_argument("--experiments-dir", type=Path, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--unique-postfix", type=str, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--teacher-backend", choices=("dino_vits8", "fixed_random_patch"), default=None)
    parser.add_argument("--sampled-tokens", type=int, default=None)
    parser.add_argument("--eval-sampled-tokens", type=int, default=None)
    parser.add_argument("--validation-sampling-seed", type=int, default=None)
    parser.add_argument("--test-sampling-seed", type=int, default=None)
    parser.add_argument("--alignment-weight", type=float, default=None)
    parser.add_argument("--code-temperature", type=float, default=None)
    parser.add_argument("--teacher-temperature", type=float, default=None)
    parser.add_argument("--decoder-num-epochs", type=int, default=None)
    parser.add_argument("--decoder-reconstruction-log-epochs", type=int, default=None)
    parser.add_argument("--similarity-diagnostic-epochs", type=int, default=None)
    parser.add_argument("--similarity-diagnostic-heatmap-size", type=int, default=None)
    parser.add_argument("--disable-tensorboard", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-teacher-download", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    args: Namespace = parser.parse_args()

    cfg: Any = importlib.import_module(
        f"relational_compression.experiments.teacher_image_compression.configs.{args.config}"
    )

    for arg_name, config_name in (
        ("num_experiments", "num_experiments"),
        ("num_epochs", "num_epochs"),
        ("seed", "seed"),
        ("batch_size", "batch_size"),
        ("num_workers", "num_workers"),
        ("sampled_tokens", "sampled_tokens"),
        ("eval_sampled_tokens", "eval_sampled_tokens"),
        ("validation_sampling_seed", "validation_sampling_seed"),
        ("test_sampling_seed", "test_sampling_seed"),
        ("alignment_weight", "alignment_weight"),
        ("code_temperature", "code_temperature"),
        ("teacher_temperature", "teacher_temperature"),
        ("decoder_num_epochs", "decoder_num_epochs"),
        ("decoder_reconstruction_log_epochs", "decoder_reconstruction_log_epochs"),
        ("similarity_diagnostic_epochs", "similarity_diagnostic_epochs"),
        ("similarity_diagnostic_heatmap_size", "similarity_diagnostic_heatmap_size"),
        ("max_train_batches", "max_train_batches"),
        ("max_val_batches", "max_val_batches"),
    ):
        value = getattr(args, arg_name)
        if value is not None:
            setattr(cfg, config_name, value)

    if args.dataset_root_dir is not None:
        cfg.dataset_root_dir = args.dataset_root_dir.absolute()
    if args.experiments_dir is not None:
        cfg.experiments_dir = args.experiments_dir.absolute()
    if args.experiment_name is not None:
        cfg.experiment_name = args.experiment_name
    if args.unique_postfix is not None:
        cfg.unique_postfix = args.unique_postfix
    if args.teacher_backend is not None:
        cfg.teacher_backend = args.teacher_backend
    if args.disable_tensorboard:
        cfg.log_to_tensorboard_global = False
    if args.no_download:
        cfg.download = False
    if args.no_teacher_download:
        cfg.teacher_download = False

    cfg.device = args.device
    return cfg


def main() -> None:
    """Run the command-line entry point."""
    cfg = get_experiment_config()
    if int(cfg.num_experiments) < 1 or int(cfg.num_epochs) < 1:
        raise ValueError("num_experiments and num_epochs must be positive")

    device = torch.device(cfg.device)
    experiment_name = _resolve_experiment_name(cfg)
    experiment_dir = get_experiment_dir(experiment_name, experiments_dir=cfg.experiments_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cfg.config_file, experiment_dir / cfg.config_file.name)
    (experiment_dir / "effective_config.json").write_text(
        json.dumps(_config_dict(cfg), indent=2, default=_json_default)
    )

    results = []
    for run_index in range(int(cfg.num_experiments)):
        run_name = f"run_{run_index:02d}_deterministic"
        run_dir = experiment_dir / run_name
        results.append(
            run_single_flowers102_experiment(
                experiment_dir=run_dir,
                checkpoint_dir=_run_checkpoint_dir(experiment_root=experiment_dir, run_name=run_name),
                experiment_config=cfg,
                run_index=run_index,
                device=device,
            )
        )

    (experiment_dir / "all_run_results.json").write_text(json.dumps(results, indent=2, default=_json_default))
    print(f"Wrote teacher-image experiment results to {experiment_dir}")


if __name__ == "__main__":
    main()
