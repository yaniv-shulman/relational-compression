"""Experiment support code for relational compression studies."""

import copy
import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import torch
from pytorch_msssim import ms_ssim
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from relational_compression.experiments.teacher_image_compression.metrics import psnr
from relational_compression.experiments.teacher_image_compression.models import BinaryPatchEncoder, make_decoder
from relational_compression.models.image_autoencoder import Decoder
from relational_compression.randomness import capture_rng_state, restore_rng_state


def _batch_ms_ssim(prediction: Tensor, target: Tensor) -> float:
    """Compute batch MS-SSIM when the image size supports it."""
    if min(target.shape[-2:]) < 160:
        return float("nan")
    return float(ms_ssim(X=prediction, Y=target, data_range=1.0, size_average=True))


def _clone_state_dict_to_cpu(model: nn.Module) -> dict[str, Tensor]:
    """Clone state dict to cpu."""
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _atomic_torch_save(value: Any, path: Path) -> None:
    """Atomically save atomic torch save."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    torch.save(obj=value, f=temporary_path)
    temporary_path.replace(path)


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


def reconstruction_from_hard_codes(student: BinaryPatchEncoder, decoder: Decoder, images: Tensor) -> Tensor:
    """Reconstruct images from deterministic student codewords."""
    student.eval()
    with torch.no_grad():
        hard_codes = student.hard_codes(images)
    return cast(Tensor, torch.sigmoid(decoder(hard_codes.detach())))


def _make_input_reconstruction_grid(images: Tensor, reconstructions: Tensor) -> Tensor:
    """Create make input reconstruction grid."""
    pair = torch.stack((images[0], reconstructions[0])).detach().cpu().clamp(0.0, 1.0)
    return cast(Tensor, make_grid(pair, nrow=2))


def _log_decoder_reconstruction_images(
    *,
    writer: SummaryWriter,
    student: BinaryPatchEncoder,
    decoder: Decoder,
    loader: DataLoader,
    device: torch.device,
    epoch: int,
) -> None:
    """Log decoder reconstruction examples to TensorBoard."""
    was_student_training = student.training
    was_decoder_training = decoder.training
    student.eval()
    decoder.eval()

    try:
        images = next(iter(loader)).to(device)
        with torch.no_grad():
            reconstruction = reconstruction_from_hard_codes(student=student, decoder=decoder, images=images)
        writer.add_image(
            tag="decoder/validate_input_reconstruction",
            img_tensor=_make_input_reconstruction_grid(images=images, reconstructions=reconstruction),
            global_step=epoch,
        )
    finally:
        student.train(was_student_training)
        decoder.train(was_decoder_training)


def validate_decoder(
    *,
    student: BinaryPatchEncoder,
    decoder: Decoder,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None,
) -> dict[str, float]:
    """Validate decoder."""
    student.eval()
    decoder.eval()
    mse_loss = nn.MSELoss(reduction="sum")
    squared_error_sum = 0.0
    element_count = 0
    ms_ssim_sum = 0.0
    ms_ssim_count = 0

    with torch.no_grad():
        for batch_idx, images in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(device)
            reconstruction = reconstruction_from_hard_codes(student=student, decoder=decoder, images=images)
            squared_error_sum += float(mse_loss(input=reconstruction, target=images))
            element_count += images.numel()
            batch_ms_ssim = _batch_ms_ssim(prediction=reconstruction, target=images)
            if math.isfinite(batch_ms_ssim):
                ms_ssim_sum += batch_ms_ssim * images.shape[0]
                ms_ssim_count += images.shape[0]

    if element_count == 0:
        raise RuntimeError("decoder validation loader produced no batches")

    mse = squared_error_sum / element_count
    return {
        "mse": mse,
        "psnr": psnr(mse),
        "ms_ssim": ms_ssim_sum / ms_ssim_count if ms_ssim_count else float("nan"),
    }


def train_decoder_stage(
    *,
    student: BinaryPatchEncoder,
    config: Any,
    run_dir: Path,
    checkpoint_dir: Path,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    writer: SummaryWriter | None,
    on_epoch_end: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Train decoder stage."""
    if int(config.decoder_num_epochs) < 1:
        return {}

    student.eval()
    for parameter in student.parameters():
        parameter.requires_grad_(False)

    decoder_dir = run_dir / "decoder"
    decoder_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir / "latest.pt"
    decoder = make_decoder(config).to(device)
    optimizer = AdamW(decoder.parameters(), lr=config.decoder_learning_rate, weight_decay=config.decoder_weight_decay)
    steps_per_epoch = len(train_loader)
    if config.max_train_batches is not None:
        steps_per_epoch = min(steps_per_epoch, config.max_train_batches)
    scheduler = make_cosine_lr_scheduler(
        optimizer,
        total_steps=max(1, int(config.decoder_num_epochs) * max(1, steps_per_epoch)),
        min_learning_rate=float(config.min_learning_rate),
        warmup_steps=int(config.learning_rate_warmup_steps),
        warmup_start_factor=float(config.learning_rate_warmup_start_factor),
    )

    mse_loss = nn.MSELoss()
    global_step = 0
    start_epoch = 1
    best_val_mse = float("inf")
    best_metrics: dict[str, float] | None = None
    best_state: dict[str, Tensor] | None = None
    best_epoch: int | None = None
    elapsed_before_restart = 0.0

    if latest_path.exists():
        checkpoint = torch.load(latest_path, map_location=device, weights_only=True)
        if checkpoint.get("config") != _config_dict_for_decoder_checkpoint(config):
            raise ValueError(f"Checkpoint config does not match this run: {latest_path}")
        decoder.load_state_dict(checkpoint["decoder_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "rng_state" not in checkpoint:
            raise ValueError(f"Checkpoint does not contain RNG state and cannot be resumed exactly: {latest_path}")
        restore_rng_state(state=checkpoint["rng_state"], restore_cuda=device.type == "cuda")
        global_step = int(checkpoint["global_step"])
        start_epoch = int(checkpoint["completed_epoch"]) + 1
        best_val_mse = float(checkpoint["best_val_mse"])
        best_metrics = checkpoint.get("best_metrics")
        best_state = checkpoint.get("best_decoder_state_dict")
        best_epoch = checkpoint.get("best_epoch")
        elapsed_before_restart = float(checkpoint.get("wall_clock_time_seconds", 0.0))
        print(f"Resuming teacher-image decoder from epoch {start_epoch} using {latest_path}")
    else:
        print(f"Starting teacher-image decoder training for {config.decoder_num_epochs} epochs")

    start_time = time.time()
    if writer is not None and int(config.tensorboard_log_steps) <= 0:
        writer = None

    for epoch in range(start_epoch, int(config.decoder_num_epochs) + 1):
        decoder.train()
        for batch_idx, images in enumerate(train_loader):
            if config.max_train_batches is not None and batch_idx >= config.max_train_batches:
                break

            images = images.to(device)
            reconstruction = reconstruction_from_hard_codes(student=student, decoder=decoder, images=images)
            loss = mse_loss(input=reconstruction, target=images)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if writer is not None and global_step % int(config.tensorboard_log_steps) == 0:
                writer.add_scalar(tag="decoder/train_mse", scalar_value=loss, global_step=global_step)
                writer.add_scalar(
                    tag="decoder/learning_rate", scalar_value=optimizer.param_groups[0]["lr"], global_step=global_step
                )
            global_step += 1

        metrics = validate_decoder(
            student=student,
            decoder=decoder,
            loader=valid_loader,
            device=device,
            max_batches=config.max_val_batches,
        )
        if writer is not None:
            for name, value in metrics.items():
                if math.isfinite(value):
                    writer.add_scalar(tag=f"decoder/validate_{name}", scalar_value=value, global_step=epoch)
            decoder_reconstruction_log_epochs = int(getattr(config, "decoder_reconstruction_log_epochs", 1))
            if decoder_reconstruction_log_epochs > 0 and epoch % decoder_reconstruction_log_epochs == 0:
                _log_decoder_reconstruction_images(
                    writer=writer,
                    student=student,
                    decoder=decoder,
                    loader=valid_loader,
                    device=device,
                    epoch=epoch,
                )
            writer.flush()

        if metrics["mse"] < best_val_mse:
            best_val_mse = metrics["mse"]
            best_metrics = copy.deepcopy(metrics)
            best_state = _clone_state_dict_to_cpu(decoder)
            best_epoch = epoch
            _atomic_torch_save(
                value={
                    "decoder_state_dict": best_state,
                    "config": _config_dict_for_decoder_checkpoint(config),
                    "best_epoch": best_epoch,
                    "best_validation": best_metrics,
                },
                path=checkpoint_dir / "best_model_checkpoint.pt",
            )

        _atomic_torch_save(
            value={
                "decoder_state_dict": decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "rng_state": capture_rng_state(include_cuda=device.type == "cuda"),
                "config": _config_dict_for_decoder_checkpoint(config),
                "global_step": global_step,
                "completed_epoch": epoch,
                "best_val_mse": best_val_mse,
                "best_metrics": best_metrics,
                "best_decoder_state_dict": best_state,
                "best_epoch": best_epoch,
                "wall_clock_time_seconds": elapsed_before_restart + time.time() - start_time,
            },
            path=latest_path,
        )
        if on_epoch_end is not None:
            on_epoch_end()

    if best_state is None or best_metrics is None or best_epoch is None:
        raise RuntimeError("decoder stage did not complete any validation epochs")

    decoder.load_state_dict(best_state)
    test_metrics = validate_decoder(
        student=student,
        decoder=decoder,
        loader=test_loader,
        device=device,
        max_batches=config.max_val_batches,
    )
    result = {
        "best_epoch": best_epoch,
        "best_validation": best_metrics,
        "test": test_metrics,
    }
    (decoder_dir / "decoder_result.json").write_text(json.dumps(result, indent=2))
    return result


_REPRESENTATION_CHECKPOINT_CONFIG_FIELDS = (
    "task_model_name",
    "dataset_name",
    "dataset_version",
    "num_epochs",
    "image_size",
    "batch_size",
    "num_workers",
    "seed",
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
    "eval_sampled_tokens",
    "prefer_cross_image_neighbors",
    "validation_sampling_seed",
    "test_sampling_seed",
    "teacher_backend",
    "teacher_repo",
    "teacher_model_name",
    "fixed_random_teacher_embedding_dim",
    "max_train_batches",
    "max_val_batches",
)

_DECODER_CHECKPOINT_CONFIG_FIELDS = (
    "task_model_name",
    "dataset_name",
    "dataset_version",
    "image_size",
    "batch_size",
    "num_workers",
    "seed",
    "min_learning_rate",
    "learning_rate_warmup_steps",
    "learning_rate_warmup_start_factor",
    "bits",
    "hidden",
    "downsample_layers",
    "residual_layers",
    "residual_hidden",
    "max_train_batches",
    "max_val_batches",
    "decoder_num_epochs",
    "decoder_learning_rate",
    "decoder_weight_decay",
)


def _config_dict_for_representation_checkpoint(config: Any) -> dict[str, Any]:
    """Collect configuration fields saved with a representation checkpoint."""
    return _config_dict_for_checkpoint_fields(config=config, field_names=_REPRESENTATION_CHECKPOINT_CONFIG_FIELDS)


def _config_dict_for_decoder_checkpoint(config: Any) -> dict[str, Any]:
    """Collect configuration fields saved with a decoder checkpoint."""
    return _config_dict_for_checkpoint_fields(config=config, field_names=_DECODER_CHECKPOINT_CONFIG_FIELDS)


def _config_dict_for_checkpoint_fields(config: Any, field_names: tuple[str, ...]) -> dict[str, Any]:
    """Collect the common checkpoint configuration fields."""
    result: dict[str, Any] = {}
    for name in field_names:
        if not hasattr(config, name):
            continue
        value = getattr(config, name)
        if isinstance(value, tuple):
            result[name] = list(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            result[name] = value
    return result
