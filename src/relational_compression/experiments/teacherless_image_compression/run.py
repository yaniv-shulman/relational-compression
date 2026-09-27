"""Experiment support code for relational compression studies."""

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import torch
from pytorch_msssim import ms_ssim
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from relational_compression.experiments.teacherless_image_compression.config import ExperimentConfig, config
from relational_compression.experiments.teacherless_image_compression.data import (
    ImageDirectoryDataset,
    seed_everything,
    split_dataset,
)
from relational_compression.experiments.teacherless_image_compression.metrics import latent_metrics
from relational_compression.models.image_autoencoder import BinaryImageAutoencoder


def _make_model(config: ExperimentConfig, stochastic: bool) -> BinaryImageAutoencoder:
    """Create make model."""
    return BinaryImageAutoencoder(
        bits=config.bits,
        hidden=config.hidden,
        downsample_layers=config.downsample_layers,
        residual_layers=config.residual_layers,
        residual_hidden=config.residual_hidden,
        stochastic=stochastic,
        grouping=config.grouping,
    )


def _psnr(mse: float) -> float:
    """Compute peak signal-to-noise ratio from mean squared error."""
    return -10.0 * math.log10(max(mse, 1e-12))


def _batch_ms_ssim(prediction: Tensor, target: Tensor) -> float:
    """Compute batch MS-SSIM when the image size supports it."""
    if min(target.shape[-2:]) < 160:
        return float("nan")
    return float(ms_ssim(X=prediction, Y=target, data_range=1.0, size_average=True))


def _validate(
    model: BinaryImageAutoencoder,
    loader: DataLoader,
    device: torch.device,
    config: ExperimentConfig,
    max_batches: int | None,
) -> dict[str, float]:
    """Validate the requested values."""
    model.eval()
    mse = nn.MSELoss()
    hard_loss_sum = 0.0
    relaxed_loss_sum = 0.0
    ms_ssim_sum = 0.0
    num_examples = 0
    num_ssim_examples = 0
    hard_values: list[Tensor] = []
    probability_values: list[Tensor] = []
    relaxed_values: list[Tensor] = []

    with torch.no_grad():
        for batch_idx, images in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(device)
            batch_size = images.shape[0]
            hard_output = model(images, force_hard=True)
            hard_loss = mse(hard_output.reconstruction, images)
            hard_loss_sum += float(hard_loss) * batch_size
            batch_ms_ssim = _batch_ms_ssim(prediction=hard_output.reconstruction, target=images)
            if math.isfinite(batch_ms_ssim):
                ms_ssim_sum += batch_ms_ssim * batch_size
                num_ssim_examples += batch_size

            # Keep encoder/decoder in eval mode but use the training relaxation in the quantizer.
            model.quantizer.train()
            relaxed_output = model(images)
            model.quantizer.eval()
            relaxed_loss_sum += float(mse(relaxed_output.reconstruction, images)) * batch_size
            num_examples += batch_size

            hard_values.append(hard_output.quantizer.hard.cpu())
            probability_values.append(hard_output.quantizer.probabilities.cpu())
            relaxed_values.append(relaxed_output.quantizer.relaxed.cpu())

    if num_examples == 0:
        raise RuntimeError("Validation loader produced no batches")

    latent = latent_metrics(
        hard=torch.cat(hard_values),
        probabilities=torch.cat(probability_values),
        relaxed=torch.cat(relaxed_values),
        pair_samples=config.diagnostic_pair_samples,
    )

    hard_mse = hard_loss_sum / num_examples
    relaxed_mse = relaxed_loss_sum / num_examples

    result = {
        "hard_mse": hard_mse,
        "hard_psnr": _psnr(hard_mse),
        "relaxed_mse": relaxed_mse,
        "relaxed_psnr": _psnr(relaxed_mse),
        "hard_minus_relaxed_mse": hard_mse - relaxed_mse,
        "ms_ssim": ms_ssim_sum / num_ssim_examples if num_ssim_examples > 0 else float("nan"),
    }

    result.update(asdict(latent))
    return result


def _run_one(
    config: ExperimentConfig,
    *,
    stochastic: bool,
    epochs: int,
    run_index: int,
    device: torch.device,
    max_train_batches: int | None,
    max_val_batches: int | None,
) -> dict[str, float]:
    """Run one."""
    seed = config.seed + run_index
    seed_everything(seed)
    dataset = ImageDirectoryDataset(root=config.dataset_root, image_size=config.image_size)
    train_set, val_set = split_dataset(dataset=dataset, validation_fraction=config.validation_fraction, seed=seed)

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    mode = "stochastic" if stochastic else "deterministic"
    run_dir = config.output_root / mode / f"run_{run_index:02d}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(run_dir / "tensorboard")

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "dataset_root": str(config.dataset_root),
                "output_root": str(config.output_root),
                "epochs": epochs,
                "mode": mode,
                "seed": seed,
            },
            indent=2,
        )
    )

    model = _make_model(config=config, stochastic=stochastic).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    mse = nn.MSELoss()
    global_step = 0

    for epoch in range(epochs):
        model.train()

        for batch_idx, images in enumerate(train_loader):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break
            images = images.to(device)
            output = model(images)
            distortion = mse(output.reconstruction, images)
            concentration = output.quantizer.concentration_loss
            loss = distortion + config.concentration_weight * concentration
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            writer.add_scalar(tag="train/loss", scalar_value=loss, global_step=global_step)
            writer.add_scalar(tag="train/l2_distortion", scalar_value=distortion, global_step=global_step)
            writer.add_scalar(tag="train/concentration", scalar_value=concentration, global_step=global_step)
            writer.add_scalar(tag="train/relaxed_hard_rms", scalar_value=concentration.sqrt(), global_step=global_step)
            global_step += 1

        metrics = _validate(model=model, loader=val_loader, device=device, config=config, max_batches=max_val_batches)

        for name, value in metrics.items():
            if math.isfinite(value):
                writer.add_scalar(tag=f"validate/{name}", scalar_value=value, global_step=epoch + 1)

        print(
            f"[{mode} run={run_index} epoch={epoch + 1}/{epochs}] "
            + " ".join(
                f"{k}={v:.5g}"
                for k, v in metrics.items()
                if k
                in {
                    "hard_mse",
                    "hard_psnr",
                    "relaxed_mse",
                    "hard_minus_relaxed_mse",
                    "effective_codes",
                    "self_collision",
                }
            )
        )

    torch.save(
        obj={
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "mode": mode,
            "epochs": epochs,
            "seed": seed,
        },
        f=run_dir / "model.pt",
    )

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    writer.close()
    return metrics


def _parse_args() -> argparse.Namespace:
    """Parse args."""
    parser = argparse.ArgumentParser(description="Teacherless binary image-compression experiment")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--mode", choices=("deterministic", "stochastic", "both"), default="both")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument(
        "--max-train-batches", type=int, default=None, help="Convenience/smoke-test limit; omit for full epochs"
    )

    parser.add_argument(
        "--max-val-batches", type=int, default=None, help="Convenience/smoke-test limit; omit for full validation"
    )

    return parser.parse_args()


def main() -> None:
    """Run the command-line entry point."""
    args = _parse_args()
    config_ = config

    if args.dataset_root is not None:
        config_ = replace(config_, dataset_root=args.dataset_root)

    if args.output_root is not None:
        config_ = replace(config_, output_root=args.output_root)

    if args.runs < 1 or args.epochs < 1:
        raise ValueError("--runs and --epochs must be positive")

    modes = [False, True] if args.mode == "both" else [args.mode == "stochastic"]
    device = torch.device(args.device)

    for stochastic in modes:
        for run_index in range(args.runs):
            _run_one(
                config_,
                stochastic=stochastic,
                epochs=args.epochs,
                run_index=run_index,
                device=device,
                max_train_batches=args.max_train_batches,
                max_val_batches=args.max_val_batches,
            )


if __name__ == "__main__":
    main()
