"""Command-line utilities for reproducible experiment workflows."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sized, cast

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image

from relational_compression.experiments.teacherless_image_compression.data import (
    build_flowers102_datasets,
    seed_everything,
)
from relational_compression.models.image_autoencoder import BinaryImageAutoencoder
from relational_compression.paths import get_experiments_dir

DEFAULT_EXPERIMENT_NAMES = (
    "teacherless_image_compression_flowers102_1_0_1787902232_mean_group",
    "teacherless_image_compression_flowers102_1_0_1787911579_mean_group_32_bits",
)
DEFAULT_MODEL_LABELS = ("16-bit hard", "32-bit hard")


def _parse_args() -> argparse.Namespace:
    """Parse args."""
    parser = argparse.ArgumentParser(
        description="Export Flowers102 original/reconstruction examples for the paper appendix."
    )
    parser.add_argument(
        "--experiment-name",
        action="append",
        dest="experiment_names",
        default=None,
        help="Experiment directory name below --experiments-dir. May be supplied multiple times.",
    )
    parser.add_argument(
        "--model-label",
        action="append",
        dest="model_labels",
        default=None,
        help="Column label for a corresponding --experiment-name. May be supplied multiple times.",
    )
    parser.add_argument("--experiments-dir", type=Path, default=None)
    parser.add_argument("--dataset-root-dir", type=Path, default=None)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--num-examples", type=int, default=12)
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--display-size", type=int, default=128)
    parser.add_argument("--examples-per-row", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("paper/graphics/flowers102_reconstruction_examples.png"))
    parser.add_argument("--metadata-output", type=Path, default=None)
    return parser.parse_args()


def _default_experiments_dir() -> Path:
    """Compute default experiments dir."""
    try:
        return get_experiments_dir()
    except KeyError:
        return Path("out/experiments").absolute()


def _load_json(path: Path) -> dict[str, Any]:
    """Load json."""
    return cast(dict[str, Any], json.loads(path.read_text()))


def _experiment_dir(experiments_dir: Path, experiment_name: str) -> Path:
    """Compute experiment dir."""
    path = experiments_dir / experiment_name
    if not path.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {path}")
    return path


def _load_effective_config(experiment_dir: Path) -> dict[str, Any]:
    """Load effective config."""
    path = experiment_dir / "effective_config.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing effective config: {path}")
    return _load_json(path)


def _make_model(config: dict[str, Any], *, stochastic: bool) -> BinaryImageAutoencoder:
    """Create make model."""
    return BinaryImageAutoencoder(
        bits=int(config["bits"]),
        hidden=int(config["hidden"]),
        downsample_layers=int(config["downsample_layers"]),
        residual_layers=int(config["residual_layers"]),
        residual_hidden=int(config["residual_hidden"]),
        stochastic=stochastic,
        grouping=str(config["grouping"]),
    )


def _checkpoint_path(experiment_dir: Path) -> Path:
    """Compute checkpoint path."""
    return experiment_dir / "run_00_deterministic" / "best_model_checkpoint.pt"


def _load_model(
    experiment_dir: Path, config: dict[str, Any], device: torch.device
) -> tuple[BinaryImageAutoencoder, dict[str, Any]]:
    """Load model."""
    checkpoint_path = _checkpoint_path(experiment_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing best deterministic checkpoint: {checkpoint_path}")

    checkpoint = cast(dict[str, Any], torch.load(checkpoint_path, map_location=device, weights_only=False))
    model = _make_model(config, stochastic=checkpoint.get("mode") == "stochastic").to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def _select_indices(dataset_size: int, *, num_examples: int, indices: list[int] | None, seed: int) -> list[int]:
    """Select indices."""
    if dataset_size < 1:
        raise ValueError("dataset must contain at least one example")

    if indices is not None:
        if not indices:
            raise ValueError("--indices must contain at least one index when provided")
        selected = list(indices)
    else:
        if num_examples < 1:
            raise ValueError("--num-examples must be positive")
        count = min(num_examples, dataset_size)
        generator = torch.Generator().manual_seed(seed)
        selected = torch.randperm(dataset_size, generator=generator)[:count].tolist()

    invalid = [index for index in selected if index < 0 or index >= dataset_size]
    if invalid:
        raise IndexError(f"Sample indices out of range for dataset of size {dataset_size}: {invalid}")

    return selected


def _tensor_to_pil(image: torch.Tensor, display_size: int) -> Image.Image:
    """Create tensor to pil."""
    pil_image = cast(Image.Image, to_pil_image(image.detach().cpu().clamp(0.0, 1.0)))
    if pil_image.size != (display_size, display_size):
        pil_image = pil_image.resize((display_size, display_size), Image.Resampling.LANCZOS)
    return pil_image


def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Load font."""
    for path in (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ):
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> None:
    """Compute draw centered text."""
    left, top, right, bottom = box
    text_box = draw.textbbox((0, 0), text, font=font)
    width = text_box[2] - text_box[0]
    height = text_box[3] - text_box[1]
    x = left + (right - left - width) // 2
    y = top + (bottom - top - height) // 2
    draw.text((x, y), text, fill=(20, 20, 20), font=font)


def _render_labeled_grid(
    columns: list[tuple[str, torch.Tensor]],
    output: Path,
    *,
    display_size: int,
    gap: int = 8,
    group_gap: int = 28,
    header_height: int = 34,
    examples_per_row: int = 1,
) -> None:
    """Compute render labeled grid."""
    if display_size < 1:
        raise ValueError("--display-size must be positive")
    if examples_per_row < 1:
        raise ValueError("--examples-per-row must be positive")
    if not columns:
        raise ValueError("at least one image column is required")

    example_count = int(columns[0][1].shape[0])
    if example_count < 1:
        raise ValueError("at least one image row is required")
    if any(int(images.shape[0]) != example_count for _, images in columns):
        raise ValueError("all image columns must contain the same number of rows")

    column_count = len(columns)
    row_count = (example_count + examples_per_row - 1) // examples_per_row
    group_width = column_count * display_size + (column_count - 1) * gap
    width = examples_per_row * group_width + (examples_per_row - 1) * group_gap
    height = header_height + row_count * display_size + max(0, row_count - 1) * gap
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(size=max(11, min(18, display_size // 11)))

    for example_column in range(examples_per_row):
        group_x = example_column * (group_width + group_gap)
        for column_index, (label, _) in enumerate(columns):
            x = group_x + column_index * (display_size + gap)
            _draw_centered_text(draw=draw, box=(x, 0, x + display_size, header_height), text=label, font=font)

    for example_column in range(1, examples_per_row):
        previous_group_end = (example_column - 1) * (group_width + group_gap) + group_width
        x = previous_group_end + group_gap // 2
        draw.line((x, 0, x, height), fill=(190, 190, 190), width=max(1, display_size // 48))

    for example_index in range(example_count):
        example_column = example_index % examples_per_row
        example_row = example_index // examples_per_row
        group_x = example_column * (group_width + group_gap)
        y = header_height + example_row * (display_size + gap)
        for column_index, (_, images) in enumerate(columns):
            x = group_x + column_index * (display_size + gap)
            canvas.paste(_tensor_to_pil(image=images[example_index], display_size=display_size), (x, y))

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _metadata_path_for_output(output: Path, metadata_output: Path | None) -> Path:
    """Compute metadata path for output."""
    return metadata_output if metadata_output is not None else output.with_suffix(".json")


def _build_model_specs(experiments_dir: Path, experiment_names: list[str], labels: list[str]) -> list[SimpleNamespace]:
    """Build model specs."""
    if len(experiment_names) != len(labels):
        raise ValueError("The number of --model-label values must match the number of --experiment-name values")

    specs: list[SimpleNamespace] = []
    for experiment_name, label in zip(experiment_names, labels, strict=True):
        directory = _experiment_dir(experiments_dir=experiments_dir, experiment_name=experiment_name)
        config = _load_effective_config(directory)
        specs.append(SimpleNamespace(name=experiment_name, label=label, directory=directory, config=config))
    return specs


def main() -> None:
    """Run the command-line entry point."""
    args = _parse_args()
    experiment_names = list(args.experiment_names or DEFAULT_EXPERIMENT_NAMES)
    labels = list(args.model_labels or DEFAULT_MODEL_LABELS)
    experiments_dir = (
        args.experiments_dir.absolute() if args.experiments_dir is not None else _default_experiments_dir()
    )
    specs = _build_model_specs(experiments_dir=experiments_dir, experiment_names=experiment_names, labels=labels)

    image_size = int(specs[0].config["image_size"])
    if any(int(spec.config["image_size"]) != image_size for spec in specs):
        raise ValueError("All models must use the same image_size to share examples")

    dataset_root_dir = (
        args.dataset_root_dir.absolute()
        if args.dataset_root_dir is not None
        else Path(specs[0].config["dataset_root_dir"])
    )
    seed_everything(args.seed)
    datasets = build_flowers102_datasets(root=dataset_root_dir, image_size=image_size, download=False)
    dataset = datasets[args.split]
    indices = _select_indices(
        len(cast(Sized, dataset)), num_examples=args.num_examples, indices=args.indices, seed=args.seed
    )
    images = torch.stack([dataset[index] for index in indices])

    device = torch.device(args.device)
    columns: list[tuple[str, torch.Tensor]] = [("original", images)]
    metadata: dict[str, Any] = {
        "split": args.split,
        "indices": indices,
        "display_size": args.display_size,
        "examples_per_row": args.examples_per_row,
        "models": [],
    }

    with torch.inference_mode():
        for spec in specs:
            model, checkpoint = _load_model(experiment_dir=spec.directory, config=spec.config, device=device)
            reconstructions = model.reconstruct_hard(images.to(device)).cpu()
            columns.append((spec.label, reconstructions))
            metadata["models"].append(
                {
                    "label": spec.label,
                    "experiment_name": spec.name,
                    "checkpoint": str(_checkpoint_path(spec.directory).relative_to(experiments_dir)),
                    "bits": int(spec.config["bits"]),
                    "best_epoch": checkpoint.get("best_epoch"),
                }
            )

    _render_labeled_grid(
        columns=columns,
        output=args.output,
        display_size=args.display_size,
        examples_per_row=args.examples_per_row,
    )
    metadata_output = _metadata_path_for_output(output=args.output, metadata_output=args.metadata_output)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote reconstruction examples to {args.output}")
    print(f"Wrote metadata to {metadata_output}")


if __name__ == "__main__":
    main()
