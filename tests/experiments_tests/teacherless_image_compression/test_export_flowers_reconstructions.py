import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from relational_compression.experiments.teacherless_image_compression.run_scripts import (
    export_flowers_reconstructions as export_script,
)


def test_select_indices_is_seeded_and_bounds_checked() -> None:
    first = export_script._select_indices(20, num_examples=5, indices=None, seed=7)
    second = export_script._select_indices(20, num_examples=5, indices=None, seed=7)

    assert first == second
    assert len(first) == 5
    assert len(set(first)) == 5

    assert export_script._select_indices(20, num_examples=5, indices=[3, 1], seed=999) == [3, 1]

    with pytest.raises(IndexError, match="out of range"):
        export_script._select_indices(2, num_examples=5, indices=[0, 2], seed=7)


def test_render_labeled_grid_writes_expected_png(tmp_path: Path) -> None:
    output = tmp_path / "grid.png"
    images = torch.linspace(0.0, 1.0, steps=4 * 3 * 4 * 4).view(4, 3, 4, 4)

    export_script._render_labeled_grid(
        [("original", images), ("reconstruction", 1.0 - images)],
        output,
        display_size=8,
        gap=2,
        group_gap=4,
        header_height=10,
        examples_per_row=2,
    )

    with Image.open(output) as rendered:
        assert rendered.mode == "RGB"
        assert rendered.size == (40, 28)


def test_build_model_specs_reads_effective_configs(tmp_path: Path) -> None:
    experiments_dir = tmp_path / "experiments"
    config = {"bits": 16, "image_size": 256}
    experiment_dir = experiments_dir / "flowers16"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "effective_config.json").write_text(json.dumps(config))

    specs = export_script._build_model_specs(experiments_dir, ["flowers16"], ["16-bit"])

    assert specs[0].name == "flowers16"
    assert specs[0].label == "16-bit"
    assert specs[0].directory == experiment_dir
    assert specs[0].config == config


def test_build_model_specs_requires_matching_labels(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="model-label"):
        export_script._build_model_specs(tmp_path, ["one", "two"], ["one label"])


def test_metadata_path_defaults_to_json_sidecar(tmp_path: Path) -> None:
    output = tmp_path / "examples.png"

    assert export_script._metadata_path_for_output(output, None) == tmp_path / "examples.json"
    assert export_script._metadata_path_for_output(output, tmp_path / "custom.json") == tmp_path / "custom.json"
