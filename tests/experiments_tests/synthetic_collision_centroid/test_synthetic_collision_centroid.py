import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from relational_compression.experiments.synthetic_collision_centroid.data import generate_imbalanced_gaussian_mixture
from relational_compression.experiments.synthetic_collision_centroid.metrics import (
    centroid_distortion,
    decoder_decomposition,
    normalized_collision_pair_distortion,
    pairwise_squared_distances,
    raw_collision_pair_distortion,
)
from relational_compression.experiments.synthetic_collision_centroid.models import (
    BinaryCodeEncoder,
    factorized_code_probabilities,
    hard_code_ids,
)
from relational_compression.experiments.synthetic_collision_centroid.run_scripts.run_synthetic_exp import (
    _gradient_comparison,
    run_experiment,
)


def _small_points() -> torch.Tensor:
    return torch.tensor(
        [
            [-2.0, -1.0],
            [-1.6, -0.8],
            [-1.1, 1.8],
            [-0.8, 2.2],
            [2.0, 1.5],
            [2.5, 1.2],
            [3.2, -1.4],
            [3.5, -1.0],
        ],
        dtype=torch.float32,
    )


def test_factorized_code_probabilities_are_normalized() -> None:
    logits = torch.tensor([[0.2, -0.5, 1.0], [-1.0, 0.3, 0.8]])
    probabilities = factorized_code_probabilities(logits, temperature=0.7)

    assert probabilities.shape == (2, 8)
    assert torch.all(probabilities > 0.0)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2), atol=1e-6)


def test_hard_code_ids_use_zero_threshold_binary_address() -> None:
    logits = torch.tensor([[-1.0, 2.0, -0.2], [0.1, 0.2, 3.0], [0.0, -1.0, 0.0]])

    assert torch.equal(hard_code_ids(logits), torch.tensor([2, 7, 0]))


def test_centroid_and_normalized_pair_distortion_are_equal() -> None:
    points = _small_points()
    logits = torch.tensor(
        [
            [-1.0, -0.5],
            [-0.8, -0.2],
            [-0.5, 1.0],
            [-0.2, 1.2],
            [1.0, 0.8],
            [1.2, 0.5],
            [1.4, -0.8],
            [1.7, -1.1],
        ],
        requires_grad=True,
    )
    probabilities = factorized_code_probabilities(logits, temperature=0.7)
    squared_distances = pairwise_squared_distances(points)

    centroid_value = centroid_distortion(points, probabilities)
    pair_value = normalized_collision_pair_distortion(
        points,
        probabilities,
        squared_distances=squared_distances,
    )

    assert torch.allclose(centroid_value, pair_value, atol=2e-6, rtol=2e-6)


def test_centroid_and_normalized_pair_gradients_match() -> None:
    torch.manual_seed(5)
    points = _small_points()
    logits = torch.randn(points.shape[0], 3, requires_grad=True)
    probabilities = factorized_code_probabilities(logits, temperature=0.8)
    squared_distances = pairwise_squared_distances(points)
    centroid_value = centroid_distortion(points, probabilities)
    pair_value = normalized_collision_pair_distortion(
        points,
        probabilities,
        squared_distances=squared_distances,
    )
    centroid_gradient = torch.autograd.grad(centroid_value, logits, retain_graph=True)[0]
    pair_gradient = torch.autograd.grad(pair_value, logits)[0]

    assert (
        torch.nn.functional.cosine_similarity(centroid_gradient.reshape(-1), pair_gradient.reshape(-1), dim=0) > 0.99999
    )
    relative_error = (centroid_gradient - pair_gradient).norm() / centroid_gradient.norm()
    assert relative_error < 2e-5


def test_raw_collision_pair_distortion_is_not_the_centroid_identity() -> None:
    points = _small_points()
    logits = torch.tensor(
        [
            [-4.0, -4.0],
            [-4.0, -4.0],
            [-4.0, -4.0],
            [-4.0, -4.0],
            [-4.0, 4.0],
            [-4.0, 4.0],
            [4.0, -4.0],
            [4.0, 4.0],
        ]
    )
    probabilities = factorized_code_probabilities(logits, temperature=0.5)
    squared_distances = pairwise_squared_distances(points)
    centroid_value = centroid_distortion(points, probabilities)
    normalized_pair = normalized_collision_pair_distortion(
        points,
        probabilities,
        squared_distances=squared_distances,
    )
    raw_pair = raw_collision_pair_distortion(points, probabilities, squared_distances=squared_distances)

    assert torch.allclose(centroid_value, normalized_pair, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(centroid_value, 4.0 * raw_pair, atol=1e-3, rtol=1e-3)


def test_decoder_distortion_decomposes_into_centroid_plus_gap() -> None:
    torch.manual_seed(7)
    points = _small_points()
    probabilities = factorized_code_probabilities(torch.randn(points.shape[0], 2), temperature=0.7)
    decoder_vectors = torch.randn(4, 2)

    decomposition = decoder_decomposition(points, probabilities, decoder_vectors)

    assert torch.allclose(
        decomposition.decoder_distortion,
        decomposition.centroid_distortion + decomposition.decoder_centroid_gap,
        atol=2e-6,
        rtol=2e-6,
    )
    assert abs(float(decomposition.residual)) < 2e-6


def test_synthetic_mixture_is_deterministic_and_imbalanced() -> None:
    weights = torch.tensor([0.5, 0.3, 0.2])
    means = torch.tensor([[-2.0, 0.0], [1.0, 2.0], [2.0, -1.0]])
    covariances = torch.stack((torch.eye(2) * 0.2, torch.eye(2) * 0.3, torch.eye(2) * 0.4))

    first = generate_imbalanced_gaussian_mixture(
        num_points=100,
        component_weights=weights,
        component_means=means,
        component_covariances=covariances,
        seed=11,
    )
    second = generate_imbalanced_gaussian_mixture(
        num_points=100,
        component_weights=weights,
        component_means=means,
        component_covariances=covariances,
        seed=11,
    )

    assert torch.equal(first.points, second.points)
    assert torch.equal(first.component_ids, second.component_ids)
    assert first.component_counts.tolist() == [50, 30, 20]


def test_model_gradient_equivalence_helper_reports_matching_gradients() -> None:
    torch.manual_seed(13)
    points = _small_points()
    model = BinaryCodeEncoder(input_dim=2, hidden_dim=8, bits=2)
    result = _gradient_comparison(
        model,
        points,
        pairwise_squared_distances(points),
        code_temperature=0.7,
    )

    assert math.isclose(result["centroid_distortion"], result["normalized_pair_distortion"], rel_tol=2e-6)
    assert result["gradient_cosine_similarity"] > 0.99999
    assert result["gradient_relative_error"] < 2e-5


def test_small_experiment_writes_standard_outputs(tmp_path: Path) -> None:
    config_file = tmp_path / "small_config.py"
    config_file.write_text("# synthetic test config\n")
    config = SimpleNamespace(
        config_file=config_file,
        task_model_name="synthetic_collision_centroid",
        dataset_name="test_mixture",
        dataset_version="1.0",
        num_experiments=1,
        experiments_dir=tmp_path / "experiments",
        experiment_base_name="synthetic_test",
        experiment_name="synthetic_test_run",
        unique_postfix=None,
        num_points=48,
        component_weights=(0.5, 0.3, 0.2),
        component_means=((-2.0, 0.0), (1.0, 2.0), (2.0, -1.0)),
        component_covariances=(
            ((0.2, 0.0), (0.0, 0.2)),
            ((0.3, 0.05), (0.05, 0.3)),
            ((0.4, -0.05), (-0.05, 0.25)),
        ),
        bits=2,
        hidden_dim=8,
        code_temperature=0.7,
        num_steps=3,
        learning_rate=0.02,
        momentum=0.9,
        random_state_samples=3,
        random_state_scale_min=0.25,
        random_state_scale_max=1.0,
        decoder_steps=3,
        decoder_learning_rate=0.1,
        decoder_initial_scale=1.0,
        seed=17,
        log_to_tensorboard_global=False,
        tensorboard_log_steps=1,
        figure_dpi=60,
        device="cpu",
    )

    results = run_experiment(config)
    experiment_dir = config.experiments_dir / config.experiment_name
    run_dir = experiment_dir / "run_00_deterministic"
    result = results[0]

    assert (experiment_dir / "effective_config.json").exists()
    assert (experiment_dir / "all_run_results.json").exists()
    assert (run_dir / "result.json").exists()
    assert (run_dir / "encoder_centroid_history.csv").exists()
    assert (run_dir / "encoder_pairwise_history.csv").exists()
    assert (run_dir / "random_state_equivalence.csv").exists()
    assert (run_dir / "decoder_history.csv").exists()
    assert (run_dir / "figures" / "synthetic_collision_centroid_equivalence.png").exists()
    assert (run_dir / "figures" / "decoder_centroid_decomposition.png").exists()
    assert result["random_state_equivalence"]["max_absolute_normalized_pair_error"] < 1e-4
    assert result["decoder_decomposition"]["max_absolute_residual"] < 1e-4
    loaded = json.loads((run_dir / "result.json").read_text())
    assert loaded["run_index"] == 0
    assert loaded["seed"] == 17


def test_tensorboard_log_steps_zero_disables_tensorboard_logging(tmp_path: Path) -> None:
    config_file = tmp_path / "small_config.py"
    config_file.write_text("# synthetic test config\n")
    config = SimpleNamespace(
        config_file=config_file,
        task_model_name="synthetic_collision_centroid",
        dataset_name="test_mixture",
        dataset_version="1.0",
        num_experiments=1,
        experiments_dir=tmp_path / "experiments",
        experiment_base_name="synthetic_test",
        experiment_name="synthetic_test_no_tensorboard",
        unique_postfix=None,
        num_points=24,
        component_weights=(0.5, 0.3, 0.2),
        component_means=((-2.0, 0.0), (1.0, 2.0), (2.0, -1.0)),
        component_covariances=(
            ((0.2, 0.0), (0.0, 0.2)),
            ((0.3, 0.05), (0.05, 0.3)),
            ((0.4, -0.05), (-0.05, 0.25)),
        ),
        bits=2,
        hidden_dim=8,
        code_temperature=0.7,
        num_steps=1,
        learning_rate=0.02,
        momentum=0.9,
        random_state_samples=1,
        random_state_scale_min=0.25,
        random_state_scale_max=1.0,
        decoder_steps=1,
        decoder_learning_rate=0.1,
        decoder_initial_scale=1.0,
        seed=17,
        log_to_tensorboard_global=True,
        tensorboard_log_steps=0,
        figure_dpi=60,
        device="cpu",
    )

    run_experiment(config)

    run_dir = config.experiments_dir / config.experiment_name / "run_00_deterministic"
    assert (run_dir / "result.json").exists()
    assert not (run_dir / "tensorboard_logs").exists()
