import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from relational_compression.experiments.graph_geometry import _laplacian_from_edges, _undirected_weighted_edges
from relational_compression.experiments.transductive_relational_distortion import optimize as optimize_module
from relational_compression.experiments.transductive_relational_distortion.data_sources import (
    clean_tu_graph,
    randomized_weighted_copy,
)
from relational_compression.experiments.transductive_relational_distortion.objectives import (
    SoftObjectiveResult,
    degree_weighted_marginal,
    hard_evaluation,
    normalized_edge_distortion,
    renyi2_entropy_from_marginal,
    renyi2_prior_matching_to_uniform,
    soft_objective,
)
from relational_compression.experiments.transductive_relational_distortion.optimize import (
    OptimizedPartition,
    RestartSummary,
    optimize_partition,
)
from relational_compression.experiments.transductive_relational_distortion.run_scripts import (
    run_experiment as run_module,
)
from relational_compression.experiments.transductive_relational_distortion.run_scripts.run_experiment import (
    _write_plots,
    run_experiment,
)
from relational_compression.experiments.transductive_relational_distortion.source_geometry import (
    SourceGeometry,
    compute_source_geometry,
    load_or_compute_source_geometry,
    source_collision_edge_importance,
    source_collision_entropy_edge_importance,
)


def _directed_edges(edges: list[tuple[int, int]]) -> torch.Tensor:
    directed = []
    for source, target in edges:
        directed.append((source, target))
        directed.append((target, source))
    return torch.tensor(directed, dtype=torch.long).t().contiguous()


def _toy_graph() -> Data:
    data = Data(edge_index=_directed_edges([(0, 1), (1, 2), (2, 0), (2, 3)]), num_nodes=4)
    data.original_split = "val"
    data.original_graph_index = 0
    return data


def _dense_transition(num_nodes: int, edges: np.ndarray, weights: np.ndarray) -> np.ndarray:
    degree = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(degree, edges[:, 0], weights)
    np.add.at(degree, edges[:, 1], weights)
    transition = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    for (source, target), weight in zip(edges.tolist(), weights.tolist(), strict=True):
        transition[source, target] = weight / degree[source]
        transition[target, source] = weight / degree[target]
    return transition


def test_common_collision_distortion_extremes() -> None:
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    rho = torch.tensor([0.25, 0.75])

    same = torch.nn.functional.one_hot(torch.tensor([0, 0, 0]), num_classes=2).float()
    crossing = torch.nn.functional.one_hot(torch.tensor([0, 1, 0]), num_classes=2).float()

    assert torch.allclose(normalized_edge_distortion(same, edge_index, rho), torch.tensor(0.0))
    assert torch.allclose(normalized_edge_distortion(crossing, edge_index, rho), torch.tensor(1.0))


def test_edge_distortion_rho_and_hard_extremes() -> None:
    geometry = compute_source_geometry(_toy_graph())
    same = torch.zeros(geometry.num_nodes, dtype=torch.long)
    crossing = torch.arange(geometry.num_nodes, dtype=torch.long)

    same_hard = hard_evaluation(same, geometry, num_partitions=geometry.num_nodes)
    crossing_hard = hard_evaluation(crossing, geometry, num_partitions=geometry.num_nodes)

    assert torch.allclose(geometry.rho_edge.sum(), torch.tensor(1.0), atol=1e-6)
    assert np.isclose(same_hard.edge_distortion, 0.0)
    assert np.isclose(crossing_hard.edge_distortion, 1.0)


def test_marginal_organization_uniform_and_collapsed() -> None:
    degree = torch.ones(4)
    uniform_probabilities = torch.full((4, 4), 0.25)
    collapsed = torch.nn.functional.one_hot(torch.zeros(4, dtype=torch.long), num_classes=4).float()

    uniform_q_bar = degree_weighted_marginal(uniform_probabilities, degree)
    collapsed_q_bar = degree_weighted_marginal(collapsed, degree)

    assert torch.allclose(renyi2_prior_matching_to_uniform(uniform_q_bar), torch.tensor(0.0))
    assert torch.allclose(renyi2_prior_matching_to_uniform(collapsed_q_bar), torch.tensor(math.log(4.0)))


def test_marginal_entropy_and_prior_ranges_are_numerically_clean() -> None:
    q_bars = (
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        torch.full((4,), 0.25),
        torch.tensor([0.7, 0.2, 0.1, 0.0]),
    )
    for q_bar in q_bars:
        h2 = renyi2_entropy_from_marginal(q_bar)
        d2 = renyi2_prior_matching_to_uniform(q_bar)
        k_eff = h2.exp()

        assert float(h2) >= -1e-7
        assert float(h2) <= math.log(4.0) + 1e-7
        assert float(k_eff) >= 1.0 - 1e-7
        assert float(k_eff) <= 4.0 + 1e-7
        assert float(d2) >= -1e-7
        assert float(d2) <= math.log(4.0) + 1e-7
        assert torch.allclose(h2 + d2, torch.tensor(math.log(4.0)), atol=1e-6)

    assert torch.allclose(renyi2_entropy_from_marginal(q_bars[0]), torch.tensor(0.0))
    assert torch.allclose(renyi2_entropy_from_marginal(q_bars[1]), torch.tensor(math.log(4.0)))
    assert torch.allclose(renyi2_prior_matching_to_uniform(q_bars[1]), torch.tensor(0.0))


def test_fourier_geometry_foster_sum_and_trace_form() -> None:
    data = _toy_graph()
    geometry = compute_source_geometry(data)
    assignments = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    hard = hard_evaluation(assignments, geometry, num_partitions=2)
    edges, weights = _undirected_weighted_edges(data.edge_index, int(data.num_nodes))
    laplacian = _laplacian_from_edges(int(data.num_nodes), edges, weights).toarray()
    labels = assignments.numpy()
    retained = labels[edges[:, 0]] == labels[edges[:, 1]]
    retained_laplacian = _laplacian_from_edges(int(data.num_nodes), edges[retained], weights[retained]).toarray()
    trace_form = np.trace((laplacian - retained_laplacian) @ np.linalg.pinv(laplacian)) / float(data.num_nodes - 1)

    assert np.isclose(geometry.foster_sum, float(data.num_nodes - 1))
    assert torch.allclose(geometry.rho_fourier.sum(), torch.tensor(1.0), atol=1e-6)
    assert np.isclose(hard.fourier_distortion, trace_form)


def test_collision_geometry_matches_dense_transition_trace_form() -> None:
    data = _toy_graph()
    edges, weights = _undirected_weighted_edges(data.edge_index, int(data.num_nodes))
    transition = _dense_transition(int(data.num_nodes), edges, weights)
    collision = transition @ transition.T
    expected = weights * (
        collision[edges[:, 0], edges[:, 0]]
        + collision[edges[:, 1], edges[:, 1]]
        - 2.0 * collision[edges[:, 0], edges[:, 1]]
    )

    efficient = source_collision_edge_importance(num_nodes=int(data.num_nodes), edges=edges, weights=weights)
    laplacian = _laplacian_from_edges(int(data.num_nodes), edges, weights).toarray()
    assignments = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    labels = assignments.numpy()
    retained = labels[edges[:, 0]] == labels[edges[:, 1]]
    removed_laplacian = (
        laplacian - _laplacian_from_edges(int(data.num_nodes), edges[retained], weights[retained]).toarray()
    )
    dense_distortion = np.trace(removed_laplacian @ collision) / np.trace(laplacian @ collision)
    geometry = compute_source_geometry(data)
    hard = hard_evaluation(assignments, geometry, num_partitions=2)

    assert np.allclose(efficient, expected)
    assert np.isclose(float(efficient.sum()), np.trace(laplacian @ collision))
    assert torch.allclose(geometry.rho_collision.sum(), torch.tensor(1.0), atol=1e-6)
    assert np.isclose(hard.collision_distortion, dense_distortion)


def test_weighted_collision_geometry_matches_trace_form() -> None:
    data = Data(
        edge_index=_directed_edges([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]),
        edge_weight=torch.tensor([0.5, 0.5, 1.7, 1.7, 0.9, 0.9, 2.3, 2.3, 1.1, 1.1]),
        num_nodes=4,
    )
    edges, weights = _undirected_weighted_edges(data.edge_index, int(data.num_nodes), edge_weight=data.edge_weight)
    transition = _dense_transition(int(data.num_nodes), edges, weights)
    collision = transition @ transition.T
    laplacian = _laplacian_from_edges(int(data.num_nodes), edges, weights).toarray()
    assignments = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    labels = assignments.numpy()
    retained = labels[edges[:, 0]] == labels[edges[:, 1]]
    retained_laplacian = _laplacian_from_edges(int(data.num_nodes), edges[retained], weights[retained]).toarray()
    trace_distortion = np.trace((laplacian - retained_laplacian) @ collision) / np.trace(laplacian @ collision)

    geometry = compute_source_geometry(data)
    hard = hard_evaluation(assignments, geometry, num_partitions=2)

    assert np.isclose(hard.collision_distortion, trace_distortion)


def test_collision_entropy_geometry_matches_trace_form() -> None:
    data = Data(
        edge_index=_directed_edges([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4)]),
        edge_weight=torch.tensor([1.0, 1.0, 2.0, 2.0, 1.5, 1.5, 0.5, 0.5, 3.0, 3.0]),
        num_nodes=5,
    )
    edges, weights = _undirected_weighted_edges(data.edge_index, int(data.num_nodes), edge_weight=data.edge_weight)
    importance, entropy, denominator = source_collision_entropy_edge_importance(
        num_nodes=int(data.num_nodes),
        edges=edges,
        weights=weights,
    )
    transition = _dense_transition(int(data.num_nodes), edges, weights)
    expected_entropy = -np.log(np.square(transition).sum(axis=1))
    laplacian = _laplacian_from_edges(int(data.num_nodes), edges, weights).toarray()
    assignments = torch.tensor([0, 0, 1, 1, 0], dtype=torch.long)
    labels = assignments.numpy()
    retained = labels[edges[:, 0]] == labels[edges[:, 1]]
    removed_laplacian = (
        laplacian - _laplacian_from_edges(int(data.num_nodes), edges[retained], weights[retained]).toarray()
    )
    dense_distortion = float(entropy @ removed_laplacian @ entropy / denominator)
    geometry = compute_source_geometry(data)
    hard = hard_evaluation(assignments, geometry, num_partitions=2)

    assert np.allclose(entropy, expected_entropy)
    assert np.isclose(importance.sum(), denominator)
    assert np.isclose(denominator, float(entropy @ laplacian @ entropy))
    assert geometry.rho_collision_entropy is not None
    assert torch.allclose(geometry.rho_collision_entropy.sum(), torch.tensor(1.0), atol=1e-6)
    assert np.isclose(hard.collision_entropy_distortion, dense_distortion)


def test_collision_entropy_zero_denominator_is_explicit() -> None:
    data = Data(edge_index=_directed_edges([(0, 1), (1, 2), (2, 3), (3, 0)]), num_nodes=4)
    geometry = compute_source_geometry(data)

    assert geometry.rho_collision_entropy is None
    assert "collision_entropy" not in geometry.defined_criteria
    with pytest.raises(ValueError, match="Unsupported source distortion criterion"):
        soft_objective(
            torch.zeros(geometry.num_nodes, 2),
            geometry,
            criterion="collision_entropy",
            lambda_org=0.0,
            temperature=1.0,
        )


def test_collision_geometry_handles_large_sparse_graph_without_dense_pair_matrix() -> None:
    num_nodes = 2_000
    edges = np.stack((np.arange(num_nodes - 1), np.arange(1, num_nodes)), axis=1)
    weights = np.ones(num_nodes - 1, dtype=np.float64)

    importance = source_collision_edge_importance(num_nodes=num_nodes, edges=edges, weights=weights)

    assert importance.shape == (num_nodes - 1,)
    assert np.isfinite(importance).all()
    assert float(importance.sum()) > 0.0


def test_source_geometry_cache_round_trip(tmp_path: Path) -> None:
    data = _toy_graph()

    first = load_or_compute_source_geometry(data, cache_dir=tmp_path)
    second = load_or_compute_source_geometry(data, cache_dir=tmp_path)

    assert torch.allclose(first.rho_fourier, second.rho_fourier)
    assert torch.allclose(first.rho_edge, second.rho_edge)
    assert torch.allclose(first.rho_collision, second.rho_collision)
    assert np.isclose(first.foster_sum, second.foster_sum)


def test_weighted_graph_consistency() -> None:
    data = Data(
        edge_index=_directed_edges([(0, 1), (1, 2), (2, 0), (2, 3)]),
        edge_weight=torch.tensor([0.5, 0.5, 2.0, 2.0, 1.5, 1.5, 3.0, 3.0]),
        num_nodes=4,
    )
    geometry = compute_source_geometry(data)
    edges, weights = _undirected_weighted_edges(data.edge_index, int(data.num_nodes), edge_weight=data.edge_weight)
    transition = _dense_transition(int(data.num_nodes), edges, weights)

    assert np.allclose(transition.sum(axis=1), np.ones(int(data.num_nodes)))
    assert np.isclose(geometry.foster_sum, float(data.num_nodes - 1))
    assert torch.allclose(geometry.degree, torch.tensor([2.0, 2.5, 6.5, 3.0]), atol=1e-6)
    assert torch.allclose(geometry.rho_edge.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(geometry.rho_fourier.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(geometry.rho_collision.sum(), torch.tensor(1.0), atol=1e-6)
    if geometry.rho_collision_entropy is not None:
        assert torch.allclose(geometry.rho_collision_entropy.sum(), torch.tensor(1.0), atol=1e-6)


def test_randomized_malnet_weighting_is_deterministic_and_symmetric() -> None:
    data = _toy_graph()
    first, first_seed = randomized_weighted_copy(data, global_seed=99, sigma=0.75)
    second, second_seed = randomized_weighted_copy(data, global_seed=99, sigma=0.75)
    third, third_seed = randomized_weighted_copy(data, global_seed=100, sigma=0.75)
    first_edges, first_weights = _undirected_weighted_edges(
        first.edge_index, int(first.num_nodes), edge_weight=first.edge_weight
    )
    third_edges, third_weights = _undirected_weighted_edges(
        third.edge_index, int(third.num_nodes), edge_weight=third.edge_weight
    )

    assert first_seed == second_seed
    assert third_seed != first_seed
    assert torch.allclose(first.edge_weight, second.edge_weight)
    assert torch.all(first.edge_weight > 0)
    assert np.array_equal(first_edges, third_edges)
    assert np.isclose(float(first_weights.mean()), 1.0)
    assert not np.allclose(first_weights, third_weights)


def test_tu_cleaning_ignores_labels_and_extracts_lcc() -> None:
    raw = Data(
        edge_index=torch.tensor(
            [[0, 1, 1, 2, 2, 0, 3, 3, 4, 5], [1, 0, 2, 1, 0, 2, 3, 4, 5, 4]],
            dtype=torch.long,
        ),
        y=torch.tensor([1]),
        num_nodes=6,
    )
    data, metadata = clean_tu_graph(raw, dataset_name="TEST", dataset_index=7, min_nodes=3, max_nodes=None)

    assert data is not None
    assert "y" not in data
    assert int(data.num_nodes) == 3
    assert int(data.edge_index.shape[1]) == 6
    assert metadata["retained"] is True


def test_soft_total_objective_has_finite_nonzero_gradient() -> None:
    geometry = compute_source_geometry(_toy_graph())
    logits = 0.01 * torch.randn(geometry.num_nodes, 3, generator=torch.Generator().manual_seed(3))
    logits.requires_grad_(True)

    result = soft_objective(logits, geometry, criterion="fourier", lambda_org=0.1, temperature=1.0)
    result.loss.backward()

    assert torch.isfinite(result.loss)
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.norm()) > 0.0


def test_optimize_partition_runs_without_nan() -> None:
    geometry = compute_source_geometry(_toy_graph())

    optimized = optimize_partition(
        geometry,
        criterion="collision",
        lambda_org=0.2,
        num_partitions=4,
        temperature=1.0,
        learning_rate=0.05,
        optimization_steps=10,
        init_scale=1e-2,
        restart_seeds=(11,),
        device="cpu",
    )

    assert optimized.assignments.shape == (geometry.num_nodes,)
    assert all(summary.finite for summary in optimized.restart_summaries)
    assert torch.isfinite(optimized.probabilities).all()
    assert torch.allclose(optimized.probabilities.sum(dim=-1), torch.ones(geometry.num_nodes), atol=1e-6)


def test_nonfinite_restart_keeps_earlier_best_iterate(monkeypatch) -> None:
    geometry = compute_source_geometry(_toy_graph())

    class FiniteLossNanGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, logits: torch.Tensor) -> torch.Tensor:
            return logits.new_tensor(0.0)

        @staticmethod
        def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
            del ctx, grad_output
            return torch.full((geometry.num_nodes, 3), float("nan"))

    def fake_seeded_logits(
        num_nodes: int,
        num_partitions: int,
        *,
        seed: int,
        init_scale: float,
        device: torch.device,
        collapse_bias: float | None = None,
    ) -> torch.Tensor:
        del init_scale, collapse_bias
        value = -1.0 if int(seed) == 1 else 1.0
        return torch.full((num_nodes, num_partitions), value, device=device, requires_grad=True)

    def fake_soft_objective(
        logits: torch.Tensor,
        geometry: SourceGeometry,
        *,
        criterion: str,
        lambda_org: float,
        temperature: float,
    ) -> SoftObjectiveResult:
        del geometry, criterion, lambda_org, temperature
        q_bar = torch.full((logits.shape[1],), 1.0 / float(logits.shape[1]), device=logits.device)
        if float(logits.detach().cpu()[0, 0]) < 0.0:
            loss = FiniteLossNanGradient.apply(logits)
        else:
            loss = logits.sum() * 0.0 + 1.0
        return SoftObjectiveResult(
            loss=loss,
            own_distortion=loss,
            edge_distortion=loss,
            fourier_distortion=loss,
            collision_distortion=loss,
            collision_entropy_distortion=loss,
            marginal_d2=loss * 0.0,
            soft_h2=loss * 0.0,
            soft_k_eff=loss * 0.0 + 1.0,
            q_bar=q_bar,
            assignment_confidence=loss * 0.0 + 1.0,
            assignment_entropy=loss * 0.0,
        )

    monkeypatch.setattr(optimize_module, "_seeded_logits", fake_seeded_logits)
    monkeypatch.setattr(optimize_module, "soft_objective", fake_soft_objective)

    optimized = optimize_module.optimize_partition(
        geometry,
        criterion="fourier",
        lambda_org=0.0,
        num_partitions=3,
        temperature=1.0,
        learning_rate=0.1,
        optimization_steps=1,
        init_scale=1e-2,
        restart_seeds=(1, 2),
        device="cpu",
    )

    assert optimized.restart_summaries[0].finite
    assert optimized.restart_summaries[0].encountered_nonfinite
    assert optimized.restart_summaries[0].best_loss == 0.0
    assert optimized.restart_summaries[0].final_loss == 0.0
    assert optimized.selected_seed == 1


def test_final_target_candidate_closure_selects_best_cross_candidate() -> None:
    geometry = compute_source_geometry(_toy_graph())

    def logits_for(assignments: torch.Tensor) -> torch.Tensor:
        logits = torch.full((geometry.num_nodes, 2), -8.0)
        logits[torch.arange(geometry.num_nodes), assignments] = 8.0
        return logits

    def summary(loss: float) -> RestartSummary:
        return RestartSummary(
            restart_index=0,
            init_kind="test",
            seed=0,
            best_loss=loss,
            best_step=0,
            best_own_distortion=loss,
            best_marginal_d2=0.0,
            best_soft_k_eff=1.0,
            best_assignment_confidence=1.0,
            best_assignment_entropy=0.0,
            final_loss=loss,
            final_own_distortion=loss,
            final_marginal_d2=0.0,
            final_soft_k_eff=1.0,
            assignment_confidence=1.0,
            assignment_entropy=0.0,
            finite=True,
            encountered_nonfinite=False,
        )

    def partition(criterion: str, assignments: torch.Tensor) -> OptimizedPartition:
        logits = logits_for(assignments)
        soft = soft_objective(logits, geometry, criterion=criterion, lambda_org=0.0, temperature=1.0)
        probabilities = torch.softmax(logits, dim=-1)
        return OptimizedPartition(
            criterion=criterion,
            lambda_org=0.0,
            selected_restart=0,
            selected_seed=0,
            logits=logits,
            probabilities=probabilities,
            assignments=probabilities.argmax(dim=-1),
            soft_result=soft,
            restart_summaries=[summary(float(soft.loss))],
        )

    edge_bad = partition("edge", torch.tensor([0, 1, 0, 1]))
    fourier_good = partition("fourier", torch.tensor([0, 0, 0, 0]))

    closed = run_module._final_target_candidate_closure(
        geometry,
        criteria=("edge", "fourier"),
        lambda_org=0.0,
        candidates={"edge": [("ordinary_restarts", edge_bad)], "fourier": [("ordinary_restarts", fourier_good)]},
        config=SimpleNamespace(assignment_temperature=1.0),
        device=torch.device("cpu"),
    )

    selected_kind, selected_edge, _, _ = closed["edge"]

    assert selected_kind.startswith("final_candidate_from_fourier")
    assert float(selected_edge.soft_result.loss) <= float(edge_bad.soft_result.loss) - 1e-3


def test_plot_writer_keeps_splits_separate(tmp_path: Path) -> None:
    aggregate = [
        {
            "collection": "toy",
            "source_variant": "unweighted",
            "split": "val",
            "criterion": "fourier",
            "lambda_org": 0.0,
            "hard_h2_mean": 0.0,
            "hard_k_eff_mean": 1.0,
            "hard_D_F_mean": 0.0,
            "soft_h2_mean": 0.0,
            "soft_D_F_mean": 0.0,
        },
        {
            "collection": "toy",
            "source_variant": "unweighted",
            "split": "val",
            "criterion": "fourier",
            "lambda_org": 0.2,
            "hard_h2_mean": 2.0,
            "hard_k_eff_mean": 7.5,
            "hard_D_F_mean": 0.2,
            "soft_h2_mean": 1.8,
            "soft_D_F_mean": 0.18,
        },
        {
            "collection": "toy",
            "source_variant": "unweighted",
            "split": "test",
            "criterion": "fourier",
            "lambda_org": 0.0,
            "hard_h2_mean": 0.0,
            "hard_k_eff_mean": 1.0,
            "hard_D_F_mean": 0.0,
            "soft_h2_mean": 0.0,
            "soft_D_F_mean": 0.0,
        },
        {
            "collection": "toy",
            "source_variant": "unweighted",
            "split": "test",
            "criterion": "fourier",
            "lambda_org": 0.2,
            "hard_h2_mean": 1.9,
            "hard_k_eff_mean": 7.0,
            "hard_D_F_mean": 0.25,
            "soft_h2_mean": 1.7,
            "soft_D_F_mean": 0.23,
        },
    ]

    _write_plots(tmp_path, aggregate)

    assert (tmp_path / "plots" / "toy_unweighted_val_hard_own_complexity_distortion.png").exists()
    assert (tmp_path / "plots" / "toy_unweighted_test_hard_own_complexity_distortion.png").exists()


def test_synthetic_smoke_run_writes_expected_outputs(tmp_path: Path) -> None:
    config = SimpleNamespace(
        seed=0,
        dataset_backend="synthetic",
        experiments_dir=tmp_path,
        experiment_base_name="trd_smoke",
        experiment_name="trd_smoke",
        unique_postfix="",
        splits=("val",),
        max_graphs_per_split=1,
        criteria=("edge", "fourier", "collision", "collision_entropy"),
        lambda_org_values=(0.0,),
        num_partitions=4,
        assignment_temperature=1.0,
        learning_rate=0.05,
        optimization_steps=5,
        init_scale=1e-2,
        restart_seeds=(7,),
        device="cpu",
        source_geometry_cache_version="test_v1",
        source_geometry_solve_batch_size=64,
        random_partition_count=4,
        random_partition_seed=0,
        cross_warm_start=True,
        cross_warm_start_closure_rounds=1,
        write_plots=False,
        source_collections=("synthetic",),
        graphs_per_collection=1,
        graph_seed=0,
    )

    result = run_experiment(config)
    experiment_dir = Path(result["experiment_dir"])

    assert (experiment_dir / "effective_config.json").exists()
    assert (experiment_dir / "per_graph_results.csv").exists()
    assert (experiment_dir / "source_rho_correlations.csv").exists()
    assert (experiment_dir / "random_partition_distortion_correlations.csv").exists()
    assert (experiment_dir / "cross_objective_diagnostics.csv").exists()
    assert (
        experiment_dir
        / "partition_artifacts"
        / "synthetic"
        / "unweighted"
        / "val"
        / "graph_000000"
        / "fourier_lambda0.pt"
    ).exists()
    assert (
        experiment_dir
        / "partition_artifacts"
        / "synthetic"
        / "unweighted"
        / "val"
        / "graph_000000"
        / "collision_lambda0.pt"
    ).exists()
    assert (experiment_dir / "aggregate_results.json").exists()
    aggregate = json.loads((experiment_dir / "aggregate_results.json").read_text())
    assert len(aggregate) == 4
