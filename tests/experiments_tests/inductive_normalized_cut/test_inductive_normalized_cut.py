import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv

from relational_compression.experiments.inductive_normalized_cut import baselines as ncut_baselines
from relational_compression.experiments.inductive_normalized_cut.baselines import (
    SpectralNormalizedCutConfig,
    spectral_normalized_cut_partition,
)
from relational_compression.experiments.inductive_normalized_cut.data import (
    PreprocessConfig,
    clean_to_undirected_lcc,
    directed_pagerank,
    directed_structural_node_features,
    hutchinson_random_walk_return_probabilities,
    make_synthetic_community_graph,
    preprocess_graph,
    random_walk_transition_apply,
    structural_node_features,
)
from relational_compression.experiments.inductive_normalized_cut.metrics import hard_normalized_cut, soft_normalized_cut
from relational_compression.experiments.inductive_normalized_cut.models import (
    GraphGPSEfficientAttentionPartitionEncoder,
    GraphPartitionEncoder,
    make_model,
)
from relational_compression.experiments.inductive_normalized_cut.run_scripts.run_malnet_tiny_exp import (
    _checkpoint_config_dict,
    evaluate_model,
    run_experiment,
)


def _directed_edges(edges: list[tuple[int, int]]) -> torch.Tensor:
    directed = []
    for source, target in edges:
        directed.append((source, target))
        directed.append((target, source))
    return torch.tensor(directed, dtype=torch.long).t().contiguous()


def _toy_graph() -> Data:
    edge_index = _directed_edges([(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)])
    x = structural_node_features(edge_index, 4, random_walk_steps=2)
    return Data(x=x, edge_index=edge_index, num_nodes=4)


def _dense_transition(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    source, target = edge_index
    degree = torch.bincount(source, minlength=num_nodes).float().clamp_min(1.0)
    transition = torch.zeros(num_nodes, num_nodes)
    transition[source, target] += 1.0 / degree[source]
    return transition


def _dense_directed_pagerank(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    alpha: float = 0.85,
    max_iter: int = 200,
    tolerance: float = 1e-12,
) -> torch.Tensor:
    non_self = edge_index[0] != edge_index[1]
    edges = torch.unique(edge_index[:, non_self], dim=1)
    source, target = edges
    out_degree = torch.bincount(source, minlength=num_nodes).float()
    transition = torch.zeros(num_nodes, num_nodes)
    if source.numel() > 0:
        transition[source, target] = 1.0 / out_degree[source].clamp_min(1.0)

    rank = torch.full((num_nodes,), 1.0 / num_nodes)
    teleport = (1.0 - alpha) / num_nodes
    for _ in range(max_iter):
        next_rank = torch.full_like(rank, teleport)
        next_rank += alpha * (rank @ transition)
        next_rank += alpha * rank[out_degree == 0].sum() / num_nodes
        if float(torch.abs(next_rank - rank).sum()) <= tolerance:
            rank = next_rank
            break
        rank = next_rank
    return rank / rank.sum()


def _exhaustive_rademacher_probes(num_nodes: int) -> torch.Tensor:
    rows = []
    for mask in range(2**num_nodes):
        rows.append([1.0 if mask & (1 << node) else -1.0 for node in range(num_nodes)])
    return torch.tensor(rows, dtype=torch.float32).t().contiguous()


def _one_hot(assignments: torch.Tensor, num_partitions: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(assignments, num_classes=num_partitions).float()


def _direct_cut_volume_ncut(
    assignments: torch.Tensor, edge_index: torch.Tensor, num_partitions: int
) -> tuple[float, float]:
    degree = torch.bincount(edge_index[0], minlength=assignments.numel()).float()
    nassoc = 0.0
    for partition in range(num_partitions):
        members = assignments == partition
        volume = float(degree[members].sum())
        if volume == 0.0:
            continue
        internal = 0.0
        for source, target in edge_index.t().tolist():
            if members[source] and members[target]:
                internal += 1.0
        nassoc += internal / volume
    return float(num_partitions) - nassoc, nassoc


def test_one_hot_soft_normalized_cut_equals_direct_classical_normalized_cut() -> None:
    data = _toy_graph()
    assignments = torch.tensor([0, 0, 1, 2])
    probabilities = _one_hot(assignments, num_partitions=3)

    soft = soft_normalized_cut(probabilities, data.edge_index)
    hard = hard_normalized_cut(assignments, data.edge_index, num_partitions=3)
    direct_ncut, direct_nassoc = _direct_cut_volume_ncut(assignments, data.edge_index, num_partitions=3)

    assert torch.allclose(soft.nassoc_per_graph, hard.nassoc_per_graph, atol=1e-6)
    assert torch.allclose(soft.ncut_per_graph, hard.ncut_per_graph, atol=1e-6)
    assert math.isclose(float(soft.nassoc_per_graph[0]), direct_nassoc, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(float(soft.ncut_per_graph[0]), direct_ncut, rel_tol=1e-6, abs_tol=1e-6)


def test_batched_loss_equals_average_of_separate_graph_losses() -> None:
    first = _toy_graph()
    second = make_synthetic_community_graph(
        num_communities=2,
        nodes_per_community=4,
        intra_edges_per_node=1,
        inter_edges_per_community_pair=1,
        random_walk_steps=2,
        seed=4,
    )
    batch = Batch.from_data_list([first, second])
    torch.manual_seed(0)
    logits = torch.randn(batch.num_nodes, 4)
    probabilities = torch.softmax(logits, dim=-1)

    batched = soft_normalized_cut(probabilities, batch.edge_index, batch=batch.batch)
    separate_first = soft_normalized_cut(probabilities[: first.num_nodes], first.edge_index)
    separate_second = soft_normalized_cut(probabilities[first.num_nodes :], second.edge_index)
    expected = torch.stack((separate_first.loss, separate_second.loss)).mean()

    assert torch.allclose(batched.loss, expected, atol=1e-6)


def test_undirected_preprocessing_removes_loops_duplicates_and_selects_lcc() -> None:
    raw_edges = torch.tensor(
        [
            [0, 0, 0, 1, 1, 2, 4, 5],
            [0, 1, 1, 0, 2, 3, 5, 4],
        ],
        dtype=torch.long,
    )

    edge_index, metadata = clean_to_undirected_lcc(raw_edges, num_nodes=6)

    assert metadata["lcc_num_nodes"] == 4
    assert metadata["lcc_num_edges"] == 3
    assert not torch.any(edge_index[0] == edge_index[1])
    assert edge_index.shape[1] == torch.unique(edge_index, dim=1).shape[1]
    edges = {tuple(edge) for edge in edge_index.t().tolist()}
    assert all((target, source) in edges for source, target in edges)


def test_preprocess_graph_removes_label_and_records_filter_metadata(tmp_path: Path) -> None:
    del tmp_path
    edge_index = _directed_edges([(0, 1), (1, 2), (2, 3)])
    raw = Data(edge_index=edge_index, num_nodes=4, y=torch.tensor([2]))
    config = PreprocessConfig(dataset_root_dir=Path("/unused"), min_nodes_per_graph=3, random_walk_steps=2)

    data, metadata = preprocess_graph(raw, split="train", original_graph_index=7, config=config)

    assert data is not None
    assert "y" not in data
    assert metadata.retained
    assert metadata.original_graph_index == 7
    assert data.x.shape == (4, 6)


def test_directed_features_survive_lcc_reindexing() -> None:
    raw_edges = torch.tensor(
        [
            [0, 0, 2, 3, 3, 4],
            [1, 2, 0, 0, 2, 5],
        ],
        dtype=torch.long,
    )
    raw = Data(edge_index=raw_edges, num_nodes=6, y=torch.tensor([1]))
    config = PreprocessConfig(
        dataset_root_dir=Path("/unused"),
        min_nodes_per_graph=4,
        random_walk_steps=0,
        include_directed_features=True,
        pagerank_max_iter=100,
        pagerank_tolerance=1e-10,
    )

    data, metadata = preprocess_graph(raw, split="train", original_graph_index=3, config=config)

    assert data is not None
    assert metadata.lcc_num_nodes == 4
    assert data.x.shape == (4, 14)
    directed = data.x[:, 4:14]
    expected_in = torch.tensor([2.0, 1.0, 2.0, 0.0])
    expected_out = torch.tensor([2.0, 0.0, 1.0, 2.0])
    expected_total = expected_in + expected_out
    full_directed = directed_structural_node_features(
        raw_edges,
        6,
        pagerank_max_iter=100,
        pagerank_tolerance=1e-10,
    )
    assert torch.allclose(directed[:, 0], expected_in)
    assert torch.allclose(directed[:, 1], expected_out)
    assert torch.allclose(directed[:, 2], expected_total)
    assert torch.allclose(directed[:, 3], torch.log1p(expected_in))
    assert torch.allclose(directed[:, 4], torch.log1p(expected_out))
    assert torch.allclose(directed[:, 5], torch.log1p(expected_total))
    assert torch.allclose(directed[:, 6], expected_in / expected_in.max())
    assert torch.allclose(directed[:, 7], expected_out / expected_out.max())
    assert torch.allclose(directed[:, 8], expected_total / expected_total.max())
    assert torch.allclose(directed[:, 9], full_directed[:4, 9], atol=1e-6)


def test_directed_pagerank_matches_dense_reference_and_features_are_deterministic() -> None:
    edge_index = torch.tensor(
        [
            [0, 0, 1, 2, 2, 3],
            [1, 2, 2, 0, 3, 2],
        ],
        dtype=torch.long,
    )

    first_rank = directed_pagerank(edge_index, 4, max_iter=200, tolerance=1e-10)
    second_rank = directed_pagerank(edge_index, 4, max_iter=200, tolerance=1e-10)
    dense_rank = _dense_directed_pagerank(edge_index, 4, max_iter=200, tolerance=1e-10)
    first_features = directed_structural_node_features(edge_index, 4, pagerank_max_iter=200, pagerank_tolerance=1e-10)
    second_features = directed_structural_node_features(edge_index, 4, pagerank_max_iter=200, pagerank_tolerance=1e-10)

    assert torch.allclose(first_rank, second_rank)
    assert torch.allclose(first_rank, dense_rank, atol=1e-6)
    assert torch.allclose(first_features, second_features)
    assert torch.allclose(first_rank.sum(), torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(first_features[:, 9], first_rank * 4.0, atol=1e-6)
    assert torch.allclose(first_features[:, 9].mean(), torch.tensor(1.0), atol=1e-6)
    assert torch.isfinite(first_features).all()


def test_random_walk_transition_sparse_apply_matches_dense_orientation() -> None:
    data = _toy_graph()
    values = torch.tensor([0.25, -1.0, 0.5, 2.0])

    sparse_result = random_walk_transition_apply(data.edge_index, int(data.num_nodes), values)
    dense_result = _dense_transition(data.edge_index, int(data.num_nodes)) @ values

    assert torch.allclose(sparse_result, dense_result, atol=1e-6)


def test_hutchinson_random_walk_return_probabilities_match_dense_with_exhaustive_probes() -> None:
    data = _toy_graph()
    num_nodes = int(data.num_nodes)
    probes = _exhaustive_rademacher_probes(num_nodes)

    estimated = hutchinson_random_walk_return_probabilities(
        data.edge_index,
        num_nodes,
        steps=3,
        num_probes=probes.shape[1],
        seed=0,
        probes=probes,
    )

    transition = _dense_transition(data.edge_index, num_nodes)
    power = transition.clone()
    expected = []
    for _ in range(3):
        expected.append(torch.diag(power))
        power = power @ transition
    expected_tensor = torch.stack(expected, dim=-1)

    assert torch.allclose(estimated, expected_tensor, atol=1e-6)


def test_structural_hutchinson_features_are_deterministic_finite_and_shaped() -> None:
    data = _toy_graph()

    first = structural_node_features(
        data.edge_index,
        int(data.num_nodes),
        random_walk_steps=5,
        random_walk_num_probes=4,
        positional_encoding_seed=11,
    )
    second = structural_node_features(
        data.edge_index,
        int(data.num_nodes),
        random_walk_steps=5,
        random_walk_num_probes=4,
        positional_encoding_seed=11,
    )

    assert first.shape == (data.num_nodes, 9)
    assert torch.isfinite(first).all()
    assert torch.allclose(first, second)


def test_hutchinson_features_scale_to_large_sparse_graph_without_dense_pairs() -> None:
    num_nodes = 3_000
    sources = torch.arange(num_nodes, dtype=torch.long)
    targets = (sources + 1) % num_nodes
    edge_index = torch.cat(
        (
            torch.stack((sources, targets), dim=0),
            torch.stack((targets, sources), dim=0),
        ),
        dim=1,
    )

    features = hutchinson_random_walk_return_probabilities(
        edge_index,
        num_nodes,
        steps=4,
        num_probes=3,
        seed=123,
    )

    assert features.shape == (num_nodes, 4)
    assert torch.isfinite(features).all()


def test_graph_relabeling_leaves_objective_values_unchanged() -> None:
    data = _toy_graph()
    assignments = torch.tensor([0, 0, 1, 2])
    probabilities = _one_hot(assignments, 3)
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    relabeled_edges = inverse[data.edge_index]
    relabeled_probabilities = probabilities[permutation]
    relabeled_assignments = assignments[permutation]

    original_soft = soft_normalized_cut(probabilities, data.edge_index)
    relabeled_soft = soft_normalized_cut(relabeled_probabilities, relabeled_edges)
    original_hard = hard_normalized_cut(assignments, data.edge_index, num_partitions=3)
    relabeled_hard = hard_normalized_cut(relabeled_assignments, relabeled_edges, num_partitions=3)

    assert torch.allclose(original_soft.loss, relabeled_soft.loss, atol=1e-6)
    assert torch.allclose(original_hard.ncut_per_graph, relabeled_hard.ncut_per_graph, atol=1e-6)


def test_soft_normalized_cut_gradients_are_finite() -> None:
    data = _toy_graph()
    logits = torch.randn(data.num_nodes, 3, requires_grad=True)
    probabilities = torch.softmax(logits, dim=-1)
    result = soft_normalized_cut(probabilities, data.edge_index)

    result.loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_graph_local_separation_d2_is_zero_for_uniform_q_z() -> None:
    data = _toy_graph()
    probabilities = torch.full((data.num_nodes, 4), 0.25)

    result = soft_normalized_cut(probabilities, data.edge_index)

    assert torch.allclose(result.separation_d2_per_graph, torch.zeros(1), atol=1e-6)
    assert torch.allclose(result.effective_partitions_per_graph, torch.tensor([4.0]), atol=1e-6)


def test_graph_local_separation_d2_is_log_k_for_collapsed_q_z() -> None:
    data = _toy_graph()
    probabilities = torch.zeros(data.num_nodes, 4)
    probabilities[:, 0] = 1.0

    result = soft_normalized_cut(probabilities, data.edge_index)

    assert torch.allclose(result.separation_d2_per_graph, torch.tensor([math.log(4.0)]), atol=1e-6)
    assert torch.allclose(result.effective_partitions_per_graph, torch.tensor([1.0]), atol=1e-6)


def test_separation_d2_is_averaged_per_graph_not_from_batch_aggregate() -> None:
    first = _toy_graph()
    second = _toy_graph()
    batch = Batch.from_data_list([first, second])
    first_probabilities = torch.full((first.num_nodes, 2), 0.5)
    second_probabilities = torch.zeros(second.num_nodes, 2)
    second_probabilities[:, 0] = 1.0
    probabilities = torch.cat((first_probabilities, second_probabilities), dim=0)

    batched = soft_normalized_cut(probabilities, batch.edge_index, batch=batch.batch, separation_weight=0.3)
    separate_first = soft_normalized_cut(first_probabilities, first.edge_index)
    separate_second = soft_normalized_cut(second_probabilities, second.edge_index)
    expected_d2 = torch.stack((separate_first.separation_d2_per_graph[0], separate_second.separation_d2_per_graph[0]))
    expected_loss = batched.ncut_per_graph.mean() + 0.3 * expected_d2.mean()

    assert torch.allclose(batched.separation_d2_per_graph, expected_d2, atol=1e-6)
    assert torch.allclose(batched.separation_d2_per_graph.mean(), torch.tensor(0.5 * math.log(2.0)), atol=1e-6)
    assert torch.allclose(batched.loss, expected_loss, atol=1e-6)


def test_separation_weight_zero_preserves_soft_ncut_loss() -> None:
    data = _toy_graph()
    logits = torch.randn(data.num_nodes, 3)
    probabilities = torch.softmax(logits, dim=-1)

    result = soft_normalized_cut(probabilities, data.edge_index, separation_weight=0.0)

    assert result.loss is result.ncut_loss
    assert torch.allclose(result.loss, result.ncut_per_graph.mean(), atol=0.0, rtol=0.0)
    assert torch.allclose(result.separation_loss, torch.tensor(0.0), atol=0.0, rtol=0.0)


def test_positive_separation_weight_adds_mean_graph_local_d2() -> None:
    data = _toy_graph()
    logits = torch.randn(data.num_nodes, 3)
    probabilities = torch.softmax(logits, dim=-1)
    weight = 0.17

    result = soft_normalized_cut(probabilities, data.edge_index, separation_weight=weight)

    expected = result.ncut_per_graph.mean() + weight * result.separation_d2_per_graph.mean()
    assert torch.allclose(result.loss, expected, atol=1e-6)
    assert torch.allclose(result.separation_loss, weight * result.separation_d2_per_graph.mean(), atol=1e-6)


def test_positive_separation_weight_gradients_are_finite() -> None:
    data = _toy_graph()
    logits = torch.randn(data.num_nodes, 4, requires_grad=True)
    probabilities = torch.softmax(logits, dim=-1)

    result = soft_normalized_cut(probabilities, data.edge_index, separation_weight=0.1)
    result.loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_checkpoint_config_includes_separation_weight() -> None:
    config = SimpleNamespace(
        separation_weight=0.03,
        model_name="graphgps_efficient_attention",
        include_directed_features=True,
        pagerank_alpha=0.85,
        pagerank_max_iter=50,
        pagerank_tolerance=1e-6,
        activation_name="silu",
        output_head_hidden_multiplier=1,
        gradient_clip_norm=1.0,
        gps_heads=4,
        gps_dropout=0.0,
        gps_ffn_multiplier=4,
    )

    checkpoint_config = _checkpoint_config_dict(config)

    assert checkpoint_config["separation_weight"] == 0.03
    assert checkpoint_config["model_name"] == "graphgps_efficient_attention"
    assert checkpoint_config["include_directed_features"] is True
    assert checkpoint_config["pagerank_alpha"] == 0.85
    assert checkpoint_config["pagerank_max_iter"] == 50
    assert checkpoint_config["pagerank_tolerance"] == 1e-6
    assert checkpoint_config["activation_name"] == "silu"
    assert checkpoint_config["output_head_hidden_multiplier"] == 1
    assert checkpoint_config["gradient_clip_norm"] == 1.0
    assert checkpoint_config["gps_heads"] == 4
    assert checkpoint_config["gps_dropout"] == 0.0
    assert checkpoint_config["gps_ffn_multiplier"] == 4
    assert "gps_ffn_mode" not in checkpoint_config
    assert "gps_local_model" not in checkpoint_config
    assert "gps_merge_mode" not in checkpoint_config


def test_evaluate_model_reports_separation_diagnostics() -> None:
    data = make_synthetic_community_graph(
        num_communities=3,
        nodes_per_community=4,
        intra_edges_per_node=1,
        inter_edges_per_community_pair=1,
        random_walk_steps=2,
        random_walk_num_probes=4,
        seed=11,
    )
    loader = DataLoader([data], batch_size=1)
    model = GraphPartitionEncoder(
        input_dim=data.x.shape[-1],
        num_partitions=3,
        hidden_dim=12,
        num_layers=2,
    )
    config = SimpleNamespace(q_z_floor=1e-12, separation_weight=0.25, num_partitions=3)

    metrics = evaluate_model(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        config=config,
        max_batches=None,
    )

    assert "separation_d2_mean" in metrics
    assert "effective_partitions_mean" in metrics
    assert math.isfinite(metrics["separation_d2_mean"])
    assert math.isfinite(metrics["effective_partitions_mean"])


def test_default_config_uses_selected_gps_batch80_setup(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RELCO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RELCO_OUT_DIR", str(tmp_path / "out"))

    config = importlib.import_module("relational_compression.experiments.inductive_normalized_cut.configs.default")

    assert config.model_name == "graphgps_efficient_attention"
    assert math.isclose(float(config.separation_weight), 0.10)
    assert int(config.batch_size) == 80
    assert math.isclose(float(config.learning_rate), 5e-4)
    assert config.include_directed_features is True
    assert str(config.preprocessing_version) == "lcc_structural_directed_relative_pagerank_hutch_rwse_v5"
    assert str(config.experiment_base_name).endswith("_gps_efficient_attention_directed_sep010_batch80")
    assert not hasattr(config, "gps_ffn_mode")
    assert not hasattr(config, "gps_local_model")
    assert not hasattr(config, "gps_merge_mode")


def test_default_config_imports_with_directed_feature_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("RELCO_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RELCO_OUT_DIR", str(tmp_path / "out"))

    config = importlib.import_module("relational_compression.experiments.inductive_normalized_cut.configs.default")
    preprocess_config = PreprocessConfig(
        dataset_root_dir=Path(config.dataset_root_dir),
        min_nodes_per_graph=int(config.min_nodes_per_graph),
        max_nodes_per_graph=config.max_nodes_per_graph,
        random_walk_steps=int(config.random_walk_steps),
        random_walk_num_probes=int(config.random_walk_num_probes),
        positional_encoding_seed=int(config.positional_encoding_seed),
        preprocessing_version=str(config.preprocessing_version),
        include_directed_features=bool(config.include_directed_features),
        pagerank_alpha=float(config.pagerank_alpha),
        pagerank_max_iter=int(config.pagerank_max_iter),
        pagerank_tolerance=float(config.pagerank_tolerance),
    )

    assert config.model_name == "graphgps_efficient_attention"
    assert math.isclose(float(config.separation_weight), 0.10)
    assert config.include_directed_features is True
    assert "relative_pagerank" in str(preprocess_config.cache_dir)
    assert str(config.experiment_base_name).endswith("_gps_efficient_attention_directed_sep010_batch80")


def test_hard_evaluation_penalizes_empty_partitions() -> None:
    data = _toy_graph()
    assignments = torch.zeros(data.num_nodes, dtype=torch.long)
    result = hard_normalized_cut(assignments, data.edge_index, num_partitions=3)

    assert torch.allclose(result.nassoc_per_graph, torch.tensor([1.0]))
    assert torch.allclose(result.ncut_per_graph, torch.tensor([2.0]))
    assert torch.allclose(result.active_partitions_per_graph, torch.tensor([1.0]))


def test_model_produces_logits_and_valid_categorical_probabilities() -> None:
    data = Batch.from_data_list([_toy_graph(), _toy_graph()])
    model = GraphPartitionEncoder(
        input_dim=data.x.shape[-1],
        num_partitions=4,
        hidden_dim=16,
        num_layers=2,
        assignment_temperature=0.8,
    )

    output = model(data)

    assert output.logits.shape == (data.num_nodes, 4)
    assert output.probabilities.shape == output.logits.shape
    assert output.hard_ids.shape == (data.num_nodes,)
    assert torch.allclose(output.probabilities.sum(dim=-1), torch.ones(data.num_nodes), atol=1e-6)


def test_graphgps_efficient_attention_model_produces_batched_logits_and_finite_loss_gradients() -> None:
    data = Batch.from_data_list([_toy_graph(), _toy_graph()])
    model = GraphGPSEfficientAttentionPartitionEncoder(
        input_dim=data.x.shape[-1],
        num_partitions=4,
        hidden_dim=16,
        num_layers=2,
        assignment_temperature=1.0,
        gps_heads=2,
    )

    output = model(data)
    loss = soft_normalized_cut(
        output.probabilities,
        data.edge_index,
        batch=data.batch,
        num_graphs=data.num_graphs,
        separation_weight=0.05,
    ).loss
    loss.backward()

    assert output.logits.shape == (data.num_nodes, 4)
    assert output.probabilities.shape == output.logits.shape
    assert output.hard_ids.shape == (data.num_nodes,)
    assert torch.allclose(output.probabilities.sum(dim=-1), torch.ones(data.num_nodes), atol=1e-6)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_graphgps_efficient_attention_model_is_equivariant_to_graph_relabeling_in_eval_mode() -> None:
    data = _toy_graph()
    permutation = torch.tensor([2, 0, 3, 1])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    relabeled = Data(
        x=data.x[permutation],
        edge_index=inverse[data.edge_index],
        num_nodes=data.num_nodes,
    )
    torch.manual_seed(2)
    model = GraphGPSEfficientAttentionPartitionEncoder(
        input_dim=data.x.shape[-1],
        num_partitions=3,
        hidden_dim=16,
        num_layers=1,
        gps_heads=2,
        use_graph_context=False,
    )
    model.eval()

    with torch.no_grad():
        original = model(data).logits
        permuted = model(relabeled).logits

    assert torch.allclose(original[permutation], permuted, atol=1e-5)


def test_graphgps_efficient_attention_is_invariant_to_padding_from_other_graphs() -> None:
    small = _toy_graph()
    large = make_synthetic_community_graph(
        num_communities=2,
        nodes_per_community=9,
        intra_edges_per_node=2,
        inter_edges_per_community_pair=1,
        random_walk_steps=2,
        seed=41,
    )
    torch.manual_seed(7)
    model = GraphGPSEfficientAttentionPartitionEncoder(
        input_dim=small.x.shape[-1],
        num_partitions=3,
        hidden_dim=16,
        num_layers=2,
        gps_heads=2,
        use_graph_context=True,
    )
    model.eval()

    with torch.no_grad():
        alone = model(Batch.from_data_list([small])).logits
        co_batched = model(Batch.from_data_list([small, large])).logits[: small.num_nodes]

    assert torch.allclose(alone, co_batched, atol=1e-5, rtol=1e-5)


def test_make_model_selects_graphgps_efficient_attention() -> None:
    data = _toy_graph()
    config = SimpleNamespace(
        model_name="graphgps_efficient_attention",
        num_partitions=3,
        hidden_dim=16,
        num_layers=1,
        assignment_temperature=1.0,
        use_graph_context=True,
        activation_name="silu",
        output_head_hidden_multiplier=1,
        gps_heads=2,
        gps_dropout=0.0,
        gps_ffn_multiplier=3,
    )

    model = make_model(config, input_dim=data.x.shape[-1])
    output = model(Batch.from_data_list([data, data]))

    assert isinstance(model, GraphGPSEfficientAttentionPartitionEncoder)
    assert isinstance(model.activation, torch.nn.SiLU)
    assert isinstance(model.layers[0].local_conv, GINConv)
    assert isinstance(model.layers[0].ffn[1], torch.nn.SiLU)
    assert isinstance(model.output, torch.nn.Sequential)
    assert model.layers[0].ffn[0].out_features == 3 * 16
    assert output.logits.shape == (2 * data.num_nodes, 3)
    assert torch.allclose(output.probabilities.sum(dim=-1), torch.ones(2 * data.num_nodes), atol=1e-6)


def test_graph_labels_are_not_consumed_by_objective_or_model() -> None:
    data = _toy_graph()
    first = Batch.from_data_list(
        [Data(x=data.x, edge_index=data.edge_index, num_nodes=data.num_nodes, y=torch.tensor([0]))]
    )
    second = Batch.from_data_list(
        [Data(x=data.x, edge_index=data.edge_index, num_nodes=data.num_nodes, y=torch.tensor([99]))]
    )
    torch.manual_seed(9)
    model = GraphPartitionEncoder(input_dim=data.x.shape[-1], num_partitions=3, hidden_dim=12, num_layers=2)

    first_output = model(first)
    second_output = model(second)
    first_loss = soft_normalized_cut(first_output.probabilities, first.edge_index, batch=first.batch).loss
    second_loss = soft_normalized_cut(second_output.probabilities, second.edge_index, batch=second.batch).loss

    assert torch.allclose(first_output.logits, second_output.logits, atol=1e-7)
    assert torch.allclose(first_loss, second_loss, atol=1e-7)


def test_graphgps_efficient_attention_model_is_compatible_with_evaluation_pipeline() -> None:
    data = make_synthetic_community_graph(
        num_communities=3,
        nodes_per_community=5,
        intra_edges_per_node=2,
        inter_edges_per_community_pair=1,
        random_walk_steps=2,
        random_walk_num_probes=4,
        seed=12,
    )
    loader = DataLoader([data], batch_size=1)
    model = GraphGPSEfficientAttentionPartitionEncoder(
        input_dim=data.x.shape[-1],
        num_partitions=3,
        hidden_dim=16,
        num_layers=1,
        gps_heads=2,
    )
    config = SimpleNamespace(q_z_floor=1e-12, separation_weight=0.05, num_partitions=3)

    metrics = evaluate_model(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
        config=config,
        max_batches=None,
    )

    assert math.isfinite(metrics["hard_ncut_mean"])
    assert math.isfinite(metrics["separation_d2_mean"])


def test_small_synthetic_community_graph_optimizes_without_nans() -> None:
    data = Batch.from_data_list(
        [
            make_synthetic_community_graph(
                num_communities=2,
                nodes_per_community=8,
                intra_edges_per_node=2,
                inter_edges_per_community_pair=1,
                random_walk_steps=2,
                seed=3,
            )
        ]
    )
    model = GraphPartitionEncoder(input_dim=data.x.shape[-1], num_partitions=2, hidden_dim=16, num_layers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    losses = []
    for _ in range(8):
        output = model(data)
        loss = soft_normalized_cut(output.probabilities, data.edge_index, batch=data.batch).loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    assert all(math.isfinite(value) for value in losses)


def test_spectral_baseline_returns_valid_partition() -> None:
    data = make_synthetic_community_graph(
        num_communities=3,
        nodes_per_community=6,
        intra_edges_per_node=2,
        inter_edges_per_community_pair=1,
        random_walk_steps=2,
        seed=8,
    )
    labels = spectral_normalized_cut_partition(
        data,
        SpectralNormalizedCutConfig(num_partitions=3, seed=1, n_init=2, max_iter=10),
    )

    assert labels.shape == (data.num_nodes,)
    assert labels.min() >= 0
    assert labels.max() < 3


def test_spectral_clustering_uses_sklearn_kmeans_settings(monkeypatch) -> None:
    calls: dict[str, Any] = {}

    class FakeKMeans:
        def __init__(self, **kwargs: object) -> None:
            calls.update(kwargs)

        def fit_predict(self, embedding: np.ndarray) -> np.ndarray:
            calls["embedding_shape"] = embedding.shape
            return np.arange(embedding.shape[0]) % calls["n_clusters"]

    monkeypatch.setattr(ncut_baselines, "KMeans", FakeKMeans)
    config = SpectralNormalizedCutConfig(
        num_partitions=3,
        seed=4,
        n_init=5,
        max_iter=6,
        tolerance=1e-4,
    )

    labels = ncut_baselines.cluster_spectral_embedding(np.ones((7, 3), dtype=np.float64), config)

    assert calls["n_clusters"] == 3
    assert calls["init"] == "k-means++"
    assert calls["n_init"] == 5
    assert calls["max_iter"] == 6
    assert calls["tol"] == 1e-4
    assert calls["random_state"] == 4
    assert calls["embedding_shape"] == (7, 3)
    assert labels.shape == (7,)


def test_spectral_baseline_finds_low_ncut_on_clear_community_graph() -> None:
    data = make_synthetic_community_graph(
        num_communities=3,
        nodes_per_community=10,
        intra_edges_per_node=4,
        inter_edges_per_community_pair=1,
        random_walk_steps=2,
        seed=10,
    )
    labels = spectral_normalized_cut_partition(
        data,
        SpectralNormalizedCutConfig(num_partitions=3, seed=2, n_init=4, max_iter=50),
    )

    result = hard_normalized_cut(labels, data.edge_index, num_partitions=3)

    assert float(result.ncut_per_graph[0]) < 0.5


def test_spectral_cache_key_includes_solver_settings() -> None:
    data = _toy_graph()
    data.original_split = "val"
    data.original_graph_index = 5
    base = SpectralNormalizedCutConfig(
        num_partitions=3,
        seed=1,
        n_init=2,
        max_iter=10,
        tolerance=1e-5,
        cache_version="unit",
    )

    keys = {
        ncut_baselines._graph_cache_key(data, base),
        ncut_baselines._graph_cache_key(data, SpectralNormalizedCutConfig(**{**base.__dict__, "seed": 2})),
        ncut_baselines._graph_cache_key(data, SpectralNormalizedCutConfig(**{**base.__dict__, "n_init": 3})),
        ncut_baselines._graph_cache_key(data, SpectralNormalizedCutConfig(**{**base.__dict__, "max_iter": 11})),
        ncut_baselines._graph_cache_key(data, SpectralNormalizedCutConfig(**{**base.__dict__, "tolerance": 1e-4})),
        ncut_baselines._graph_cache_key(
            data, SpectralNormalizedCutConfig(**{**base.__dict__, "cache_version": "unit_v2"})
        ),
    }

    assert len(keys) == 6


def test_small_experiment_writes_standard_outputs(tmp_path: Path) -> None:
    config_file = tmp_path / "config.py"
    config_file.write_text("# inductive normalized cut test config\n")
    config = SimpleNamespace(
        config_file=config_file,
        task_model_name="inductive_normalized_cut",
        dataset_name="synthetic",
        dataset_version="1.0",
        num_experiments=1,
        dataset_root_dir=tmp_path / "data",
        preprocessing_version="test",
        min_nodes_per_graph=1,
        max_nodes_per_graph=None,
        random_walk_steps=2,
        random_walk_num_probes=4,
        positional_encoding_seed=21,
        dataset_backend="synthetic",
        experiments_dir=tmp_path / "experiments",
        experiment_base_name="synthetic_inductive_ncut",
        experiment_name="synthetic_inductive_ncut_test",
        unique_postfix=None,
        num_epochs=2,
        batch_size=2,
        num_workers=0,
        learning_rate=0.01,
        min_learning_rate=0.001,
        learning_rate_warmup_steps=0,
        learning_rate_warmup_start_factor=0.1,
        weight_decay=0.0,
        num_partitions=3,
        hidden_dim=16,
        num_layers=2,
        assignment_temperature=1.0,
        use_graph_context=True,
        q_z_floor=1e-12,
        separation_weight=0.0,
        run_spectral_baseline=False,
        spectral_seed=1,
        spectral_n_init=2,
        spectral_max_iter=10,
        spectral_tolerance=1e-5,
        spectral_cache_version="test",
        seed=21,
        log_to_tensorboard_global=False,
        tensorboard_log_steps=1,
        max_train_batches=1,
        max_val_batches=1,
        device="cpu",
    )

    result = run_experiment(config)[0]
    experiment_dir = config.experiments_dir / config.experiment_name
    run_dir = experiment_dir / "run_00_categorical"

    assert (experiment_dir / "effective_config.json").exists()
    assert (experiment_dir / "all_run_results.json").exists()
    assert (run_dir / "result.json").exists()
    assert (run_dir / "history.csv").exists()
    assert (experiment_dir / "checkpoints" / "run_00_categorical" / "latest.pt").exists()
    assert (experiment_dir / "checkpoints" / "run_00_categorical" / "best_model_checkpoint.pt").exists()
    assert result["checkpoint_metric"] == "hard_ncut_mean"
    assert "best_validation" in result
    assert "test" in result
    loaded = json.loads((run_dir / "result.json").read_text())
    assert loaded["run_index"] == 0
    assert loaded["seed"] == 21
