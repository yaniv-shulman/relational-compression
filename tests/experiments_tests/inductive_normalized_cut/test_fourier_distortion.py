import numpy as np
import torch
from torch_geometric.data import Data

from relational_compression.experiments.inductive_normalized_cut.run_scripts.compute_fourier_distortion import (
    _laplacian_from_edges,
    _undirected_weighted_edges,
    compute_edge_effective_resistances,
    fourier_dirichlet_distortion,
)


def _directed_edge_index(edges: list[tuple[int, int]]) -> torch.Tensor:
    directed = []
    for source, target in edges:
        directed.append((source, target))
        directed.append((target, source))
    return torch.tensor(directed, dtype=torch.long).t().contiguous()


def test_fourier_distortion_is_zero_for_all_one_cluster() -> None:
    data = Data(edge_index=_directed_edge_index([(0, 1), (1, 2)]), num_nodes=3)
    edge_resistance = compute_edge_effective_resistances(data)

    value = fourier_dirichlet_distortion(torch.zeros(3, dtype=torch.long), edge_resistance, num_nodes=3)

    assert value == 0.0


def test_fourier_distortion_is_one_when_all_tree_edges_are_cut() -> None:
    data = Data(edge_index=_directed_edge_index([(0, 1), (1, 2), (2, 3)]), num_nodes=4)
    edge_resistance = compute_edge_effective_resistances(data)

    value = fourier_dirichlet_distortion(torch.arange(4), edge_resistance, num_nodes=4)

    assert np.isclose(edge_resistance.foster_sum, 3.0)
    assert np.isclose(value, 1.0)


def test_cut_edge_effective_resistance_matches_trace_form() -> None:
    data = Data(edge_index=_directed_edge_index([(0, 1), (1, 2), (2, 0), (2, 3)]), num_nodes=4)
    assignments = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    edge_resistance = compute_edge_effective_resistances(data)

    edge_form = fourier_dirichlet_distortion(assignments, edge_resistance, num_nodes=4)

    edges, weights = _undirected_weighted_edges(data.edge_index, int(data.num_nodes))
    laplacian = _laplacian_from_edges(int(data.num_nodes), edges, weights).toarray()
    labels = assignments.numpy()
    retained = labels[edges[:, 0]] == labels[edges[:, 1]]
    retained_laplacian = _laplacian_from_edges(int(data.num_nodes), edges[retained], weights[retained]).toarray()
    trace_form = float(np.trace((laplacian - retained_laplacian) @ np.linalg.pinv(laplacian)) / 3.0)

    assert np.isclose(edge_resistance.foster_sum, 3.0)
    assert np.isclose(edge_form, trace_form)
