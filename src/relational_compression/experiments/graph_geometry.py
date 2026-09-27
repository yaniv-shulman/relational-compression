"""Effective-resistance utilities for graph-based fidelity calculations."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from torch import Tensor
from torch_geometric.data import Data


@dataclass(frozen=True)
class EdgeResistanceData:
    """Store undirected edges, weights, effective resistances, and their total mass."""

    edges: np.ndarray
    weights: np.ndarray
    resistances: np.ndarray
    foster_sum: float


def _undirected_weighted_edges(
    edge_index: Tensor,
    num_nodes: int,
    edge_weight: Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Coalesce directed edge storage into weighted undirected edges."""
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    source = edge_index[0].detach().cpu().numpy().astype(np.int64, copy=False)
    target = edge_index[1].detach().cpu().numpy().astype(np.int64, copy=False)
    if edge_weight is None:
        weights = np.ones(source.shape[0], dtype=np.float64)
    else:
        weights = edge_weight.detach().cpu().numpy().astype(np.float64, copy=False)
    keep = source != target
    source = source[keep]
    target = target[keep]
    weights = weights[keep]
    row = np.minimum(source, target)
    col = np.maximum(source, target)
    edge_sum = sparse.coo_matrix((weights, (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    edge_count = sparse.coo_matrix((np.ones_like(weights), (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    edge_sum.sum_duplicates()
    edge_count.sum_duplicates()
    edge_sum.eliminate_zeros()
    edge_count.eliminate_zeros()
    coo_sum = edge_sum.tocoo()
    coo_count = edge_count.tocoo()
    if not (np.array_equal(coo_sum.row, coo_count.row) and np.array_equal(coo_sum.col, coo_count.col)):
        raise RuntimeError("edge weight and edge count sparsity patterns diverged")
    edges = np.stack((coo_sum.row, coo_sum.col), axis=1).astype(np.int64, copy=False)
    undirected_weights = (coo_sum.data / coo_count.data).astype(np.float64, copy=False)
    return edges, undirected_weights


def _laplacian_from_edges(num_nodes: int, edges: np.ndarray, weights: np.ndarray) -> sparse.csr_matrix:
    """Construct a weighted graph Laplacian from undirected edge records."""
    row = np.concatenate((edges[:, 0], edges[:, 1], edges[:, 0], edges[:, 1]))
    col = np.concatenate((edges[:, 1], edges[:, 0], edges[:, 0], edges[:, 1]))
    data = np.concatenate((-weights, -weights, weights, weights))
    laplacian = sparse.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    laplacian.sum_duplicates()
    return laplacian


def compute_edge_effective_resistances(
    data: Data,
    *,
    solve_batch_size: int = 256,
) -> EdgeResistanceData:
    """Compute exact edge effective resistances by grounding one Laplacian vertex.

    Args:
        data: Graph containing its edge index and optional edge weights.
        solve_batch_size: Number of edge right-hand sides solved at once.

    Returns:
        Coalesced edges with their weights, resistances, and Foster sum.

    """
    num_nodes = int(data.num_nodes)
    if num_nodes <= 1:
        raise ValueError("effective resistance requires at least two nodes")
    if solve_batch_size <= 0:
        raise ValueError("solve_batch_size must be positive")
    edge_weight = getattr(data, "edge_weight", None)
    edges, weights = _undirected_weighted_edges(
        edge_index=data.edge_index, num_nodes=num_nodes, edge_weight=edge_weight
    )
    if edges.size == 0:
        raise ValueError("effective resistance requires at least one edge")

    laplacian = _laplacian_from_edges(num_nodes=num_nodes, edges=edges, weights=weights)
    grounded = num_nodes - 1
    reduced = laplacian[:grounded, :grounded].tocsc()
    factor = splu(reduced)

    resistances = np.empty(edges.shape[0], dtype=np.float64)
    for start in range(0, edges.shape[0], int(solve_batch_size)):
        end = min(start + int(solve_batch_size), edges.shape[0])
        batch_edges = edges[start:end]
        rhs = np.zeros((grounded, end - start), dtype=np.float64)
        columns = np.arange(end - start)
        source = batch_edges[:, 0]
        target = batch_edges[:, 1]
        mask_source = source != grounded
        mask_target = target != grounded
        rhs[source[mask_source], columns[mask_source]] += 1.0
        rhs[target[mask_target], columns[mask_target]] -= 1.0
        solved = factor.solve(rhs)
        resistances[start:end] = np.sum(rhs * solved, axis=0)

    resistances = np.maximum(resistances, 0.0)
    foster_sum = float(np.dot(weights, resistances))
    return EdgeResistanceData(edges=edges, weights=weights, resistances=resistances, foster_sum=foster_sum)


def fourier_dirichlet_distortion(assignments: Tensor, edge_resistance: EdgeResistanceData, *, num_nodes: int) -> float:
    """Compute normalized all-mode Dirichlet distortion for hard assignments."""
    if assignments.ndim != 1:
        raise ValueError("assignments must have shape [num_nodes]")
    if int(assignments.shape[0]) != int(num_nodes):
        raise ValueError("assignments length must match num_nodes")
    if num_nodes <= 1:
        return 0.0
    labels = assignments.detach().cpu().numpy()
    cut = labels[edge_resistance.edges[:, 0]] != labels[edge_resistance.edges[:, 1]]
    removed_trace = float(np.dot(edge_resistance.weights[cut], edge_resistance.resistances[cut]))
    return removed_trace / float(num_nodes - 1)
