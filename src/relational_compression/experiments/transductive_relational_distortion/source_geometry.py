"""Source-derived graph-fidelity geometry and reusable cached summaries."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data

from relational_compression.experiments.graph_geometry import (
    EdgeResistanceData,
    _undirected_weighted_edges,
    compute_edge_effective_resistances,
)

SOURCE_GEOMETRY_CACHE_VERSION = "trd_source_geometry_v3"
EDGE = "edge"
FOURIER = "fourier"
COLLISION = "collision"
COLLISION_ENTROPY = "collision_entropy"
SOURCE_CRITERIA = (EDGE, FOURIER, COLLISION, COLLISION_ENTROPY)


@dataclass(frozen=True)
class SourceGeometry:
    """Store source graph quantities used by the transductive fidelity criteria."""

    num_nodes: int
    edge_index: Tensor
    edge_weight: Tensor
    degree: Tensor
    rho_edge: Tensor
    rho_fourier: Tensor
    rho_collision: Tensor
    rho_collision_entropy: Tensor | None
    effective_resistance: Tensor
    foster_sum: float
    collision_trace: float
    collision_entropy_denominator: float
    collision_entropy: Tensor
    metadata: dict[str, Any]

    def to(self, device: torch.device | str) -> "SourceGeometry":
        """Move tensor-valued source geometry to a target device."""
        return SourceGeometry(
            num_nodes=self.num_nodes,
            edge_index=self.edge_index.to(device=device),
            edge_weight=self.edge_weight.to(device=device),
            degree=self.degree.to(device=device),
            rho_edge=self.rho_edge.to(device=device),
            rho_fourier=self.rho_fourier.to(device=device),
            rho_collision=self.rho_collision.to(device=device),
            rho_collision_entropy=None
            if self.rho_collision_entropy is None
            else self.rho_collision_entropy.to(device=device),
            effective_resistance=self.effective_resistance.to(device=device),
            foster_sum=self.foster_sum,
            collision_trace=self.collision_trace,
            collision_entropy_denominator=self.collision_entropy_denominator,
            collision_entropy=self.collision_entropy.to(device=device),
            metadata=self.metadata,
        )

    @property
    def defined_criteria(self) -> tuple[str, ...]:
        """Return source-fidelity criteria available for this graph."""
        criteria = [EDGE, FOURIER, COLLISION]
        if self.rho_collision_entropy is not None:
            criteria.append(COLLISION_ENTROPY)
        return tuple(criteria)

    def rho_for(self, criterion: str) -> Tensor:
        """Return edge-importance weights for one declared criterion."""
        if criterion == EDGE:
            return self.rho_edge
        if criterion == FOURIER:
            return self.rho_fourier
        if criterion == COLLISION:
            return self.rho_collision
        if criterion == COLLISION_ENTROPY and self.rho_collision_entropy is not None:
            return self.rho_collision_entropy
        raise ValueError(f"Unsupported source distortion criterion: {criterion}")


def _edge_degree(num_nodes: int, edges: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Compute edge degree."""
    degree = np.zeros(num_nodes, dtype=np.float64)
    np.add.at(degree, edges[:, 0], weights)
    np.add.at(degree, edges[:, 1], weights)
    return degree


def _adjacency_maps(num_nodes: int, edges: np.ndarray, weights: np.ndarray) -> list[dict[int, float]]:
    """Compute adjacency maps."""
    adjacency: list[dict[int, float]] = [dict() for _ in range(num_nodes)]
    for (source, target), weight in zip(edges.tolist(), weights.tolist(), strict=True):
        adjacency[source][target] = adjacency[source].get(target, 0.0) + float(weight)
        adjacency[target][source] = adjacency[target].get(source, 0.0) + float(weight)
    return adjacency


def _row_self_collision(row: dict[int, float], degree: float) -> float:
    """Compute row self collision."""
    if degree <= 0.0:
        return 0.0
    inv_degree = 1.0 / degree
    return float(sum((weight * inv_degree) ** 2 for weight in row.values()))


def _row_collision_dot(
    first: dict[int, float],
    second: dict[int, float],
    first_degree: float,
    second_degree: float,
) -> float:
    """Compute row collision dot."""
    if first_degree <= 0.0 or second_degree <= 0.0:
        return 0.0
    if len(first) > len(second):
        first, second = second, first
        first_degree, second_degree = second_degree, first_degree
    scale = 1.0 / (first_degree * second_degree)
    return float(sum(weight * second.get(node, 0.0) * scale for node, weight in first.items()))


def source_collision_edge_importance(
    *,
    num_nodes: int,
    edges: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Compute w_ij ||P_i - P_j||_2^2 for canonical undirected edges."""
    degree = _edge_degree(num_nodes=num_nodes, edges=edges, weights=weights)
    if np.any(degree <= 0.0):
        raise ValueError("source collision geometry requires positive degree for every node")
    adjacency = _adjacency_maps(num_nodes=num_nodes, edges=edges, weights=weights)
    self_collision = np.asarray(
        [_row_self_collision(row=adjacency[node], degree=degree[node]) for node in range(num_nodes)],
        dtype=np.float64,
    )
    importance = np.empty(edges.shape[0], dtype=np.float64)
    for edge_index, (source, target) in enumerate(edges.tolist()):
        cross_collision = _row_collision_dot(
            first=adjacency[source], second=adjacency[target], first_degree=degree[source], second_degree=degree[target]
        )
        delta = max(0.0, self_collision[source] + self_collision[target] - 2.0 * cross_collision)
        importance[edge_index] = weights[edge_index] * delta
    return importance


def source_collision_entropy_edge_importance(
    *,
    num_nodes: int,
    edges: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute w_ij (H_2(P_i) - H_2(P_j))^2 for canonical undirected edges."""
    degree = _edge_degree(num_nodes=num_nodes, edges=edges, weights=weights)
    if np.any(degree <= 0.0):
        raise ValueError("source collision-entropy geometry requires positive degree for every node")
    adjacency = _adjacency_maps(num_nodes=num_nodes, edges=edges, weights=weights)
    local_collision = np.asarray(
        [_row_self_collision(row=adjacency[node], degree=degree[node]) for node in range(num_nodes)],
        dtype=np.float64,
    )
    local_entropy = -np.log(np.maximum(local_collision, np.finfo(np.float64).tiny))
    differences = local_entropy[edges[:, 0]] - local_entropy[edges[:, 1]]
    importance = weights * differences * differences
    return (
        importance.astype(np.float64, copy=False),
        local_entropy.astype(np.float64, copy=False),
        float(importance.sum()),
    )


def _torch_geometry(
    *,
    data: Data,
    edge_resistance: EdgeResistanceData,
    edge_importance: np.ndarray,
    fourier_importance: np.ndarray,
    collision_importance: np.ndarray,
    collision_trace: float,
    collision_entropy_importance: np.ndarray,
    collision_entropy: np.ndarray,
    collision_entropy_denominator: float,
) -> SourceGeometry:
    """Compute torch geometry."""
    num_nodes = int(data.num_nodes)
    edges = edge_resistance.edges
    weights = edge_resistance.weights
    degree = _edge_degree(num_nodes=num_nodes, edges=edges, weights=weights)
    edge_denominator = float(edge_importance.sum())
    fourier_denominator = float(fourier_importance.sum())
    if edge_denominator <= 0.0:
        raise ValueError("D_E source geometry has zero normalization denominator")
    if fourier_denominator <= 0.0:
        raise ValueError("D_F source geometry has zero normalization denominator")
    if collision_trace <= 0.0:
        raise ValueError("D_C source geometry has zero normalization denominator")
    collision_entropy_defined = bool(collision_entropy_denominator > 0.0)

    metadata = {
        "source_geometry_cache_version": SOURCE_GEOMETRY_CACHE_VERSION,
        "collection": str(getattr(data, "collection", "unknown")),
        "source_variant": str(getattr(data, "source_variant", "unknown")),
        "original_split": str(getattr(data, "original_split", "unknown")),
        "dataset_index": int(getattr(data, "dataset_index", -1)),
        "original_graph_index": int(getattr(data, "original_graph_index", -1)),
        "weight_seed": int(getattr(data, "weight_seed", -1)),
        "weight_sigma": float(getattr(data, "weight_sigma", float("nan"))),
        "num_nodes": num_nodes,
        "num_undirected_edges": int(edges.shape[0]),
        "foster_sum": float(edge_resistance.foster_sum),
        "foster_expected": float(max(0, num_nodes - 1)),
        "edge_denominator": edge_denominator,
        "fourier_denominator": fourier_denominator,
        "collision_trace": float(collision_trace),
        "collision_entropy_denominator": float(collision_entropy_denominator),
        "collision_entropy_defined": collision_entropy_defined,
    }
    return SourceGeometry(
        num_nodes=num_nodes,
        edge_index=torch.as_tensor(edges.T, dtype=torch.long).contiguous(),
        edge_weight=torch.as_tensor(weights, dtype=torch.float32),
        degree=torch.as_tensor(degree, dtype=torch.float32),
        rho_edge=torch.as_tensor(edge_importance / edge_denominator, dtype=torch.float32),
        rho_fourier=torch.as_tensor(fourier_importance / fourier_denominator, dtype=torch.float32),
        rho_collision=torch.as_tensor(collision_importance / float(collision_trace), dtype=torch.float32),
        rho_collision_entropy=None
        if not collision_entropy_defined
        else torch.as_tensor(collision_entropy_importance / float(collision_entropy_denominator), dtype=torch.float32),
        effective_resistance=torch.as_tensor(edge_resistance.resistances, dtype=torch.float64),
        foster_sum=float(edge_resistance.foster_sum),
        collision_trace=float(collision_trace),
        collision_entropy_denominator=float(collision_entropy_denominator),
        collision_entropy=torch.as_tensor(collision_entropy, dtype=torch.float64),
        metadata=metadata,
    )


def compute_source_geometry(data: Data, *, solve_batch_size: int = 256) -> SourceGeometry:
    """Compute source geometry."""
    edge_resistance = compute_edge_effective_resistances(data, solve_batch_size=solve_batch_size)
    edge_importance = edge_resistance.weights.copy()
    collision_importance = source_collision_edge_importance(
        num_nodes=int(data.num_nodes),
        edges=edge_resistance.edges,
        weights=edge_resistance.weights,
    )
    collision_entropy_importance, collision_entropy, collision_entropy_denominator = (
        source_collision_entropy_edge_importance(
            num_nodes=int(data.num_nodes),
            edges=edge_resistance.edges,
            weights=edge_resistance.weights,
        )
    )
    fourier_importance = edge_resistance.weights * edge_resistance.resistances
    collision_trace = float(collision_importance.sum())
    return _torch_geometry(
        data=data,
        edge_resistance=edge_resistance,
        edge_importance=edge_importance,
        fourier_importance=fourier_importance,
        collision_importance=collision_importance,
        collision_trace=collision_trace,
        collision_entropy_importance=collision_entropy_importance,
        collision_entropy=collision_entropy,
        collision_entropy_denominator=collision_entropy_denominator,
    )


def _topology_digest(edges: np.ndarray, weights: np.ndarray) -> str:
    """Compute topology digest."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(edges, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(weights, dtype=np.float64).tobytes())
    return digest.hexdigest()[:16]


def source_geometry_cache_path(data: Data, *, cache_dir: Path, cache_version: str) -> Path:
    """Compute source geometry cache path."""
    num_nodes = int(data.num_nodes)
    edge_weight = getattr(data, "edge_weight", None)
    edges, weights = _undirected_weighted_edges(
        edge_index=data.edge_index, num_nodes=num_nodes, edge_weight=edge_weight
    )
    digest = _topology_digest(edges=edges, weights=weights)
    collection = str(getattr(data, "collection", "unknown"))
    source_variant = str(getattr(data, "source_variant", "unknown"))
    split = str(getattr(data, "original_split", "unknown"))
    graph_index = int(getattr(data, "dataset_index", getattr(data, "original_graph_index", -1)))
    filename = f"{split}_{graph_index:06d}_n{num_nodes}_e{edges.shape[0]}_{digest}.npz"
    return Path(cache_dir) / cache_version / collection / source_variant / split / filename


def save_source_geometry(path: Path, geometry: SourceGeometry) -> None:
    """Save source geometry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        edges=geometry.edge_index.detach().cpu().numpy().T.astype(np.int64, copy=False),
        weights=geometry.edge_weight.detach().cpu().numpy().astype(np.float64, copy=False),
        degree=geometry.degree.detach().cpu().numpy().astype(np.float64, copy=False),
        rho_edge=geometry.rho_edge.detach().cpu().numpy().astype(np.float64, copy=False),
        rho_fourier=geometry.rho_fourier.detach().cpu().numpy().astype(np.float64, copy=False),
        rho_collision=geometry.rho_collision.detach().cpu().numpy().astype(np.float64, copy=False),
        rho_collision_entropy=np.asarray([], dtype=np.float64)
        if geometry.rho_collision_entropy is None
        else geometry.rho_collision_entropy.detach().cpu().numpy().astype(np.float64, copy=False),
        resistances=geometry.effective_resistance.detach().cpu().numpy().astype(np.float64, copy=False),
        collision_entropy=geometry.collision_entropy.detach().cpu().numpy().astype(np.float64, copy=False),
        foster_sum=np.asarray(geometry.foster_sum, dtype=np.float64),
        collision_trace=np.asarray(geometry.collision_trace, dtype=np.float64),
        collision_entropy_denominator=np.asarray(geometry.collision_entropy_denominator, dtype=np.float64),
        metadata_json=np.asarray(json.dumps(geometry.metadata)),
    )


def load_source_geometry(path: Path) -> SourceGeometry:
    """Load source geometry without unpickling cache content."""
    with np.load(path, allow_pickle=False) as loaded:
        edges = loaded["edges"].astype(np.int64, copy=False)
        metadata = json.loads(str(loaded["metadata_json"].item()))
        rho_collision_entropy = loaded["rho_collision_entropy"].astype(np.float64, copy=False)
        return SourceGeometry(
            num_nodes=int(metadata["num_nodes"]),
            edge_index=torch.as_tensor(edges.T, dtype=torch.long).contiguous(),
            edge_weight=torch.as_tensor(loaded["weights"], dtype=torch.float32),
            degree=torch.as_tensor(loaded["degree"], dtype=torch.float32),
            rho_edge=torch.as_tensor(loaded["rho_edge"], dtype=torch.float32),
            rho_fourier=torch.as_tensor(loaded["rho_fourier"], dtype=torch.float32),
            rho_collision=torch.as_tensor(loaded["rho_collision"], dtype=torch.float32),
            rho_collision_entropy=None
            if rho_collision_entropy.size == 0
            else torch.as_tensor(rho_collision_entropy, dtype=torch.float32),
            effective_resistance=torch.as_tensor(loaded["resistances"], dtype=torch.float64),
            foster_sum=float(loaded["foster_sum"]),
            collision_trace=float(loaded["collision_trace"]),
            collision_entropy_denominator=float(loaded["collision_entropy_denominator"]),
            collision_entropy=torch.as_tensor(loaded["collision_entropy"], dtype=torch.float64),
            metadata=metadata,
        )


def load_or_compute_source_geometry(
    data: Data,
    *,
    cache_dir: Path,
    cache_version: str = SOURCE_GEOMETRY_CACHE_VERSION,
    solve_batch_size: int = 256,
) -> SourceGeometry:
    """Load or compute source geometry."""
    cache_path = source_geometry_cache_path(data, cache_dir=cache_dir, cache_version=cache_version)
    if cache_path.exists():
        return load_source_geometry(cache_path)
    geometry = compute_source_geometry(data, solve_batch_size=solve_batch_size)
    save_source_geometry(path=cache_path, geometry=geometry)
    return geometry
