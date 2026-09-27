"""Spectral normalized-cut reference construction and result caching."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from torch import Tensor
from torch_geometric.data import Data

from relational_compression.experiments.inductive_normalized_cut.metrics import hard_normalized_cut


@dataclass(frozen=True)
class SpectralNormalizedCutConfig:
    """Configure the per-graph spectral normalized-cut reference."""

    num_partitions: int = 8
    seed: int = 1337
    n_init: int = 8
    max_iter: int = 100
    tolerance: float = 1e-5
    cache_version: str = "spectral_ncut_v1"


def _adjacency_from_edge_index(edge_index: Tensor, num_nodes: int) -> sparse.csr_matrix:
    """Build an unweighted sparse adjacency matrix from edge incidences."""
    row = edge_index[0].detach().cpu().numpy()
    col = edge_index[1].detach().cpu().numpy()
    values = np.ones(edge_index.shape[1], dtype=np.float64)
    adjacency = sparse.coo_matrix((values, (row, col)), shape=(num_nodes, num_nodes)).tocsr()
    adjacency.sum_duplicates()
    adjacency.data[:] = 1.0
    adjacency.setdiag(0.0)
    adjacency.eliminate_zeros()
    return adjacency


def _normalized_laplacian(adjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    """Construct the symmetric normalized Laplacian of an adjacency matrix."""
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    inv_sqrt = np.divide(1.0, np.sqrt(degree), out=np.zeros_like(degree), where=degree > 0.0)
    normalized_adjacency = sparse.diags(inv_sqrt).dot(adjacency).dot(sparse.diags(inv_sqrt))
    return sparse.eye(adjacency.shape[0], dtype=np.float64, format="csr") - normalized_adjacency.tocsr()


def _row_normalize(values: np.ndarray) -> np.ndarray:
    """Normalize nonzero rows of an embedding to unit Euclidean length."""
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return np.divide(values, norms, out=np.zeros_like(values), where=norms > 0.0)


def cluster_spectral_embedding(embedding: np.ndarray, config: SpectralNormalizedCutConfig) -> np.ndarray:
    """Cluster a spectral embedding with the configured seeded K-means run."""
    return cast(
        np.ndarray,
        KMeans(
            n_clusters=int(config.num_partitions),
            init="k-means++",
            n_init=int(config.n_init),
            max_iter=int(config.max_iter),
            tol=float(config.tolerance),
            random_state=int(config.seed),
        ).fit_predict(embedding),
    )


def spectral_normalized_cut_partition(data: Data, config: SpectralNormalizedCutConfig) -> Tensor:
    """Return a hard partition from the spectral normalized-cut reference."""
    num_nodes = int(data.num_nodes)
    num_partitions = int(config.num_partitions)
    if num_nodes <= num_partitions:
        raise ValueError("spectral normalized cut requires more nodes than partitions")

    adjacency = _adjacency_from_edge_index(edge_index=data.edge_index, num_nodes=num_nodes)
    laplacian = _normalized_laplacian(adjacency)
    try:
        eigenvalues, eigenvectors = eigsh(
            laplacian,
            k=num_partitions,
            which="SM",
            tol=config.tolerance,
            maxiter=max(5 * num_nodes, 1000),
            v0=np.ones(num_nodes, dtype=np.float64),
        )
    except Exception as exc:  # noqa: BLE001
        if num_nodes > 512:
            raise RuntimeError(
                f"Sparse spectral normalized-cut solver failed for graph with {num_nodes} nodes"
            ) from exc
        dense = laplacian.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(dense)
        eigenvectors = eigenvectors[:, :num_partitions]

    order = np.argsort(eigenvalues)[:num_partitions]
    embedding = _row_normalize(eigenvectors[:, order])
    labels = cluster_spectral_embedding(embedding=embedding, config=config)
    return torch.from_numpy(labels).long()


def _graph_cache_key(data: Data, config: SpectralNormalizedCutConfig) -> str:
    """Create a stable cache filename for one graph and reference configuration."""
    split = str(getattr(data, "original_split", "unknown"))
    index = int(getattr(data, "original_graph_index", -1))
    nodes = int(data.num_nodes)
    edges = int(data.edge_index.shape[1] // 2)
    tolerance = f"{float(config.tolerance):.12g}".replace("-", "m").replace(".", "p")
    return (
        f"{config.cache_version}_{split}_{index:06d}_n{nodes}_e{edges}_"
        f"k{config.num_partitions}_seed{config.seed}_ninit{config.n_init}_"
        f"maxiter{config.max_iter}_tol{tolerance}.pt"
    )


def evaluate_spectral_normalized_cut(
    data: Data,
    *,
    cache_dir: Path | None,
    config: SpectralNormalizedCutConfig,
) -> dict[str, Any]:
    """Evaluate or load the cached spectral normalized-cut reference."""
    cache_path = None if cache_dir is None else cache_dir / _graph_cache_key(data=data, config=config)
    if cache_path is not None and cache_path.exists():
        cached = torch.load(cache_path, weights_only=False)
        return cast(dict[str, Any], cached)

    labels = spectral_normalized_cut_partition(data=data, config=config)
    metrics = hard_normalized_cut(
        assignments=labels,
        edge_index=data.edge_index,
        num_partitions=int(config.num_partitions),
    )
    result: dict[str, Any] = {
        "config": asdict(config),
        "labels": labels,
        "hard_ncut": float(metrics.ncut_per_graph[0].cpu()),
        "hard_nassoc": float(metrics.nassoc_per_graph[0].cpu()),
        "active_partitions": float(metrics.active_partitions_per_graph[0].cpu()),
        "hard_min_volume_fraction": float(metrics.min_partition_volume_fraction_per_graph[0].cpu()),
        "hard_max_volume_fraction": float(metrics.max_partition_volume_fraction_per_graph[0].cpu()),
        "hard_within_edge_fraction": float(metrics.within_edge_fraction_per_graph[0].cpu()),
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(obj=result, f=cache_path)
        metadata_path = cache_path.with_suffix(".json")
        serializable = {key: value for key, value in result.items() if key != "labels"}
        metadata_path.write_text(json.dumps(serializable, indent=2))
    return result
