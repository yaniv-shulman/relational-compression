"""Soft and hard normalized-cut objectives and reporting summaries."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SoftNormalizedCutResult:
    """Collect per-graph outputs from the soft normalized-cut objective."""

    loss: Tensor
    ncut_loss: Tensor
    separation_loss: Tensor
    ncut_per_graph: Tensor
    nassoc_per_graph: Tensor
    separation_d2_per_graph: Tensor
    effective_partitions_per_graph: Tensor
    active_partitions_per_graph: Tensor
    assignment_entropy_per_graph: Tensor
    assignment_confidence_per_graph: Tensor
    min_partition_volume_fraction_per_graph: Tensor
    max_partition_volume_fraction_per_graph: Tensor
    within_edge_fraction_per_graph: Tensor
    q_z: Tensor


@dataclass(frozen=True)
class HardNormalizedCutResult:
    """Collect per-graph metrics for hard partition assignments."""

    ncut_per_graph: Tensor
    nassoc_per_graph: Tensor
    active_partitions_per_graph: Tensor
    min_partition_volume_fraction_per_graph: Tensor
    max_partition_volume_fraction_per_graph: Tensor
    within_edge_fraction_per_graph: Tensor
    volume_fractions: Tensor


def _edge_weights(edge_index: Tensor, edge_weight: Tensor | None) -> Tensor:
    """Return supplied edge weights or unit weights for every edge."""
    if edge_weight is None:
        return torch.ones(edge_index.shape[1], device=edge_index.device, dtype=torch.float32)
    return edge_weight.to(device=edge_index.device, dtype=torch.float32)


def _node_degrees(
    *,
    edge_index: Tensor,
    num_nodes: int,
    edge_weight: Tensor | None = None,
) -> Tensor:
    """Compute weighted out-degrees from directed graph incidences."""
    weights = _edge_weights(edge_index=edge_index, edge_weight=edge_weight)
    degree = torch.zeros(num_nodes, device=edge_index.device, dtype=weights.dtype)
    return degree.index_add(0, edge_index[0], weights)


def _graph_count(batch: Tensor | None, num_graphs: int | None = None) -> int:
    """Resolve the number of graphs represented by a batch vector."""
    if num_graphs is not None:
        return int(num_graphs)
    if batch is None or batch.numel() == 0:
        return 1
    return int(batch.max().detach().cpu()) + 1


def _batch_vector(num_nodes: int, batch: Tensor | None, device: torch.device) -> Tensor:
    """Return graph identifiers for nodes in a single graph or batch."""
    if batch is None:
        return torch.zeros(num_nodes, device=device, dtype=torch.long)
    return batch.to(device=device, dtype=torch.long)


def _scatter_sum_by_graph(values: Tensor, graph_ids: Tensor, num_graphs: int) -> Tensor:
    """Sum values into graph-indexed bins."""
    result = torch.zeros(num_graphs, device=values.device, dtype=values.dtype)
    return result.index_add(0, graph_ids, values)


def soft_normalized_cut(
    probabilities: Tensor,
    edge_index: Tensor,
    *,
    batch: Tensor | None = None,
    num_graphs: int | None = None,
    edge_weight: Tensor | None = None,
    q_z_floor: float = 1e-12,
    separation_weight: float = 0.0,
) -> SoftNormalizedCutResult:
    """Compute the q_Z-normalized soft normalized-cut objective.

    The graph must use directed symmetric edge incidences. For a PyG batch, q_Z
    and volume are computed per graph and losses are averaged across graphs.

    Args:
        probabilities: Node-to-partition assignment probabilities.
        edge_index: Directed symmetric edge incidences.
        batch: Optional graph identifier for every node.
        num_graphs: Optional explicit number of graphs in the batch.
        edge_weight: Optional nonnegative weight for every directed edge.
        q_z_floor: Lower bound for inverse aggregate partition masses.
        separation_weight: Weight for the marginal organization term.

    Returns:
        Objective losses and per-graph normalized-cut diagnostics.

    """
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [num_nodes, num_partitions]")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")

    num_nodes, num_partitions = probabilities.shape
    device = probabilities.device
    node_batch = _batch_vector(num_nodes=num_nodes, batch=batch, device=device)
    num_graphs = _graph_count(batch=node_batch, num_graphs=num_graphs)
    weights = _edge_weights(edge_index=edge_index, edge_weight=edge_weight).to(device=device, dtype=probabilities.dtype)
    edge_index = edge_index.to(device=device)
    source = edge_index[0]
    target = edge_index[1]
    edge_graph = node_batch[source]

    degree = _node_degrees(edge_index=edge_index, num_nodes=num_nodes, edge_weight=weights).to(probabilities.dtype)
    volume = _scatter_sum_by_graph(values=degree, graph_ids=node_batch, num_graphs=num_graphs).clamp_min(
        torch.finfo(probabilities.dtype).tiny
    )

    mass = torch.zeros(num_graphs, num_partitions, device=device, dtype=probabilities.dtype)
    mass.index_add_(0, node_batch, degree.unsqueeze(-1) * probabilities)
    q_z = mass / volume.unsqueeze(-1)

    inv_q_z = q_z.clamp_min(float(q_z_floor)).reciprocal()
    edge_affinity = (probabilities[source] * probabilities[target] * inv_q_z[edge_graph]).sum(dim=-1)
    nassoc = _scatter_sum_by_graph(values=edge_affinity * weights, graph_ids=edge_graph, num_graphs=num_graphs) / volume
    ncut = float(num_partitions) - nassoc
    aggregate_collision = q_z.square().sum(dim=-1)
    separation_d2 = (float(num_partitions) * aggregate_collision).clamp_min(torch.finfo(probabilities.dtype).tiny).log()
    effective_partitions = aggregate_collision.clamp_min(torch.finfo(probabilities.dtype).tiny).reciprocal()
    ncut_loss = ncut.mean()
    separation_loss = float(separation_weight) * separation_d2.mean()
    loss = ncut_loss if float(separation_weight) == 0.0 else ncut_loss + separation_loss

    entropy = -(probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log() * probabilities).sum(dim=-1)
    confidence = probabilities.max(dim=-1).values
    entropy_by_graph = _scatter_sum_by_graph(
        values=entropy, graph_ids=node_batch, num_graphs=num_graphs
    ) / torch.bincount(node_batch, minlength=num_graphs).to(device=device, dtype=probabilities.dtype).clamp_min(1.0)
    confidence_by_graph = _scatter_sum_by_graph(
        values=confidence, graph_ids=node_batch, num_graphs=num_graphs
    ) / torch.bincount(node_batch, minlength=num_graphs).to(device=device, dtype=probabilities.dtype).clamp_min(1.0)

    same_soft = (probabilities[source] * probabilities[target]).sum(dim=-1)
    within_edge_fraction = (
        _scatter_sum_by_graph(values=same_soft * weights, graph_ids=edge_graph, num_graphs=num_graphs) / volume
    )
    active = (q_z > 1.0 / max(1, 10 * num_partitions)).sum(dim=-1).to(probabilities.dtype)

    return SoftNormalizedCutResult(
        loss=loss,
        ncut_loss=ncut_loss,
        separation_loss=separation_loss,
        ncut_per_graph=ncut,
        nassoc_per_graph=nassoc,
        separation_d2_per_graph=separation_d2,
        effective_partitions_per_graph=effective_partitions,
        active_partitions_per_graph=active,
        assignment_entropy_per_graph=entropy_by_graph,
        assignment_confidence_per_graph=confidence_by_graph,
        min_partition_volume_fraction_per_graph=q_z.min(dim=-1).values,
        max_partition_volume_fraction_per_graph=q_z.max(dim=-1).values,
        within_edge_fraction_per_graph=within_edge_fraction,
        q_z=q_z,
    )


def hard_normalized_cut(
    assignments: Tensor,
    edge_index: Tensor,
    *,
    num_partitions: int,
    batch: Tensor | None = None,
    num_graphs: int | None = None,
    edge_weight: Tensor | None = None,
) -> HardNormalizedCutResult:
    """Evaluate hard K-way normalized cut, penalizing empty partitions.

    Args:
        assignments: Hard partition identifier for each node.
        edge_index: Directed symmetric edge incidences.
        num_partitions: Declared number of partition labels.
        batch: Optional graph identifier for every node.
        num_graphs: Optional explicit number of graphs in the batch.
        edge_weight: Optional nonnegative weight for every directed edge.

    Returns:
        Per-graph hard normalized-cut diagnostics.

    """
    if assignments.ndim != 1:
        raise ValueError("assignments must have shape [num_nodes]")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")

    num_nodes = assignments.shape[0]
    device = assignments.device
    node_batch = _batch_vector(num_nodes=num_nodes, batch=batch, device=device)
    num_graphs = _graph_count(batch=node_batch, num_graphs=num_graphs)
    assignments = assignments.to(device=device, dtype=torch.long)
    edge_index = edge_index.to(device=device)
    weights = _edge_weights(edge_index=edge_index, edge_weight=edge_weight).to(device=device)
    source = edge_index[0]
    target = edge_index[1]
    edge_graph = node_batch[source]

    degree = _node_degrees(edge_index=edge_index, num_nodes=num_nodes, edge_weight=weights)
    volume = _scatter_sum_by_graph(values=degree, graph_ids=node_batch, num_graphs=num_graphs).clamp_min(
        torch.finfo(degree.dtype).tiny
    )

    flat_partition = node_batch * num_partitions + assignments
    volume_by_partition = torch.zeros(num_graphs * num_partitions, device=device, dtype=degree.dtype)
    volume_by_partition.index_add_(0, flat_partition, degree)
    volume_by_partition = volume_by_partition.reshape(num_graphs, num_partitions)

    same = assignments[source] == assignments[target]
    internal_assoc = torch.zeros(num_graphs * num_partitions, device=device, dtype=degree.dtype)
    internal_partition = edge_graph * num_partitions + assignments[source]
    internal_assoc.index_add_(0, internal_partition[same], weights[same])
    internal_assoc = internal_assoc.reshape(num_graphs, num_partitions)

    nonempty = volume_by_partition > 0
    nassoc_by_partition = torch.where(
        condition=nonempty,
        input=internal_assoc / volume_by_partition.clamp_min(torch.finfo(degree.dtype).tiny),
        other=torch.zeros_like(internal_assoc),
    )
    nassoc = nassoc_by_partition.sum(dim=-1)
    ncut = float(num_partitions) - nassoc
    volume_fractions = volume_by_partition / volume.unsqueeze(-1)
    within_edge_fraction = (
        _scatter_sum_by_graph(values=same.to(degree.dtype) * weights, graph_ids=edge_graph, num_graphs=num_graphs)
        / volume
    )

    return HardNormalizedCutResult(
        ncut_per_graph=ncut,
        nassoc_per_graph=nassoc,
        active_partitions_per_graph=nonempty.sum(dim=-1).to(degree.dtype),
        min_partition_volume_fraction_per_graph=volume_fractions.min(dim=-1).values,
        max_partition_volume_fraction_per_graph=volume_fractions.max(dim=-1).values,
        within_edge_fraction_per_graph=within_edge_fraction,
        volume_fractions=volume_fractions,
    )


def summarize_tensor(values: Tensor, prefix: str) -> dict[str, float]:
    """Summarize a tensor with scalar reporting statistics."""
    finite = values.detach().float().cpu()
    if finite.numel() == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_mean": float(finite.mean()),
        f"{prefix}_median": float(torch.quantile(input=finite, q=0.5)),
        f"{prefix}_std": float(finite.std(unbiased=False)),
        f"{prefix}_min": float(finite.min()),
        f"{prefix}_max": float(finite.max()),
    }


def summarize_soft_result(result: SoftNormalizedCutResult, prefix: str = "") -> dict[str, float]:
    """Convert soft normalized-cut outputs into named scalar summaries."""
    stem = f"{prefix}_" if prefix != "" else ""
    values: dict[str, float] = {}
    values.update(summarize_tensor(values=result.ncut_per_graph, prefix=f"{stem}soft_ncut"))
    values.update(summarize_tensor(values=result.nassoc_per_graph, prefix=f"{stem}soft_nassoc"))
    values.update(summarize_tensor(values=result.separation_d2_per_graph, prefix=f"{stem}separation_d2"))
    values.update(summarize_tensor(values=result.effective_partitions_per_graph, prefix=f"{stem}effective_partitions"))
    values.update(summarize_tensor(values=result.active_partitions_per_graph, prefix=f"{stem}soft_active_partitions"))
    values.update(summarize_tensor(values=result.assignment_entropy_per_graph, prefix=f"{stem}assignment_entropy"))
    values.update(
        summarize_tensor(values=result.assignment_confidence_per_graph, prefix=f"{stem}assignment_confidence")
    )
    values.update(
        summarize_tensor(values=result.min_partition_volume_fraction_per_graph, prefix=f"{stem}min_volume_fraction")
    )
    values.update(
        summarize_tensor(values=result.max_partition_volume_fraction_per_graph, prefix=f"{stem}max_volume_fraction")
    )
    values.update(
        summarize_tensor(values=result.within_edge_fraction_per_graph, prefix=f"{stem}soft_within_edge_fraction")
    )
    values[f"{stem}soft_ncut_loss"] = float(result.ncut_loss.detach().cpu())
    values[f"{stem}separation_loss"] = float(result.separation_loss.detach().cpu())
    values[f"{stem}loss"] = float(result.loss.detach().cpu())
    return values


def summarize_hard_result(result: HardNormalizedCutResult, prefix: str = "") -> dict[str, float]:
    """Convert hard partition outputs into named scalar summaries."""
    stem = f"{prefix}_" if prefix != "" else ""
    values: dict[str, float] = {}
    values.update(summarize_tensor(values=result.ncut_per_graph, prefix=f"{stem}hard_ncut"))
    values.update(summarize_tensor(values=result.nassoc_per_graph, prefix=f"{stem}hard_nassoc"))
    values.update(summarize_tensor(values=result.active_partitions_per_graph, prefix=f"{stem}hard_active_partitions"))
    values.update(
        summarize_tensor(
            values=result.min_partition_volume_fraction_per_graph, prefix=f"{stem}hard_min_volume_fraction"
        )
    )
    values.update(
        summarize_tensor(
            values=result.max_partition_volume_fraction_per_graph, prefix=f"{stem}hard_max_volume_fraction"
        )
    )
    values.update(
        summarize_tensor(values=result.within_edge_fraction_per_graph, prefix=f"{stem}hard_within_edge_fraction")
    )
    return values
