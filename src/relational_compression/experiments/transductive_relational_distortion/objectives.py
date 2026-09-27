"""Experiment support code for relational compression studies."""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from relational_compression.experiments.transductive_relational_distortion.source_geometry import (
    COLLISION,
    COLLISION_ENTROPY,
    EDGE,
    FOURIER,
    SOURCE_CRITERIA,
    SourceGeometry,
)


@dataclass(frozen=True)
class SoftObjectiveResult:
    """Store Soft Objective Result values."""

    loss: Tensor
    own_distortion: Tensor
    edge_distortion: Tensor
    fourier_distortion: Tensor
    collision_distortion: Tensor
    collision_entropy_distortion: Tensor
    marginal_d2: Tensor
    soft_h2: Tensor
    soft_k_eff: Tensor
    q_bar: Tensor
    assignment_confidence: Tensor
    assignment_entropy: Tensor


@dataclass(frozen=True)
class HardEvaluationResult:
    """Store Hard Evaluation Result values."""

    edge_distortion: float
    fourier_distortion: float
    collision_distortion: float
    collision_entropy_distortion: float
    hard_h2: float
    hard_k_eff: float
    active_partitions: int
    max_volume_fraction: float
    min_volume_fraction: float
    volume_fractions: Tensor


def assignment_probabilities(logits: Tensor, *, temperature: float) -> Tensor:
    """Compute assignment probabilities."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [num_nodes, num_partitions]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    return torch.softmax(logits / float(temperature), dim=-1)


def edge_collision(probabilities: Tensor, edge_index: Tensor) -> Tensor:
    """Compute edge collision."""
    source, target = edge_index.to(device=probabilities.device)
    return (probabilities[source] * probabilities[target]).sum(dim=-1)


def normalized_edge_distortion(probabilities: Tensor, edge_index: Tensor, rho: Tensor) -> Tensor:
    """Compute normalized edge distortion."""
    rho = rho.to(device=probabilities.device, dtype=probabilities.dtype)
    return (rho * (1.0 - edge_collision(probabilities=probabilities, edge_index=edge_index))).sum()


def hard_normalized_edge_distortion(assignments: Tensor, edge_index: Tensor, rho: Tensor) -> float:
    """Compute hard normalized edge distortion."""
    labels = assignments.to(device=edge_index.device, dtype=torch.long)
    source, target = edge_index.to(device=labels.device)
    cut = labels[source] != labels[target]
    return float(rho.to(device=labels.device, dtype=torch.float64)[cut].sum().detach().cpu())


def _nan_like(reference: Tensor) -> Tensor:
    """Compute nan like."""
    return reference.new_tensor(float("nan"))


def _soft_distortion_for(probabilities: Tensor, geometry: SourceGeometry, criterion: str) -> Tensor:
    """Compute soft distortion for."""
    try:
        return normalized_edge_distortion(
            probabilities=probabilities, edge_index=geometry.edge_index, rho=geometry.rho_for(criterion)
        )
    except ValueError:
        return _nan_like(probabilities)


def _hard_distortion_for(assignments: Tensor, geometry: SourceGeometry, criterion: str) -> float:
    """Compute hard distortion for."""
    try:
        return hard_normalized_edge_distortion(
            assignments=assignments, edge_index=geometry.edge_index, rho=geometry.rho_for(criterion)
        )
    except ValueError:
        return float("nan")


def degree_weighted_marginal(probabilities: Tensor, degree: Tensor) -> Tensor:
    """Compute degree weighted marginal."""
    degree = degree.to(device=probabilities.device, dtype=probabilities.dtype)
    volume = degree.sum().clamp_min(torch.finfo(probabilities.dtype).tiny)
    q_bar = (degree.unsqueeze(-1) * probabilities).sum(dim=0) / volume
    q_bar = q_bar.clamp_min(0.0)
    return q_bar / q_bar.sum().clamp_min(torch.finfo(q_bar.dtype).tiny)


def _marginal_collision(q_bar: Tensor) -> Tensor:
    """Compute marginal collision."""
    q_bar = q_bar.clamp_min(0.0)
    q_bar = q_bar / q_bar.sum().clamp_min(torch.finfo(q_bar.dtype).tiny)
    collision = q_bar.square().sum()
    return collision.clamp(min=1.0 / float(q_bar.numel()), max=1.0)


def _zero_tiny(value: Tensor) -> Tensor:
    """Compute zero tiny."""
    tolerance = 32.0 * torch.finfo(value.dtype).eps
    return torch.where(condition=value.abs() < tolerance, input=value.new_zeros(()), other=value)


def renyi2_entropy_from_marginal(q_bar: Tensor) -> Tensor:
    """Compute renyi2 entropy from marginal."""
    collision = _marginal_collision(q_bar)
    return _zero_tiny(-collision.log())


def renyi2_prior_matching_to_uniform(q_bar: Tensor) -> Tensor:
    """Compute renyi2 prior matching to uniform."""
    collision = _marginal_collision(q_bar)
    return _zero_tiny((float(q_bar.numel()) * collision).log())


def soft_objective(
    logits: Tensor,
    geometry: SourceGeometry,
    *,
    criterion: str,
    lambda_org: float,
    temperature: float,
) -> SoftObjectiveResult:
    """Compute soft objective."""
    probabilities = assignment_probabilities(logits, temperature=temperature)
    if criterion not in SOURCE_CRITERIA or criterion not in geometry.defined_criteria:
        raise ValueError(f"Unsupported source distortion criterion: {criterion}")
    edge = _soft_distortion_for(probabilities=probabilities, geometry=geometry, criterion=EDGE)
    fourier = _soft_distortion_for(probabilities=probabilities, geometry=geometry, criterion=FOURIER)
    collision = _soft_distortion_for(probabilities=probabilities, geometry=geometry, criterion=COLLISION)
    collision_entropy = _soft_distortion_for(
        probabilities=probabilities, geometry=geometry, criterion=COLLISION_ENTROPY
    )
    own = {
        EDGE: edge,
        FOURIER: fourier,
        COLLISION: collision,
        COLLISION_ENTROPY: collision_entropy,
    }[criterion]

    q_bar = degree_weighted_marginal(probabilities=probabilities, degree=geometry.degree)
    marginal_d2 = renyi2_prior_matching_to_uniform(q_bar)
    soft_h2 = renyi2_entropy_from_marginal(q_bar)
    loss = own + float(lambda_org) * marginal_d2
    node_entropy = -(probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log() * probabilities).sum(dim=-1)

    return SoftObjectiveResult(
        loss=loss,
        own_distortion=own,
        edge_distortion=edge,
        fourier_distortion=fourier,
        collision_distortion=collision,
        collision_entropy_distortion=collision_entropy,
        marginal_d2=marginal_d2,
        soft_h2=soft_h2,
        soft_k_eff=soft_h2.exp(),
        q_bar=q_bar,
        assignment_confidence=probabilities.max(dim=-1).values.mean(),
        assignment_entropy=node_entropy.mean(),
    )


def hard_evaluation(assignments: Tensor, geometry: SourceGeometry, *, num_partitions: int) -> HardEvaluationResult:
    """Compute hard evaluation."""
    if assignments.ndim != 1:
        raise ValueError("assignments must have shape [num_nodes]")
    if int(assignments.shape[0]) != geometry.num_nodes:
        raise ValueError("assignments length must match graph node count")
    assignments = assignments.to(device=geometry.edge_index.device, dtype=torch.long)
    one_hot = F.one_hot(assignments, num_classes=int(num_partitions)).to(dtype=geometry.degree.dtype)
    q_bar = degree_weighted_marginal(probabilities=one_hot, degree=geometry.degree)
    hard_h2 = renyi2_entropy_from_marginal(q_bar)
    return HardEvaluationResult(
        edge_distortion=_hard_distortion_for(assignments=assignments, geometry=geometry, criterion=EDGE),
        fourier_distortion=_hard_distortion_for(assignments=assignments, geometry=geometry, criterion=FOURIER),
        collision_distortion=_hard_distortion_for(assignments=assignments, geometry=geometry, criterion=COLLISION),
        collision_entropy_distortion=_hard_distortion_for(
            assignments=assignments, geometry=geometry, criterion=COLLISION_ENTROPY
        ),
        hard_h2=float(hard_h2.detach().cpu()),
        hard_k_eff=float(hard_h2.exp().detach().cpu()),
        active_partitions=int((q_bar > 0.0).sum().detach().cpu()),
        max_volume_fraction=float(q_bar.max().detach().cpu()),
        min_volume_fraction=float(q_bar.min().detach().cpu()),
        volume_fractions=q_bar.detach().cpu(),
    )
