"""Synthetic mixture data used to validate centroid--collision identities."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SyntheticMixture:
    """Store sampled points, component labels, and component counts."""

    points: Tensor
    component_ids: Tensor
    component_counts: Tensor


def _integer_component_counts(num_points: int, component_weights: Tensor) -> Tensor:
    """Allocate integer component counts while preserving the requested total."""
    if num_points < 1:
        raise ValueError("num_points must be positive")
    if component_weights.ndim != 1 or component_weights.numel() < 1:
        raise ValueError("component_weights must be a nonempty vector")
    if torch.any(component_weights <= 0.0):
        raise ValueError("component_weights must be positive")

    normalized = component_weights / component_weights.sum()
    expected = normalized * num_points
    counts = torch.floor(expected).to(torch.long)
    remainder = num_points - int(counts.sum())
    if remainder > 0:
        fractional = expected - counts.to(expected.dtype)
        _, indices = torch.topk(fractional, k=remainder)
        counts[indices] += 1
    return counts


def generate_imbalanced_gaussian_mixture(
    *,
    num_points: int,
    component_weights: Tensor,
    component_means: Tensor,
    component_covariances: Tensor,
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> SyntheticMixture:
    """Sample a deterministic imbalanced Gaussian mixture.

    Args:
        num_points: Total number of observations to sample.
        component_weights: Positive mixture weights.
        component_means: Per-component mean vectors.
        component_covariances: Per-component covariance matrices.
        seed: Random seed for the CPU generator.
        dtype: Output floating-point dtype for the sampled points.

    Returns:
        Sampled points with their shuffled component labels and counts.

    """
    if component_means.ndim != 2:
        raise ValueError("component_means must have shape [components, dimensions]")
    if component_covariances.ndim != 3:
        raise ValueError("component_covariances must have shape [components, dimensions, dimensions]")
    if component_weights.shape[0] != component_means.shape[0]:
        raise ValueError("component_weights and component_means must have matching component counts")
    if component_covariances.shape[0] != component_means.shape[0]:
        raise ValueError("component_covariances and component_means must have matching component counts")
    if component_covariances.shape[1:] != (component_means.shape[1], component_means.shape[1]):
        raise ValueError("component_covariances have incompatible dimensions")

    weights = component_weights.detach().cpu().to(dtype=torch.float64)
    means = component_means.detach().cpu().to(dtype=torch.float64)
    covariances = component_covariances.detach().cpu().to(dtype=torch.float64)
    counts = _integer_component_counts(num_points=num_points, component_weights=weights)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))

    point_parts: list[Tensor] = []
    label_parts: list[Tensor] = []
    for component_id, count in enumerate(counts.tolist()):
        if count == 0:
            continue
        covariance = covariances[component_id]
        cholesky = torch.linalg.cholesky(covariance)
        noise = torch.randn(count, means.shape[1], generator=generator, dtype=torch.float64)
        samples = means[component_id] + noise @ cholesky.T
        point_parts.append(samples)
        label_parts.append(torch.full(size=(count,), fill_value=component_id, dtype=torch.long))

    points = torch.cat(point_parts, dim=0)
    component_ids = torch.cat(label_parts, dim=0)
    permutation = torch.randperm(points.shape[0], generator=generator)
    return SyntheticMixture(
        points=points[permutation].to(dtype=dtype),
        component_ids=component_ids[permutation],
        component_counts=counts,
    )
