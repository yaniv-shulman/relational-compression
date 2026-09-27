"""Centroid and pairwise distortion calculations for the synthetic study."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class CentroidStatistics:
    """Store assignment masses and their weighted reconstruction centroids."""

    masses: Tensor
    centroids: Tensor


@dataclass(frozen=True)
class DecoderDecomposition:
    """Store centroid distortion, decoder excess, and the decomposition residual."""

    decoder_distortion: Tensor
    centroid_distortion: Tensor
    decoder_centroid_gap: Tensor
    residual: Tensor


def pairwise_squared_distances(points: Tensor) -> Tensor:
    """Return all squared Euclidean distances between input points."""
    if points.ndim != 2:
        raise ValueError("points must have shape [points, dimensions]")
    squared_norm = points.square().sum(dim=-1, keepdim=True)
    return (squared_norm + squared_norm.T - 2.0 * (points @ points.T)).clamp_min(0.0)


def centroid_statistics(points: Tensor, code_probabilities: Tensor) -> CentroidStatistics:
    """Compute assignment masses and weighted centroids for each codeword."""
    if points.ndim != 2 or code_probabilities.ndim != 2:
        raise ValueError("points and code_probabilities must both be rank 2")
    if points.shape[0] != code_probabilities.shape[0]:
        raise ValueError("points and code_probabilities must have matching point counts")

    masses = code_probabilities.sum(dim=0)
    denominator = masses.clamp_min(torch.finfo(code_probabilities.dtype).tiny)
    centroids = (code_probabilities.T @ points) / denominator[:, None]
    return CentroidStatistics(masses=masses, centroids=centroids)


def centroid_distortion(points: Tensor, code_probabilities: Tensor) -> Tensor:
    """Compute squared reconstruction distortion using assignment centroids."""
    statistics = centroid_statistics(points=points, code_probabilities=code_probabilities)
    squared_distance = (points[:, None, :] - statistics.centroids[None, :, :]).square().sum(dim=-1)
    return (code_probabilities * squared_distance).sum() / points.shape[0]


def normalized_collision_pair_distortion(
    points: Tensor,
    code_probabilities: Tensor,
    *,
    squared_distances: Tensor | None = None,
) -> Tensor:
    """Compute inverse-mass-weighted pairwise squared-distance distortion."""
    if squared_distances is None:
        squared_distances = pairwise_squared_distances(points)
    if squared_distances.shape != (points.shape[0], points.shape[0]):
        raise ValueError("squared_distances must have shape [points, points]")

    statistics = centroid_statistics(points=points, code_probabilities=code_probabilities)
    inverse_mass = statistics.masses.clamp_min(torch.finfo(code_probabilities.dtype).tiny).reciprocal()
    normalized_collision_kernel = (code_probabilities * inverse_mass[None, :]) @ code_probabilities.T
    return (normalized_collision_kernel * squared_distances).sum() / (2.0 * points.shape[0])


def raw_collision_pair_distortion(
    points: Tensor,
    code_probabilities: Tensor,
    *,
    squared_distances: Tensor | None = None,
) -> Tensor:
    """Compute the unnormalized collision-weighted pairwise distortion."""
    if squared_distances is None:
        squared_distances = pairwise_squared_distances(points)
    if squared_distances.shape != (points.shape[0], points.shape[0]):
        raise ValueError("squared_distances must have shape [points, points]")

    collision_kernel = code_probabilities @ code_probabilities.T
    normalizer = 2.0 * points.shape[0] * points.shape[0]
    return (collision_kernel * squared_distances).sum() / normalizer


def decoder_distortion(points: Tensor, code_probabilities: Tensor, decoder_vectors: Tensor) -> Tensor:
    """Compute squared distortion under arbitrary codeword reconstruction vectors."""
    if decoder_vectors.ndim != 2:
        raise ValueError("decoder_vectors must have shape [codes, dimensions]")
    if decoder_vectors.shape != (code_probabilities.shape[1], points.shape[1]):
        raise ValueError("decoder_vectors have incompatible shape")
    squared_distance = (points[:, None, :] - decoder_vectors[None, :, :]).square().sum(dim=-1)
    return (code_probabilities * squared_distance).sum() / points.shape[0]


def decoder_centroid_gap(points: Tensor, code_probabilities: Tensor, decoder_vectors: Tensor) -> Tensor:
    """Compute excess decoder distortion above the centroid reconstruction."""
    statistics = centroid_statistics(points=points, code_probabilities=code_probabilities)
    squared_gap = (statistics.centroids - decoder_vectors).square().sum(dim=-1)
    return (statistics.masses * squared_gap).sum() / points.shape[0]


def decoder_decomposition(
    points: Tensor,
    code_probabilities: Tensor,
    decoder_vectors: Tensor,
) -> DecoderDecomposition:
    """Decompose decoder distortion into centroid distortion and decoder excess."""
    decoder_value = decoder_distortion(
        points=points, code_probabilities=code_probabilities, decoder_vectors=decoder_vectors
    )
    centroid_value = centroid_distortion(points=points, code_probabilities=code_probabilities)
    gap_value = decoder_centroid_gap(
        points=points, code_probabilities=code_probabilities, decoder_vectors=decoder_vectors
    )
    residual = decoder_value - centroid_value - gap_value
    return DecoderDecomposition(
        decoder_distortion=decoder_value,
        centroid_distortion=centroid_value,
        decoder_centroid_gap=gap_value,
        residual=residual,
    )
