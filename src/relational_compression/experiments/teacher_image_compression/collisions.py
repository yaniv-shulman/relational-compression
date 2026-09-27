"""Experiment support code for relational compression studies."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class SignedTeacherGraph:
    """Store pairwise teacher alignment and separation requirements."""

    pairs: Tensor
    similarities: Tensor
    teacher_probabilities: Tensor
    neutral_probabilities: Tensor
    positive_weights: Tensor
    negative_weights: Tensor


@dataclass(frozen=True)
class SignedPartitionLoss:
    """Store signed teacher-defined loss diagnostics."""

    loss: Tensor
    positive_loss: Tensor
    negative_loss: Tensor
    positive_weight_mass: Tensor
    negative_weight_mass: Tensor
    same_probability: Tensor
    positive_same_probability: Tensor
    negative_same_probability: Tensor


def soft_bit_probabilities(logits: Tensor, code_temperature: float) -> Tensor:
    """Convert binary logits to independent soft bit probabilities.

    Args:
        logits: Binary code logits for each token.
        code_temperature: Positive temperature applied before the sigmoid.

    Returns:
        Per-token Bernoulli probabilities for each bit.

    """
    if code_temperature <= 0.0:
        raise ValueError("code_temperature must be positive")
    return torch.sigmoid(logits / float(code_temperature))


def same_code_log_probability(soft_bit_probabilities: Tensor, pairs: Tensor) -> Tensor:
    """Compute log same-codeword probabilities for selected token pairs.

    Args:
        soft_bit_probabilities: Independent Bernoulli bit probabilities by token.
        pairs: Row pairs identifying the tokens to compare.

    Returns:
        Log probability that every bit agrees for each pair.

    """
    if soft_bit_probabilities.ndim != 2:
        raise ValueError("soft_bit_probabilities must have shape [N, bits]")
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must have shape [P, 2]")
    if pairs.numel() == 0:
        return soft_bit_probabilities.new_empty((0,))

    left = soft_bit_probabilities[pairs[:, 0]]
    right = soft_bit_probabilities[pairs[:, 1]]
    agreement = left * right + (1.0 - left) * (1.0 - right)
    return agreement.clamp_min(torch.finfo(agreement.dtype).tiny).log().sum(dim=-1)


def same_code_probability(soft_bit_probabilities: Tensor, pairs: Tensor) -> Tensor:
    """Compute same-codeword probabilities for selected token pairs.

    Args:
        soft_bit_probabilities: Independent Bernoulli bit probabilities by token.
        pairs: Row pairs identifying the tokens to compare.

    Returns:
        Whole-code agreement probability for each pair.

    """
    return same_code_log_probability(soft_bit_probabilities=soft_bit_probabilities, pairs=pairs).exp()


def signed_teacher_graph(
    teacher_tokens: Tensor,
    image_indices: Tensor,
    *,
    teacher_temperature: float,
    prefer_cross_image: bool = True,
) -> SignedTeacherGraph:
    """Construct pairwise teacher alignment and separation requirements.

    Args:
        teacher_tokens: Normalized teacher token features.
        image_indices: Source-image index for each token.
        teacher_temperature: Temperature used to normalize teacher similarities.
        prefer_cross_image: Whether to exclude pairs from the same source image.

    Returns:
        Pair indices, teacher similarities, and signed relation weights.

    """
    if teacher_tokens.ndim != 2:
        raise ValueError("teacher_tokens must have shape [N, D]")
    if image_indices.shape != (teacher_tokens.shape[0],):
        raise ValueError("image_indices must have shape [N]")
    if teacher_temperature <= 0.0:
        raise ValueError("teacher_temperature must be positive")

    token_count = teacher_tokens.shape[0]
    device = teacher_tokens.device
    empty_index = torch.empty(0, 2, dtype=torch.long, device=device)
    empty_float = teacher_tokens.new_empty((0,))
    if token_count < 2:
        return SignedTeacherGraph(
            pairs=empty_index,
            similarities=empty_float,
            teacher_probabilities=empty_float,
            neutral_probabilities=empty_float,
            positive_weights=empty_float,
            negative_weights=empty_float,
        )

    similarities = teacher_tokens @ teacher_tokens.T
    valid = ~torch.eye(token_count, dtype=torch.bool, device=device)
    if prefer_cross_image:
        valid = valid & (image_indices[:, None] != image_indices[None, :])

    valid_rows = valid.sum(dim=1) > 0
    row_indices = torch.arange(token_count, device=device)[valid_rows]
    col_indices = torch.arange(token_count, device=device)
    row_valid = valid[valid_rows]
    row_logits = (similarities[valid_rows] / float(teacher_temperature)).masked_fill(~row_valid, -torch.inf)
    row_probabilities = torch.softmax(row_logits, dim=1)

    row_grid = row_indices.view(-1, 1).expand_as(row_probabilities)
    col_grid = col_indices.view(1, -1).expand_as(row_probabilities)
    pairs = torch.stack((row_grid[row_valid], col_grid[row_valid]), dim=1)
    teacher_probabilities = row_probabilities[row_valid]
    neutral_probabilities = row_valid.sum(dim=1, keepdim=True).to(dtype=teacher_tokens.dtype).reciprocal()
    neutral_probabilities = neutral_probabilities.expand_as(row_probabilities)[row_valid]

    delta = (teacher_probabilities - neutral_probabilities).detach()
    return SignedTeacherGraph(
        pairs=pairs,
        similarities=similarities[pairs[:, 0], pairs[:, 1]].detach(),
        teacher_probabilities=teacher_probabilities.detach(),
        neutral_probabilities=neutral_probabilities.detach(),
        positive_weights=delta.clamp_min(0.0),
        negative_weights=(-delta).clamp_min(0.0),
    )


def signed_partition_loss(
    soft_bit_probabilities: Tensor,
    graph: SignedTeacherGraph,
    *,
    bits: int,
) -> SignedPartitionLoss:
    """Compute the signed teacher-defined partitioning loss."""
    if bits < 1:
        raise ValueError("bits must be positive")
    if soft_bit_probabilities.ndim != 2:
        raise ValueError("soft_bit_probabilities must have shape [N, bits]")
    if soft_bit_probabilities.shape[1] != bits:
        raise ValueError("soft_bit_probabilities last dimension must equal bits")

    zero = soft_bit_probabilities.sum() * 0.0
    log_same_probability = same_code_log_probability(soft_bit_probabilities=soft_bit_probabilities, pairs=graph.pairs)
    if log_same_probability.numel() == 0:
        empty = soft_bit_probabilities.new_empty((0,))
        return SignedPartitionLoss(
            loss=zero,
            positive_loss=zero,
            negative_loss=zero,
            positive_weight_mass=zero.detach(),
            negative_weight_mass=zero.detach(),
            same_probability=empty,
            positive_same_probability=empty,
            negative_same_probability=empty,
        )

    prior_probability = math.exp(-int(bits) * math.log(2.0))
    probability_floor = torch.finfo(soft_bit_probabilities.dtype).eps
    if probability_floor >= prior_probability:
        raise ValueError(
            "collision clipping floor must be below the neutral same-code probability; "
            f"got dtype={soft_bit_probabilities.dtype} and bits={bits}; "
            "use a higher-precision loss dtype or reduce the code width"
        )
    same_probability = log_same_probability.exp().clamp(min=probability_floor, max=1.0 - probability_floor)
    same_logit = torch.logit(same_probability)
    prior_logit = math.log(prior_probability) - math.log1p(-prior_probability)
    margin = same_logit - prior_logit

    positive_weight_mass = graph.positive_weights.sum()
    negative_weight_mass = graph.negative_weights.sum()
    positive_numerator = (graph.positive_weights * F.softplus(-margin)).sum()
    negative_numerator = (graph.negative_weights * F.softplus(margin)).sum()
    denominator_floor = torch.finfo(margin.dtype).tiny
    positive_active = positive_weight_mass > 0.0
    negative_active = negative_weight_mass > 0.0
    positive_active_float = positive_active.to(dtype=margin.dtype)
    negative_active_float = negative_active.to(dtype=margin.dtype)
    positive_loss = positive_numerator / positive_weight_mass.clamp_min(denominator_floor)
    negative_loss = negative_numerator / negative_weight_mass.clamp_min(denominator_floor)
    active_side_count = (positive_active_float + negative_active_float).clamp_min(1.0)
    loss = (positive_active_float * positive_loss + negative_active_float * negative_loss) / active_side_count
    loss = loss + zero

    return SignedPartitionLoss(
        loss=loss,
        positive_loss=positive_loss,
        negative_loss=negative_loss,
        positive_weight_mass=positive_weight_mass.detach(),
        negative_weight_mass=negative_weight_mass.detach(),
        same_probability=same_probability,
        positive_same_probability=same_probability[graph.positive_weights > 0.0],
        negative_same_probability=same_probability[graph.negative_weights > 0.0],
    )
