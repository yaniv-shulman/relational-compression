"""Experiment support code for relational compression studies."""

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from relational_compression.experiments.teacher_image_compression.collisions import (
    signed_partition_loss,
    signed_teacher_graph,
    soft_bit_probabilities,
)
from relational_compression.experiments.teacher_image_compression.sampling import sample_tokens


@dataclass
class HardCodeMetrics:
    """Store hard-code occupancy and collision metrics."""

    bit_occupancy_mean: float
    bit_occupancy_min: float
    bit_occupancy_max: float
    hard_h1: float
    hard_h2: float
    effective_codes: float
    active_codes: int
    empirical_collision: float


def psnr(mse: float) -> float:
    """Compute peak signal-to-noise ratio from mean squared error.

    Args:
        mse: Reconstruction mean squared error.

    Returns:
        Peak signal-to-noise ratio in decibels.

    """
    return -10.0 * math.log10(max(mse, 1e-12))


def pack_hard_codes(hard_tokens: Tensor) -> Tensor:
    """Pack hard codes."""
    bits = (hard_tokens > 0).to(torch.int64)
    if bits.ndim != 2:
        raise ValueError("hard_tokens must have shape [N, bits]")
    if bits.shape[1] > 62:
        raise ValueError("Exact hard-code entropy diagnostics currently support at most 62 bits")

    weights = (1 << torch.arange(bits.shape[1], device=bits.device, dtype=torch.int64)).view(1, -1)
    return (bits * weights).sum(dim=1)


def hard_code_metrics(hard_tokens: Tensor) -> HardCodeMetrics:
    """Compute hard-code occupancy and collision metrics.

    Args:
        hard_tokens: Hard binary signs indexed by batch, token, and bit.

    Returns:
        Aggregate hard-code utilization metrics.

    """
    if hard_tokens.ndim != 2:
        raise ValueError("hard_tokens must have shape [N, bits]")
    if hard_tokens.shape[0] < 1:
        raise ValueError("at least one hard token is required")

    occupancy = (hard_tokens > 0).float().mean(dim=0)
    codes = pack_hard_codes(hard_tokens)
    _, counts = torch.unique(codes, return_counts=True)
    q = counts.float() / counts.sum()
    h1 = -(q * q.clamp_min(torch.finfo(q.dtype).tiny).log()).sum()
    collision = q.square().sum()
    h2 = -collision.log()

    return HardCodeMetrics(
        bit_occupancy_mean=float(occupancy.mean()),
        bit_occupancy_min=float(occupancy.min()),
        bit_occupancy_max=float(occupancy.max()),
        hard_h1=float(h1),
        hard_h2=float(h2),
        effective_codes=float(math.exp(float(h2))),
        active_codes=int(counts.numel()),
        empirical_collision=float(collision),
    )


def hard_collision_rate(hard_tokens: Tensor, pairs: Tensor) -> float:
    """Compute hard collision rate."""
    if pairs.numel() == 0:
        return float("nan")

    collisions = torch.all(hard_tokens[pairs[:, 0]] == hard_tokens[pairs[:, 1]], dim=1)
    return float(collisions.float().mean())


def weighted_hard_collision_sum(hard_tokens: Tensor, pairs: Tensor, weights: Tensor) -> tuple[float, float]:
    """Compute weighted hard collision sum."""
    if pairs.numel() == 0:
        return 0.0, 0.0
    if weights.shape != (pairs.shape[0],):
        raise ValueError("weights must have shape [P]")

    collisions = torch.all(hard_tokens[pairs[:, 0]] == hard_tokens[pairs[:, 1]], dim=1).float()
    weight_values = weights.detach().float()
    return float((weight_values * collisions).sum()), float(weight_values.sum())


def weighted_hard_collision_rate(hard_tokens: Tensor, pairs: Tensor, weights: Tensor) -> float:
    """Compute weighted hard collision rate."""
    numerator, denominator = weighted_hard_collision_sum(hard_tokens=hard_tokens, pairs=pairs, weights=weights)
    if denominator <= 0.0:
        return float("nan")
    return numerator / denominator


def _weighted_summary(prefix: str, value_parts: list[Tensor], weight_parts: list[Tensor]) -> dict[str, float]:
    """Compute weighted summary."""
    if len(value_parts) == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
        }

    values = torch.cat(value_parts).float()
    weights = torch.cat(weight_parts).float()
    weight_sum = weights.sum()
    if values.numel() == 0 or float(weight_sum) <= 0.0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
        }

    mean = (weights * values).sum() / weight_sum
    variance = (weights * (values - mean).square()).sum() / weight_sum
    return {
        f"{prefix}_mean": float(mean),
        f"{prefix}_std": float(variance.clamp_min(0.0).sqrt()),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
    }


def _flatten_hard(hard: Tensor) -> Tensor:
    """Flatten hard."""
    return hard.permute(0, 2, 3, 1).reshape(-1, hard.shape[1])


def _append_weighted_values(
    value_parts: list[Tensor], weight_parts: list[Tensor], values: Tensor, weights: Tensor
) -> None:
    """Append weighted values."""
    if values.shape != weights.shape:
        raise ValueError("values and weights must have matching shapes")
    mask = weights > 0.0
    value_parts.append(values[mask].detach().cpu().float())
    weight_parts.append(weights[mask].detach().cpu().float())


def evaluate_representation(
    *,
    student: nn.Module,
    teacher: nn.Module,
    loader: Any,
    device: torch.device,
    bits: int,
    sampled_tokens: int,
    code_temperature: float,
    teacher_temperature: float,
    prefer_cross_image: bool,
    max_batches: int | None,
    seed: int,
) -> dict[str, float]:
    """Evaluate representation."""
    student.eval()
    teacher.eval()
    generator = torch.Generator().manual_seed(seed)

    all_hard: list[Tensor] = []
    positive_loss_sum = 0.0
    negative_loss_sum = 0.0
    positive_loss_weight_sum = 0.0
    negative_loss_weight_sum = 0.0
    positive_collision_sum = 0.0
    positive_weight_sum = 0.0
    negative_collision_sum = 0.0
    negative_weight_sum = 0.0
    positive_cosine_values: list[Tensor] = []
    positive_cosine_weights: list[Tensor] = []
    negative_cosine_values: list[Tensor] = []
    negative_cosine_weights: list[Tensor] = []

    with torch.no_grad():
        for batch_idx, images in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(device)
            teacher_grid = teacher(images)
            output = student(images, force_hard=True)
            all_hard.append(_flatten_hard(output.hard).cpu())

            sample = sample_tokens(
                teacher_grid=teacher_grid,
                logits=output.logits,
                hard=output.hard,
                max_tokens=sampled_tokens,
                generator=generator,
            )
            graph = signed_teacher_graph(
                teacher_tokens=sample.teacher_tokens,
                image_indices=sample.image_indices,
                teacher_temperature=teacher_temperature,
                prefer_cross_image=prefer_cross_image,
            )
            probabilities = soft_bit_probabilities(logits=sample.logits, code_temperature=code_temperature)
            loss_values = signed_partition_loss(soft_bit_probabilities=probabilities, graph=graph, bits=bits)
            positive_loss_sum += float((loss_values.positive_loss * loss_values.positive_weight_mass).detach().cpu())
            negative_loss_sum += float((loss_values.negative_loss * loss_values.negative_weight_mass).detach().cpu())
            positive_loss_weight_sum += float(loss_values.positive_weight_mass.detach().cpu())
            negative_loss_weight_sum += float(loss_values.negative_weight_mass.detach().cpu())

            numerator, denominator = weighted_hard_collision_sum(
                hard_tokens=sample.hard, pairs=graph.pairs, weights=graph.positive_weights
            )
            positive_collision_sum += numerator
            positive_weight_sum += denominator
            numerator, denominator = weighted_hard_collision_sum(
                hard_tokens=sample.hard, pairs=graph.pairs, weights=graph.negative_weights
            )
            negative_collision_sum += numerator
            negative_weight_sum += denominator
            _append_weighted_values(
                value_parts=positive_cosine_values,
                weight_parts=positive_cosine_weights,
                values=graph.similarities,
                weights=graph.positive_weights,
            )
            _append_weighted_values(
                value_parts=negative_cosine_values,
                weight_parts=negative_cosine_weights,
                values=graph.similarities,
                weights=graph.negative_weights,
            )

    if len(all_hard) == 0:
        raise RuntimeError("evaluation loader produced no batches")

    hard_metrics = asdict(hard_code_metrics(torch.cat(all_hard)))
    positive_collision = positive_collision_sum / positive_weight_sum if positive_weight_sum > 0.0 else float("nan")
    negative_collision = negative_collision_sum / negative_weight_sum if negative_weight_sum > 0.0 else float("nan")
    partition_gap = positive_collision - negative_collision
    empirical_collision = max(float(hard_metrics["empirical_collision"]), torch.finfo(torch.float32).tiny)
    positive_enrichment = positive_collision / empirical_collision
    negative_suppression = 1.0 - negative_collision / empirical_collision
    positive_loss = positive_loss_sum / positive_loss_weight_sum if positive_loss_weight_sum > 0.0 else 0.0
    negative_loss = negative_loss_sum / negative_loss_weight_sum if negative_loss_weight_sum > 0.0 else 0.0
    active_sides = int(positive_loss_weight_sum > 0.0) + int(negative_loss_weight_sum > 0.0)
    teacher_loss = (positive_loss + negative_loss) / active_sides if active_sides > 0 else 0.0

    result = {
        **hard_metrics,
        "teacher_loss": teacher_loss,
        "teacher_positive_loss": positive_loss,
        "teacher_negative_loss": negative_loss,
        "teacher_positive_weight_mass": positive_weight_sum,
        "teacher_negative_weight_mass": negative_weight_sum,
        "teacher_positive_hard_collision_rate": positive_collision,
        "teacher_negative_hard_collision_rate": negative_collision,
        "teacher_partition_gap": partition_gap,
        "teacher_positive_collision_enrichment": positive_enrichment,
        "teacher_negative_collision_suppression": negative_suppression,
    }
    result.update(
        _weighted_summary(
            prefix="teacher_positive_cosine", value_parts=positive_cosine_values, weight_parts=positive_cosine_weights
        )
    )
    result.update(
        _weighted_summary(
            prefix="teacher_negative_cosine", value_parts=negative_cosine_values, weight_parts=negative_cosine_weights
        )
    )
    return result
