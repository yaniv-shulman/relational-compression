"""Experiment support code for relational compression studies."""

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class LatentMetrics:
    """Store occupancy and concentration diagnostics for image latents."""

    bit_occupancy_mean: float
    bit_occupancy_min: float
    bit_occupancy_max: float
    hard_h1: float
    hard_h2: float
    effective_codes: float
    active_codes: int
    empirical_collision: float
    soft_pair_collision: float
    self_collision: float
    relaxed_hard_rms: float


def _pack_codes(hard: Tensor) -> Tensor:
    """Pack codes."""
    bits = (hard > 0).to(torch.int64)
    bits = bits.permute(0, 2, 3, 1).reshape(-1, bits.shape[1])

    if bits.shape[1] > 62:
        raise ValueError("Exact hard-code entropy diagnostics currently support at most 62 bits")

    weights = (1 << torch.arange(bits.shape[1], device=bits.device, dtype=torch.int64)).view(1, -1)
    return (bits * weights).sum(dim=1)


def latent_metrics(hard: Tensor, probabilities: Tensor, relaxed: Tensor, pair_samples: int) -> LatentMetrics:
    """Compute occupancy and concentration diagnostics for image latents."""
    occupancy = (hard > 0).float().mean(dim=(0, 2, 3))
    codes = _pack_codes(hard)
    _, counts = torch.unique(codes, return_counts=True)
    q = counts.float() / counts.sum()
    h1 = -(q * q.clamp_min(torch.finfo(q.dtype).tiny).log()).sum()
    collision = q.square().sum()
    h2 = -collision.log()

    flat_p = probabilities.permute(0, 2, 3, 1).reshape(-1, probabilities.shape[1])
    n = flat_p.shape[0]
    sample_count = min(pair_samples, max(1, n * 4))
    i = torch.randint(high=n, size=(sample_count,), device=flat_p.device)
    j = torch.randint(high=n, size=(sample_count,), device=flat_p.device)
    pi = flat_p[i]
    pj = flat_p[j]
    agreement = pi * pj + (1.0 - pi) * (1.0 - pj)
    soft_pair_collision = agreement.clamp_min(torch.finfo(agreement.dtype).tiny).log().sum(dim=1).exp().mean()

    self_agreement = probabilities.square() + (1.0 - probabilities).square()
    self_collision = self_agreement.clamp_min(torch.finfo(self_agreement.dtype).tiny).log().sum(dim=1).exp().mean()
    relaxed_hard_rms = torch.square(relaxed - hard).mean().sqrt()

    return LatentMetrics(
        bit_occupancy_mean=float(occupancy.mean()),
        bit_occupancy_min=float(occupancy.min()),
        bit_occupancy_max=float(occupancy.max()),
        hard_h1=float(h1),
        hard_h2=float(h2),
        effective_codes=float(math.exp(float(h2))),
        active_codes=int(counts.numel()),
        empirical_collision=float(collision),
        soft_pair_collision=float(soft_pair_collision),
        self_collision=float(self_collision),
        relaxed_hard_rms=float(relaxed_hard_rms),
    )
