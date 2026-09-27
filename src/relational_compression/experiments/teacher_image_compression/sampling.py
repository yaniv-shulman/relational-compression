"""Experiment support code for relational compression studies."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class TokenSample:
    """Store sampled token positions and their source image indices."""

    teacher_tokens: Tensor
    logits: Tensor
    hard: Tensor
    image_indices: Tensor
    spatial_indices: Tensor


def sample_token_indices(
    *,
    batch_size: int,
    grid_h: int,
    grid_w: int,
    max_tokens: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample token indices."""
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")

    total = batch_size * grid_h * grid_w
    count = min(max_tokens, total)
    flat = torch.randperm(total, generator=generator, device=torch.device("cpu"))[:count].to(device)
    image_indices = flat // (grid_h * grid_w)
    spatial_indices = flat % (grid_h * grid_w)
    return image_indices, spatial_indices


def gather_tokens(
    *,
    teacher_grid: Tensor,
    logits: Tensor,
    hard: Tensor,
    image_indices: Tensor,
    spatial_indices: Tensor,
) -> TokenSample:
    """Gather tokens."""
    if teacher_grid.ndim != 4:
        raise ValueError("teacher_grid must have shape [B, H, W, D]")
    if logits.ndim != 4 or hard.ndim != 4:
        raise ValueError("logits and hard tensors must have shape [B, bits, H, W]")

    batch_size, grid_h, grid_w, _ = teacher_grid.shape
    if logits.shape[0] != batch_size or hard.shape[0] != batch_size:
        raise ValueError("student and teacher batch sizes do not match")
    if logits.shape[2:] != (grid_h, grid_w) or hard.shape[2:] != (grid_h, grid_w):
        raise ValueError(
            f"student latent grid {tuple(logits.shape[2:])} does not match teacher grid {(grid_h, grid_w)}"
        )

    row = spatial_indices // grid_w
    col = spatial_indices % grid_w
    return TokenSample(
        teacher_tokens=teacher_grid[image_indices, row, col],
        logits=logits.permute(0, 2, 3, 1)[image_indices, row, col],
        hard=hard.permute(0, 2, 3, 1)[image_indices, row, col],
        image_indices=image_indices,
        spatial_indices=spatial_indices,
    )


def sample_tokens(
    *,
    teacher_grid: Tensor,
    logits: Tensor,
    hard: Tensor,
    max_tokens: int,
    generator: torch.Generator | None = None,
) -> TokenSample:
    """Sample tokens."""
    image_indices, spatial_indices = sample_token_indices(
        batch_size=teacher_grid.shape[0],
        grid_h=teacher_grid.shape[1],
        grid_w=teacher_grid.shape[2],
        max_tokens=max_tokens,
        device=teacher_grid.device,
        generator=generator,
    )
    return gather_tokens(
        teacher_grid=teacher_grid,
        logits=logits,
        hard=hard,
        image_indices=image_indices,
        spatial_indices=spatial_indices,
    )
