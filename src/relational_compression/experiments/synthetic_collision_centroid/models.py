"""Encoder and decoder building blocks for synthetic finite-code experiments."""

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BinaryCodeEncoder(nn.Module):
    """Map point coordinates to logits for a factorized binary code."""

    def __init__(self, *, input_dim: int, hidden_dim: int, bits: int) -> None:
        """Initialize the point encoder.

        Args:
            input_dim: Number of point coordinates.
            hidden_dim: Width of each hidden layer.
            bits: Number of binary code bits.

        """
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or bits < 1:
            raise ValueError("input_dim, hidden_dim, and bits must be positive")
        self.bits = bits
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, bits),
        )

    def forward(self, points: Tensor) -> Tensor:
        """Return binary-code logits for a batch of points."""
        return cast(Tensor, self.network(points))


class CodebookDecoder(nn.Module):
    """Store one learnable reconstruction vector for every finite codeword."""

    def __init__(self, *, num_codes: int, output_dim: int, initial_scale: float, seed: int) -> None:
        """Initialize the codebook with deterministic Gaussian vectors.

        Args:
            num_codes: Number of complete binary codewords.
            output_dim: Dimension of every reconstruction vector.
            initial_scale: Standard deviation of the initial codebook.
            seed: Random seed used for initialization.

        """
        super().__init__()
        if num_codes < 1 or output_dim < 1:
            raise ValueError("num_codes and output_dim must be positive")
        if initial_scale <= 0.0:
            raise ValueError("initial_scale must be positive")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        initial = torch.randn(num_codes, output_dim, generator=generator) * float(initial_scale)
        self.code_vectors = nn.Parameter(initial)


def enumerate_binary_codes(bits: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Enumerate all binary words as rows of a tensor."""
    if bits < 1:
        raise ValueError("bits must be positive")
    code_ids = torch.arange(2**bits, device=device, dtype=torch.long)
    bit_ids = torch.arange(bits, device=device, dtype=torch.long)
    return ((code_ids[:, None] >> bit_ids[None, :]) & 1).to(dtype=dtype)


def factorized_code_probabilities(logits: Tensor, *, temperature: float) -> Tensor:
    """Compute complete-word probabilities under independent binary logits."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [points, bits]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    binary_codes = enumerate_binary_codes(logits.shape[1], device=logits.device, dtype=logits.dtype)
    scaled_logits = logits / float(temperature)
    log_positive = F.logsigmoid(scaled_logits)
    log_negative = F.logsigmoid(-scaled_logits)
    log_probabilities = (
        binary_codes[None, :, :] * log_positive[:, None, :]
        + (1.0 - binary_codes[None, :, :]) * log_negative[:, None, :]
    ).sum(dim=-1)
    return torch.softmax(log_probabilities, dim=-1)


def hard_code_ids(logits: Tensor) -> Tensor:
    """Pack zero-threshold binary logits into complete-word identifiers."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [points, bits]")
    bit_weights = 2 ** torch.arange(logits.shape[1], device=logits.device, dtype=torch.long)
    return ((logits > 0.0).to(torch.long) * bit_weights[None, :]).sum(dim=-1)
