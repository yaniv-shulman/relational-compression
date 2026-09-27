"""Binary quantization layers for discrete image representations."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class QuantizerOutput:
    """Collect relaxed and hard codes with their training diagnostics."""

    relaxed: Tensor
    hard: Tensor
    probabilities: Tensor
    concentration_loss: Tensor


class MeanGroupBinaryQuantizer(nn.Module):
    """Relax a binary bottleneck by centering values within sign-defined groups.

    The module emits hard signs at evaluation and a differentiable group-coupled
    surrogate while training.
    """

    def __init__(
        self,
        *,
        stochastic: bool,
        grouping: str = "per_channel",
    ) -> None:
        """Initialize the quantizer.

        Args:
            stochastic: Whether training samples binary signs from sigmoid probabilities.
            grouping: Whether means are computed per channel or over all values.

        """
        super().__init__()

        if grouping not in {"per_channel", "global"}:
            raise ValueError("grouping must be 'per_channel' or 'global'")

        self.stochastic = stochastic
        self.grouping = grouping

    @staticmethod
    def hard_from_logits(logits: Tensor) -> Tensor:
        """Convert logits to deterministic signs using a zero threshold."""
        return torch.where(condition=logits > 0, input=torch.ones_like(logits), other=-torch.ones_like(logits))

    def _group_mean(self, values: Tensor, mask: Tensor) -> Tensor:
        """Compute a masked mean over the configured coupling dimensions."""
        if self.grouping == "global":
            dims = tuple(range(values.ndim))
        else:
            if values.ndim < 2:
                raise ValueError("per_channel grouping expects a channel dimension")

            dims = (0, *range(2, values.ndim))

        weight = mask.to(values.dtype)
        count = weight.sum(dim=dims, keepdim=True)
        total = (values * weight).sum(dim=dims, keepdim=True)

        # Empty sign groups can occur early in training. Falling back to zero keeps
        # the transform finite without coupling the two groups.
        return torch.where(condition=count > 0, input=total / count.clamp_min(1.0), other=torch.zeros_like(total))

    def _relax(self, values: Tensor, hard: Tensor) -> Tensor:
        """Construct the differentiable sign-group relaxation for one batch."""
        positive = hard > 0
        negative = ~positive
        positive_mean = self._group_mean(values, positive)
        negative_mean = self._group_mean(values, negative)
        centered = torch.where(condition=positive, input=values - positive_mean, other=values - negative_mean)
        return hard + centered

    def forward(self, logits: Tensor, *, force_hard: bool = False) -> QuantizerOutput:
        """Quantize logits into hard signs and a training-time relaxed representation.

        Args:
            logits: Pre-quantization activations.
            force_hard: Whether to bypass the training relaxation.

        Returns:
            Hard and relaxed codes, probabilities, and the concentration loss.

        """
        probabilities = torch.sigmoid(logits)
        deterministic_hard = self.hard_from_logits(logits)

        if force_hard or not self.training:
            hard = deterministic_hard
            relaxed = hard
        else:
            if self.stochastic:
                sample_values = probabilities - torch.rand_like(probabilities)
                hard = torch.where(
                    condition=sample_values > 0, input=torch.ones_like(logits), other=-torch.ones_like(logits)
                )
                group_values = sample_values
            else:
                hard = deterministic_hard
                group_values = logits

            relaxed = self._relax(group_values, hard)

        concentration_loss = torch.square(relaxed - hard).mean()

        return QuantizerOutput(
            relaxed=relaxed,
            hard=hard,
            probabilities=probabilities,
            concentration_loss=concentration_loss,
        )
