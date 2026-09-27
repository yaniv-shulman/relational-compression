"""Convolutional binary autoencoder components for image experiments."""

from dataclasses import dataclass
from typing import cast

import torch
from compressai.layers import GDN1
from torch import Tensor, nn

from relational_compression.models.quantizers import MeanGroupBinaryQuantizer, QuantizerOutput


class ResidualStack(nn.Module):
    """Apply residual convolutional blocks at a fixed channel width."""

    def __init__(self, channels: int, layers: int, hidden: int) -> None:
        """Initialize the residual blocks.

        Args:
            channels: Input and output channel count for each block.
            layers: Number of residual blocks.
            hidden: Hidden channel count within each block.

        """
        super().__init__()

        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(channels, hidden, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(hidden, channels, 1),
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Transform feature maps through the residual stack."""
        for layer in self.layers:
            x = x + layer(x)

        return torch.relu(x)


class Encoder(nn.Module):
    """Encode an RGB image into spatial binary-code logits."""

    def __init__(
        self,
        in_channels: int,
        hidden: int,
        downsample_layers: int,
        residual_layers: int,
        residual_hidden: int,
        bits: int,
    ) -> None:
        """Initialize the convolutional analysis transform.

        Args:
            in_channels: Number of image input channels.
            hidden: Main convolutional channel width.
            downsample_layers: Number of stride-two analysis stages.
            residual_layers: Number of residual blocks after analysis.
            residual_hidden: Hidden width within each residual block.
            bits: Number of output binary-logit channels.

        """
        super().__init__()
        modules: list[nn.Module] = []
        current = in_channels

        for i in range(downsample_layers):
            out = hidden // 2 if i == 0 else hidden
            modules.extend([nn.Conv2d(current, out, 4, stride=2, padding=1), GDN1(out, inverse=False)])
            current = out

        modules.append(nn.Conv2d(current, hidden, 3, padding=1))

        self.conv = nn.Sequential(*modules)
        self.residual = ResidualStack(channels=hidden, layers=residual_layers, hidden=residual_hidden)
        self.to_logits = nn.Conv2d(hidden, bits, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Encode an image batch into spatial binary-logit maps."""
        return cast(Tensor, self.to_logits(self.residual(self.conv(x))))


class Decoder(nn.Module):
    """Decode spatial binary codes into image logits."""

    def __init__(
        self,
        bits: int,
        hidden: int,
        upsample_layers: int,
        residual_layers: int,
        residual_hidden: int,
        out_channels: int,
    ) -> None:
        """Initialize the convolutional synthesis transform.

        Args:
            bits: Number of binary-code channels.
            hidden: Main convolutional channel width.
            upsample_layers: Number of transposed-convolution stages.
            residual_layers: Number of residual blocks before synthesis.
            residual_hidden: Hidden width within each residual block.
            out_channels: Number of reconstructed image channels.

        """
        super().__init__()
        self.in_conv = nn.Conv2d(bits, hidden, 3, padding=1)
        self.residual = ResidualStack(channels=hidden, layers=residual_layers, hidden=residual_hidden)
        modules: list[nn.Module] = []
        current = hidden

        for i in range(upsample_layers):
            final = i == upsample_layers - 1
            out = out_channels if final else (hidden // 2 if i == upsample_layers - 2 else hidden)
            modules.append(nn.ConvTranspose2d(current, out, 4, stride=2, padding=1))

            if not final:
                modules.append(GDN1(out, inverse=True))

            current = out

        self.upconv = nn.Sequential(*modules)

    def forward(self, z: Tensor) -> Tensor:
        """Decode a batch of spatial latent codes into image logits."""
        return cast(Tensor, self.upconv(self.residual(self.in_conv(z))))


@dataclass
class AutoencoderOutput:
    """Collect an image reconstruction, encoder logits, and quantizer result."""

    reconstruction: Tensor
    logits: Tensor
    quantizer: QuantizerOutput


class BinaryImageAutoencoder(nn.Module):
    """Learn a hard-sign spatial image bottleneck with a convolutional decoder."""

    threshold_scale_ema: Tensor
    threshold_scale_ema_initialized: Tensor

    def __init__(
        self,
        *,
        bits: int,
        hidden: int,
        downsample_layers: int,
        residual_layers: int,
        residual_hidden: int,
        stochastic: bool,
        grouping: str,
    ) -> None:
        """Initialize the binary image autoencoder.

        Args:
            bits: Number of binary latent channels.
            hidden: Main convolutional channel width.
            downsample_layers: Number of analysis and synthesis stages.
            residual_layers: Number of residual blocks per transform.
            residual_hidden: Hidden width within each residual block.
            stochastic: Whether the quantizer samples signs while training.
            grouping: Sign-group coupling rule for the quantizer.

        """
        super().__init__()
        self.encoder = Encoder(
            in_channels=3,
            hidden=hidden,
            downsample_layers=downsample_layers,
            residual_layers=residual_layers,
            residual_hidden=residual_hidden,
            bits=bits,
        )

        self.quantizer = MeanGroupBinaryQuantizer(
            stochastic=stochastic,
            grouping=grouping,
        )

        self.decoder = Decoder(
            bits=bits,
            hidden=hidden,
            upsample_layers=downsample_layers,
            residual_layers=residual_layers,
            residual_hidden=residual_hidden,
            out_channels=3,
        )
        self.register_buffer("threshold_scale_ema", torch.ones(bits))
        self.register_buffer("threshold_scale_ema_initialized", torch.tensor(False))

    def threshold_scales(
        self,
        logits: Tensor,
        *,
        ema_decay: float,
        scale_min: float,
        update: bool,
    ) -> Tensor:
        """Return per-bit threshold scales, optionally updating their moving average.

        Args:
            logits: Current pre-quantization activations.
            ema_decay: Moving-average retention factor.
            scale_min: Lower bound that keeps normalization finite.
            update: Whether to update the moving average in training mode.

        Returns:
            Broadcastable threshold scales for the supplied logits.

        """
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if scale_min <= 0.0:
            raise ValueError("scale_min must be positive")

        if update and self.training:
            with torch.no_grad():
                batch_scale = logits.detach().float().square().mean(dim=(0, 2, 3)).sqrt().clamp_min(scale_min)
                if bool(self.threshold_scale_ema_initialized):
                    self.threshold_scale_ema = self.threshold_scale_ema * ema_decay + batch_scale * (1.0 - ema_decay)
                else:
                    self.threshold_scale_ema = batch_scale
                    self.threshold_scale_ema_initialized = torch.tensor(True, device=batch_scale.device)

        return cast(
            Tensor,
            self.threshold_scale_ema.clamp_min(scale_min)
            .to(device=logits.device, dtype=logits.dtype)
            .view(1, -1, 1, 1),
        )

    def forward(self, x: Tensor, *, force_hard: bool = False) -> AutoencoderOutput:
        """Encode, quantize, and reconstruct an image batch.

        Args:
            x: Input image batch.
            force_hard: Whether to decode deterministic hard signs.

        Returns:
            Reconstruction, logits, and quantization diagnostics.

        """
        logits = self.encoder(x)
        q = self.quantizer(logits, force_hard=force_hard)
        latent = q.hard if force_hard else q.relaxed
        reconstruction = torch.sigmoid(self.decoder(latent))
        return AutoencoderOutput(reconstruction=reconstruction, logits=logits, quantizer=q)

    def reconstruct_hard(self, x: Tensor) -> Tensor:
        """Reconstruct images using deterministic hard signs at the bottleneck."""
        return self.forward(x, force_hard=True).reconstruction
