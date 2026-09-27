"""Student, teacher, and decoder models for teacher-defined image compression."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from relational_compression.models.image_autoencoder import Decoder, Encoder

DINO_VITS8_SOURCE_REVISION = "7c446df5b9f45747937fb0d72314eb9f7b66930a"
DINO_VITS8_REPO = f"facebookresearch/dino:{DINO_VITS8_SOURCE_REVISION}"
DINO_VITS8_MODEL = "dino_vits8"
DINO_VITS8_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain.pth"
DINO_VITS8_PATCH_SIZE = 8
DINO_INPUT_MEAN = (0.485, 0.456, 0.406)
DINO_INPUT_STD = (0.229, 0.224, 0.225)


@dataclass
class StudentOutput:
    """Collect logits and hard signs emitted by the student encoder."""

    logits: Tensor
    hard: Tensor


class BinaryPatchEncoder(nn.Module):
    """Convolutional binary encoder without a reconstruction decoder."""

    def __init__(
        self,
        *,
        bits: int,
        hidden: int,
        downsample_layers: int,
        residual_layers: int,
        residual_hidden: int,
    ) -> None:
        """Initialize the instance."""
        super().__init__()
        self.bits = bits
        self.encoder = Encoder(
            in_channels=3,
            hidden=hidden,
            downsample_layers=downsample_layers,
            residual_layers=residual_layers,
            residual_hidden=residual_hidden,
            bits=bits,
        )

    def forward(self, images: Tensor, *, force_hard: bool = False) -> StudentOutput:
        """Run the module forward pass."""
        del force_hard
        logits = self.encoder(images)
        hard = torch.where(condition=logits > 0.0, input=torch.ones_like(logits), other=-torch.ones_like(logits))
        return StudentOutput(logits=logits, hard=hard)

    def hard_codes(self, images: Tensor) -> Tensor:
        """Compute hard codes."""
        return self.forward(images, force_hard=True).hard


class DinoVitS8PatchTeacher(nn.Module):
    """Frozen DINO ViT-S/8 patch-token teacher.

    The default loader uses the official ``facebookresearch/dino`` PyTorch Hub
    entry ``dino_vits8`` pinned to ``DINO_VITS8_SOURCE_REVISION`` and the
    checkpoint URL recorded in ``DINO_VITS8_CHECKPOINT_URL``.
    """

    def __init__(
        self,
        backbone: nn.Module,
        *,
        patch_size: int = DINO_VITS8_PATCH_SIZE,
        input_mean: tuple[float, float, float] = DINO_INPUT_MEAN,
        input_std: tuple[float, float, float] = DINO_INPUT_STD,
    ) -> None:
        """Initialize the instance."""
        super().__init__()
        self.backbone = backbone
        self.input_mean: Tensor
        self.input_std: Tensor
        self.patch_size = patch_size
        self.register_buffer("input_mean", torch.tensor(input_mean).view(1, 3, 1, 1))
        self.register_buffer("input_std", torch.tensor(input_std).view(1, 3, 1, 1))
        self.freeze()

    @classmethod
    def from_torch_hub(
        cls,
        *,
        cache_dir: Path,
        allow_download: bool = True,
        repo: str = DINO_VITS8_REPO,
        model_name: str = DINO_VITS8_MODEL,
    ) -> "DinoVitS8PatchTeacher":
        """Load a frozen DINO teacher from Torch Hub."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not allow_download:
            checkpoint_path = cache_dir / "checkpoints" / Path(DINO_VITS8_CHECKPOINT_URL).name
            if not (cache_dir / torch_hub_repo_cache_name(repo)).exists() or not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"DINO torch.hub repo and checkpoint are not cached under {cache_dir}; "
                    "rerun with teacher download enabled"
                )

        previous_hub_dir = torch.hub.get_dir()
        torch.hub.set_dir(str(cache_dir))
        try:
            backbone = torch.hub.load(
                repo_or_dir=repo,
                model=model_name,
                pretrained=True,
                trust_repo=True,
                skip_validation=True,
            )
        finally:
            torch.hub.set_dir(previous_hub_dir)

        return cls(cast(nn.Module, backbone))

    def freeze(self) -> None:
        """Disable gradient updates for all teacher parameters."""
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "DinoVitS8PatchTeacher":
        """Train the requested values."""
        super().train(False)
        return self

    def _normalized_input(self, images: Tensor) -> Tensor:
        """Normalize images with the teacher input statistics."""
        return (images - self.input_mean.to(device=images.device, dtype=images.dtype)) / self.input_std.to(
            device=images.device,
            dtype=images.dtype,
        )

    def _extract_sequence(self, images: Tensor) -> Tensor:
        """Extract the sequence representation produced by the teacher backbone."""
        normalized = self._normalized_input(images)
        if hasattr(self.backbone, "get_intermediate_layers"):
            sequence = cast(Any, self.backbone).get_intermediate_layers(normalized, n=1)[0]
        else:
            sequence = cast(Any, self.backbone)(normalized)

        if isinstance(sequence, (tuple, list)):
            sequence = sequence[0]
        if sequence.ndim != 3:
            raise ValueError(f"expected teacher sequence with shape [B, tokens, D], got {tuple(sequence.shape)}")

        return cast(Tensor, sequence)

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        """Run the module forward pass."""
        sequence = self._extract_sequence(images)
        expected_patches = (images.shape[-2] // self.patch_size) * (images.shape[-1] // self.patch_size)
        patch_tokens = sequence[:, 1:, :] if sequence.shape[1] == expected_patches + 1 else sequence
        if patch_tokens.shape[1] != expected_patches:
            raise ValueError(
                f"expected {expected_patches} DINO patch tokens for input shape {tuple(images.shape)}, "
                f"got {patch_tokens.shape[1]}"
            )

        grid_h = images.shape[-2] // self.patch_size
        grid_w = images.shape[-1] // self.patch_size
        tokens = patch_tokens.reshape(images.shape[0], grid_h, grid_w, patch_tokens.shape[-1])
        return cast(Tensor, F.normalize(tokens, dim=-1))


class FixedRandomPatchTeacher(nn.Module):
    """Network-free frozen teacher used only for tests and smoke runs."""

    def __init__(self, *, embedding_dim: int = 32, patch_size: int = DINO_VITS8_PATCH_SIZE, seed: int = 0) -> None:
        """Initialize the instance."""
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        projection = torch.randn(3, embedding_dim, generator=generator) / math.sqrt(3.0)
        bias = torch.randn(embedding_dim, generator=generator) / math.sqrt(embedding_dim)
        self.patch_size = patch_size
        self.projection: Tensor
        self.bias: Tensor
        self.register_buffer("projection", projection)
        self.register_buffer("bias", bias)

    def train(self, mode: bool = True) -> "FixedRandomPatchTeacher":
        """Train the requested values."""
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, images: Tensor) -> Tensor:
        """Run the module forward pass."""
        pooled = F.avg_pool2d(images, kernel_size=self.patch_size, stride=self.patch_size)
        tokens = torch.einsum("bchw,cd->bhwd", pooled, self.projection.to(device=images.device, dtype=images.dtype))
        tokens = tokens + self.bias.to(device=images.device, dtype=images.dtype)
        return F.normalize(tokens, dim=-1)


def make_student(config: Any) -> BinaryPatchEncoder:
    """Create make student."""
    return BinaryPatchEncoder(
        bits=int(config.bits),
        hidden=int(config.hidden),
        downsample_layers=int(config.downsample_layers),
        residual_layers=int(config.residual_layers),
        residual_hidden=int(config.residual_hidden),
    )


def torch_hub_repo_cache_name(repo: str) -> str:
    """Compute torch hub repo cache name."""
    repo_without_scheme = repo.removeprefix("https://github.com/").removeprefix("http://github.com/")
    repo_name, _, revision = repo_without_scheme.partition(":")
    parts = repo_name.split("/")
    if len(parts) < 2:
        raise ValueError(f"unsupported torch.hub repo name: {repo}")
    owner, name = parts[0], parts[1]
    revision = revision or "main"
    return f"{owner}_{name}_{revision.replace('/', '_')}"


def make_decoder(config: Any) -> Decoder:
    """Create make decoder."""
    return Decoder(
        bits=int(config.bits),
        hidden=int(config.hidden),
        upsample_layers=int(config.downsample_layers),
        residual_layers=int(config.residual_layers),
        residual_hidden=int(config.residual_hidden),
        out_channels=3,
    )
