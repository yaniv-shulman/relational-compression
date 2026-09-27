"""Graph partition encoders used by the inductive normalized-cut study."""

from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, SAGEConv, global_mean_pool
from torch_geometric.utils import to_dense_batch


def _make_activation(name: str) -> nn.Module:
    """Create an activation module from a configured name."""
    normalized = name.lower()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation_name: {name}")


def _make_output_head(
    *,
    input_dim: int,
    hidden_dim: int,
    num_partitions: int,
    activation_name: str,
    hidden_multiplier: int = 0,
) -> nn.Module:
    """Create the configured partition-logit output head."""
    if hidden_multiplier < 0:
        raise ValueError("output_head_hidden_multiplier must be non-negative")
    if hidden_multiplier == 0:
        return nn.Linear(input_dim, num_partitions)
    output_hidden_dim = int(hidden_multiplier) * hidden_dim
    return nn.Sequential(
        nn.Linear(input_dim, output_hidden_dim),
        _make_activation(activation_name),
        nn.Linear(output_hidden_dim, num_partitions),
    )


@dataclass(frozen=True)
class PartitionEncoderOutput:
    """Collect soft and hard assignments emitted by a graph encoder."""

    logits: Tensor
    probabilities: Tensor
    hard_ids: Tensor


class GraphSAGEResidualBlock(nn.Module):
    """Apply a residual GraphSAGE update at one hidden width."""

    def __init__(self, hidden_dim: int, *, activation_name: str = "gelu") -> None:
        """Initialize the GraphSAGE residual block."""
        super().__init__()
        self.conv = SAGEConv(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = _make_activation(activation_name)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Update node features using graph-neighborhood information."""
        update = self.conv(x, edge_index)
        update = self.norm(update)
        return x + cast(Tensor, self.activation(update))


class GraphPartitionEncoder(nn.Module):
    """Produce categorical graph partitions with GraphSAGE message passing."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_partitions: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 4,
        assignment_temperature: float = 1.0,
        use_graph_context: bool = True,
        activation_name: str = "gelu",
        output_head_hidden_multiplier: int = 0,
    ) -> None:
        """Initialize the GraphSAGE partition encoder."""
        super().__init__()
        if num_partitions < 2:
            raise ValueError("num_partitions must be at least 2")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if assignment_temperature <= 0.0:
            raise ValueError("assignment_temperature must be positive")

        self.num_partitions = int(num_partitions)
        self.assignment_temperature = float(assignment_temperature)
        self.use_graph_context = bool(use_graph_context)
        self.activation = _make_activation(activation_name)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            GraphSAGEResidualBlock(hidden_dim, activation_name=activation_name) for _ in range(num_layers)
        )
        output_dim = hidden_dim * 2 if self.use_graph_context else hidden_dim
        self.output = _make_output_head(
            input_dim=output_dim,
            hidden_dim=hidden_dim,
            num_partitions=num_partitions,
            activation_name=activation_name,
            hidden_multiplier=int(output_head_hidden_multiplier),
        )

    def forward(self, data: Data) -> PartitionEncoderOutput:
        """Encode graph nodes into soft and hard partition assignments."""
        x = cast(Tensor, self.activation(self.input_projection(data.x.float())))
        for layer in self.layers:
            x = layer(x, data.edge_index)

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        if self.use_graph_context:
            context = global_mean_pool(x, batch)
            x = torch.cat((x, context[batch]), dim=-1)

        logits = cast(Tensor, self.output(x))
        probabilities = torch.softmax(logits / self.assignment_temperature, dim=-1)
        hard_ids = probabilities.argmax(dim=-1)
        return PartitionEncoderOutput(logits=logits, probabilities=probabilities, hard_ids=hard_ids)


class SoftmaxEfficientAttention(nn.Module):
    """Factorized softmax Efficient Attention with explicit padding masks.

    Queries are normalized over feature channels, keys are normalized over
    valid sequence positions, and the resulting context is formed without
    constructing an N x N attention map.
    """

    def __init__(self, hidden_dim: int, *, heads: int = 4, dropout: float = 0.0) -> None:
        """Initialize factorized multi-head attention."""
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if heads < 1:
            raise ValueError("heads must be positive")
        if hidden_dim % heads != 0:
            raise ValueError("hidden_dim must be divisible by heads")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.hidden_dim = int(hidden_dim)
        self.heads = int(heads)
        self.head_dim = int(hidden_dim // heads)
        self.to_qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        """Apply masked factorized attention to dense node features."""
        if x.ndim != 3:
            raise ValueError("x must have shape [batch, nodes, hidden_dim]")
        if x.shape[-1] != self.hidden_dim:
            raise ValueError("x hidden dimension does not match the attention module")
        if mask is not None and mask.shape != x.shape[:2]:
            raise ValueError("mask must have shape [batch, nodes]")

        batch_size, num_nodes, _ = x.shape
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q = q.reshape(batch_size, num_nodes, self.heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(batch_size, num_nodes, self.heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(batch_size, num_nodes, self.heads, self.head_dim).permute(0, 2, 1, 3)
        output_dtype = v.dtype

        # Keep the factorized softmax and contractions in FP32 for numerical
        # stability, while returning to the model dtype afterward.
        with torch.autocast(device_type=x.device.type, enabled=False):
            q = q.float()
            k = k.float()
            v = v.float()
            valid_mask = None if mask is None else mask.to(device=x.device, dtype=torch.bool)

            q_norm = torch.softmax(q, dim=-1)
            if valid_mask is not None:
                invalid_keys = ~valid_mask[:, None, :, None]
                k = k.masked_fill(invalid_keys, torch.finfo(k.dtype).min)
                v = v * valid_mask[:, None, :, None].to(dtype=v.dtype)

            k_norm = torch.softmax(k, dim=-2)
            if valid_mask is not None:
                k_norm = k_norm * valid_mask[:, None, :, None].to(dtype=k_norm.dtype)

            context = torch.einsum("bhnd,bhne->bhde", k_norm, v)
            out = torch.einsum("bhnd,bhde->bhne", q_norm, context)
            if valid_mask is not None:
                out = out * valid_mask[:, None, :, None].to(dtype=out.dtype)

        out = out.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, self.hidden_dim)
        out = out.to(dtype=output_dtype)
        return cast(Tensor, self.dropout(self.output(out)))


class GraphGPSEfficientAttentionBlock(nn.Module):
    """GraphGPS-style local GIN + global Efficient Attention block."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        heads: int = 4,
        dropout: float = 0.0,
        ffn_multiplier: int = 2,
        activation_name: str = "gelu",
    ) -> None:
        """Initialize the local-message and global-attention block."""
        super().__init__()
        if ffn_multiplier < 1:
            raise ValueError("ffn_multiplier must be positive")
        self.dropout = float(dropout)
        ffn_dim = int(ffn_multiplier) * hidden_dim
        self.local_conv = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                _make_activation(activation_name),
                nn.Linear(hidden_dim, hidden_dim),
            ),
            train_eps=True,
        )
        self.global_attention = SoftmaxEfficientAttention(hidden_dim, heads=heads, dropout=dropout)
        # Explicit nn.LayerNorm gives node-wise Transformer-style normalization.
        self.local_norm = nn.LayerNorm(hidden_dim)
        self.global_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            _make_activation(activation_name),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
        """Combine local graph updates with masked global attention."""
        local_update = self.local_conv(x, edge_index)
        local_update = F.dropout(local_update, p=self.dropout, training=self.training)
        local = self.local_norm(x + local_update)

        dense_x, mask = to_dense_batch(x, batch)
        global_update = self.global_attention(dense_x, mask=mask)[mask]
        global_update = F.dropout(global_update, p=self.dropout, training=self.training)
        global_x = self.global_norm(x + global_update)
        out = local + global_x
        return cast(Tensor, self.output_norm(out + self.ffn(out)))


class GraphGPSEfficientAttentionPartitionEncoder(nn.Module):
    """Produce graph partitions with local GIN and global efficient attention."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_partitions: int = 8,
        hidden_dim: int = 128,
        num_layers: int = 4,
        assignment_temperature: float = 1.0,
        use_graph_context: bool = True,
        gps_heads: int = 4,
        gps_dropout: float = 0.0,
        gps_ffn_multiplier: int = 2,
        activation_name: str = "gelu",
        output_head_hidden_multiplier: int = 0,
    ) -> None:
        """Initialize the GraphGPS efficient-attention partition encoder."""
        super().__init__()
        if num_partitions < 2:
            raise ValueError("num_partitions must be at least 2")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if assignment_temperature <= 0.0:
            raise ValueError("assignment_temperature must be positive")
        if gps_heads < 1:
            raise ValueError("gps_heads must be positive")
        if hidden_dim % gps_heads != 0:
            raise ValueError("hidden_dim must be divisible by gps_heads")
        if gps_dropout < 0.0 or gps_dropout >= 1.0:
            raise ValueError("gps_dropout must be in [0, 1)")
        if gps_ffn_multiplier < 1:
            raise ValueError("gps_ffn_multiplier must be positive")

        self.num_partitions = int(num_partitions)
        self.assignment_temperature = float(assignment_temperature)
        self.use_graph_context = bool(use_graph_context)
        self.activation = _make_activation(activation_name)
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            GraphGPSEfficientAttentionBlock(
                hidden_dim,
                heads=int(gps_heads),
                dropout=float(gps_dropout),
                ffn_multiplier=int(gps_ffn_multiplier),
                activation_name=activation_name,
            )
            for _ in range(num_layers)
        )
        output_dim = hidden_dim * 2 if self.use_graph_context else hidden_dim
        self.output = _make_output_head(
            input_dim=output_dim,
            hidden_dim=hidden_dim,
            num_partitions=num_partitions,
            activation_name=activation_name,
            hidden_multiplier=int(output_head_hidden_multiplier),
        )

    def forward(self, data: Data) -> PartitionEncoderOutput:
        """Encode graph nodes into soft and hard partition assignments."""
        x = cast(Tensor, self.activation(self.input_projection(data.x.float())))
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        for layer in self.layers:
            x = layer(x, data.edge_index, batch)

        if self.use_graph_context:
            context = global_mean_pool(x, batch)
            x = torch.cat((x, context[batch]), dim=-1)

        logits = self.output(x)
        probabilities = torch.softmax(logits / self.assignment_temperature, dim=-1)
        hard_ids = probabilities.argmax(dim=-1)
        return PartitionEncoderOutput(logits=logits, probabilities=probabilities, hard_ids=hard_ids)


def make_model(config: Any, *, input_dim: int) -> GraphPartitionEncoder | GraphGPSEfficientAttentionPartitionEncoder:
    """Build the configured inductive graph partition encoder."""
    model_name = str(getattr(config, "model_name", "graphsage")).lower()
    if model_name in {"graphsage", "sage"}:
        return GraphPartitionEncoder(
            input_dim=input_dim,
            num_partitions=int(config.num_partitions),
            hidden_dim=int(config.hidden_dim),
            num_layers=int(config.num_layers),
            assignment_temperature=float(config.assignment_temperature),
            use_graph_context=bool(getattr(config, "use_graph_context", True)),
            activation_name=str(getattr(config, "activation_name", "gelu")),
            output_head_hidden_multiplier=int(getattr(config, "output_head_hidden_multiplier", 0)),
        )
    if model_name in {"graphgps_efficient_attention", "gps_efficient_attention", "gin_efficient_attention"}:
        return GraphGPSEfficientAttentionPartitionEncoder(
            input_dim=input_dim,
            num_partitions=int(config.num_partitions),
            hidden_dim=int(config.hidden_dim),
            num_layers=int(config.num_layers),
            assignment_temperature=float(config.assignment_temperature),
            use_graph_context=bool(getattr(config, "use_graph_context", True)),
            gps_heads=int(getattr(config, "gps_heads", 4)),
            gps_dropout=float(getattr(config, "gps_dropout", 0.0)),
            gps_ffn_multiplier=int(getattr(config, "gps_ffn_multiplier", 2)),
            activation_name=str(getattr(config, "activation_name", "gelu")),
            output_head_hidden_multiplier=int(getattr(config, "output_head_hidden_multiplier", 0)),
        )
    raise ValueError(f"Unsupported inductive normalized-cut model_name: {model_name}")
