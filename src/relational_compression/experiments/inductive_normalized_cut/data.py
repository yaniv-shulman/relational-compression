"""Experiment support code for relational compression studies."""

import json
import random
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, subgraph

PREPROCESSING_VERSION = "lcc_structural_hutch_rwse_v3"
SPLITS = ("train", "val", "test")
_SPLIT_SEED_OFFSETS = {"train": 0, "val": 1_000_000, "test": 2_000_000}


@dataclass(frozen=True)
class PreprocessConfig:
    """Configure graph preprocessing and feature construction."""

    dataset_root_dir: Path
    min_nodes_per_graph: int = 512
    max_nodes_per_graph: int | None = None
    random_walk_steps: int = 16
    random_walk_num_probes: int = 8
    positional_encoding_seed: int = 1337
    preprocessing_version: str = PREPROCESSING_VERSION
    include_directed_features: bool = False
    pagerank_alpha: float = 0.85
    pagerank_max_iter: int = 50
    pagerank_tolerance: float = 1e-6

    @property
    def cache_dir(self) -> Path:
        """Return the cache directory for this preprocessing configuration."""
        max_nodes = "none" if self.max_nodes_per_graph is None else str(self.max_nodes_per_graph)
        name = (
            f"{self.preprocessing_version}_min{self.min_nodes_per_graph}_max{max_nodes}_"
            f"rw{self.random_walk_steps}_probes{self.random_walk_num_probes}_seed{self.positional_encoding_seed}"
        )
        if self.include_directed_features:
            alpha = f"{float(self.pagerank_alpha):.6g}".replace("-", "m").replace(".", "p")
            tolerance = f"{float(self.pagerank_tolerance):.6g}".replace("-", "m").replace(".", "p")
            name = f"{name}_directed_pralpha{alpha}_priters{self.pagerank_max_iter}_prtol{tolerance}"
        return self.dataset_root_dir / "relational_compression_cache" / name


@dataclass(frozen=True)
class GraphMetadata:
    """Store preprocessing metadata for one graph."""

    split: str
    original_graph_index: int
    original_num_nodes: int
    original_num_edges: int
    cleaned_num_nodes: int
    cleaned_num_edges: int
    lcc_num_nodes: int
    lcc_num_edges: int
    lcc_fraction: float
    retained: bool
    exclusion_reason: str | None


class CachedMalNetTinyNormalizedCutDataset(Dataset[Data]):
    """Load cached MalNet-Tiny graphs for normalized-cut experiments."""

    def __init__(self, cache_dir: Path, split: str) -> None:
        """Initialize the instance."""
        if split not in SPLITS:
            raise ValueError(f"Unsupported split: {split}")
        self.cache_dir = Path(cache_dir)
        self.split = split
        index_path = self.cache_dir / split / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Missing MalNet-Tiny Relational Compression cache index: {index_path}")
        self.index: list[dict[str, Any]] = json.loads(index_path.read_text())

    def __len__(self) -> int:
        """Return the number of available graph records."""
        return len(self.index)

    def __getitem__(self, index: int) -> Data:
        """Load one cached graph record."""
        item = self.index[index]
        # Cached PyG Data records are custom objects rather than tensor/dict checkpoints.
        data = torch.load(self.cache_dir / self.split / item["path"], weights_only=False)
        if "y" in data:
            delattr(data, "y")
        return data


def seed_everything(seed: int) -> None:
    """Seed everything."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _connected_components(edge_index: Tensor, num_nodes: int) -> list[list[int]]:
    """Find connected components in an undirected graph."""
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for source, target in edge_index.t().cpu().tolist():
        adjacency[source].append(target)

    seen = [False] * num_nodes
    components: list[list[int]] = []
    for start in range(num_nodes):
        if seen[start]:
            continue
        seen[start] = True
        queue: deque[int] = deque([start])
        component: list[int] = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in adjacency[node]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    queue.append(neighbor)
        components.append(component)
    return components


def clean_to_undirected_lcc_with_nodes(
    edge_index: Tensor, num_nodes: int
) -> tuple[Tensor, dict[str, int | float], Tensor]:
    """Clean to undirected lcc with nodes."""
    if edge_index.numel() == 0:
        raise ValueError("Cannot extract an LCC from an edgeless graph")

    edge_index = edge_index.to(dtype=torch.long, device="cpu")
    non_self = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, non_self]
    reversed_edges = edge_index.flip(0)
    undirected = torch.cat((edge_index, reversed_edges), dim=1)
    undirected = coalesce(undirected, num_nodes=num_nodes)
    cleaned_edges = int(undirected.shape[1] // 2)
    components = _connected_components(edge_index=undirected, num_nodes=num_nodes)
    largest = max(components, key=len)
    largest_tensor = torch.tensor(sorted(largest), dtype=torch.long)
    lcc_edge_index, _ = subgraph(largest_tensor, undirected, relabel_nodes=True, num_nodes=num_nodes)
    lcc_edge_index = coalesce(lcc_edge_index, num_nodes=largest_tensor.numel())
    metadata = {
        "cleaned_num_nodes": int(num_nodes),
        "cleaned_num_edges": cleaned_edges,
        "lcc_num_nodes": int(largest_tensor.numel()),
        "lcc_num_edges": int(lcc_edge_index.shape[1] // 2),
        "lcc_fraction": float(largest_tensor.numel() / max(1, num_nodes)),
    }
    return lcc_edge_index, metadata, largest_tensor


def clean_to_undirected_lcc(edge_index: Tensor, num_nodes: int) -> tuple[Tensor, dict[str, int | float]]:
    """Clean to undirected lcc."""
    lcc_edge_index, metadata, _ = clean_to_undirected_lcc_with_nodes(edge_index=edge_index, num_nodes=num_nodes)
    return lcc_edge_index, metadata


def _simple_directed_edges(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Remove self-loops and coalesce directed graph edges."""
    edge_index = edge_index.to(dtype=torch.long, device="cpu")
    if edge_index.numel() == 0:
        return edge_index.reshape(2, 0)
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]
    return cast(Tensor, coalesce(edge_index, num_nodes=num_nodes))


def directed_pagerank(
    edge_index: Tensor,
    num_nodes: int,
    *,
    alpha: float = 0.85,
    max_iter: int = 50,
    tolerance: float = 1e-6,
) -> Tensor:
    """Compute PageRank features for a directed graph."""
    if num_nodes < 1:
        raise ValueError("num_nodes must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("pagerank_alpha must be in (0, 1)")
    if max_iter < 1:
        raise ValueError("pagerank_max_iter must be positive")
    if tolerance < 0.0:
        raise ValueError("pagerank_tolerance must be non-negative")

    edge_index = _simple_directed_edges(edge_index=edge_index, num_nodes=num_nodes)
    source, target = edge_index
    out_degree = torch.bincount(source, minlength=num_nodes).float()
    rank = torch.full(size=(num_nodes,), fill_value=1.0 / float(num_nodes), dtype=torch.float32)
    teleport = (1.0 - float(alpha)) / float(num_nodes)

    for _ in range(int(max_iter)):
        next_rank = torch.full_like(input=rank, fill_value=teleport)
        if source.numel() > 0:
            next_rank.index_add_(0, target, float(alpha) * rank[source] / out_degree[source].clamp_min(1.0))
        dangling_mass = rank[out_degree == 0].sum()
        if float(dangling_mass) > 0.0:
            next_rank += float(alpha) * dangling_mass / float(num_nodes)
        delta = torch.abs(next_rank - rank).sum()
        rank = next_rank
        if float(delta) <= float(tolerance):
            break
    return rank / rank.sum().clamp_min(torch.finfo(rank.dtype).tiny)


def directed_structural_node_features(
    edge_index: Tensor,
    num_nodes: int,
    *,
    pagerank_alpha: float = 0.85,
    pagerank_max_iter: int = 50,
    pagerank_tolerance: float = 1e-6,
) -> Tensor:
    """Construct directed structural node features."""
    edge_index = _simple_directed_edges(edge_index=edge_index, num_nodes=num_nodes)
    source, target = edge_index
    in_degree = torch.bincount(target, minlength=num_nodes).float()
    out_degree = torch.bincount(source, minlength=num_nodes).float()
    total_degree = in_degree + out_degree
    directed_degrees = (in_degree, out_degree, total_degree)
    pagerank = directed_pagerank(
        edge_index=edge_index,
        num_nodes=num_nodes,
        alpha=pagerank_alpha,
        max_iter=pagerank_max_iter,
        tolerance=pagerank_tolerance,
    )
    # Use PageRank relative to the uniform 1/N baseline so the feature has a
    # stable cross-graph scale: uniform PageRank maps to one for every graph.
    relative_pagerank = pagerank * float(num_nodes)
    return torch.stack(
        (
            *directed_degrees,
            *(torch.log1p(values) for values in directed_degrees),
            *(values / values.max().clamp_min(1.0) for values in directed_degrees),
            relative_pagerank,
        ),
        dim=-1,
    )


def random_walk_transition_apply(edge_index: Tensor, num_nodes: int, values: Tensor) -> Tensor:
    """Apply P = D^{-1} A to one or more node signals without forming P."""
    if values.shape[0] != num_nodes:
        raise ValueError("values first dimension must match num_nodes")
    source, target = edge_index
    degree = torch.bincount(source, minlength=num_nodes).to(device=values.device, dtype=values.dtype).clamp_min(1.0)
    if values.ndim == 1:
        output = torch.zeros_like(values)
        output.index_add_(0, source, values[target] / degree[source])
        return output
    if values.ndim == 2:
        output = torch.zeros_like(values)
        output.index_add_(0, source, values[target] / degree[source].unsqueeze(-1))
        return output
    raise ValueError("values must have shape [num_nodes] or [num_nodes, num_signals]")


def _rademacher_probes(num_nodes: int, num_probes: int, seed: int) -> Tensor:
    """Generate seeded Rademacher probe signals."""
    if num_probes < 1:
        raise ValueError("random_walk_num_probes must be positive")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return (
        torch.randint(low=0, high=2, size=(num_nodes, num_probes), generator=generator, dtype=torch.float32)
        .mul_(2.0)
        .sub_(1.0)
    )


def hutchinson_random_walk_return_probabilities(
    edge_index: Tensor,
    num_nodes: int,
    *,
    steps: int,
    num_probes: int,
    seed: int,
    probes: Tensor | None = None,
) -> Tensor:
    """Estimate diag(P^t) for t=1..steps using Rademacher Hutchinson probes."""
    if steps <= 0:
        return torch.empty(num_nodes, 0, dtype=torch.float32)
    if probes is None:
        probes = _rademacher_probes(num_nodes=num_nodes, num_probes=num_probes, seed=seed)
    else:
        if probes.ndim != 2 or probes.shape[0] != num_nodes:
            raise ValueError("probes must have shape [num_nodes, num_probes]")
        probes = probes.float()

    state = probes
    features: list[Tensor] = []
    for _ in range(steps):
        state = random_walk_transition_apply(edge_index=edge_index, num_nodes=num_nodes, values=state)
        features.append((probes * state).mean(dim=1))
    return torch.stack(features, dim=-1)


def structural_node_features(
    edge_index: Tensor,
    num_nodes: int,
    random_walk_steps: int = 16,
    random_walk_num_probes: int = 8,
    positional_encoding_seed: int = 1337,
    *,
    probes: Tensor | None = None,
    directed_features: Tensor | None = None,
) -> Tensor:
    """Construct structural node features for an undirected graph."""
    degree = torch.bincount(edge_index[0], minlength=num_nodes).float()
    degree_norm = degree / degree.max().clamp_min(1.0)
    base = torch.stack(
        (
            torch.ones_like(degree),
            degree,
            torch.log1p(degree),
            degree_norm,
        ),
        dim=-1,
    )
    pieces = [base]
    if directed_features is not None:
        if directed_features.shape[0] != num_nodes or directed_features.ndim != 2:
            raise ValueError("directed_features must have shape [num_nodes, num_features]")
        pieces.append(directed_features.float())

    rwse = hutchinson_random_walk_return_probabilities(
        edge_index=edge_index,
        num_nodes=num_nodes,
        steps=random_walk_steps,
        num_probes=random_walk_num_probes,
        seed=positional_encoding_seed,
        probes=probes,
    )
    pieces.append(rwse)
    return torch.cat(pieces, dim=-1).contiguous()


def preprocess_graph(
    raw_data: Data,
    *,
    split: str,
    original_graph_index: int,
    config: PreprocessConfig,
    build_features: bool = True,
) -> tuple[Data | None, GraphMetadata]:
    """Clean a graph and attach its derived node features."""
    original_num_nodes = int(raw_data.num_nodes)
    original_num_edges = int(raw_data.edge_index.shape[1])
    try:
        edge_index, lcc, lcc_nodes = clean_to_undirected_lcc_with_nodes(
            edge_index=raw_data.edge_index, num_nodes=original_num_nodes
        )
    except ValueError:
        metadata = GraphMetadata(
            split=split,
            original_graph_index=original_graph_index,
            original_num_nodes=original_num_nodes,
            original_num_edges=original_num_edges,
            cleaned_num_nodes=original_num_nodes,
            cleaned_num_edges=0,
            lcc_num_nodes=0,
            lcc_num_edges=0,
            lcc_fraction=0.0,
            retained=False,
            exclusion_reason="edgeless",
        )
        return None, metadata

    exclusion_reason = None
    if int(lcc["lcc_num_nodes"]) < int(config.min_nodes_per_graph):
        exclusion_reason = "below_min_nodes"
    elif config.max_nodes_per_graph is not None and int(lcc["lcc_num_nodes"]) > int(config.max_nodes_per_graph):
        exclusion_reason = "above_max_nodes"

    metadata = GraphMetadata(
        split=split,
        original_graph_index=original_graph_index,
        original_num_nodes=original_num_nodes,
        original_num_edges=original_num_edges,
        cleaned_num_nodes=int(lcc["cleaned_num_nodes"]),
        cleaned_num_edges=int(lcc["cleaned_num_edges"]),
        lcc_num_nodes=int(lcc["lcc_num_nodes"]),
        lcc_num_edges=int(lcc["lcc_num_edges"]),
        lcc_fraction=float(lcc["lcc_fraction"]),
        retained=exclusion_reason is None,
        exclusion_reason=exclusion_reason,
    )
    if exclusion_reason is not None:
        return None, metadata

    if build_features:
        feature_seed = (
            int(config.positional_encoding_seed) + _SPLIT_SEED_OFFSETS.get(split, 3_000_000) + int(original_graph_index)
        )
        directed_features = None
        if bool(config.include_directed_features):
            directed_features = directed_structural_node_features(
                edge_index=raw_data.edge_index,
                num_nodes=original_num_nodes,
                pagerank_alpha=float(config.pagerank_alpha),
                pagerank_max_iter=int(config.pagerank_max_iter),
                pagerank_tolerance=float(config.pagerank_tolerance),
            )[lcc_nodes]
        features = structural_node_features(
            edge_index=edge_index,
            num_nodes=int(lcc["lcc_num_nodes"]),
            random_walk_steps=int(config.random_walk_steps),
            random_walk_num_probes=int(config.random_walk_num_probes),
            positional_encoding_seed=feature_seed,
            directed_features=directed_features,
        )
    else:
        features = torch.empty(int(lcc["lcc_num_nodes"]), 0, dtype=torch.float32)
    data = Data(x=features, edge_index=edge_index, num_nodes=int(lcc["lcc_num_nodes"]))
    data.original_split = split
    data.original_graph_index = int(original_graph_index)
    data.original_num_nodes = original_num_nodes
    data.original_num_edges = original_num_edges
    data.lcc_fraction = float(lcc["lcc_fraction"])
    return data, metadata


def _stats(values: list[int | float]) -> dict[str, float | int | None]:
    """Summarize a numeric sequence for graph-preprocessing metadata."""
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p10": float(np.percentile(a=array, q=10)),
        "p90": float(np.percentile(a=array, q=90)),
    }


def dataset_summary(metadata_by_split: dict[str, list[GraphMetadata]], config: PreprocessConfig) -> dict[str, Any]:
    """Summarize a collection of preprocessed graph records."""
    summary: dict[str, Any] = {
        "preprocessing_version": config.preprocessing_version,
        "dataset_root_dir": str(config.dataset_root_dir),
        "min_nodes_per_graph": config.min_nodes_per_graph,
        "max_nodes_per_graph": config.max_nodes_per_graph,
        "random_walk_steps": config.random_walk_steps,
        "random_walk_num_probes": config.random_walk_num_probes,
        "positional_encoding_seed": config.positional_encoding_seed,
        "include_directed_features": config.include_directed_features,
        "pagerank_alpha": config.pagerank_alpha,
        "pagerank_max_iter": config.pagerank_max_iter,
        "pagerank_tolerance": config.pagerank_tolerance,
        "splits": {},
    }
    for split, metadata in metadata_by_split.items():
        retained = [item for item in metadata if item.retained]
        excluded = Counter(item.exclusion_reason for item in metadata if not item.retained)
        summary["splits"][split] = {
            "raw_graph_count": len(metadata),
            "retained_graph_count": len(retained),
            "excluded_counts": {str(key): int(value) for key, value in excluded.items()},
            "raw_node_count": _stats([item.original_num_nodes for item in metadata]),
            "raw_edge_count": _stats([item.original_num_edges for item in metadata]),
            "lcc_node_count": _stats([item.lcc_num_nodes for item in metadata]),
            "lcc_edge_count": _stats([item.lcc_num_edges for item in metadata]),
            "lcc_fraction": _stats([item.lcc_fraction for item in metadata]),
        }
    return summary


def prepare_malnet_tiny_cache(config: PreprocessConfig, *, download: bool = True) -> dict[str, Any]:
    """Prepare malnet tiny cache."""
    from torch_geometric.datasets import MalNetTiny

    if not download:
        raw_dir = config.dataset_root_dir / "raw"
        processed_dir = config.dataset_root_dir / "processed"
        has_official_cache = (raw_dir.exists() and any(raw_dir.iterdir())) or (
            processed_dir.exists() and any(processed_dir.iterdir())
        )
        if not has_official_cache:
            raise FileNotFoundError(
                f"No official PyG MalNet-Tiny cache found under {config.dataset_root_dir}; "
                "rerun without --no-download to allow the official loader to download it."
            )

    cache_dir = config.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_by_split: dict[str, list[GraphMetadata]] = {}

    for split in SPLITS:
        pyg_split = "val" if split == "val" else split
        dataset = MalNetTiny(root=str(config.dataset_root_dir), split=pyg_split)

        split_dir = cache_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        index: list[dict[str, Any]] = []
        split_metadata: list[GraphMetadata] = []
        for graph_index, raw_data in enumerate(dataset):
            filename = f"graph_{graph_index:06d}.pt"
            graph_path = split_dir / filename
            data, metadata = preprocess_graph(
                raw_data,
                split=split,
                original_graph_index=graph_index,
                config=config,
                build_features=not graph_path.exists(),
            )
            split_metadata.append(metadata)
            retained_so_far = len(index) + int(data is not None)
            if (graph_index + 1) % 100 == 0:
                print(
                    f"[prepare {split}] processed={graph_index + 1}/{len(dataset)} retained={retained_so_far}",
                    flush=True,
                )
            if data is None:
                continue
            if not graph_path.exists():
                torch.save(obj=data, f=graph_path)
            index.append(
                {
                    "path": filename,
                    "original_graph_index": int(metadata.original_graph_index),
                    "num_nodes": int(metadata.lcc_num_nodes),
                    "num_edges": int(metadata.lcc_num_edges),
                    "lcc_fraction": float(metadata.lcc_fraction),
                }
            )

        (split_dir / "index.json").write_text(json.dumps(index, indent=2))
        (split_dir / "metadata.json").write_text(json.dumps([asdict(item) for item in split_metadata], indent=2))
        metadata_by_split[split] = split_metadata

    summary = dataset_summary(metadata_by_split=metadata_by_split, config=config)
    (cache_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def load_cached_splits(cache_dir: Path) -> dict[str, CachedMalNetTinyNormalizedCutDataset]:
    """Load cached splits."""
    return {split: CachedMalNetTinyNormalizedCutDataset(cache_dir=cache_dir, split=split) for split in SPLITS}


def make_synthetic_community_graph(
    *,
    num_communities: int = 2,
    nodes_per_community: int = 12,
    intra_edges_per_node: int = 4,
    inter_edges_per_community_pair: int = 2,
    random_walk_steps: int = 4,
    random_walk_num_probes: int = 8,
    positional_encoding_seed: int = 1337,
    seed: int = 0,
) -> Data:
    """Create make synthetic community graph."""
    rng = random.Random(seed)
    edges: set[tuple[int, int]] = set()
    total_nodes = num_communities * nodes_per_community
    for community in range(num_communities):
        offset = community * nodes_per_community
        for local_node in range(nodes_per_community):
            source = offset + local_node
            for step in range(1, intra_edges_per_node + 1):
                target = offset + ((local_node + step) % nodes_per_community)
                if source != target:
                    edges.add((min(source, target), max(source, target)))
    for first in range(num_communities):
        for second in range(first + 1, num_communities):
            for _ in range(inter_edges_per_community_pair):
                source = first * nodes_per_community + rng.randrange(nodes_per_community)
                target = second * nodes_per_community + rng.randrange(nodes_per_community)
                edges.add((min(source, target), max(source, target)))

    directed_edges = []
    for source, target in sorted(edges):
        directed_edges.append((source, target))
        directed_edges.append((target, source))
    edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
    x = structural_node_features(
        edge_index=edge_index,
        num_nodes=total_nodes,
        random_walk_steps=random_walk_steps,
        random_walk_num_probes=random_walk_num_probes,
        positional_encoding_seed=positional_encoding_seed,
    )
    return Data(x=x, edge_index=edge_index, num_nodes=total_nodes)


def make_synthetic_splits(
    *,
    random_walk_steps: int = 4,
    random_walk_num_probes: int = 8,
    positional_encoding_seed: int = 1337,
    seed: int = 0,
) -> dict[str, list[Data]]:
    """Create make synthetic splits."""
    splits = {
        "train": [
            make_synthetic_community_graph(
                random_walk_steps=random_walk_steps,
                random_walk_num_probes=random_walk_num_probes,
                positional_encoding_seed=positional_encoding_seed + _SPLIT_SEED_OFFSETS["train"] + index,
                seed=seed + index,
            )
            for index in range(8)
        ],
        "val": [
            make_synthetic_community_graph(
                random_walk_steps=random_walk_steps,
                random_walk_num_probes=random_walk_num_probes,
                positional_encoding_seed=positional_encoding_seed + _SPLIT_SEED_OFFSETS["val"] + index,
                seed=seed + 100 + index,
            )
            for index in range(3)
        ],
        "test": [
            make_synthetic_community_graph(
                random_walk_steps=random_walk_steps,
                random_walk_num_probes=random_walk_num_probes,
                positional_encoding_seed=positional_encoding_seed + _SPLIT_SEED_OFFSETS["test"] + index,
                seed=seed + 200 + index,
            )
            for index in range(3)
        ],
    }
    for split, graphs in splits.items():
        for index, graph in enumerate(graphs):
            graph.original_split = split
            graph.original_graph_index = index
    return splits
