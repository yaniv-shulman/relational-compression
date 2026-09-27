"""Experiment support code for relational compression studies."""

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch_geometric.data import Data

from relational_compression.experiments.graph_geometry import _undirected_weighted_edges
from relational_compression.experiments.inductive_normalized_cut.data import (
    PreprocessConfig,
    clean_to_undirected_lcc_with_nodes,
    load_cached_splits,
    make_synthetic_splits,
)

MALNET_UNWEIGHTED = "malnet_unweighted"
MALNET_WEIGHTED = "malnet_weighted"
COLLAB = "collab"
PROTEINS = "proteins"
SYNTHETIC = "synthetic"


@dataclass(frozen=True)
class SourceGraphRecord:
    """Store Source Graph Record values."""

    collection: str
    source_variant: str
    split: str
    dataset_index: int
    original_graph_index: int
    data: Data
    weight_seed: int | None = None
    weight_sigma: float | None = None
    cleaning: dict[str, Any] | None = None


def deterministic_subset_indices(total: int, count: int | None, seed: int) -> list[int]:
    """Compute deterministic subset indices."""
    if total < 0:
        raise ValueError("total must be non-negative")
    if count is None or int(count) >= int(total):
        return list(range(int(total)))
    if int(count) < 0:
        raise ValueError("count must be non-negative")
    rng = np.random.default_rng(int(seed))
    return sorted(int(index) for index in rng.choice(int(total), size=int(count), replace=False))


def _stable_uint64(*parts: object) -> int:
    """Compute stable uint64."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest()[:8], "little", signed=False)


def randomized_weighted_copy(
    data: Data,
    *,
    global_seed: int,
    sigma: float,
) -> tuple[Data, int]:
    """Return a topology-identical graph with deterministic positive undirected weights."""
    if sigma < 0.0:
        raise ValueError("weight sigma must be non-negative")
    graph_seed = _stable_uint64(
        int(global_seed),
        getattr(data, "original_split", "unknown"),
        int(getattr(data, "original_graph_index", -1)),
        int(data.num_nodes),
        int(data.edge_index.shape[1]),
    )
    edges, _ = _undirected_weighted_edges(edge_index=data.edge_index, num_nodes=int(data.num_nodes))
    rng = np.random.default_rng(graph_seed)
    weights = np.exp(rng.normal(0.0, float(sigma), size=edges.shape[0])).astype(np.float64, copy=False)
    weights = weights / max(float(weights.mean()), np.finfo(np.float64).tiny)
    directed_edges = np.concatenate((edges, edges[:, ::-1]), axis=0)
    directed_weights = np.concatenate((weights, weights), axis=0)
    result = Data(
        edge_index=torch.as_tensor(directed_edges.T, dtype=torch.long).contiguous(),
        edge_weight=torch.as_tensor(directed_weights, dtype=torch.float32),
        num_nodes=int(data.num_nodes),
    )
    for name in (
        "original_split",
        "original_graph_index",
        "original_num_nodes",
        "original_num_edges",
        "lcc_fraction",
        "dataset_index",
    ):
        if hasattr(data, name):
            setattr(result, name, getattr(data, name))
    return result, int(graph_seed)


def graph_statistics(data: Data) -> dict[str, float | int]:
    """Compute graph statistics."""
    edge_weight = getattr(data, "edge_weight", None)
    edges, weights = _undirected_weighted_edges(
        edge_index=data.edge_index, num_nodes=int(data.num_nodes), edge_weight=edge_weight
    )
    degree = np.zeros(int(data.num_nodes), dtype=np.float64)
    np.add.at(degree, edges[:, 0], weights)
    np.add.at(degree, edges[:, 1], weights)
    mean_degree = float(degree.mean()) if degree.size else float("nan")
    std_degree = float(degree.std()) if degree.size else float("nan")
    density = 0.0
    if int(data.num_nodes) > 1:
        density = float(2.0 * edges.shape[0] / (int(data.num_nodes) * (int(data.num_nodes) - 1)))
    if edge_weight is None:
        weight_cv = 0.0
    else:
        weight_cv = float(weights.std() / max(float(weights.mean()), np.finfo(np.float64).tiny))

    clustering = float("nan")
    try:
        import networkx as nx

        graph = nx.Graph()
        graph.add_nodes_from(range(int(data.num_nodes)))
        graph.add_edges_from((int(source), int(target)) for source, target in edges.tolist())
        clustering = float(nx.transitivity(graph))
    except ImportError:
        pass

    return {
        "num_nodes": int(data.num_nodes),
        "num_edges": int(edges.shape[0]),
        "average_degree": mean_degree,
        "degree_coefficient_of_variation": std_degree / max(mean_degree, float(np.finfo(np.float64).tiny)),
        "graph_density": density,
        "transitivity": clustering,
        "edge_weight_coefficient_of_variation": weight_cv,
    }


def _annotate(
    data: Data,
    *,
    collection: str,
    source_variant: str,
    split: str,
    dataset_index: int,
    original_graph_index: int,
    weight_seed: int | None = None,
    weight_sigma: float | None = None,
) -> Data:
    """Compute annotate."""
    data.collection = collection
    data.source_variant = source_variant
    data.original_split = split
    data.dataset_index = int(dataset_index)
    data.original_graph_index = int(original_graph_index)
    data.weight_seed = -1 if weight_seed is None else int(weight_seed)
    data.weight_sigma = float("nan") if weight_sigma is None else float(weight_sigma)
    if hasattr(data, "y"):
        delattr(data, "y")
    return data


def _malnet_preprocess_config(config: Any) -> PreprocessConfig:
    """Compute malnet preprocess config."""
    return PreprocessConfig(
        dataset_root_dir=Path(config.dataset_root_dir),
        min_nodes_per_graph=int(config.min_nodes_per_graph),
        max_nodes_per_graph=config.max_nodes_per_graph,
        random_walk_steps=int(config.random_walk_steps),
        random_walk_num_probes=int(config.random_walk_num_probes),
        positional_encoding_seed=int(config.positional_encoding_seed),
        preprocessing_version=str(config.preprocessing_version),
        include_directed_features=bool(config.include_directed_features),
        pagerank_alpha=float(config.pagerank_alpha),
        pagerank_max_iter=int(config.pagerank_max_iter),
        pagerank_tolerance=float(config.pagerank_tolerance),
    )


def _load_malnet_records(config: Any, *, weighted: bool) -> tuple[list[SourceGraphRecord], dict[str, Any]]:
    """Load malnet records."""
    preprocess_config = _malnet_preprocess_config(config)
    try:
        splits = load_cached_splits(preprocess_config.cache_dir)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"{exc}\nPrepare the shared Relational Compression MalNet-Tiny cache first:\n"
            "  poetry run python -m relational_compression.experiments.inductive_normalized_cut.run_scripts.prepare_malnet_tiny"
        ) from exc

    split = str(config.malnet_split)
    dataset = splits[split]
    indices = deterministic_subset_indices(
        total=len(dataset), count=int(config.graphs_per_collection), seed=int(config.graph_seed)
    )
    records: list[SourceGraphRecord] = []
    for dataset_index in indices:
        data = dataset[dataset_index]
        original_graph_index = int(getattr(data, "original_graph_index", dataset_index))
        if weighted:
            data, weight_seed = randomized_weighted_copy(
                data,
                global_seed=int(config.weight_seed),
                sigma=float(config.weight_sigma),
            )
            source_variant = "random_log_normal_weights"
            collection = MALNET_WEIGHTED
        else:
            weight_seed = None
            source_variant = "unweighted"
            collection = MALNET_UNWEIGHTED
        data = _annotate(
            data,
            collection=collection,
            source_variant=source_variant,
            split=split,
            dataset_index=dataset_index,
            original_graph_index=original_graph_index,
            weight_seed=weight_seed,
            weight_sigma=float(config.weight_sigma) if weighted else None,
        )
        records.append(
            SourceGraphRecord(
                collection=collection,
                source_variant=source_variant,
                split=split,
                dataset_index=dataset_index,
                original_graph_index=original_graph_index,
                data=data,
                weight_seed=weight_seed,
                weight_sigma=float(config.weight_sigma) if weighted else None,
                cleaning={
                    "cache_dir": str(preprocess_config.cache_dir),
                    "lcc_fraction": float(getattr(data, "lcc_fraction", float("nan"))),
                },
            )
        )
    return records, {"selected_indices": indices, "preprocess_cache_dir": str(preprocess_config.cache_dir)}


def clean_tu_graph(
    raw_data: Data,
    *,
    dataset_name: str,
    dataset_index: int,
    min_nodes: int,
    max_nodes: int | None,
) -> tuple[Data | None, dict[str, Any]]:
    """Clean tu graph."""
    num_nodes = int(raw_data.num_nodes)
    original_edges = int(raw_data.edge_index.shape[1])
    try:
        edge_index, lcc, _ = clean_to_undirected_lcc_with_nodes(edge_index=raw_data.edge_index, num_nodes=num_nodes)
    except ValueError:
        return None, {
            "dataset_name": dataset_name,
            "dataset_index": int(dataset_index),
            "original_num_nodes": num_nodes,
            "original_num_edges": original_edges,
            "retained": False,
            "exclusion_reason": "edgeless",
        }
    exclusion_reason = None
    if int(lcc["lcc_num_nodes"]) < int(min_nodes):
        exclusion_reason = "below_min_nodes"
    elif max_nodes is not None and int(lcc["lcc_num_nodes"]) > int(max_nodes):
        exclusion_reason = "above_max_nodes"
    metadata = {
        "dataset_name": dataset_name,
        "dataset_index": int(dataset_index),
        "original_num_nodes": num_nodes,
        "original_num_edges": original_edges,
        "cleaned_num_nodes": int(lcc["cleaned_num_nodes"]),
        "cleaned_num_edges": int(lcc["cleaned_num_edges"]),
        "lcc_num_nodes": int(lcc["lcc_num_nodes"]),
        "lcc_num_edges": int(lcc["lcc_edges"] if "lcc_edges" in lcc else lcc["lcc_num_edges"]),
        "lcc_fraction": float(lcc["lcc_fraction"]),
        "retained": exclusion_reason is None,
        "exclusion_reason": exclusion_reason,
    }
    if exclusion_reason is not None:
        return None, metadata
    data = Data(edge_index=edge_index, num_nodes=int(lcc["lcc_num_nodes"]))
    data.original_num_nodes = num_nodes
    data.original_num_edges = original_edges
    data.lcc_fraction = float(lcc["lcc_fraction"])
    return data, metadata


def _load_tu_records(config: Any, *, name: str) -> tuple[list[SourceGraphRecord], dict[str, Any]]:
    """Load tu records."""
    from torch_geometric.datasets import TUDataset

    dataset_root = Path(config.tu_dataset_root_dir)
    dataset = TUDataset(root=str(dataset_root), name=name.upper())
    metadata_rows: list[dict[str, Any]] = []
    retained: list[tuple[int, Data, dict[str, Any]]] = []
    for dataset_index, raw_data in enumerate(dataset):
        data, metadata = clean_tu_graph(
            raw_data,
            dataset_name=name.upper(),
            dataset_index=dataset_index,
            min_nodes=int(config.tu_min_nodes_per_graph),
            max_nodes=config.tu_max_nodes_per_graph,
        )
        metadata_rows.append(metadata)
        if data is not None:
            retained.append((dataset_index, data, metadata))

    selected_positions = deterministic_subset_indices(
        total=len(retained),
        count=int(config.graphs_per_collection),
        seed=int(config.graph_seed) + _stable_uint64(name) % 1_000_000,
    )
    records: list[SourceGraphRecord] = []
    for retained_position in selected_positions:
        dataset_index, data, metadata = retained[retained_position]
        collection = name.lower()
        data = _annotate(
            data,
            collection=collection,
            source_variant="unweighted",
            split="test",
            dataset_index=dataset_index,
            original_graph_index=dataset_index,
        )
        records.append(
            SourceGraphRecord(
                collection=collection,
                source_variant="unweighted",
                split="test",
                dataset_index=dataset_index,
                original_graph_index=dataset_index,
                data=data,
                cleaning=metadata,
            )
        )

    excluded = Counter(str(row.get("exclusion_reason")) for row in metadata_rows if not row.get("retained", False))
    return records, {
        "dataset_root": str(dataset_root),
        "raw_graph_count": len(dataset),
        "eligible_graph_count": len(retained),
        "selected_indices": [record.dataset_index for record in records],
        "excluded_counts": {key: int(value) for key, value in excluded.items()},
        "metadata": metadata_rows,
    }


def _load_synthetic_records(config: Any) -> tuple[list[SourceGraphRecord], dict[str, Any]]:
    """Load synthetic records."""
    split = "val"
    graphs = make_synthetic_splits(seed=int(config.seed))[split]
    indices = deterministic_subset_indices(
        total=len(graphs), count=int(config.graphs_per_collection), seed=int(config.graph_seed)
    )
    records: list[SourceGraphRecord] = []
    for dataset_index in indices:
        data = graphs[dataset_index]
        data = _annotate(
            data,
            collection=SYNTHETIC,
            source_variant="unweighted",
            split=split,
            dataset_index=dataset_index,
            original_graph_index=dataset_index,
        )
        records.append(
            SourceGraphRecord(
                collection=SYNTHETIC,
                source_variant="unweighted",
                split=split,
                dataset_index=dataset_index,
                original_graph_index=dataset_index,
                data=data,
            )
        )
    return records, {"selected_indices": indices}


def load_source_graph_records(config: Any) -> tuple[list[SourceGraphRecord], dict[str, Any]]:
    """Load source graph records."""
    records: list[SourceGraphRecord] = []
    metadata: dict[str, Any] = {}
    for collection in tuple(config.source_collections):
        collection = str(collection)
        if collection == SYNTHETIC:
            loaded, info = _load_synthetic_records(config)
        elif collection == MALNET_UNWEIGHTED:
            loaded, info = _load_malnet_records(config, weighted=False)
        elif collection == MALNET_WEIGHTED:
            loaded, info = _load_malnet_records(config, weighted=True)
        elif collection == COLLAB:
            loaded, info = _load_tu_records(config, name="COLLAB")
        elif collection == PROTEINS:
            loaded, info = _load_tu_records(config, name="PROTEINS")
        else:
            raise ValueError(f"Unsupported source collection: {collection}")
        records.extend(loaded)
        metadata[collection] = info
    return records, metadata


def graph_record_row(record: SourceGraphRecord) -> dict[str, Any]:
    """Compute graph record row."""
    stats = graph_statistics(record.data)
    return {
        "collection": record.collection,
        "source_variant": record.source_variant,
        "split": record.split,
        "dataset_index": int(record.dataset_index),
        "original_graph_index": int(record.original_graph_index),
        "weight_seed": "" if record.weight_seed is None else int(record.weight_seed),
        "weight_sigma": "" if record.weight_sigma is None else float(record.weight_sigma),
        **stats,
        "cleaning": "" if record.cleaning is None else record.cleaning,
    }
