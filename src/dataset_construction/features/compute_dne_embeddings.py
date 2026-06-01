import sys
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional

import networkx as nx
import numpy as np

sys.path.append("./src/dataset_construction/DNE/code")


logger = logging.getLogger(__name__)


def load_or_compute_dne_embeddings(
    graph: nx.Graph,
    embeddings_path: str,
    dne_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """
    Return DNE embeddings, loading from cache if available.

    Args:
        graph: NetworkX graph with concept string node IDs and weighted edges
        embeddings_path: path to .pkl cache file
        dne_cfg: hyperparameter overrides (hidden_dim, epochs, walk_number, etc.)

    Returns:
        dict {concept_name: np.ndarray of shape (hidden_dim,)}
    """
    if dne_cfg is None:
        dne_cfg = {}

    path = Path(embeddings_path)
    if path.exists():
        logger.info(f"Loading cached DNE embeddings from {path}")
        with open(path, "rb") as f:
            embeddings = pickle.load(f)
        sample_dim = len(next(iter(embeddings.values()))) if embeddings else 0
        logger.info(f"Loaded {len(embeddings):,} DNE embeddings (dim={sample_dim})")
        return embeddings

    embeddings = _train_dne(graph, dne_cfg)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(embeddings, f)
    logger.info(f"Saved {len(embeddings):,} DNE embeddings to {path}")
    return embeddings


def _train_dne(
    graph: nx.Graph,
    dne_cfg: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """Train DNE on graph and return {concept: embedding_vector}."""

    from models.dne import DNE

    hidden_dim = dne_cfg.get("hidden_dim", 128)
    num_pos_feat = dne_cfg.get("num_pos_features", 128)
    epochs = dne_cfg.get("epochs", 5)
    walk_number = dne_cfg.get("walk_number", 50)
    walk_length = dne_cfg.get("walk_length", 10)
    batch_size = dne_cfg.get("batch_size", 1000)
    p = dne_cfg.get("p", 1.0)
    q = dne_cfg.get("q", 1.0)

    if graph.number_of_edges() == 0:
        logger.warning("DNE: graph has no edges, returning empty embeddings")
        return {}

    num_pos_feat = min(num_pos_feat, graph.number_of_nodes() - 1)

    logger.info(
        f"DNE training: {graph.number_of_nodes():,} nodes, "
        f"{graph.number_of_edges():,} edges | "
        f"hidden_dim={hidden_dim}, num_pos_features={num_pos_feat}, epochs={epochs}"
    )

    dne = DNE(graph, feat=None, hidden_dim=hidden_dim, num_pos_features=num_pos_feat)
    dne.train(
        batch_size=batch_size,
        epochs=epochs,
        walk_number=walk_number,
        walk_length=walk_length,
        p=p,
        q=q,
    )
    embeddings = dne.get_embeddings()
    logger.info(f"DNE: produced {len(embeddings):,} embeddings (dim={hidden_dim})")
    return embeddings
