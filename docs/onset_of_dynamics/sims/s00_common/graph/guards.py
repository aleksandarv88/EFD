# docs/onset_of_dynamics/sims/s00_common/graph/guards.py
from __future__ import annotations

import random
from typing import Iterable, List

import networkx as nx
import numpy as np

from . import Pair


def assert_avg_degree_monotone(avg_degrees: Iterable[float], tol: float = 1e-6) -> None:
    """Ensure average degree is non-decreasing as refinements increase."""
    prev = None
    for value in avg_degrees:
        if prev is not None and value + tol < prev:
            raise AssertionError(
                f"Average degree must be monotone; got drop {prev:.3f} -> {value:.3f}"
            )
        prev = value


def assert_edges_euclidean(G: nx.Graph, pos: np.ndarray, sample_k: int = 500) -> None:
    """Verify stored edge weights match Euclidean distances for a random subset."""
    edges = list(G.edges(data="weight"))
    if not edges:
        return
    rng = random.Random(0xEFD)
    sample = rng.sample(edges, k=min(sample_k, len(edges)))
    for u, v, w in sample:
        dist = float(np.linalg.norm(pos[u] - pos[v]))
        if abs(w - dist) > 1e-6:
            raise AssertionError(
                f"Edge ({u},{v}) weight mismatch: stored={w:.6f}, euclid={dist:.6f}"
            )


def assert_direct_pair_ratio(pairs: List[Pair], G: nx.Graph, max_ratio: float = 0.05) -> None:
    """Ensure only a small fraction of sampled pairs are direct neighbors."""
    if not pairs:
        return
    direct = sum(1 for p in pairs if G.has_edge(p.u, p.v))
    ratio = direct / len(pairs)
    if ratio > max_ratio:
        raise AssertionError(
            f"Direct pair ratio too high ({ratio:.2%} > {max_ratio:.2%}); radius may be too large."
        )

