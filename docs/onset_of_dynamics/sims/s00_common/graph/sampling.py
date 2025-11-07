# docs/onset_of_dynamics/sims/s00_common/graph/sampling.py
from __future__ import annotations

import random
from typing import List, Tuple

import networkx as nx
import numpy as np

from . import Pair


def sample_pairs_band(
    G: nx.Graph,
    pos: np.ndarray,
    band: Tuple[float, float],
    k: int,
    min_hops: int,
    seed: int,
) -> List[Pair]:
    """Random pairs with euclidean distance in band and hop >= min_hops."""
    if k <= 0:
        return []

    band_lo, band_hi = sorted(band)
    rng = random.Random(seed)
    nodes = list(G.nodes())
    N = len(nodes)
    if N < 2:
        return []

    pairs: List[Pair] = []
    max_attempts = max(1000, 20 * k)
    attempts = 0

    while len(pairs) < k and attempts < max_attempts:
        attempts += 1
        u, v = rng.sample(nodes, 2)
        diff = pos[u] - pos[v]
        d_euclid = float(np.linalg.norm(diff))
        if not (band_lo <= d_euclid <= band_hi):
            continue
        try:
            hops = nx.shortest_path_length(G, u, v)
        except nx.NetworkXNoPath:
            continue
        if hops < min_hops:
            continue
        pairs.append(Pair(u=u, v=v, d_euclid=d_euclid, hops=hops))

    if len(pairs) < k:
        raise RuntimeError(
            f"Could not sample enough pairs within band {band_lo}-{band_hi} "
            f"(wanted {k}, got {len(pairs)}). Consider widening the band."
        )
    return pairs

