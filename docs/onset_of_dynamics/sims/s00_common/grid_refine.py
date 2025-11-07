"""
grid_refine.py — Stage-5 helpers for building grid refinements and sampling node pairs.

Creates 2D grids on the unit square with either 4- or 8-neighborhoods.
Adds node attributes:
  - "pos": (x, y) in [0,1]^2
  - "ij":  (i, j) integer lattice indices
Optionally tags edges with Euclidean length under "len".

Also provides utilities to sample node pairs for distance tests.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional
import math
import numpy as np
import networkx as nx

# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GridSpec:
    nx: int              # columns (x)
    ny: int              # rows (y)
    diag: bool = False   # 8-neighbors if True, else 4-neighbors
    tag_edge_len: bool = True

def _add_edges(G: nx.Graph, spec: GridSpec) -> None:
    nx_, ny_ = spec.nx, spec.ny
    for j in range(ny_):
        for i in range(nx_):
            u = (i, j)
            if i + 1 < nx_:
                v = (i + 1, j)
                G.add_edge(u, v)
            if j + 1 < ny_:
                v = (i, j + 1)
                G.add_edge(u, v)
            if spec.diag:
                if i + 1 < nx_ and j + 1 < ny_:
                    G.add_edge(u, (i + 1, j + 1))
                if i + 1 < nx_ and j - 1 >= 0:
                    G.add_edge(u, (i + 1, j - 1))

def _tag_edge_lengths(G: nx.Graph) -> None:
    for u, v in G.edges():
        x1, y1 = G.nodes[u]["pos"]
        x2, y2 = G.nodes[v]["pos"]
        G.edges[u, v]["len"] = math.hypot(x2 - x1, y2 - y1)

def build_grid(spec: GridSpec) -> nx.Graph:
    """
    Build a grid graph on [0,1]^2 with (spec.nx x spec.ny) lattice.
    Nodes are keyed by (i,j). Attribute 'pos' holds normalized coords.

    Returns
    -------
    G : networkx.Graph
    """
    assert spec.nx >= 2 and spec.ny >= 2, "Need at least 2x2 grid."
    G = nx.Graph()
    for j in range(spec.ny):
        y = j / (spec.ny - 1)
        for i in range(spec.nx):
            x = i / (spec.nx - 1)
            G.add_node((i, j), pos=(float(x), float(y)), ij=(i, j))
    _add_edges(G, spec)
    if spec.tag_edge_len:
        _tag_edge_lengths(G)
    G.graph["kind"] = "grid"
    G.graph["diag"] = spec.diag
    G.graph["shape"] = (spec.nx, spec.ny)
    return G

# ---------------------------------------------------------------------------
# Pair sampling

def _hop_distance(G: nx.Graph, src: Tuple[int, int]) -> dict:
    """Unweighted BFS distance in hops."""
    return nx.single_source_shortest_path_length(G, src)

def sample_pairs_by_min_hops(
    G: nx.Graph,
    k: int,
    rng: np.random.Generator,
    min_hops: int = 4,
    max_tries: int = 50_000,
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Sample k pairs of nodes (u,v) with BFS-hop distance >= min_hops.
    Tries up to max_tries random draws; returns fewer than k if needed.
    """
    nodes = list(G.nodes())
    n = len(nodes)
    result: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    tried = 0
    while len(result) < k and tried < max_tries:
        u = nodes[rng.integers(0, n)]
        v = nodes[rng.integers(0, n)]
        if u == v:
            tried += 1
            continue
        d = _hop_distance(G, u).get(v, None)
        if d is not None and d >= min_hops:
            result.append((u, v))
        tried += 1
    return result

def sample_pairs_by_spacing(
    G: nx.Graph,
    k: int,
    rng: np.random.Generator,
    min_sep: float = 0.2,
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Sample k pairs with Euclidean separation (in 'pos') >= min_sep.
    """
    nodes = list(G.nodes())
    pos = nx.get_node_attributes(G, "pos")
    n = len(nodes)
    result: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    tries = 0
    while len(result) < k and tries < 50_000:
        u = nodes[rng.integers(0, n)]
        v = nodes[rng.integers(0, n)]
        if u == v:
            tries += 1
            continue
        (x1, y1), (x2, y2) = pos[u], pos[v]
        if math.hypot(x2 - x1, y2 - y1) >= min_sep:
            result.append((u, v))
        tries += 1
    return result
