# docs/onset_of_dynamics/sims/s00_common/graph/builders.py
from __future__ import annotations

import math
from typing import Literal

import networkx as nx
import numpy as np

from . import GridSpec, GraphPack


def build_radius_graph(spec: GridSpec, radius: float) -> GraphPack:
    """Grid nodes on [0,1]^2, connect pairs within radius, weight = Euclidean."""
    if radius <= 0:
        raise ValueError("radius must be positive")

    xs = np.linspace(0.0, 1.0, spec.nx, dtype=np.float64)
    ys = np.linspace(0.0, 1.0, spec.ny, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel()], axis=1)
    n_nodes = pos.shape[0]

    builder: Literal["scipy.KDTree", "numpy_ball"]
    edges = []
    try:
        from scipy.spatial import cKDTree as KDTree  # type: ignore

        tree = KDTree(pos)
        neighborhoods = tree.query_ball_point(pos, radius)
        builder = "scipy.KDTree"
        for i, neigh in enumerate(neighborhoods):
            for j in neigh:
                if j <= i:
                    continue
                dist = float(np.linalg.norm(pos[i] - pos[j]))
                if dist <= radius * (1 + 1e-9):
                    edges.append((i, j, dist))
    except Exception:
        # Fallback: numpy box pruning
        builder = "numpy_ball"
        radius2 = radius * radius
        for i in range(n_nodes - 1):
            diff = pos[i + 1 :] - pos[i]
            np.abs(diff, out=diff)
            # quick axis-aligned prune
            mask = (diff[:, 0] <= radius) & (diff[:, 1] <= radius)
            if not np.any(mask):
                continue
            mask_idx = np.nonzero(mask)[0]
            selected = pos[i + 1 :][mask_idx] - pos[i]
            dist2 = np.sum(selected * selected, axis=1)
            within = np.nonzero(dist2 <= radius2)[0]
            for idx in within:
                j = mask_idx[idx] + i + 1
                dist = math.sqrt(float(dist2[idx]))
                edges.append((i, j, dist))

    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)

    avg_deg = 0.0 if n_nodes == 0 else 2.0 * len(edges) / n_nodes
    return GraphPack(G=G, pos=pos.astype(np.float32), avg_degree=avg_deg, builder=builder)
