# docs/onset_of_dynamics/sims/s00_common/grid_refine.py
from __future__ import annotations
from dataclasses import dataclass
import math
import networkx as nx

@dataclass
class GridSpec:
    nx: int
    ny: int
    diag: bool = False            # use 8-neighbour (if radius_phys is None)
    tag_edge_len: bool = True     # tag edges with 'len' weight
    radius_phys: float | None = None  # NEW: connect all nodes within this physical radius

def _add_edge(G: nx.Graph, u: int, v: int, pos: dict, tag_len: bool):
    (x1, y1) = pos[u]
    (x2, y2) = pos[v]
    if tag_len:
        d = math.hypot(x2 - x1, y2 - y1)
        G.add_edge(u, v, len=d)
    else:
        G.add_edge(u, v)

def build_grid(spec: GridSpec) -> nx.Graph:
    """
    Unit-square grid [0,1]x[0,1] with (nx x ny) nodes.
    If radius_phys is set, connect all pairs within that physical radius (Euclidean),
    which densifies directions as h -> 0 (good for continuum limits).
    Otherwise, build standard 4- or 8-neighbour stencils.
    """
    nxg, nyg = spec.nx, spec.ny
    assert nxg >= 2 and nyg >= 2, "nx, ny must be >= 2"

    G = nx.Graph()
    # nodes + positions
    for j in range(nyg):
        y = j / (nyg - 1)
        for i in range(nxg):
            x = i / (nxg - 1)
            u = j * nxg + i
            G.add_node(u, pos=(x, y))

    pos = nx.get_node_attributes(G, "pos")

    # --- dense radius-based stencil (NEW) ---
    if spec.radius_phys is not None and spec.radius_phys > 0.0:
        r = float(spec.radius_phys)
        hx = 1.0 / (nxg - 1)
        hy = 1.0 / (nyg - 1)
        rx = int(math.ceil(r / hx))
        ry = int(math.ceil(r / hy))
        for j in range(nyg):
            for i in range(nxg):
                u = j * nxg + i
                x_u, y_u = pos[u]
                for dj in range(-ry, ry + 1):
                    vj = j + dj
                    if vj < 0 or vj >= nyg:
                        continue
                    for di in range(-rx, rx + 1):
                        vi = i + di
                        if di == 0 and dj == 0:
                            continue
                        if vi < 0 or vi >= nxg:
                            continue
                        v = vj * nxg + vi
                        x_v, y_v = pos[v]
                        d = math.hypot(x_v - x_u, y_v - y_u)
                        if d <= r + 1e-12:
                            _add_edge(G, u, v, pos, spec.tag_edge_len)
        return G

    # --- fallback: classic 4- / 8-neighbour grid ---
    def inside(i: int, j: int) -> bool:
        return 0 <= i < nxg and 0 <= j < nyg

    # 4-neighbour
    for j in range(nyg):
        for i in range(nxg):
            u = j * nxg + i
            for di, dj in ((1, 0), (0, 1)):
                vi, vj = i + di, j + dj
                if inside(vi, vj):
                    v = vj * nxg + vi
                    _add_edge(G, u, v, pos, spec.tag_edge_len)

    # add diagonals if requested
    if spec.diag:
        for j in range(nyg):
            for i in range(nxg):
                u = j * nxg + i
                for di, dj in ((1, 1), (1, -1)):
                    vi, vj = i + di, j + dj
                    if inside(vi, vj):
                        v = vj * nxg + vi
                        _add_edge(G, u, v, pos, spec.tag_edge_len)

    return G
