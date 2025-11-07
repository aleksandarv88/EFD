"""
loops.py — Small-loop builders and flag-complex (clique) area estimators.

Utilities:
  - simple 3- and 4-cycles around a node (grid/mesh friendly)
  - enumerate short cycles near a node
  - estimate loop "area" using:
      (A) clique/flag-complex 2-simplex count, or
      (B) geometric area via node 'pos' (shoelace), if available
"""

from __future__ import annotations
from typing import Iterable, List, Tuple, Dict, Set
import math
import itertools
import networkx as nx

Node = Tuple[int, int]

# ---------------------------------------------------------------------------
# Local cycle enumeration

def local_triangles(G: nx.Graph, center: Node) -> List[List[Node]]:
    """Return all triangles (3-cycles) that include 'center'."""
    nbrs = set(G.neighbors(center))
    tri: List[List[Node]] = []
    for u, v in itertools.combinations(sorted(nbrs), 2):
        if G.has_edge(u, v):
            tri.append([center, u, v, center])
    return tri

def local_quads_grid_like(G: nx.Graph, center: Node) -> List[List[Node]]:
    """
    Try to build 4-cycles using neighbor-of-neighbor patterns around center.
    Works best for grids or rectified meshes with diagonals.
    """
    quads: List[List[Node]] = []
    nbrs = list(G.neighbors(center))
    nbrs_set = set(nbrs)
    for u in nbrs:
        for v in G.neighbors(u):
            if v == center or v in nbrs_set:
                continue
            # find a w that closes center -> u -> v -> w -> center
            for w in G.neighbors(v):
                if w != center and G.has_edge(w, center):
                    cycle = [center, u, v, w, center]
                    # canonicalize to avoid duplicates
                    if _is_new_cycle(quads, cycle):
                        quads.append(cycle)
    return quads

def _is_new_cycle(existing: List[List[Node]], cycle: List[Node]) -> bool:
    """Avoid storing rotationally identical cycles."""
    core = cycle[:-1]
    rotations = [tuple(core[i:] + core[:i]) for i in range(len(core))]
    rev = list(reversed(core))
    rotations += [tuple(rev[i:] + rev[:i]) for i in range(len(core))]
    existing_cores = {tuple(c[:-1]) for c in existing}
    return all(r not in existing_cores for r in rotations)

def simple_cycles_local(G: nx.Graph, center: Node, max_len: int = 6) -> List[List[Node]]:
    """
    Gather small cycles through 'center': triangles, quads, and some length-5/6 cycles.
    """
    cycles: List[List[Node]] = []
    cycles += local_triangles(G, center)
    cycles += local_quads_grid_like(G, center)

    if max_len >= 5:
        # Expand triangles by inserting a neighbor on one edge to produce 4/5/6-cycles
        tri = [c[:-1] for c in local_triangles(G, center)]
        for a, b, c in tri:
            for x in G.neighbors(b):
                if x in (a, c, center):
                    continue
                if G.has_edge(x, c) and G.has_edge(center, a):
                    cyc = [center, a, b, x, c, center]
                    if len(cyc) - 1 <= max_len and _is_new_cycle(cycles, cyc):
                        cycles.append(cyc)
    return cycles

# ---------------------------------------------------------------------------
# Flag-complex "area" approximations

def flag_complex_area_count(G: nx.Graph, cycle: List[Node]) -> float:
    """
    Approximate area by the minimal number of 2-simplices (triangles) needed
    to fill the loop, assuming the flag complex (clique complex) of G.

    For general loops we use (k-2) as a lower bound (triangulation of a k-gon).
    If all triangle chords exist between consecutive triples, this equals k-2.
    """
    loop = cycle[:-1]
    k = len(loop)
    if k < 3:
        return 0.0
    # Check existence of chords that allow fan triangulation from loop[0]
    anchor = loop[0]
    chords_ok = True
    for m in range(2, k - 0):
        u = loop[m - 1]
        v = loop[m % k]
        if not (G.has_edge(anchor, u) and G.has_edge(anchor, v)):
            chords_ok = False
            break
    return float(k - 2) if chords_ok else float(max(1, k - 2))

def polygon_area_from_pos(G: nx.Graph, cycle: List[Node]) -> float:
    """
    If nodes have 2D 'pos', return geometric area via the shoelace formula.
    Assumes cycle nodes are ordered around the loop and cycle[-1]==cycle[0].
    """
    coords = [G.nodes[n].get("pos", None) for n in cycle]
    if any(c is None for c in coords):
        return flag_complex_area_count(G, cycle)
    area = 0.0
    for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5
