"""
meshes.py — Simple parametric mesh generators (graph view) for sphere,
cylinder, and saddle patches. Produces an undirected graph with:

Node attrs:
  - 'pos3': (x,y,z) in R^3
  - 'uv'  : (u,v) parameter in [0,1]x[0,1] (except cylinder u ∈ [0,1), v∈[0,1])
  - 'pos' : optional projection to 2D (u,v) for planar operations

Edges connect 4-neighborhood on parameter grid (wrap in u for cylinder).
Faces list (triangulation) is returned for convenience.
"""

from __future__ import annotations
from typing import List, Tuple
import math
import networkx as nx

Vec3 = Tuple[float, float, float]
Vec2 = Tuple[float, float]

def _add_face_tris(faces: List[Tuple[Tuple[int,int], Tuple[int,int], Tuple[int,int]]],
                   i: int, j: int, wrap_i: bool, wrap_j: bool,
                   Nu: int, Nv: int) -> None:
    """Split a cell (i,j)-(i+1,j+1) into two triangles."""
    ip = (i + 1) % Nu if wrap_i else i + 1
    jp = (j + 1) % Nv if wrap_j else j + 1
    if ip >= Nu or jp >= Nv:
        return
    a = (i, j); b = (ip, j); c = (ip, jp); d = (i, jp)
    faces.append((a, b, c))
    faces.append((a, c, d))

# ---------------------------------------------------------------------------
# Sphere (radius R), parameterized by (u,v) -> θ,φ
# u in [0,1] → φ∈[0,2π), v in [0,1] → θ∈[0,π]

def sphere_graph(Nu: int, Nv: int, R: float = 1.0) -> Tuple[nx.Graph, List[Tuple]]:
    assert Nu >= 3 and Nv >= 3
    G = nx.Graph()
    faces: List[Tuple] = []
    for j in range(Nv):
        v = j / (Nv - 1)
        theta = math.pi * v            # polar (0..π)
        for i in range(Nu):
            u = i / Nu                 # wrap in u
            phi = 2.0 * math.pi * u    # azimuth
            x = R * math.sin(theta) * math.cos(phi)
            y = R * math.sin(theta) * math.sin(phi)
            z = R * math.cos(theta)
            G.add_node((i, j), pos3=(x, y, z), uv=(u, v), pos=(u, v))
    # edges
    for j in range(Nv):
        for i in range(Nu):
            ip = (i + 1) % Nu
            if j + 1 < Nv:
                G.add_edge((i, j), (i, j + 1))
            G.add_edge((i, j), (ip, j))
            if j + 1 < Nv:
                G.add_edge((i, j), (ip, j + 1))  # diagonal improves connectivity
    # faces
    for j in range(Nv - 1):
        for i in range(Nu):
            _add_face_tris(faces, i, j, wrap_i=True, wrap_j=False, Nu=Nu, Nv=Nv)
    G.graph["kind"] = "sphere"
    G.graph["R"] = R
    return G, faces

# ---------------------------------------------------------------------------
# Cylinder (radius R, height H) — wrap in u

def cylinder_graph(Nu: int, Nv: int, R: float = 1.0, H: float = 1.0) -> Tuple[nx.Graph, List[Tuple]]:
    assert Nu >= 3 and Nv >= 2
    G = nx.Graph()
    faces: List[Tuple] = []
    for j in range(Nv):
        v = j / (Nv - 1)       # 0..1
        z = H * (v - 0.5)      # center at 0
        for i in range(Nu):
            u = i / Nu
            phi = 2.0 * math.pi * u
            x = R * math.cos(phi)
            y = R * math.sin(phi)
            G.add_node((i, j), pos3=(x, y, z), uv=(u, v), pos=(u, v))
    for j in range(Nv):
        for i in range(Nu):
            ip = (i + 1) % Nu
            if j + 1 < Nv:
                G.add_edge((i, j), (i, j + 1))
            G.add_edge((i, j), (ip, j))
            if j + 1 < Nv:
                G.add_edge((i, j), (ip, j + 1))
    for j in range(Nv - 1):
        for i in range(Nu):
            _add_face_tris(faces, i, j, wrap_i=True, wrap_j=False, Nu=Nu, Nv=Nv)
    G.graph["kind"] = "cylinder"
    G.graph["R"] = R
    G.graph["H"] = H
    return G, faces

# ---------------------------------------------------------------------------
# Saddle (hyperbolic paraboloid) z = a*x^2 - b*y^2 over square patch

def saddle_graph(Nx: int, Ny: int, a: float = 1.0, b: float = 1.0, extent: float = 1.0) -> Tuple[nx.Graph, List[Tuple]]:
    assert Nx >= 3 and Ny >= 3
    G = nx.Graph()
    faces: List[Tuple] = []
    for j in range(Ny):
        v = j / (Ny - 1)
        y = (2*v - 1) * extent
        for i in range(Nx):
            u = i / (Nx - 1)
            x = (2*u - 1) * extent
            z = a * x * x - b * y * y
            G.add_node((i, j), pos3=(x, y, z), uv=(u, v), pos=(u, v))
    # 4-neighborhood + diag
    for j in range(Ny):
        for i in range(Nx):
            if i + 1 < Nx:
                G.add_edge((i, j), (i + 1, j))
            if j + 1 < Ny:
                G.add_edge((i, j), (i, j + 1))
            if i + 1 < Nx and j + 1 < Ny:
                G.add_edge((i, j), (i + 1, j + 1))
    for j in range(Ny - 1):
        for i in range(Nx - 1):
            _add_face_tris(faces, i, j, wrap_i=False, wrap_j=False, Nu=Nx, Nv=Ny)
    G.graph["kind"] = "saddle"
    G.graph["params"] = {"a": a, "b": b, "extent": extent}
    return G, faces
