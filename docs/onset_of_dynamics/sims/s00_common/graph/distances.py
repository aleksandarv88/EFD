# docs/onset_of_dynamics/sims/s00_common/graph/distances.py
from __future__ import annotations

from dataclasses import dataclass
import math
import os
import pickle
import tempfile
from typing import Dict, Iterable

import networkx as nx
import numpy as np

from ..parallel import _init_worker, get_global, map_process


def sssp_lengths(
    G: nx.Graph,
    sources: Iterable[int],
    *,
    max_workers: int | None = None,
) -> Dict[int, Dict[int, float]]:
    """Return map: src -> {node: dist}. Prefer SciPy CSR Dijkstra; fallback NX."""
    src_list = list(dict.fromkeys(sources))
    if not src_list:
        return {}

    nodes = list(G.nodes())
    node_index = {node: idx for idx, node in enumerate(nodes)}
    src_indices = [node_index[s] for s in src_list]

    # First try SciPy (fast + multi-source)
    try:
        from scipy.sparse.csgraph import dijkstra

        try:
            matrix = nx.to_scipy_sparse_array(
                G, nodelist=nodes, weight="weight", dtype=np.float64, format="csr"
            )
        except AttributeError:
            matrix = nx.to_scipy_sparse_matrix(
                G, nodelist=nodes, weight="weight", dtype=np.float64, format="csr"
            )
        dist = dijkstra(matrix, directed=False, indices=src_indices, return_predecessors=False)
        dist = np.atleast_2d(dist)
        out: Dict[int, Dict[int, float]] = {}
        for row, src in zip(dist, src_list):
            src_map = {}
            for node, d in zip(nodes, row):
                if math.isfinite(d):
                    src_map[node] = float(d)
            out[src] = src_map
        return out
    except Exception:
        pass

    # Fallback: NetworkX Dijkstra, optionally parallelised
    max_workers = _choose_workers(max_workers, len(src_list))
    if max_workers <= 1:
        return {s: nx.single_source_dijkstra_path_length(G, s, weight="weight") for s in src_list}

    with _pickled_graph(G) as graph_pickle:
        results = map_process(
            _worker_sssp,
            src_list,
            max_workers=max_workers,
            desc="sssp",
            initializer=_init_worker,
            initargs=(0, None, graph_pickle, "graph", None, None),
        )
    dist_map: Dict[int, Dict[int, float]] = {}
    for src, res in zip(src_list, results):
        if isinstance(res, Exception):
            raise res
        dist_map[src] = res
    return dist_map


def _choose_workers(max_workers: int | None, tasks: int) -> int:
    if tasks <= 1:
        return 1
    if max_workers is None or max_workers <= 0:
        return min(os.cpu_count() or 1, tasks)
    return min(max_workers, tasks)


def _worker_sssp(src: int):
    G = get_global("graph")
    if G is None:
        raise RuntimeError("Graph was not loaded in worker.")
    return nx.single_source_dijkstra_path_length(G, src, weight="weight")


@dataclass
class _PickledGraph:
    path: str

    def __enter__(self):
        return self.path

    def __exit__(self, exc_type, exc, tb):
        try:
            os.remove(self.path)
        except OSError:
            pass


def _pickled_graph(G: nx.Graph) -> _PickledGraph:
    fd, path = tempfile.mkstemp(prefix="efd_graph_", suffix=".pkl")
    os.close(fd)
    with open(path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    return _PickledGraph(path)
