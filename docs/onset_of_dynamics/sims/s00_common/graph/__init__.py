# docs/onset_of_dynamics/sims/s00_common/graph/__init__.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import networkx as nx
import numpy as np


@dataclass(frozen=True)
class GridSpec:
    nx: int
    ny: int

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.nx, self.ny)


@dataclass(frozen=True)
class GraphPack:
    G: nx.Graph
    pos: np.ndarray  # shape (N, 2), dtype float32
    avg_degree: float
    builder: str


@dataclass(frozen=True)
class Pair:
    u: int
    v: int
    d_euclid: float
    hops: int


@dataclass(frozen=True)
class RunConfig:
    refinements: List[Tuple[int, int]]
    pairs_per_level: int
    band: Tuple[float, float]
    radius_phys: float
    radius_h_factor: float
    min_hops: int
    seed: int
    max_workers: int | None = None

