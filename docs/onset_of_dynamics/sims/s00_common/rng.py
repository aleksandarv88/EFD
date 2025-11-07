"""
rng.py — Seeded RNG helpers for deterministic sims.

Provides:
  - make_rng(seed)             → np.random.Generator
  - split_rng(rng, n)          → list of child Generators
  - temp_seed(seed)            → context manager to temporarily set bitgen state
"""

from __future__ import annotations
from contextlib import contextmanager
from typing import List
import numpy as np

def make_rng(seed: int | None) -> np.random.Generator:
    """Construct a PCG64-based Generator. If seed is None, uses entropy."""
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(np.random.PCG64(seed))

def split_rng(rng: np.random.Generator, n: int) -> List[np.random.Generator]:
    """
    Fast child streams via SeedSequence spawning.
    """
    ss = np.random.SeedSequence(rng.integers(0, 2**32 - 1, dtype=np.uint32))
    children = ss.spawn(n)
    return [np.random.default_rng(np.random.PCG64(s)) for s in children]

@contextmanager
def temp_seed(rng: np.random.Generator, seed: int):
    """
    Temporarily set RNG to a child stream from a given seed, restoring after.
    Useful for reproducible sub-blocks inside a sim.
    """
    state = rng.bit_generator.state
    try:
        rng.bit_generator = np.random.PCG64(seed)
        yield rng
    finally:
        rng.bit_generator.state = state
