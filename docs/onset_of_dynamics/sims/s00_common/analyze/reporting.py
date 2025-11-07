# docs/onset_of_dynamics/sims/s00_common/analyze/reporting.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


def make_metrics_dict(
    h_list: List[float],
    L1: List[float],
    L2: List[float],
    Linf: List[float],
    slope: float,
    band: tuple[float, float],
    band_levels: List[tuple[float, float]] | None,
    radius_used: List[float],
    images: Iterable[Path],
    levels: List[tuple[int, int]],
    pairs_total: int,
) -> dict:
    """Return flat dict ready for sim_results.append()."""
    return {
        "levels": levels,
        "h": h_list,
        "L1": L1,
        "L2": L2,
        "Linf": Linf,
        "slope": slope,
        "band": list(band),
        "band_levels": [list(b) for b in (band_levels or [])],
        "radius": radius_used,
        "images": [str(p) for p in images],
        "pairs_total": pairs_total,
    }
