# docs/onset_of_dynamics/sims/s00_common/analyze/errors.py
from __future__ import annotations

import numpy as np


def l1_l2_linf(abs_errs: np.ndarray) -> tuple[float, float, float]:
    if abs_errs.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    L1 = float(np.mean(abs_errs))
    L2 = float(np.sqrt(np.mean(abs_errs**2)))
    Linf = float(np.max(abs_errs))
    return L1, L2, Linf


def fit_loglog(h: np.ndarray, e: np.ndarray) -> float:
    """Slope of log10(e) vs log10(h), ignore nonpositive entries."""
    mask = (h > 0) & (e > 0)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    h_sel = np.log10(h[mask])
    e_sel = np.log10(e[mask])
    slope, _ = np.polyfit(h_sel, e_sel, 1)
    return float(slope)

