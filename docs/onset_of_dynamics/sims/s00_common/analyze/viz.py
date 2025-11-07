# docs/onset_of_dynamics/sims/s00_common/analyze/viz.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

from .errors import fit_loglog


def plot_scatter(
    figpath: Path,
    euclid_all: np.ndarray,
    graph_all: np.ndarray,
    figsize: Tuple[float, float],
    dpi: int,
) -> None:
    """Scatter of graph distances vs Euclidean distances."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.scatter(euclid_all, graph_all, s=12, alpha=0.5, edgecolor="none")
    max_eu = float(np.max(euclid_all)) if euclid_all.size else 0.0
    max_gr = float(np.max(graph_all)) if graph_all.size else 0.0
    lims = np.array([0.0, max(max_eu, max_gr, 1e-6)])
    ax.plot(lims, lims, color="black", linewidth=1.0, linestyle="--", label="y=x")
    ax.set_xlabel("Euclidean distance")
    ax.set_ylabel("Graph distance")
    ax.set_title("Graph vs Euclidean distances")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    figpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figpath)
    plt.close(fig)


def plot_convergence(
    figpath: Path,
    h_list: Iterable[float],
    L2_list: Iterable[float],
    figsize: Tuple[float, float],
    dpi: int,
) -> float:
    """Log-log convergence curve; returns fitted slope."""
    import matplotlib.pyplot as plt

    h_arr = np.array(list(h_list), dtype=np.float64)
    L2_arr = np.array(list(L2_list), dtype=np.float64)
    slope = fit_loglog(h_arr, L2_arr)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.loglog(h_arr, L2_arr, marker="o", linewidth=1.5)
    ax.set_xlabel("h (grid spacing)")
    ax.set_ylabel("L2 error")
    ax.set_title(f"Convergence (slope ≈ {slope:.3f})")
    ax.grid(which="both", alpha=0.3)
    fig.tight_layout()
    figpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figpath)
    plt.close(fig)
    return slope
