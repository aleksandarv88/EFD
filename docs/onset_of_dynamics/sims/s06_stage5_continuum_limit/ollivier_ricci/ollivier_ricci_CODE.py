#!/usr/bin/env python3
"""
Stage 5 – Ollivier–Ricci curvature on radius graphs sampled from refined spheres.
Builds radius-limited sphere graphs, samples edges, solves local W1 transport,
and reports κ statistics + convergence plots.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import yaml

# Project utilities (your existing modules)
from lib.efd.io_utils import ensure_version_dir, next_fig_path, sha256_of_file
from docs.onset_of_dynamics.sims.s00_common.sim_paths import SimPaths, sim_root
from docs.onset_of_dynamics.sims.s00_common.sim_results import ResultsWriter
from docs.onset_of_dynamics.sims.s00_common.progress import pbar

# Optional LP solver for W1
try:
    from scipy.optimize import linprog
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# Stage doc mapping (for RESULTS block)
STAGE_DOCS = {
    "S1": Path("docs/onset_of_dynamics/STAGE_1_Re-CoherenceOrderAndInternalTime.txt"),
    "S2": Path("docs/onset_of_dynamics/STAGE_2_PropagationAndSpeedLimit.txt"),
    "S3": Path("docs/onset_of_dynamics/STAGE_3_MetricAndCurvatureFromCoherence.txt"),
    "S4": Path("docs/onset_of_dynamics/STAGE_4_CoherenceCostAndEnergyConservation.txt"),
    "S5": Path("docs/onset_of_dynamics/STAGE_5.txt"),
}


@dataclass
class RunConfig:
    sphere_refine: list[dict]
    R: float
    radius_factor: float
    radius_phys: float | None
    edges_per_level: int
    lazy_alpha: float
    zcut_abs: float
    min_deg: int
    seed: int
    figsize: tuple[float, float]
    dpi: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 5 Ollivier–Ricci convergence.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="YAML config path (default: config/default.yaml)",
    )
    return parser.parse_args()


def load_config(path: Path) -> RunConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    section = raw.get("run", raw) or {}

    def _refine(seq):
        default = [
            {"Nu": 32, "Nv": 16},
            {"Nu": 64, "Nv": 32},
            {"Nu": 128, "Nv": 64},
            {"Nu": 256, "Nv": 128},
        ]
        items = seq if isinstance(seq, list) and seq else default
        return [{"Nu": int(item["Nu"]), "Nv": int(item["Nv"])} for item in items]

    radius_phys_val = section.get("radius_phys")
    return RunConfig(
        sphere_refine=_refine(section.get("sphere_refine")),
        R=float(section.get("R", 1.0)),
        radius_factor=float(section.get("radius_factor", 2.2)),
        radius_phys=None if radius_phys_val is None else float(radius_phys_val),
        edges_per_level=int(section.get("edges_per_level", 512)),
        lazy_alpha=float(section.get("lazy_alpha", 0.5)),
        zcut_abs=float(section.get("zcut_abs", 0.7)),
        min_deg=int(section.get("min_deg", 6)),
        seed=int(section.get("seed", 12345)),
        figsize=tuple(section.get("figsize", (6.5, 4.2))),
        dpi=int(section.get("dpi", 140)),
    )



# ---------------------------------------------------------------------------
# Geometry + graph utilities
# ---------------------------------------------------------------------------


@dataclass
class LocalGraph:
    pos: np.ndarray
    nbrs: list[np.ndarray]
    W: dict[tuple[int, int], float]
    edges: np.ndarray
    R: float


def sphere_uv(R: float, Nu: int, Nv: int) -> np.ndarray:
    us = np.linspace(0, 2 * math.pi, Nu, endpoint=False)
    vs = np.linspace(0, math.pi, Nv)
    verts = []
    for v in vs:
        sv, cv = math.sin(v), math.cos(v)
        for u in us:
            cu, su = math.cos(u), math.sin(u)
            verts.append([R * sv * cu, R * sv * su, R * cv])
    return np.array(verts, dtype=np.float64)


def great_circle_dist(p: np.ndarray, q: np.ndarray, R: float) -> float:
    dot = float(np.dot(p, q)) / (R * R)
    dot = max(-1.0, min(1.0, dot))
    return R * math.acos(dot)


def build_radius_graph_on_sphere(R: float, Nu: int, Nv: int, radius: float) -> LocalGraph:
    V = sphere_uv(R=R, Nu=Nu, Nv=Nv)
    pos = V
    N = V.shape[0]

    def idx(iu: int, iv: int) -> int:
        return iv * Nu + iu

    if radius <= 0:
        raise ValueError("radius must be > 0")

    angle_max = min(radius / R, math.pi)
    max_du = max(1, int(math.ceil((angle_max / (2 * math.pi)) * Nu * 2.0)))
    max_dv = max(1, int(math.ceil((angle_max / math.pi) * Nv * 2.0)))

    edges = []
    nbrs = [set() for _ in range(N)]
    W: dict[tuple[int, int], float] = {}

    for iv in range(Nv):
        for iu in range(Nu):
            i = idx(iu, iv)
            for dv in range(-max_dv, max_dv + 1):
                jv = iv + dv
                if jv < 0 or jv >= Nv:
                    continue
                for du in range(-max_du, max_du + 1):
                    if du == 0 and dv == 0:
                        continue
                    ju = (iu + du) % Nu
                    j = idx(ju, jv)
                    d_geo = great_circle_dist(pos[i], pos[j], R)
                    if d_geo <= radius:
                        a, b = (i, j) if i < j else (j, i)
                        if (a, b) not in W:
                            W[(a, b)] = d_geo
                            edges.append((a, b))
                            nbrs[a].add(b)
                            nbrs[b].add(a)

    edges_arr = np.asarray(edges, dtype=np.int32) if edges else np.empty((0, 2), dtype=np.int32)
    nbr_arrays = [np.fromiter(ns, dtype=np.int32) if ns else np.empty((0,), dtype=np.int32) for ns in nbrs]
    return LocalGraph(pos=pos, nbrs=nbr_arrays, W=W, edges=edges_arr, R=R)


def cost_matrix_geo(pos: np.ndarray, supp_u: np.ndarray, supp_v: np.ndarray, R: float) -> np.ndarray:
    Pu = pos[supp_u]
    Pv = pos[supp_v]
    Su, Sv = Pu.shape[0], Pv.shape[0]
    D = np.empty((Su, Sv), dtype=float)
    for i in range(Su):
        for j in range(Sv):
            D[i, j] = great_circle_dist(Pu[i], Pv[j], R)
    return D


def wasserstein_1(m_src: np.ndarray, m_dst: np.ndarray, D: np.ndarray) -> float:
    """
    W1(m_src, m_dst | cost=D). Uses LP if enabled, otherwise robust greedy fallback.
    Includes guards to avoid implausible values and FP pathologies.
    """
    # Normalize defensively
    m_src = np.clip(m_src, 0.0, None)
    m_dst = np.clip(m_dst, 0.0, None)
    s_sum = float(m_src.sum())
    t_sum = float(m_dst.sum())
    if s_sum <= 0 or t_sum <= 0:
        return float("nan")
    m_src = m_src / s_sum
    m_dst = m_dst / t_sum

    S, T = D.shape
    if S == 0 or T == 0:
        return float("nan")

    # quick sanity bound
    Dmax = float(np.nanmax(D))
    if not np.isfinite(Dmax):
        return float("nan")

    c = D.reshape(-1)
    Aeq = []
    beq = []
    # row sums = m_src
    for i in range(S):
        row = np.zeros(S * T)
        row[i * T:(i + 1) * T] = 1.0
        Aeq.append(row); beq.append(m_src[i])
    # col sums = m_dst
    for j in range(T):
        col = np.zeros(S * T)
        col[j::T] = 1.0
        Aeq.append(col); beq.append(m_dst[j])

    Aeq = np.vstack(Aeq)
    beq = np.array(beq)
    bounds = [(0, None)] * (S * T)

    if SCIPY_OK:
        res = linprog(
            c, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs",
            options={"presolve": True}
        )
        if res.success and np.isfinite(res.fun) and (res.fun <= Dmax + 1e-9):
            return float(res.fun)
        # else: fall through to greedy

    # Greedy upper bound fallback
    ms = m_src.copy()
    mt = m_dst.copy()
    i = j = 0
    cost = 0.0
    while i < S and j < T:
        flow = min(ms[i], mt[j])
        cost += flow * D[i, j]
        ms[i] -= flow
        mt[j] -= flow
        if ms[i] <= 1e-15:
            i += 1
        if mt[j] <= 1e-15:
            j += 1

    if not np.isfinite(cost) or cost < 0 or cost > Dmax + 1e-9:
        return float("nan")
    return float(cost)


def lazy_measure(
    u: int,
    nbrs: list[np.ndarray],
    alpha: float,
    pos: np.ndarray,
    R: float,
    area_weighted: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """m_u = alpha*δ_u + (1-alpha)*weighted(N(u))"""
    deg = int(len(nbrs[u]))
    if deg == 0:
        return np.array([1.0]), np.array([u])
    support = np.concatenate(([u], nbrs[u]))
    m = np.empty(support.shape[0], dtype=float)
    m[0] = alpha
    if not area_weighted:
        m[1:] = (1.0 - alpha) / deg
    else:
        Ns = nbrs[u]
        z = pos[Ns, 2] / R
        w = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
        w_sum = float(w.sum())
        if w_sum <= 0:
            m[1:] = (1.0 - alpha) / deg
        else:
            m[1:] = (1.0 - alpha) * (w / w_sum)
    return m, support
def ricci_edge(u: int, v: int, lg: LocalGraph, alpha: float, R: float) -> float:
    """
    κ_OR(u,v) = 1 - W1(m_u, m_v) / d(u,v),
    with W1 computed on geodesic ground cost between supports.
    """
    m_u, supp_u = lazy_measure(u, lg.nbrs, alpha, lg.pos, R)
    m_v, supp_v = lazy_measure(v, lg.nbrs, alpha, lg.pos, R)
    D = cost_matrix_geo(lg.pos, supp_u, supp_v, R)

    a, b = (u, v) if u < v else (v, u)
    d_uv = lg.W.get((a, b))
    if d_uv is None or d_uv <= 0:
        return float("nan")
    W1 = wasserstein_1(m_u, m_v, D)

    if not np.isfinite(W1):
        return float("nan")
    if W1 < 0:
        W1 = 0.0  # clamp tiny negatives from FP

    return 1.0 - (W1 / d_uv)


# ---------------------------------------------------------------------------
# Plotting + metrics
# ---------------------------------------------------------------------------

def plot_histogram(figdir: Path, values: np.ndarray, figsize: tuple[float, float], dpi: int) -> Path:
    path = next_fig_path(figdir, prefix="ollivier_hist")
    plt.figure(figsize=figsize, dpi=dpi)
    plt.hist(values, bins=40)
    plt.xlabel("κ_OR (edges)")
    plt.ylabel("count")
    plt.title("Ollivier–Ricci distribution (finest level)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_convergence(figdir: Path, hs: Sequence[float], means: Sequence[float],
                     stds: Sequence[float], figsize: tuple[float, float], dpi: int) -> Path:
    path = next_fig_path(figdir, prefix="ollivier_convergence")
    plt.figure(figsize=figsize, dpi=dpi)
    plt.errorbar(hs, means, yerr=stds, fmt="o-")
    plt.gca().invert_xaxis()
    plt.xscale("log")
    plt.xlabel("h (≈ πR/Nu)")
    plt.ylabel("mean κ_OR ± std")
    plt.title("Ollivier–Ricci vs mesh spacing")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def stage_doc_map(sim_dir: Path) -> dict[str, Path]:
    # Your recent fix: parents[2] (not [3])
    docs_root = sim_dir.parents[2]
    return {k: docs_root / path.name for k, path in STAGE_DOCS.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    sim_dir = sim_root(__file__)
    cfg_path = args.config if args.config.is_absolute() else sim_dir / args.config
    cfg = load_config(cfg_path)

    if not SCIPY_OK:
        print("WARNING: scipy unavailable, falling back to greedy W1 (less accurate).")

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    paths = SimPaths(sim_dir, results_name="ollivier_ricci_RESULTS.txt")
    images_dir = ensure_version_dir(paths.images_dir)
    writer = ResultsWriter(paths.results_path, stage_doc_map(sim_dir))

    R = cfg.R
    records: list[dict] = []

    with pbar(total=len(cfg.sphere_refine), desc="levels") as bar:
        for level in cfg.sphere_refine:
            Nu = int(level["Nu"])
            Nv = int(level["Nv"])
            h = math.pi * R / float(Nu)
            radius = cfg.radius_factor * h
            if cfg.radius_phys is not None:
                radius = cfg.radius_phys

            lg = build_radius_graph_on_sphere(R=R, Nu=Nu, Nv=Nv, radius=radius)

            valid_edges = lg.edges
            if valid_edges.size:
                zcut = cfg.zcut_abs
                mask = (
                    (np.abs(lg.pos[valid_edges[:, 0], 2]) < zcut * R)
                    & (np.abs(lg.pos[valid_edges[:, 1], 2]) < zcut * R)
                )
                valid_edges = valid_edges[mask]
                if valid_edges.size:
                    degrees = np.array([len(nbr) for nbr in lg.nbrs], dtype=int)
                    deg_mask = np.minimum(degrees[valid_edges[:, 0]], degrees[valid_edges[:, 1]]) >= cfg.min_deg
                    valid_edges = valid_edges[deg_mask]

            M = valid_edges.shape[0]
            take = min(cfg.edges_per_level, M) if M > 0 else 0
            if take == 0:
                kappa = np.array([float("nan")])
            else:
                idx = np.random.choice(M, size=take, replace=False)
                kappa = np.empty(take, dtype=float)
                for i, edge_idx in enumerate(idx):
                    u, v = int(valid_edges[edge_idx, 0]), int(valid_edges[edge_idx, 1])
                    kappa[i] = ricci_edge(u, v, lg, alpha=cfg.lazy_alpha, R=R)

            finite = kappa[np.isfinite(kappa)]
            mean = float(np.mean(finite)) if finite.size else float("nan")
            std = float(np.std(finite)) if finite.size else float("nan")

            records.append({
                "Nu": Nu, "Nv": Nv, "h": h, "radius": radius,
                "mean": mean, "std": std, "count": int(take)
            })
            bar.update(1)

    records.sort(key=lambda r: r["h"], reverse=True)

    # Finest histogram (recompute with up to 2000 edges for smoother plot)
    finest = records[-1]
    lg_finest = build_radius_graph_on_sphere(
        R=R, Nu=finest["Nu"], Nv=finest["Nv"], radius=finest["radius"]
    )
    valid_edges_f = lg_finest.edges
    if valid_edges_f.size:
        zcut = cfg.zcut_abs
        mask_f = (
            (np.abs(lg_finest.pos[valid_edges_f[:, 0], 2]) < zcut * R)
            & (np.abs(lg_finest.pos[valid_edges_f[:, 1], 2]) < zcut * R)
        )
        valid_edges_f = valid_edges_f[mask_f]
        if valid_edges_f.size:
            degrees_f = np.array([len(nbr) for nbr in lg_finest.nbrs], dtype=int)
            deg_mask_f = np.minimum(degrees_f[valid_edges_f[:, 0]], degrees_f[valid_edges_f[:, 1]]) >= cfg.min_deg
            valid_edges_f = valid_edges_f[deg_mask_f]
    M_f = valid_edges_f.shape[0]
    sample = min(2000, M_f)
    idx_f = np.random.choice(M_f, size=sample, replace=False) if M_f else []
    kappa_finest = []
    for ei in idx_f:
        u, v = int(valid_edges_f[ei, 0]), int(valid_edges_f[ei, 1])
        kappa_finest.append(ricci_edge(u, v, lg_finest, alpha=cfg.lazy_alpha, R=R))
    kappa_finest = np.asarray(kappa_finest, dtype=float)
    kappa_finest = kappa_finest[np.isfinite(kappa_finest)]
    if kappa_finest.size == 0:
        kappa_finest = np.array([0.0], dtype=float)

    hist_path = plot_histogram(images_dir, kappa_finest, cfg.figsize, cfg.dpi)
    conv_path = plot_convergence(
        images_dir,
        [r["h"] for r in records],
        [r["mean"] for r in records],
        [r["std"] for r in records],
        cfg.figsize,
        cfg.dpi,
    )

    metrics = {
        "levels": [(r["Nu"], r["Nv"]) for r in records],
        "h": [r["h"] for r in records],
        "radius": [r["radius"] for r in records],
        "mean_kappa": [r["mean"] for r in records],
        "std_kappa": [r["std"] for r in records],
        "count": [r["count"] for r in records],
        "scipy_available": SCIPY_OK,
    }
    normK = [
        (mk / (rad * rad)) if rad and np.isfinite(mk) else float("nan")
        for mk, rad in zip(metrics["mean_kappa"], metrics["radius"])
    ]
    metrics["mean_kappa_over_r2"] = normK
    print("K_hat (mean_kappa / radius^2) =", normK)
    cfg_snapshot = {"config_path": cfg_path.as_posix(), "run": asdict(cfg)}

    writer.append_run_block(
        images_dir,
        cfg_snapshot,
        metrics,
        [hist_path, conv_path],
        seed=cfg.seed,
        code_sha=sha256_of_file(Path(__file__)),
    )

    print("Ollivier–Ricci sim complete.")


if __name__ == "__main__":
    main()
