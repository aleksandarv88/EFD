#!/usr/bin/env python3
"""
Stage 5 – OR vs. Defect correlation on refined spheres.

For each refinement level:
  1) Build (V,F) sphere mesh and a radius-limited geodesic graph.
  2) Compute per-vertex angle defect and barycentric vertex area => K_v = defect_v / A_v.
  3) Sample edges (u,v), compute κ_OR(u,v) with geodesic ground cost.
  4) For each sampled edge use K_edge = 0.5*(K_u+K_v) and X = K_edge * r^2 (dimensionless).
  5) Plot scatter κ_OR vs X, fit κ ≈ a*X + b, report slope/intercept and Pearson r.

Outputs:
  - scatter (with fitted line) per finest level
  - convergence-ish summary over levels in RESULTS
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import yaml

# Project utils
from lib.efd.io_utils import ensure_version_dir, next_fig_path, sha256_of_file
from docs.onset_of_dynamics.sims.s00_common.sim_paths import SimPaths, sim_root
from docs.onset_of_dynamics.sims.s00_common.sim_results import ResultsWriter
from docs.onset_of_dynamics.sims.s00_common.progress import pbar


try:
    from scipy.optimize import linprog
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

STAGE_DOCS = {
    "S1": Path("docs/onset_of_dynamics/STAGE_1_Re-CoherenceOrderAndInternalTime.txt"),
    "S2": Path("docs/onset_of_dynamics/STAGE_2_PropagationAndSpeedLimit.txt"),
    "S3": Path("docs/onset_of_dynamics/STAGE_3_MetricAndCurvatureFromCoherence.txt"),
    "S4": Path("docs/onset_of_dynamics/STAGE_4_CoherenceCostAndEnergyConservation.txt"),
    "S5": Path("docs/onset_of_dynamics/STAGE_5.txt"),
}

# -----------------------
# Config
# -----------------------

@dataclass
class RunConfig:
    sphere_refine: list[dict]
    R: float
    radius_factor: float
    radius_phys: float | None
    edges_per_level: int
    lazy_alpha: float
    seed: int
    figsize: Tuple[float, float]
    dpi: int
    zcut: float
    deg_min: int

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OR vs. Defect correlation (sphere).")
    p.add_argument("--config", type=Path, default=Path("config/default.yaml"),
                   help="YAML config path (default: config/default.yaml)")
    return p.parse_args()

def load_config(path: Path) -> RunConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    r = raw.get("run", raw) or {}
    def _refine(seq):
        default = [
            {"Nu": 64, "Nv": 32},
            {"Nu": 128, "Nv": 64},
            {"Nu": 256, "Nv": 128},
        ]
        items = seq if isinstance(seq, list) and seq else default
        return [{"Nu": int(x["Nu"]), "Nv": int(x["Nv"])} for x in items]
    rp = r.get("radius_phys")
    return RunConfig(
        sphere_refine=_refine(r.get("sphere_refine")),
        R=float(r.get("R", 1.0)),
        radius_factor=float(r.get("radius_factor", 1.8)),
        radius_phys=None if rp is None else float(rp),
        edges_per_level=int(r.get("edges_per_level", 512)),
        lazy_alpha=float(r.get("lazy_alpha", 0.5)),
        seed=int(r.get("seed", 12345)),
        figsize=tuple(r.get("figsize", (6.5, 4.2))),
        dpi=int(r.get("dpi", 140)),
        zcut=float(r.get("zcut", 0.7)),
        deg_min=int(r.get("deg_min", 6)),
    )
def sphere_uv(R: float, Nu: int, Nv: int):
    """
    Return (V,F) for a UV grid sphere:
      V shape (Nu*Nv, 3), ordered by rows iv=0..Nv-1, columns iu=0..Nu-1
      F triangulates quads between consecutive latitude rings.
    NOTE: Poles are represented by Nu coincident vertices (degenerate ring);
          our angle/area code guards zero-length edges, so this is OK and
          keeps indexing consistent with OR graph construction.
    """
    us = np.linspace(0.0, 2.0 * math.pi, Nu, endpoint=False)
    vs = np.linspace(0.0, math.pi, Nv)

    verts = []
    for v in vs:
        sv, cv = math.sin(v), math.cos(v)
        for u in us:
            cu, su = math.cos(u), math.sin(u)
            verts.append([R * sv * cu, R * sv * su, R * cv])
    V = np.asarray(verts, dtype=np.float64)

    faces = []
    for iv in range(Nv - 1):
        base0 = iv * Nu
        base1 = (iv + 1) * Nu
        for iu in range(Nu):
            a = base0 + iu
            b = base0 + ((iu + 1) % Nu)
            c = base1 + iu
            d = base1 + ((iu + 1) % Nu)
            faces.append([a, c, b])
            faces.append([b, c, d])
    F = np.asarray(faces, dtype=np.int32)
    return V, F
def stage_doc_map(sim_dir: Path) -> dict[str, Path]:
    docs_root = sim_dir.parents[2]
    return {k: docs_root / p.name for k, p in STAGE_DOCS.items()}

# -----------------------
# Geometry helpers
# -----------------------

def great_circle_dist(p: np.ndarray, q: np.ndarray, R: float) -> float:
    dot = float(np.dot(p, q)) / (R * R)
    dot = max(-1.0, min(1.0, dot))
    return R * math.acos(dot)

def tri_angles(p0, p1, p2) -> Tuple[float, float, float]:
    # returns angles at (p0,p1,p2)
    def ang(a,b,c):
        u = a - b
        v = c - b
        nu = np.linalg.norm(u); nv = np.linalg.norm(v)
        if nu == 0 or nv == 0:
            return 0.0
        cosv = np.dot(u, v) / (nu * nv)
        cosv = float(np.clip(cosv, -1.0, 1.0))
        return float(np.arccos(cosv))
    return ang(p1, p0, p2), ang(p0, p1, p2), ang(p0, p2, p1)

def tri_area(p0, p1, p2) -> float:
    # 3D triangle area
    return 0.5 * float(np.linalg.norm(np.cross(p1 - p0, p2 - p0)))

def vertex_defect_and_area(V: np.ndarray, F: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Angle defect per vertex and barycentric vertex area (1/3 per incident face).
    """
    n = V.shape[0]
    defect = np.zeros(n, dtype=float)
    A_v = np.zeros(n, dtype=float)
    for f in F:
        i, j, k = int(f[0]), int(f[1]), int(f[2])
        a0, a1, a2 = tri_angles(V[i], V[j], V[k])
        defect[i] += a0
        defect[j] += a1
        defect[k] += a2
        A = tri_area(V[i], V[j], V[k])
        A_v[i] += A / 3.0
        A_v[j] += A / 3.0
        A_v[k] += A / 3.0
    defect = (2.0 * math.pi) - defect
    return defect, A_v

# -----------------------
# OR machinery (local)
# -----------------------

def build_radius_graph_on_sphere(V: np.ndarray, R: float, Nu: int, Nv: int, radius: float):
    """
    Build radius-limited graph using geodesic threshold on the UV grid adjacency.
    Returns: nbrs (list of np.ndarray), edges (E,2), W dict[(i,j)]=geo_dist
    """
    N = V.shape[0]
    def idx(iu: int, iv: int) -> int:
        return iv * Nu + iu
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
                    dgeo = great_circle_dist(V[i], V[j], R)
                    if dgeo <= radius:
                        a, b = (i, j) if i < j else (j, i)
                        if (a, b) not in W:
                            W[(a, b)] = dgeo
                            edges.append((a, b))
                            nbrs[a].add(b)
                            nbrs[b].add(a)
    E = np.asarray(edges, dtype=np.int32) if edges else np.empty((0, 2), dtype=np.int32)
    nbrs = [np.fromiter(s, dtype=np.int32) if s else np.empty((0,), dtype=np.int32) for s in nbrs]
    return nbrs, E, W

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
    # normalize defensively
    m_src = np.clip(m_src, 0.0, None); m_dst = np.clip(m_dst, 0.0, None)
    ss, tt = float(m_src.sum()), float(m_dst.sum())
    if ss <= 0 or tt <= 0:
        return float("nan")
    m_src /= ss; m_dst /= tt
    S, T = D.shape
    if S == 0 or T == 0:
        return float("nan")
    c = D.reshape(-1)
    Aeq = []
    beq = []
    for i in range(S):
        row = np.zeros(S*T); row[i*T:(i+1)*T] = 1.0
        Aeq.append(row); beq.append(m_src[i])
    for j in range(T):
        col = np.zeros(S*T); col[j::T] = 1.0
        Aeq.append(col); beq.append(m_dst[j])
    Aeq = np.vstack(Aeq); beq = np.array(beq)
    bounds = [(0, None)] * (S*T)
    if SCIPY_OK:
        res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs", options={"presolve": True})
        if res.success and np.isfinite(res.fun):
            return float(res.fun)
    # greedy fallback
    ms = m_src.copy(); mt = m_dst.copy()
    i = j = 0; cost = 0.0
    while i < S and j < T:
        flow = min(ms[i], mt[j])
        cost += flow * D[i, j]
        ms[i] -= flow; mt[j] -= flow
        if ms[i] <= 1e-15: i += 1
        if mt[j] <= 1e-15: j += 1
    return float(cost)

def lazy_measure(u: int, nbrs: list[np.ndarray], alpha: float, pos: np.ndarray, R: float, area_weighted=True):
    deg = len(nbrs[u])
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
        w = np.sqrt(np.clip(1.0 - z*z, 0.0, 1.0))
        sw = float(w.sum())
        m[1:] = (1.0 - alpha) * (w / sw if sw > 0 else 1.0/deg)
    return m, support

def ricci_edge(u: int, v: int, pos: np.ndarray, nbrs: list[np.ndarray], W: dict, alpha: float, R: float) -> float:
    a, b = (u, v) if u < v else (v, u)
    d_uv = W.get((a, b))
    if d_uv is None or d_uv <= 0:
        return float("nan")
    m_u, su = lazy_measure(u, nbrs, alpha, pos, R, area_weighted=True)
    m_v, sv = lazy_measure(v, nbrs, alpha, pos, R, area_weighted=True)
    D = cost_matrix_geo(pos, su, sv, R)
    W1 = wasserstein_1(m_u, m_v, D)
    if not np.isfinite(W1):  # guard
        return float("nan")
    return 1.0 - (W1 / d_uv)

# -----------------------
# Plotting
# -----------------------

def scatter_with_fit(figdir: Path, X: np.ndarray, Y: np.ndarray, figsize, dpi) -> Path:
    path = next_fig_path(figdir, prefix="or_vs_defect_scatter")
    plt.figure(figsize=figsize, dpi=dpi)
    plt.scatter(X, Y, s=10, alpha=0.5)
    if X.size >= 2 and np.all(np.isfinite(X)) and np.all(np.isfinite(Y)):
        a, b = np.polyfit(X, Y, 1)
        xs = np.linspace(np.min(X), np.max(X), 200)
        plt.plot(xs, a*xs + b)
        lbl = f"fit: κ ≈ {a:.3f}·(K*r²) + {b:.3f}"
        plt.legend([lbl], loc="best")
    plt.xlabel("K_edge · r²   (dimensionless)")
    plt.ylabel("κ_OR (edge)")
    plt.title("Ollivier–Ricci vs. local Gaussian curvature (sphere)")
    plt.tight_layout()
    plt.savefig(path); plt.close()
    return path

# -----------------------
# Main
# -----------------------

def main():
    args = parse_args()
    sim_dir = sim_root(__file__)
    cfg_path = args.config if args.config.is_absolute() else sim_dir / args.config
    cfg = load_config(cfg_path)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    paths = SimPaths(sim_dir, results_name="or_vs_defect_RESULTS.txt")
    images_dir = ensure_version_dir(paths.images_dir)
    writer = ResultsWriter(paths.results_path, stage_doc_map(sim_dir))

    records = []
    all_levels = []
    with pbar(total=len(cfg.sphere_refine), desc="levels") as bar:
        for level in cfg.sphere_refine:
            Nu, Nv = int(level["Nu"]), int(level["Nv"])
            # Mesh
            V, F = sphere_uv(R=cfg.R, Nu=Nu, Nv=Nv)  # (N,3), (M,3)
            # Characteristic spacing
            h = math.pi * cfg.R / float(Nu)
            radius = cfg.radius_phys if cfg.radius_phys is not None else cfg.radius_factor * h

            # Defect + vertex area -> K_v
            defect, A_v = vertex_defect_and_area(V, F)
            K_v = np.divide(defect, A_v, out=np.full_like(defect, np.nan), where=A_v>0)

            # Neighborhood graph
            nbrs, E, W = build_radius_graph_on_sphere(V, cfg.R, Nu, Nv, radius)

            # Edge filters (optional but recommended)
            valid = E
            if valid.size:
                z = V[:, 2]
                mask = (np.abs(z[valid[:, 0]]) < cfg.zcut * cfg.R) & (np.abs(z[valid[:, 1]]) < cfg.zcut * cfg.R)
                valid = valid[mask]
                if valid.size:
                    degs = np.array([len(n) for n in nbrs], dtype=int)
                    mask_deg = np.minimum(degs[valid[:, 0]], degs[valid[:, 1]]) >= cfg.deg_min
                    valid = valid[mask_deg]

            M = valid.shape[0]
            take = min(cfg.edges_per_level, M) if M > 0 else 0
            if take == 0:
                metrics = {
                    "Nu": Nu, "Nv": Nv, "h": h, "radius": radius,
                    "count": 0, "slope": float("nan"), "intercept": float("nan"),
                    "pearson_r": float("nan"), "pearson_r2": float("nan"),
                    "mean_kappa": float("nan"), "std_kappa": float("nan"),
                }
                records.append(metrics)
                all_levels.append({"X": [], "Y": []})
                bar.update(1)
                continue

            idx = np.random.choice(M, size=take, replace=False)
            Ed = valid[idx]

            # Compute OR and K_edge * r^2
            kappas = np.empty(take, dtype=float)
            X = np.empty(take, dtype=float)
            r2 = radius * radius
            for i, (u, v) in enumerate(Ed):
                kappas[i] = ricci_edge(int(u), int(v), V, nbrs, W, cfg.lazy_alpha, cfg.R)
                Ku = K_v[int(u)]; Kv = K_v[int(v)]
                X[i] = 0.5 * (Ku + Kv) * r2  # dimensionless predictor

            # Clean NaNs
            finite = np.isfinite(kappas) & np.isfinite(X)
            Yc = kappas[finite]; Xc = X[finite]

            # Fit + correlation
            if Xc.size >= 2:
                a, b = np.polyfit(Xc, Yc, 1)
                r = float(np.corrcoef(Xc, Yc)[0, 1]) if Xc.size >= 2 else float("nan")
            else:
                a = b = r = float("nan")

            metrics = {
                "Nu": Nu, "Nv": Nv, "h": h, "radius": radius,
                "count": int(Xc.size),
                "slope": float(a), "intercept": float(b),
                "pearson_r": r, "pearson_r2": (r*r if np.isfinite(r) else float("nan")),
                "mean_kappa": float(np.mean(Yc)) if Yc.size else float("nan"),
                "std_kappa": float(np.std(Yc)) if Yc.size else float("nan"),
            }
            records.append(metrics)
            all_levels.append({"X": Xc, "Y": Yc})
            bar.update(1)

    # Choose finest level for plot
    records.sort(key=lambda m: m["h"])
    finest = records[0]
    finest_idx = None
    for i, m in enumerate(records):
        if m["Nu"] == finest["Nu"] and m["Nv"] == finest["Nv"]:
            finest_idx = i; break
    if finest_idx is None:
        finest_idx = len(records) - 1

    Xf = np.asarray(all_levels[finest_idx]["X"], dtype=float)
    Yf = np.asarray(all_levels[finest_idx]["Y"], dtype=float)
    scatter_path = scatter_with_fit(images_dir, Xf, Yf, cfg.figsize, cfg.dpi)

    # RESULTS block
    cfg_snapshot = {"config_path": cfg_path.as_posix(), "run": asdict(cfg)}
    metrics = {
        "levels": [(m["Nu"], m["Nv"]) for m in records],
        "h": [m["h"] for m in records],
        "radius": [m["radius"] for m in records],
        "count": [m["count"] for m in records],
        "slope": [m["slope"] for m in records],
        "intercept": [m["intercept"] for m in records],
        "pearson_r": [m["pearson_r"] for m in records],
        "pearson_r2": [m["pearson_r2"] for m in records],
        "mean_kappa": [m["mean_kappa"] for m in records],
        "std_kappa": [m["std_kappa"] for m in records],
        "scipy_available": SCIPY_OK,
    }

    writer.append_run_block(
        images_dir,
        cfg_snapshot,
        metrics,
        [scatter_path],
        seed=cfg.seed,
        code_sha=sha256_of_file(Path(__file__)),
    )

    print("OR vs Defect correlation complete.")

if __name__ == "__main__":
    main()
