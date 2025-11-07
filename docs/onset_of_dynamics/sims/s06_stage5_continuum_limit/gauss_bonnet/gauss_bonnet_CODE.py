#!/usr/bin/env python3
"""
Stage 5 — Gauss–Bonnet global check on refined spheres.

We verify:
  1) Sum of angle defects  Σ_v defect(v)  →  4π (χ=2 for S^2), independent of R.
  2) Mean Gaussian curvature K̄ ≈ (Σ_v defect(v)) / Area  →  1/R^2.

Self-contained: includes a small UV-sphere triangulator; no import from meshes.py.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
import yaml

# Project utilities
from lib.efd.io_utils import ensure_version_dir, next_fig_path, sha256_of_file
from docs.onset_of_dynamics.sims.s00_common.sim_paths import SimPaths, sim_root
from docs.onset_of_dynamics.sims.s00_common.sim_results import ResultsWriter
from docs.onset_of_dynamics.sims.s00_common.progress import pbar

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
    sphere_refine: list[dict]          # e.g. [{"Nu":32,"Nv":16}, ...]
    R_list: list[float]                # e.g. [0.5, 1.0, 2.0]
    seed: int
    figsize: tuple[float, float]
    dpi: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gauss–Bonnet global check on refined spheres")
    p.add_argument("--config", type=Path, default=Path("config/default.yaml"),
                   help="YAML config path (default: config/default.yaml)")
    return p.parse_args()


def load_config(path: Path) -> RunConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    section = raw.get("run", raw) or {}

    def _refine(seq):
        default = [
            {"Nu": 32,  "Nv": 16},
            {"Nu": 64,  "Nv": 32},
            {"Nu": 128, "Nv": 64},
            {"Nu": 256, "Nv": 128},
        ]
        items = seq if isinstance(seq, list) and seq else default
        return [{"Nu": int(it["Nu"]), "Nv": int(it["Nv"])} for it in items]

    R_list = section.get("R_list", [0.5, 1.0, 2.0])
    R_list = [float(r) for r in R_list]

    return RunConfig(
        sphere_refine=_refine(section.get("sphere_refine")),
        R_list=R_list,
        seed=int(section.get("seed", 12345)),
        figsize=tuple(section.get("figsize", (6.5, 4.2))),
        dpi=int(section.get("dpi", 140)),
    )


# ---------- Self-contained UV-sphere (vertices + triangles) ----------

def sphere_uv_with_faces(R: float, Nu: int, Nv: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a UV-sphere with:
      - Nu longitudes (u ∈ [0, 2π), seam wrapped)
      - Nv latitudes including both poles (v ∈ [0, π], poles at v=0,π)
    Returns:
      V: (N,3) float64
      F: (M,3) int32  (CCW triangles)
    Notes:
      Nv must be >= 3 to have caps; Nu >= 3 for meaningful rings.
    """
    if Nu < 3 or Nv < 3:
        raise ValueError("Require Nu>=3, Nv>=3")

    # Latitudes (include both poles)
    v_vals = np.linspace(0.0, math.pi, Nv, endpoint=True)
    u_vals = np.linspace(0.0, 2.0 * math.pi, Nu, endpoint=False)

    verts = []
    # north pole
    verts.append([0.0, 0.0, R])

    # interior rings (skip poles)
    for iv in range(1, Nv - 1):
        v = v_vals[iv]
        sv, cv = math.sin(v), math.cos(v)
        for iu in range(Nu):
            u = u_vals[iu]
            cu, su = math.cos(u), math.sin(u)
            verts.append([R * sv * cu, R * sv * su, R * cv])

    # south pole
    verts.append([0.0, 0.0, -R])

    V = np.asarray(verts, dtype=np.float64)
    north = 0
    south = V.shape[0] - 1

    def ring_index(iv: int, iu: int) -> int:
        """
        Return vertex index in V for ring iv (1..Nv-2), longitude iu (0..Nu-1).
        """
        return 1 + (iv - 1) * Nu + iu

    faces = []

    # top cap: triangles from north pole to ring 1
    for iu in range(Nu):
        a = north
        b = ring_index(1, iu)
        c = ring_index(1, (iu + 1) % Nu)
        faces.append((a, b, c))  # CCW seen from outside

    # quads between interior rings -> split into two triangles
    for iv in range(1, Nv - 2):
        for iu in range(Nu):
            a = ring_index(iv, iu)
            b = ring_index(iv, (iu + 1) % Nu)
            c = ring_index(iv + 1, iu)
            d = ring_index(iv + 1, (iu + 1) % Nu)
            faces.append((a, c, b))  # tri 1
            faces.append((b, c, d))  # tri 2

    # bottom cap: triangles from last ring to south pole
    for iu in range(Nu):
        a = ring_index(Nv - 2, iu)
        b = ring_index(Nv - 2, (iu + 1) % Nu)
        c = south
        faces.append((a, b, c))  # CCW seen from outside

    F = np.asarray(faces, dtype=np.int32)
    return V, F


# ---------- Geometry helpers (angle defect & area) ----------

def triangle_area(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
    return 0.5 * np.linalg.norm(np.cross(q - p, r - p))


def triangle_angles(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> tuple[float, float, float]:
    """Return angles at (p,q,r) using law of cosines on Euclidean edge lengths."""
    a = np.linalg.norm(q - r)
    b = np.linalg.norm(p - r)
    c = np.linalg.norm(p - q)
    def angle(opposite: float, s1: float, s2: float) -> float:
        denom = 2.0 * s1 * s2
        if denom <= 0:
            return 0.0
        x = (s1*s1 + s2*s2 - opposite*opposite) / denom
        x = max(-1.0, min(1.0, x))
        return math.acos(x)
    A = angle(a, b, c)  # at p
    B = angle(b, a, c)  # at q
    C = angle(c, a, b)  # at r
    return A, B, C


def angle_defects_and_area(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Compute per-vertex angle defect (2π - sum of incident angles) and total mesh area.
    """
    nV = V.shape[0]
    angle_sum = np.zeros(nV, dtype=float)
    total_area = 0.0
    for f0, f1, f2 in F:
        p, q, r = V[f0], V[f1], V[f2]
        A, B, C = triangle_angles(p, q, r)
        angle_sum[f0] += A
        angle_sum[f1] += B
        angle_sum[f2] += C
        total_area += triangle_area(p, q, r)
    defect = 2.0 * math.pi - angle_sum
    return defect, total_area


# ---------- Plotting ----------

def plot_gb_error(figdir: Path, recs: list[dict], figsize: tuple[float, float], dpi: int) -> Path:
    """
    |Σ defect - 4π| vs h for each R, on one figure.
    h ≈ πR/Nu (characteristic spacing along u).
    """
    path = next_fig_path(figdir, prefix="gauss_bonnet_error")
    plt.figure(figsize=figsize, dpi=dpi)
    byR: dict[float, list[dict]] = {}
    for r in recs:
        byR.setdefault(r["R"], []).append(r)
    for R, rows in byR.items():
        rows = sorted(rows, key=lambda x: x["h"], reverse=True)
        hs   = [x["h"] for x in rows]
        errs = [abs(x["sum_defect"] - 4.0 * math.pi) for x in rows]
        plt.plot(hs, errs, "o-", label=f"R={R}")
    plt.gca().invert_xaxis()
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("h (≈ πR/Nu)  [log]")
    plt.ylabel("|Σ defect − 4π|  [log]")
    plt.title("Gauss–Bonnet error vs mesh spacing")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_kbar(figdir: Path, recs: list[dict], figsize: tuple[float, float], dpi: int) -> Path:
    """
    K̄ = (Σ defect) / Area  vs  refinement, with the theoretical 1/R² reference per R.
    """
    path = next_fig_path(figdir, prefix="gauss_bonnet_kbar")
    plt.figure(figsize=figsize, dpi=dpi)
    byR: dict[float, list[dict]] = {}
    for r in recs:
        byR.setdefault(r["R"], []).append(r)
    for R, rows in byR.items():
        rows = sorted(rows, key=lambda x: x["h"], reverse=True)
        hs   = [x["h"] for x in rows]
        kbar = [x["kbar"] for x in rows]
        plt.plot(hs, kbar, "o-", label=f"R={R}  (theory {1.0/(R*R):.3f})")
    plt.gca().invert_xaxis()
    plt.xscale("log")
    plt.xlabel("h (≈ πR/Nu)  [log]")
    plt.ylabel("K̄ = (Σ defect) / Area")
    plt.title("Mean Gaussian curvature vs mesh spacing (sphere)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


# ---------- Stage doc mapping ----------

def stage_doc_map(sim_dir: Path) -> dict[str, Path]:
    docs_root = sim_dir.parents[2]  # matches your fixed layout
    return {k: docs_root / p.name for k, p in STAGE_DOCS.items()}


# ---------- Main ----------

def main() -> None:
    args = parse_args()
    sim_dir = sim_root(__file__)
    cfg_path = args.config if args.config.is_absolute() else sim_dir / args.config
    cfg = load_config(cfg_path)

    paths = SimPaths(sim_dir, results_name="gauss_bonnet_RESULTS.txt")
    images_dir = ensure_version_dir(paths.images_dir)
    writer = ResultsWriter(paths.results_path, stage_doc_map(sim_dir))

    records: list[dict] = []
    with pbar(total=len(cfg.R_list) * len(cfg.sphere_refine), desc="Gauss–Bonnet") as bar:
        for R in cfg.R_list:
            for level in cfg.sphere_refine:
                Nu, Nv = int(level["Nu"]), int(level["Nv"])
                h = math.pi * R / float(Nu)  # characteristic spacing in u

                V, F = sphere_uv_with_faces(R=R, Nu=Nu, Nv=Nv)
                defect, area = angle_defects_and_area(V, F)
                sum_def = float(np.sum(defect))
                kbar = sum_def / area if area > 0 else float("nan")

                records.append({
                    "R": R,
                    "Nu": Nu,
                    "Nv": Nv,
                    "h": h,
                    "sum_defect": sum_def,
                    "area": area,
                    "kbar": kbar,
                    "theory_k": 1.0 / (R * R),
                    "gb_error": abs(sum_def - 4.0 * math.pi),
                })
                bar.update(1)

    fig_err = plot_gb_error(images_dir, records, cfg.figsize, cfg.dpi)
    fig_k   = plot_kbar(images_dir, records, cfg.figsize, cfg.dpi)

    metrics = {
        "R_list": cfg.R_list,
        "levels": [(r["Nu"], r["Nv"]) for r in records],
        "h": [r["h"] for r in records],
        "sum_defect": [r["sum_defect"] for r in records],
        "area": [r["area"] for r in records],
        "kbar": [r["kbar"] for r in records],
        "theory_k": [r["theory_k"] for r in records],
        "gb_error": [r["gb_error"] for r in records],
        "images": [str(fig_err), str(fig_k)],
    }
    cfg_snapshot = {"config_path": cfg_path.as_posix(), "run": asdict(cfg)}

    writer.append_run_block(
        images_dir,
        cfg_snapshot,
        metrics,
        [fig_err, fig_k],
        seed=cfg.seed,
        code_sha=sha256_of_file(Path(__file__)),
    )

    print("Gauss–Bonnet check complete.")


if __name__ == "__main__":
    main()
