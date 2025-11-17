#!/usr/bin/env python3
"""Stage 5 – curvature via angle defect, refactored onto common sim plumbing."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml

from lib.efd.io_utils import ensure_version_dir, next_fig_path, sha256_of_file
from docs.onset_of_dynamics.sims.s00_common.sim_paths import SimPaths, sim_root
from docs.onset_of_dynamics.sims.s00_common.sim_results import ResultsWriter
from docs.onset_of_dynamics.sims.s00_common.progress import pbar


STAGE_DOCS = {
    "S1": Path("docs/onset_of_dynamics/STAGE_1_Re-CoherenceOrderAndInternalTime.txt"),
    "S2": Path("docs/onset_of_dynamics/STAGE_2_PropagationAndSpeedLimit.txt"),
    "S3": Path("docs/onset_of_dynamics/STAGE_3_MetricAndCurvatureFromCoherence.txt"),
    "S4": Path("docs/onset_of_dynamics/STAGE_4_CoherenceCostAndEnergyConservation.txt"),
    "S5": Path("docs/onset_of_dynamics/STAGE_5.txt"),
}


@dataclass
class RunConfig:
    seed: int
    plane: dict
    sphere: dict
    saddle: dict
    sphere_refine: list[dict]
    figsize: tuple[float, float]
    dpi: int


@dataclass
class CurvStats:
    L1: float
    L2: float
    Linf: float
    mean: float
    std: float
    n_used: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 5 curvature via angle defect.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="YAML config path (default: configs/default.yaml)",
    )
    return parser.parse_args()


def load_config(path: Path) -> RunConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    section = raw.get("run", raw) or {}

    def _plane(cfg):
        defaults = {"Nx": 40, "Ny": 40, "extent": 1.0}
        data = {**defaults, **(cfg or {})}
        return {"Nx": int(data["Nx"]), "Ny": int(data["Ny"]), "extent": float(data["extent"])}

    def _sphere(cfg):
        defaults = {"R": 1.0, "Nu": 64, "Nv": 32}
        data = {**defaults, **(cfg or {})}
        return {"R": float(data["R"]), "Nu": int(data["Nu"]), "Nv": int(data["Nv"])}

    def _saddle(cfg):
        defaults = {"Nx": 40, "Ny": 40, "extent": 0.6, "alpha": 0.6}
        data = {**defaults, **(cfg or {})}
        return {
            "Nx": int(data["Nx"]),
            "Ny": int(data["Ny"]),
            "extent": float(data["extent"]),
            "alpha": float(data["alpha"]),
        }

    def _sphere_refine(seq):
        default = [
            {"Nu": 16, "Nv": 8},
            {"Nu": 32, "Nv": 16},
            {"Nu": 64, "Nv": 32},
            {"Nu": 128, "Nv": 64},
        ]
        items = seq if isinstance(seq, list) and seq else default
        cleaned = []
        for item in items:
            cleaned.append({"Nu": int(item["Nu"]), "Nv": int(item["Nv"])})
        return cleaned

    figsize = tuple(section.get("figsize", (6.6, 4.2)))
    dpi = int(section.get("dpi", 140))
    seed = int(section.get("seed", 12345))

    return RunConfig(
        seed=seed,
        plane=_plane(section.get("plane")),
        sphere=_sphere(section.get("sphere")),
        saddle=_saddle(section.get("saddle")),
        sphere_refine=_sphere_refine(section.get("sphere_refine")),
        figsize=figsize,
        dpi=dpi,
    )


# ---------------------------------------------------------------------------
# Geometry helpers (mostly from the original standalone script)
# ---------------------------------------------------------------------------
def tri_area(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
    return 0.5 * np.linalg.norm(np.cross(q - p, r - p))


def angle_at(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
    v1 = p - q
    v2 = r - q
    a = np.linalg.norm(v1)
    b = np.linalg.norm(v2)
    if a == 0 or b == 0:
        return 0.0
    c = np.dot(v1, v2) / (a * b)
    c = np.clip(c, -1.0, 1.0)
    return math.acos(c)


def vertex_mixed_area(vid: int, V: np.ndarray, F: np.ndarray) -> float:
    nbr_faces = [f for f in F if vid in f]
    A = 0.0
    for f in nbr_faces:
        i, j, k = f
        p, q, r = V[i], V[j], V[k]
        if vid == i:
            a, b, c = i, j, k
        elif vid == j:
            a, b, c = j, k, i
        else:
            a, b, c = k, i, j
        pa, pb, pc = V[a], V[b], V[c]
        A_a = angle_at(pb, pa, pc)
        A_b = angle_at(pc, pb, pa)
        A_c = angle_at(pa, pc, pb)
        obtuse = (A_a > math.pi / 2) or (A_b > math.pi / 2) or (A_c > math.pi / 2)
        area_f = tri_area(p, q, r)
        if obtuse:
            if A_a > math.pi / 2:
                A += 0.5 * area_f
            else:
                A += 0.25 * area_f
        else:
            v_ab = pb - pa
            v_ac = pc - pa
            lab2 = np.dot(v_ab, v_ab)
            lac2 = np.dot(v_ac, v_ac)
            cot_b = 1.0 / math.tan(A_b) if abs(A_b) > 1e-12 else 0.0
            cot_c = 1.0 / math.tan(A_c) if abs(A_c) > 1e-12 else 0.0
            A += (lab2 * cot_c + lac2 * cot_b) / 8.0
    return A


def discrete_gaussian_curvature(V: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = V.shape[0]
    K = np.zeros(n, dtype=np.float64)
    angle_sum = np.zeros(n, dtype=np.float64)
    for (i, j, k) in F:
        Ai = angle_at(V[j], V[i], V[k])
        Aj = angle_at(V[k], V[j], V[i])
        Ak = angle_at(V[i], V[k], V[j])
        angle_sum[i] += Ai
        angle_sum[j] += Aj
        angle_sum[k] += Ak
    areas = np.zeros(n, dtype=np.float64)
    for v in range(n):
        areas[v] = max(vertex_mixed_area(v, V, F), 1e-18)
    defect = (2.0 * math.pi) - angle_sum
    K[:] = defect / areas
    return K, areas


def plane_patch(Nx: int, Ny: int, extent: float) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(-extent, extent, Nx)
    ys = np.linspace(-extent, extent, Ny)
    VV = [[xs[i], ys[j], 0.0] for j in range(Ny) for i in range(Nx)]
    V = np.array(VV, dtype=np.float64)
    F = []
    def vid(i, j): return j * Nx + i
    for j in range(Ny - 1):
        for i in range(Nx - 1):
            v00 = vid(i, j)
            v10 = vid(i + 1, j)
            v01 = vid(i, j + 1)
            v11 = vid(i + 1, j + 1)
            F.append((v00, v10, v11))
            F.append((v00, v11, v01))
    return V, np.array(F, dtype=np.int64)


def sphere_uv(R: float, Nu: int, Nv: int) -> tuple[np.ndarray, np.ndarray]:
    us = np.linspace(0, 2 * math.pi, Nu, endpoint=False)
    vs = np.linspace(0, math.pi, Nv)
    VV = []
    for j, v in enumerate(vs):
        for i, u in enumerate(us):
            x = R * math.sin(v) * math.cos(u)
            y = R * math.sin(v) * math.sin(u)
            z = R * math.cos(v)
            VV.append([x, y, z])
    V = np.array(VV, dtype=np.float64)
    F = []
    def vid(i, j): return j * Nu + (i % Nu)
    for j in range(Nv - 1):
        for i in range(Nu):
            a = vid(i, j)
            b = vid(i + 1, j)
            c = vid(i, j + 1)
            d = vid(i + 1, j + 1)
            if j == 0:
                F.append((a, d, c))
            elif j == Nv - 2:
                F.append((a, b, c))
            else:
                F.append((a, b, d))
                F.append((a, d, c))
    return V, np.array(F, dtype=np.int64)


def saddle_patch(Nx: int, Ny: int, extent: float, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    xs = np.linspace(-extent, extent, Nx)
    ys = np.linspace(-extent, extent, Ny)
    VV = []
    for j in range(Ny):
        for i in range(Nx):
            x = xs[i]
            y = ys[j]
            z = alpha * (x * x - y * y)
            VV.append([x, y, z])
    V = np.array(VV, dtype=np.float64)
    F = []
    def vid(i, j): return j * Nx + i
    for j in range(Ny - 1):
        for i in range(Nx - 1):
            v00 = vid(i, j)
            v10 = vid(i + 1, j)
            v01 = vid(i, j + 1)
            v11 = vid(i + 1, j + 1)
            F.append((v00, v10, v11))
            F.append((v00, v11, v01))
    return V, np.array(F, dtype=np.int64)


def mask_interior_plane(Nx: int, Ny: int, border: int = 1) -> np.ndarray:
    mask = np.zeros(Nx * Ny, dtype=bool)
    for j in range(border, Ny - border):
        for i in range(border, Nx - border):
            mask[j * Nx + i] = True
    return mask


def mask_interior_grid(Nu: int, Nv: int, border: int = 1) -> np.ndarray:
    mask = np.zeros(Nu * Nv, dtype=bool)
    for j in range(border, Nv - border):
        for i in range(Nu):
            mask[j * Nu + i] = True
    return mask


def stats_vs_target(K: np.ndarray, target: float, mask: np.ndarray) -> CurvStats:
    Km = K[mask]
    err = np.abs(Km - target)
    return CurvStats(
        L1=float(np.mean(err)),
        L2=float(np.sqrt(np.mean(err * err))),
        Linf=float(np.max(err)),
        mean=float(np.mean(Km)),
        std=float(np.std(Km)),
        n_used=int(Km.size),
    )


def plot_histograms(figdir: Path, datasets: Sequence[np.ndarray], labels: Sequence[str],
                    figsize: tuple[float, float], dpi: int) -> Path:
    fig_path = next_fig_path(figdir, prefix="curvature_hist")
    plt.figure(figsize=figsize, dpi=dpi)
    bins = 60
    for K, lab in zip(datasets, labels):
        finite = np.isfinite(K)
        if np.any(finite):
            plt.hist(K[finite], bins=bins, histtype="step", label=lab, density=True)
    plt.xlabel("Discrete Gaussian curvature K")
    plt.ylabel("Density")
    plt.title("Stage 5: curvature via angle-defect")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    return fig_path


def plot_convergence(figdir: Path, hs: Sequence[float], L2s: Sequence[float],
                     figsize: tuple[float, float], dpi: int) -> tuple[Path | None, float]:
    if len(hs) != len(L2s) or len(hs) < 2:
        return None, float("nan")
    x = np.array(hs, dtype=np.float64)
    y = np.array(L2s, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x = x[mask]
    y = y[mask]
    if x.size < 2:
        return None, float("nan")
    p = np.polyfit(np.log(x), np.log(y), 1)
    slope = float(p[0])
    xs = np.linspace(x.min(), x.max(), 200)
    ys = np.exp(p[1]) * xs ** slope

    fig_path = next_fig_path(figdir, prefix="curvature_convergence")
    plt.figure(figsize=figsize, dpi=dpi)
    plt.loglog(x, y, "o", label="sphere L2 error")
    plt.loglog(xs, ys, "-", label=f"fit: slope={slope:.3f}")
    plt.gca().invert_xaxis()
    plt.xlabel("h (≈ πR/Nu)")
    plt.ylabel("L2 error |K - 1/R^2|")
    plt.title("Convergence on sphere (angle-defect)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    return fig_path, slope


def stage_docs(sim_dir: Path) -> dict[str, Path]:
    docs_root = sim_dir.parents[2]
    return {k: docs_root / path.name for k, path in STAGE_DOCS.items()}


def main() -> None:
    args = parse_args()
    sim_dir = sim_root(__file__)
    cfg_path = args.config if args.config.is_absolute() else sim_dir / args.config
    cfg = load_config(cfg_path)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    paths = SimPaths(sim_dir, results_name="curvature_angle_defect_RESULTS.txt")
    images_dir = ensure_version_dir(paths.images_dir)
    writer = ResultsWriter(paths.results_path, stage_docs(sim_dir))

    # Plane (target curvature 0)
    plane_cfg = cfg.plane
    Vp, Fp = plane_patch(plane_cfg["Nx"], plane_cfg["Ny"], extent=plane_cfg["extent"])
    Kp, _ = discrete_gaussian_curvature(Vp, Fp)
    mask_p = mask_interior_plane(plane_cfg["Nx"], plane_cfg["Ny"], border=2)
    stats_plane = stats_vs_target(Kp, 0.0, mask_p)

    # Sphere (target K = 1/R^2)
    sphere_cfg = cfg.sphere
    Vs, Fs = sphere_uv(R=sphere_cfg["R"], Nu=sphere_cfg["Nu"], Nv=sphere_cfg["Nv"])
    Ks, _ = discrete_gaussian_curvature(Vs, Fs)
    mask_s = mask_interior_grid(sphere_cfg["Nu"], sphere_cfg["Nv"], border=2)
    stats_sphere = stats_vs_target(Ks, 1.0 / (sphere_cfg["R"] ** 2), mask_s)

    # Saddle (qualitative)
    saddle_cfg = cfg.saddle
    Vh, Fh = saddle_patch(
        saddle_cfg["Nx"], saddle_cfg["Ny"], extent=saddle_cfg["extent"], alpha=saddle_cfg["alpha"]
    )
    Kh, _ = discrete_gaussian_curvature(Vh, Fh)
    mask_h = mask_interior_plane(saddle_cfg["Nx"], saddle_cfg["Ny"], border=2)
    frac_neg = float(np.mean(Kh[mask_h] < 0.0))

    # Sphere refinement sweep
    hs: list[float] = []
    L2s: list[float] = []
    with pbar(total=len(cfg.sphere_refine), desc="Sphere refinements") as bar:
        for lv in cfg.sphere_refine:
            Nu_r = lv["Nu"]
            Nv_r = lv["Nv"]
            Vr, Fr = sphere_uv(R=sphere_cfg["R"], Nu=Nu_r, Nv=Nv_r)
            Kr, _ = discrete_gaussian_curvature(Vr, Fr)
            mask_r = mask_interior_grid(Nu_r, Nv_r, border=2)
            stats_r = stats_vs_target(Kr, 1.0 / (sphere_cfg["R"] ** 2), mask_r)
            h = math.pi * sphere_cfg["R"] / float(Nu_r)
            hs.append(h)
            L2s.append(stats_r.L2)
            bar.update(1)

    # Plots
    hist_path = plot_histograms(
        images_dir,
        [Kp[mask_p], Ks[mask_s], Kh[mask_h]],
        [
            "plane (target 0)",
            f"sphere R={sphere_cfg['R']} (target {1.0 / (sphere_cfg['R'] ** 2):.2f})",
            "saddle (qualitative)",
        ],
        cfg.figsize,
        cfg.dpi,
    )
    conv_path, slope = plot_convergence(images_dir, hs, L2s, cfg.figsize, cfg.dpi)

    metrics = {
        "plane": asdict(stats_plane),
        "sphere": asdict(stats_sphere),
        "saddle_frac_negative": frac_neg,
        "sphere_refine_h": hs,
        "sphere_refine_L2": L2s,
        "sphere_refine_slope": slope,
    }
    image_paths = [hist_path] + ([conv_path] if conv_path else [])
    cfg_snapshot = {"config_path": cfg_path.as_posix(), "run": asdict(cfg)}

    writer.append_run_block(
        images_dir,
        cfg_snapshot,
        metrics,
        image_paths,
        seed=cfg.seed,
        code_sha=sha256_of_file(Path(__file__)),
    )

    print("Plane stats:", stats_plane)
    print("Sphere stats:", stats_sphere)
    print("Saddle negative fraction:", frac_neg)
    print("hs:", hs)
    print("L2:", L2s)
    print("Slope:", slope)


if __name__ == "__main__":
    main()
