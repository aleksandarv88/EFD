#!/usr/bin/env python3
"""
Stage 5 – Holonomy via Gauss–Bonnet on refined spheres.
Loops are k-ring vertex neighborhoods; holonomy estimate is the sum of
angle defects inside the loop, compared against K * area (K = 1/R^2).
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

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
    sphere_refine: list[dict]
    k_rings: list[int]
    seeds_per_level: int
    R: float
    seed: int
    figsize: tuple[float, float]
    dpi: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 5 holonomy loop convergence.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="Path to YAML config (default: config/default.yaml)",
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

    return RunConfig(
        sphere_refine=_refine(section.get("sphere_refine")),
        k_rings=[int(k) for k in section.get("k_rings", [1, 2])],
        seeds_per_level=int(section.get("seeds_per_level", 64)),
        R=float(section.get("R", 1.0)),
        seed=int(section.get("seed", 12345)),
        figsize=tuple(section.get("figsize", (6.5, 4.2))),
        dpi=int(section.get("dpi", 140)),
    )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def sphere_uv(R: float, Nu: int, Nv: int) -> tuple[np.ndarray, np.ndarray]:
    us = np.linspace(0, 2 * math.pi, Nu, endpoint=False)
    vs = np.linspace(0, math.pi, Nv)
    verts = []
    for v in vs:
        for u in us:
            verts.append([
                R * math.sin(v) * math.cos(u),
                R * math.sin(v) * math.sin(u),
                R * math.cos(v),
            ])
    V = np.array(verts, dtype=np.float64)
    faces = []
    def vid(i, j): return j * Nu + (i % Nu)
    for j in range(Nv - 1):
        for i in range(Nu):
            a = vid(i, j)
            b = vid(i + 1, j)
            c = vid(i, j + 1)
            d = vid(i + 1, j + 1)
            if j == 0:
                faces.append((a, d, c))
            elif j == Nv - 2:
                faces.append((a, b, c))
            else:
                faces.append((a, b, d))
                faces.append((a, d, c))
    return V, np.array(faces, dtype=np.int64)


def vertex_adjacency(F: np.ndarray, nV: int) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(nV)]
    for (a, b, c) in F:
        for u, v in ((a, b), (b, c), (c, a)):
            if v not in adj[u]:
                adj[u].append(v)
            if u not in adj[v]:
                adj[v].append(u)
    return adj


def k_ring_vertices(adj: list[list[int]], seed: int, k: int) -> set[int]:
    frontier = {seed}
    seen = {seed}
    for _ in range(k):
        nxt = set()
        for u in frontier:
            for v in adj[u]:
                if v not in seen:
                    nxt.add(v)
                    seen.add(v)
        if not nxt:
            break
        frontier = nxt
    return seen


def angle_at(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
    v1 = p - q
    v2 = r - q
    a = np.linalg.norm(v1)
    b = np.linalg.norm(v2)
    if a == 0 or b == 0:
        return 0.0
    c = np.dot(v1, v2) / max(1e-15, a * b)
    c = np.clip(c, -1.0, 1.0)
    return math.acos(c)


def angle_defect(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    defects = np.zeros(V.shape[0], dtype=float)
    for (a, b, c) in F:
        defects[a] += angle_at(V[b], V[a], V[c])
        defects[b] += angle_at(V[c], V[b], V[a])
        defects[c] += angle_at(V[a], V[c], V[b])
    return (2.0 * math.pi) - defects


def triangle_area(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
    return 0.5 * np.linalg.norm(np.cross(q - p, r - p))


def vertex_patch_area(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    areas = np.zeros(V.shape[0], dtype=float)
    for (a, b, c) in F:
        ar = triangle_area(V[a], V[b], V[c]) / 3.0
        areas[a] += ar
        areas[b] += ar
        areas[c] += ar
    return areas


# ---------------------------------------------------------------------------
# Experiment + plotting
# ---------------------------------------------------------------------------
def holonomy_experiment(Nu: int, Nv: int, R: float, k_values: Sequence[int],
                        seeds: int, rng: random.Random) -> dict:
    V, F = sphere_uv(R=R, Nu=Nu, Nv=Nv)
    adj = vertex_adjacency(F, V.shape[0])
    defects = angle_defect(V, F)
    areas = vertex_patch_area(V, F)
    K = 1.0 / (R * R)

    candidates = [i for i, (_, _, z) in enumerate(V) if abs(z) < 0.95 * R]
    if len(candidates) < seeds:
        seeds = len(candidates)
    seeds_idx = rng.sample(candidates, seeds)

    area_list = []
    hol_est = []
    hol_tgt = []
    err_abs = []

    for k in k_values:
        for s in seeds_idx:
            region = k_ring_vertices(adj, s, k)
            area = float(np.sum(areas[list(region)]))
            hol = float(np.sum(defects[list(region)]))
            target = K * area
            area_list.append(area)
            hol_est.append(hol)
            hol_tgt.append(target)
            err_abs.append(abs(hol - target))

    return {
        "areas": np.array(area_list),
        "hol_est": np.array(hol_est),
        "hol_tgt": np.array(hol_tgt),
        "err_abs": np.array(err_abs),
    }


def fit_loglog(hs: Iterable[float], L2s: Iterable[float]) -> float:
    x = np.log(np.array(list(hs), dtype=float))
    y = np.log(np.array(list(L2s), dtype=float))
    m, _ = np.linalg.lstsq(np.vstack([x, np.ones_like(x)]).T, y, rcond=None)[0]
    return float(m)


def plot_scatter(figdir: Path, datasets: Sequence[dict], figsize: tuple[float, float], dpi: int) -> Path:
    fig_path = next_fig_path(figdir, prefix="holonomy_scatter")
    plt.figure(figsize=figsize, dpi=dpi)
    for data in datasets:
        plt.scatter(data["hol_tgt"], data["hol_est"], s=12, alpha=0.6)
    max_target = max((np.max(d["hol_tgt"]) for d in datasets), default=1.0)
    plt.plot([0, max_target], [0, max_target], color="k", linewidth=1.0, label="y = x")
    plt.xlabel("Target holonomy (K * area)")
    plt.ylabel("Estimated holonomy (∑ defects)")
    plt.title("Holonomy convergence on sphere")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    return fig_path


def plot_convergence(figdir: Path, hs: Sequence[float], L2s: Sequence[float],
                     figsize: tuple[float, float], dpi: int) -> tuple[Path, float]:
    slope = fit_loglog(hs, L2s) if len(hs) >= 2 else float("nan")
    fig_path = next_fig_path(figdir, prefix="holonomy_convergence")
    plt.figure(figsize=figsize, dpi=dpi)
    plt.loglog(hs, L2s, marker="o")
    plt.gca().invert_xaxis()
    plt.xlabel("h (≈ πR/Nu)")
    plt.ylabel("L2 error |hol_est - K * area|")
    plt.title(f"Holonomy convergence (slope ≈ {slope:.3f})")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    return fig_path, slope


def stage_doc_map(sim_dir: Path) -> dict[str, Path]:
    docs_root = sim_dir.parents[2]
    return {k: docs_root / path.name for k, path in STAGE_DOCS.items()}


def main() -> None:
    args = parse_args()
    sim_dir = sim_root(__file__)
    cfg_path = args.config if args.config.is_absolute() else sim_dir / args.config
    cfg = load_config(cfg_path)

    random.seed(cfg.seed)
    rng = random.Random(cfg.seed)

    paths = SimPaths(sim_dir, results_name="holonomy_loops_RESULTS.txt")
    images_dir = ensure_version_dir(paths.images_dir)
    writer = ResultsWriter(paths.results_path, stage_doc_map(sim_dir))

    hs: list[float] = []
    L2s: list[float] = []
    datasets: list[dict] = []

    with pbar(total=len(cfg.sphere_refine), desc="Sphere refinements") as bar:
        for entry in cfg.sphere_refine:
            Nu = int(entry["Nu"])
            Nv = int(entry["Nv"])
            result = holonomy_experiment(
                Nu=Nu,
                Nv=Nv,
                R=cfg.R,
                k_values=cfg.k_rings,
                seeds=cfg.seeds_per_level,
                rng=rng,
            )
            h = math.pi * cfg.R / float(Nu)
            hs.append(h)
            L2s.append(float(np.sqrt(np.mean(result["err_abs"] ** 2))))
            datasets.append(result)
            bar.update(1)

    scatter_path = plot_scatter(images_dir, datasets, cfg.figsize, cfg.dpi)
    conv_path, slope = plot_convergence(images_dir, hs, L2s, cfg.figsize, cfg.dpi)

    metrics = {
        "levels": [(int(entry["Nu"]), int(entry["Nv"])) for entry in cfg.sphere_refine],
        "h": hs,
        "L2": L2s,
        "slope": slope,
        "k_rings": cfg.k_rings,
        "seeds_per_level": cfg.seeds_per_level,
    }
    cfg_snapshot = {"config_path": cfg_path.as_posix(), "run": asdict(cfg)}

    writer.append_run_block(
        images_dir,
        cfg_snapshot,
        metrics,
        [scatter_path, conv_path],
        seed=cfg.seed,
        code_sha=sha256_of_file(Path(__file__)),
    )

    print("Holonomy loop sim complete.")


if __name__ == "__main__":
    main()
