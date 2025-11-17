# distance_convergence_CODE.py
# Thin orchestrator for Stage 5 continuum-limit distance convergence.
# Builds guarded radius graphs, samples point pairs, and measures how
# geodesic distances converge to Euclidean as the grid refines.

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required to run this sim (pip install pyyaml).") from exc

from lib.efd.io_utils import ensure_version_dir, next_fig_path, sha256_of_file
from docs.onset_of_dynamics.sims.s00_common.sim_paths import SimPaths, sim_root
from docs.onset_of_dynamics.sims.s00_common.sim_results import ResultsWriter
from docs.onset_of_dynamics.sims.s00_common.graph import GridSpec, RunConfig
from docs.onset_of_dynamics.sims.s00_common.graph.builders import build_radius_graph
from docs.onset_of_dynamics.sims.s00_common.graph.distances import sssp_lengths
from docs.onset_of_dynamics.sims.s00_common.graph.guards import (
    assert_avg_degree_monotone,
    assert_direct_pair_ratio,
    assert_edges_euclidean,
)
from docs.onset_of_dynamics.sims.s00_common.graph.sampling import sample_pairs_band
from docs.onset_of_dynamics.sims.s00_common.analyze.errors import l1_l2_linf
from docs.onset_of_dynamics.sims.s00_common.analyze.reporting import make_metrics_dict
from docs.onset_of_dynamics.sims.s00_common.analyze.viz import plot_convergence, plot_scatter


STAGE_DOCS = {
    "S1": Path("docs/onset_of_dynamics/STAGE_1_Re-CoherenceOrderAndInternalTime.txt"),
    "S2": Path("docs/onset_of_dynamics/STAGE_2_PropagationAndSpeedLimit.txt"),
    "S3": Path("docs/onset_of_dynamics/STAGE_3_MetricAndCurvatureFromCoherence.txt"),
    "S4": Path("docs/onset_of_dynamics/STAGE_4_CoherenceCostAndEnergyConservation.txt"),
    "S5": Path("docs/onset_of_dynamics/STAGE_5.txt"),
}


@dataclass
class LevelResult:
    spec: GridSpec
    h: float
    radius: float
    avg_degree: float
    euclid: np.ndarray
    graph: np.ndarray
    errors: np.ndarray
    band_used: tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 5 distance convergence test.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config (default: configs/default.yaml).",
    )
    return parser.parse_args()


def load_run_config(config_path: Path) -> tuple[RunConfig, dict]:
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError("Config must be a mapping.")

    run_section = raw.get("run", raw)
    if not isinstance(run_section, dict):
        raise ValueError("`run` config must be a mapping.")

    run_cfg = RunConfig(
        refinements=[tuple(map(int, pair)) for pair in run_section["refinements"]],
        pairs_per_level=int(run_section["pairs_per_level"]),
        band=tuple(float(x) for x in run_section["band"]),
        radius_phys=float(run_section["radius_phys"]),
        radius_h_factor=float(run_section["radius_h_factor"]),
        min_hops=int(run_section["min_hops"]),
        seed=int(run_section["seed"]),
        max_workers=int(run_section.get("max_workers", 0)),
    )
    extras = {
        "figsize": tuple(run_section.get("figsize", (6.5, 4.2))),
        "dpi": int(run_section.get("dpi", 140)),
    }
    run_meta = dict(run_section)
    run_meta.update(extras)
    return run_cfg, run_meta


def compute_level(spec: GridSpec, cfg: RunConfig, seed: int) -> LevelResult:
    hx = 1.0 / (spec.nx - 1)
    hy = 1.0 / (spec.ny - 1)
    h = max(hx, hy)
    min_spacing = min(hx, hy)
    radius = max(cfg.radius_phys, cfg.radius_h_factor * h)
    if radius < min_spacing:
        radius = min_spacing * 1.05
        print(f"  [radius] bumped to {radius:.4f} to ensure connectivity")

    pack = build_radius_graph(spec, radius)
    if pack.G.number_of_edges() == 0:
        for _ in range(4):
            radius *= 1.5
            pack = build_radius_graph(spec, radius)
            if pack.G.number_of_edges() > 0:
                print(f"  [radius] auto-expanded to {radius:.4f} (edges restored)")
                break
        if pack.G.number_of_edges() == 0:
            raise RuntimeError("Failed to build a connected graph; increase radius parameters.")
    assert_edges_euclidean(pack.G, pack.pos)

    band_lo, band_hi = cfg.band
    band_span = max(1e-6, band_hi - band_lo)
    band_used = (band_lo, band_hi)
    for attempt in range(5):
        try:
            pairs = sample_pairs_band(
                pack.G,
                pack.pos,
                band_used,
                cfg.pairs_per_level,
                cfg.min_hops,
                seed + attempt,
            )
            break
        except RuntimeError:
            if attempt == 4:
                raise
            pad = max(0.05, 0.25 * band_span)
            band_lo = max(0.0, band_lo - pad)
            band_hi = min(math.sqrt(2.0), band_hi + pad)
            band_used = (band_lo, band_hi)
            print(f"  [adaptive] widening band to {band_used}")
    assert_direct_pair_ratio(pairs, pack.G)

    sources = sorted({p.u for p in pairs})
    dmap = sssp_lengths(pack.G, sources, max_workers=cfg.max_workers)

    euclid_all: List[float] = []
    graph_all: List[float] = []
    errors: List[float] = []
    for pair in pairs:
        d_uv = dmap.get(pair.u, {}).get(pair.v, math.inf)
        if math.isfinite(d_uv):
            euclid_all.append(pair.d_euclid)
            graph_all.append(d_uv)
            errors.append(abs(d_uv - pair.d_euclid))

    if not euclid_all:
        raise RuntimeError("No finite graph distances computed; graph might be disconnected.")

    return LevelResult(
        spec=spec,
        h=h,
        radius=radius,
        avg_degree=pack.avg_degree,
        euclid=np.asarray(euclid_all, dtype=np.float64),
        graph=np.asarray(graph_all, dtype=np.float64),
        errors=np.asarray(errors, dtype=np.float64),
        band_used=band_used,
    )


def versioned_stage_docs(sim_dir: Path) -> dict[str, Path]:
    docs_root = sim_dir.parents[2]
    return {k: (docs_root / path.name) for k, path in STAGE_DOCS.items()}


def main() -> None:
    args = parse_args()
    sim_dir = sim_root(__file__)
    cfg_path = (sim_dir / args.config) if not args.config.is_absolute() else args.config
    cfg, raw_cfg = load_run_config(cfg_path)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    paths = SimPaths(sim_dir, results_name="distance_convergence_RESULTS.txt")
    images_dir = ensure_version_dir(paths.images_dir)
    writer = ResultsWriter(paths.results_path, versioned_stage_docs(sim_dir))

    level_results: List[LevelResult] = []
    avg_degrees: List[float] = []
    for idx, dims in enumerate(cfg.refinements):
        spec = GridSpec(*dims)
        level_seed = cfg.seed + idx
        level = compute_level(spec, cfg, level_seed)
        level_results.append(level)
        avg_degrees.append(level.avg_degree)
        assert_avg_degree_monotone(avg_degrees)
        print(
            f"[level {spec.nx}x{spec.ny}] h={level.h:.4f} "
            f"avg_deg={level.avg_degree:.2f} pairs={level.euclid.size}"
        )

    euclid_all = np.concatenate([lvl.euclid for lvl in level_results])
    graph_all = np.concatenate([lvl.graph for lvl in level_results])

    L1_list: List[float] = []
    L2_list: List[float] = []
    Linf_list: List[float] = []
    h_list: List[float] = []
    radius_used: List[float] = []
    levels: List[Tuple[int, int]] = []
    band_levels: List[Tuple[float, float]] = []
    for lvl in level_results:
        L1, L2, Linf = l1_l2_linf(lvl.errors)
        L1_list.append(L1)
        L2_list.append(L2)
        Linf_list.append(Linf)
        h_list.append(lvl.h)
        radius_used.append(lvl.radius)
        levels.append((lvl.spec.nx, lvl.spec.ny))
        band_levels.append(lvl.band_used)
        print(
            f"  L2={L2:.4e} L_inf={Linf:.4e} radius={lvl.radius:.4f}"
        )

    scatter_path = next_fig_path(images_dir, prefix="scatter")
    plot_scatter(scatter_path, euclid_all, graph_all, raw_cfg["figsize"], raw_cfg["dpi"])

    conv_path = next_fig_path(images_dir, prefix="convergence")
    slope = plot_convergence(conv_path, h_list, L2_list, raw_cfg["figsize"], raw_cfg["dpi"])

    metrics = make_metrics_dict(
        h_list=h_list,
        L1=L1_list,
        L2=L2_list,
        Linf=Linf_list,
        slope=slope,
        band=cfg.band,
        band_levels=band_levels,
        radius_used=radius_used,
        images=[scatter_path, conv_path],
        levels=levels,
        pairs_total=int(euclid_all.size),
    )

    cfg_snapshot = {
        "config_path": cfg_path.as_posix(),
        "run": raw_cfg,
    }
    writer.append_run_block(
        images_dir,
        cfg_snapshot,
        metrics,
        [scatter_path, conv_path],
        seed=cfg.seed,
        code_sha=sha256_of_file(Path(__file__)),
    )
    print(f"Wrote metrics + figures to {paths.results_path}")


if __name__ == "__main__":
    main()
