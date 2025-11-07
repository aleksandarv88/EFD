# docs/onset_of_dynamics/sims/s06_stage5_continuum_limit/distance_convergence/distance_convergence_CODE.py
from __future__ import annotations
from pathlib import Path
import sys, math, os, datetime as dt, pickle, tempfile
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict

# --- locate and import s00_common ---
HERE = Path(__file__).resolve()
SIMS_DIR = HERE.parents[2]
COMMON_DIR = SIMS_DIR / "s00_common"
sys.path.append(str(COMMON_DIR))

from grid_refine import GridSpec, build_grid
from targets import euclidean_distance
from rng import make_rng
from progress import pbar
from parallel import map_process, _init_worker, get_global

from sim_paths import sim_root, images_root, results_path
from sim_plots import start_fig_env, save_fig
from sim_results import write_run_block

STAGES_META = {
    "S1": {"Version": "v3.4", "DocSHA": "n/a"},
    "S2": {"Version": "v3.3", "DocSHA": "n/a"},
    "S3": {"Version": "v3.1", "DocSHA": "n/a"},
    "S4": {"Version": "v3.2", "DocSHA": "n/a"},
    "S5": {"Version": "v5.0", "DocSHA": "n/a"},
}

def _header(seed: int, images_dir: str, commit: str = "unknown") -> str:
    utc = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return f"=== RUN {utc} | {images_dir} | seed={seed} | commit={commit} | code_sha=n/a ==="

def _grid_h(spec: GridSpec) -> float:
    hx = 1.0 / (spec.nx - 1)
    hy = 1.0 / (spec.ny - 1)
    return max(hx, hy)

def _sample_pairs_in_band(G: nx.Graph, rng, k: int, band=(0.6, 0.8)):
    """Sampling policy (no new math): pick k node pairs with Euclidean distance in [a,b]."""
    pos = nx.get_node_attributes(G, "pos")
    nodes = list(G.nodes())
    a, b = band
    out = []
    trials = 0
    max_trials = 200_000
    n = len(nodes)
    if n < 2:
        return out
    while len(out) < k and trials < max_trials:
        trials += 1
        u = nodes[int(rng.random() * n)]
        v = nodes[int(rng.random() * n)]
        if u == v:
            continue
        d = euclidean_distance(pos[u], pos[v])
        if a <= d <= b:
            out.append((u, v))
    return out

# --------- WORKER FUNCTION (top-level, picklable) ----------
def _worker_sssp_targets(task: tuple[int, list[int], str]) -> list[tuple[int, int, float]]:
    """
    task = (source_u, targets_list, graph_key)
    Returns: [(u, v, d_uv), ...] for the given targets.
    Graph is accessed via parallel.get_global(graph_key) inside worker.
    """
    u, targets, key = task
    G: nx.Graph = get_global(key)
    dists = nx.single_source_dijkstra_path_length(
        G, u, weight=lambda a, b, d: float(d.get("len", 1.0))
    )
    out = []
    for v in targets:
        d = dists.get(v, float("inf"))
        out.append((u, v, float(d)))
    return out

def main() -> None:
    # ----------------- config (math-compliant knobs) -----------------
    seed = 12345
    # Keep modest by default for stability; you can add (256,256) later
    refinements = [(32,32), (64,64), (128,128), (256,256), (512,512)]
    pairs_per_level = 512
    band = (0.6, 0.8)
    radius_phys = 0.05          # dense physical-radius stencil (emergent; no ad-hoc math)
    # parallelism
    cpu = os.cpu_count() or 2
    max_workers = min(cpu - 2, 16)  # keep a couple of cores free
    graph_key = "_G"
    # ---------------------------------------------------------------

    rng = make_rng(seed)
    root = sim_root(__file__)
    vdir = start_fig_env(images_root(root))
    results_txt = results_path(root, "distance_convergence_grid")

    hs, l1_errs, l2_errs, linf_errs = [], [], [], []
    n_pairs_total = 0
    scat_true, scat_graph = [], []

    with pbar(total=len(refinements), desc="refinements") as bar_ref:
        for nx_g, ny_g in refinements:
            # --- build grid with radius-based local metric (Stage-3 compliant) ---
            spec = GridSpec(
                nx=nx_g, ny=ny_g,
                diag=False,                 # radius stencil replaces diag
                tag_edge_len=True,
                radius_phys=radius_phys
            )
            G = build_grid(spec)
            pos = nx.get_node_attributes(G, "pos")
            h = _grid_h(spec)
            hs.append(h)

            # sample pairs within band
            pairs = _sample_pairs_in_band(G, rng, k=pairs_per_level, band=band)
            n_pairs_total += len(pairs)

            # group by source to do 1 Dijkstra per source
            by_src: dict[int, list[int]] = defaultdict(list)
            for (u, v) in pairs:
                by_src[u].append(v)

            # --- serialize graph to a temp file (Windows-safe; avoids pipe truncation) ---
            with tempfile.NamedTemporaryFile(prefix="efd_graph_", suffix=".pkl", delete=False) as tf:
                pickle.dump(G, tf, protocol=pickle.HIGHEST_PROTOCOL)
                graph_path = tf.name

            try:
                tasks = [(u, vs, graph_key) for u, vs in by_src.items()]
                results = map_process(
                    _worker_sssp_targets,
                    tasks,
                    max_workers=max_workers,
                    desc=f"sssp {nx_g}x{ny_g} (|src|={len(tasks)})",
                    initializer=_init_worker,
                    initargs=(0, None, graph_path, graph_key),  # bytes=None, use path
                )
            finally:
                try:
                    os.remove(graph_path)
                except Exception:
                    pass

            # accumulate errors (robust to None/Exceptions)
            errs = []
            for res in results:
                if res is None:
                    continue
                if isinstance(res, Exception):
                    raise RuntimeError(f"Worker failed for grid {nx_g}x{ny_g}") from res
                for (u, v, dn) in res:
                    dt_true = euclidean_distance(pos[u], pos[v])
                    if math.isfinite(dn) and dt_true > 0:
                        rel = abs(dn - dt_true) / dt_true
                        errs.append(rel)
                        scat_true.append(dt_true)
                        scat_graph.append(dn)

            if len(errs) == 0:
                l1 = l2 = lmax = float("nan")
            else:
                arr = np.array(errs, float)
                l1 = float(arr.mean())
                l2 = float(np.sqrt((arr**2).mean()))
                lmax = float(arr.max())
            l1_errs.append(l1); l2_errs.append(l2); linf_errs.append(lmax)

            bar_ref.update(1)

    # --- Fig 1: numeric vs analytic scatter ---
    fig1 = plt.figure(figsize=(6, 6))
    plt.scatter(scat_true, scat_graph, s=10)
    if scat_true:
        m = max(max(scat_true), max(scat_graph))
        plt.plot([0, m], [0, m])
    plt.xlabel("Analytic distance (Euclidean)")
    plt.ylabel("Graph shortest-path distance")
    plt.title("Distance: numeric vs analytic (banded pairs)")
    plt.axis("equal"); plt.tight_layout()
    img1 = save_fig(fig1, vdir, prefix="fig_dist_scatter")

    # --- Fig 2: convergence (log-log) + slope on L2 ---
    # keep only finite points to fit slope
    hC    = np.array([h for h,e in zip(hs, l2_errs) if np.isfinite(e)], float)
    L1C   = np.array([e for e in l1_errs if np.isfinite(e)], float)
    L2C   = np.array([e for e in l2_errs if np.isfinite(e)], float)
    LInfC = np.array([e for e in linf_errs if np.isfinite(e)], float)
    slope = float("nan")

    fig2 = plt.figure(figsize=(6, 5))
    if len(hC) >= 2:
        plt.loglog(hC, L1C, "o-", label="L1 rel. err")
        plt.loglog(hC, L2C, "s-", label="L2 rel. err")
        plt.loglog(hC, LInfC, "^-", label="L∞ rel. err")
        xs = np.log(hC); ys = np.log(L2C)
        if np.all(np.isfinite(xs)) and np.all(np.isfinite(ys)):
            A = np.vstack([xs, np.ones_like(xs)]).T
            slope, _ = np.linalg.lstsq(A, ys, rcond=None)[0]
            plt.text(hC[-1], L2C[-1], f"slope ≈ {slope:.2f}")
    plt.gca().invert_xaxis()
    plt.xlabel("h (grid spacing)"); plt.ylabel("relative error")
    plt.title("Convergence of graph distance to Euclidean")
    plt.legend(); plt.tight_layout()
    img2 = save_fig(fig2, vdir, prefix="fig_convergence")

    # --- results block ---
    metrics = [
        f"levels = {len(refinements)}",
        f"pairs_total = {n_pairs_total}",
        "h = [" + ", ".join(f"{x:.6f}" for x in hs) + "]",
        "L1 = [" + ", ".join(f"{x:.6g}" for x in l1_errs) + "]",
        "L2 = [" + ", ".join(f"{x:.6g}" for x in l2_errs) + "]",
        "Linf = [" + ", ".join(f"{x:.6g}" for x in linf_errs) + "]",
        f"L2_fit_slope = {slope:.3f}",
        f"band = [{band[0]:.3f}, {band[1]:.3f}]",
        f"radius_phys = {radius_phys}",
        f"workers = {os.cpu_count() or 1}",
    ]
    config_dump = "\n".join([
        "distance_convergence_grid:",
        f"  refinements: {refinements}",
        f"  pairs_per_level: {pairs_per_level}",
        f"  band: [{band[0]}, {band[1]}]",
        f"  radius_phys: {radius_phys}",
    ]) + "\n"

    header_line = _header(seed, str(vdir).replace("\\", "/"))
    write_run_block(
        results_txt=results_txt,
        header_line=header_line,
        stages_meta=STAGES_META,
        config_dump=config_dump,
        metrics_lines=metrics,
        image_paths=[img1, img2],
    )

if __name__ == "__main__":
    main()
