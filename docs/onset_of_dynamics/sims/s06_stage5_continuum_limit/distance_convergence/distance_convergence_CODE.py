# distance_convergence_CODE.py
# Single-file, radius-graph, guarded continuum-limit test.
# Builds radius-based weighted graphs on [0,1]^2 grids, measures convergence
# of shortest-path distance to Euclidean distance as h -> 0.

import os, sys, math, json, time, hashlib, random
from datetime import datetime
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# --------------------------
# Config (your last request)
# --------------------------
CONFIG = {
    "refinements": [(16, 16), (32, 32), (64, 64), (128, 128), (200, 200)],  # (nx, ny)
    "pairs_per_level": 256,
    "band": (0.6, 0.8),             # pick pairs with euclidean |x| in this band [min,max]
    "radius_phys": 0.05,            # fixed physical radius
    "radius_h_factor": 0.0,         # 0 => do NOT auto-scale; pure continuum test
    "min_hops": 2,                  # enforce graph_distance(u,v) >= min_hops
    "seed": 12345,
    "out_root": None,               # auto: this file's folder
    "images_dirname": "images",     # will create versioned subfolder images/vNNN
    "figsize": (6.5, 4.2),
    "dpi": 120,
}

# --------------------------
# Paths & helpers (inline)
# --------------------------
def sha_short_of_file(path: Path, n=12):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:n]
    except Exception:
        return "n/a"

def env_line():
    try:
        import scipy  # noqa
        scipy_yes = "yes"
    except Exception:
        scipy_yes = "no"
    return {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "numpy": np.__version__,
        "networkx": nx.__version__,
        "matplotlib": plt.matplotlib.__version__,
        "scipy": scipy_yes,
        "os": os.name,
    }

def now_utc_iso():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def ensure_versioned_images_dir(out_root: Path, images_dirname: str):
    root = out_root / images_dirname
    root.mkdir(parents=True, exist_ok=True)
    # auto vNNN
    existing = [d for d in root.iterdir() if d.is_dir() and d.name.startswith("v")]
    nmax = 0
    for d in existing:
        try:
            nmax = max(nmax, int(d.name[1:]))
        except Exception:
            pass
    target = root / f"v{nmax+1:03d}"
    target.mkdir(parents=True, exist_ok=True)
    return target

def write_results_block(results_path: Path, run_dir: Path, metrics_dict, cfg_snapshot):
    run_header = f"=== RUN {now_utc_iso()} | {str(run_dir)} | seed={CONFIG['seed']} | commit=unknown | code_sha=n/a ===\n"
    stages = "STAGES: S1 v3.4 | S2 v3.3 | S3 v3.1 | S4 v3.2 | S5 v5.0\n"
    doc_shas = "DOC SHAS: S1=n/a S2=n/a S3=n/a S4=n/a S5=n/a\n"
    cfg_line = "CONFIG:\n  " + json.dumps({"distance_convergence_grid": cfg_snapshot}, indent=2) + "\n"

    metrics_lines = ["METRICS:"]
    for k, v in metrics_dict.items():
        if isinstance(v, list):
            metrics_lines.append(f"  - {k} = {v}")
        else:
            metrics_lines.append(f"  - {k} = {v}")
    metrics_txt = "\n".join(metrics_lines) + "\n"

    images_ = metrics_dict.get("images", [])
    images_lines = ["IMAGES:"] + [f"  {p}" for p in images_]
    images_txt = "\n".join(images_lines) + "\n"

    env_map = env_line()
    env_txt = "ENV:\n" + "\n".join([f"  {k}={v}" for k, v in env_map.items()]) + "\n"

    footer = "=== END RUN ===\n\n"

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(run_header)
        f.write(stages)
        f.write(doc_shas)
        f.write(cfg_line)
        f.write(metrics_txt)
        f.write(images_txt)
        f.write(env_txt)
        f.write(footer)

# --------------------------
# Graph building (radius)
# --------------------------
def build_radius_graph(nx_n, ny_n, radius):
    """
    Nodes on [0,1]^2, edges connect points <= radius, weight = Euclidean distance.
    Returns (G, pos(Nx2), avg_degree, builder_path_used)
    """
    xs = np.linspace(0.0, 1.0, nx_n)
    ys = np.linspace(0.0, 1.0, ny_n)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel()], axis=1)  # (N,2)
    N = pos.shape[0]

    G = nx.Graph()
    for i in range(N):
        G.add_node(i, xy=(float(pos[i, 0]), float(pos[i, 1])))

    used = "numpy_ball"
    try:
        # Preferred: SciPy KDTree
        from scipy.spatial import cKDTree as KDTree  # type: ignore
        tree = KDTree(pos)
        nbrs = tree.query_ball_point(pos, r=radius + 1e-12)
        used = "scipy.KDTree"
    except Exception:
        # Fallback: naive (still okay for up to ~40k nodes with basic pruning)
        nbrs = []
        # coarse box prune
        for i in range(N):
            px, py = pos[i]
            dx = np.abs(pos[:, 0] - px)
            dy = np.abs(pos[:, 1] - py)
            mask = (dx <= radius) & (dy <= radius)
            cand = np.where(mask)[0]
            d = np.hypot(pos[cand, 0] - px, pos[cand, 1] - py)
            lst = cand[(d <= radius + 1e-12)]
            nbrs.append(lst.tolist())

    # add weighted edges
    for i, lst in enumerate(nbrs):
        for j in lst:
            if j <= i:  # add once
                continue
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            w = float(np.hypot(dx, dy))
            if w > 0.0 and w <= radius + 1e-12:
                G.add_edge(i, j, weight=w)

    deg = np.array([deg for _, deg in G.degree()], dtype=float)
    avg_deg = float(deg.mean()) if deg.size else 0.0
    return G, pos, avg_deg, used

# --------------------------
# Distances
# --------------------------
def sssp_lengths(G, src_nodes):
    """
    Compute single-source shortest-path distances (weighted) for each src in src_nodes.
    Returns dict: src -> dict(node -> dist)
    Uses SciPy CSR if available; otherwise NetworkX Dijkstra per source.
    """
    # Build adjacency for scipy path
    paths = {}
    try:
        import scipy.sparse as sp  # type: ignore
        from scipy.sparse.csgraph import dijkstra as cs_dij  # type: ignore

        # CSR matrix
        n = G.number_of_nodes()
        rows, cols, data = [], [], []
        for u, v, d in G.edges(data=True):
            w = float(d.get("weight", 1.0))
            rows.append(u); cols.append(v); data.append(w)
            rows.append(v); cols.append(u); data.append(w)
        A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

        for s in src_nodes:
            dist = cs_dij(A, directed=False, indices=s, return_predecessors=False)
            # dist is np.ndarray
            paths[s] = {i: float(dist[i]) for i in range(n)}
    except Exception:
        # Fallback: per-source Dijkstra via NetworkX
        for s in src_nodes:
            dist = nx.single_source_dijkstra_path_length(G, s, weight="weight")
            paths[s] = {int(k): float(v) for k, v in dist.items()}
    return paths

# --------------------------
# Sampling pairs in a band
# --------------------------
def sample_pairs_band(G, pos, band, k, min_hops, seed):
    """
    Sample up to k pairs with Euclidean |x-y| in band and with shortest-path hops >= min_hops.
    Returns list of (u, v, d_euclid, hops).
    """
    rng = np.random.default_rng(seed)
    n = len(pos)
    pairs = []
    tries = 0
    max_tries = max(20000, k * 200)

    # Precompute an unweighted hop-length oracle with BFS on a light subgraph
    # (Use edges as unweighted)
    # We'll just rely on nx.shortest_path_length per tested pair; okay for k<=~500.
    while len(pairs) < k and tries < max_tries:
        tries += 1
        u = int(rng.integers(0, n))
        v = int(rng.integers(0, n))
        if u == v:
            continue
        dx = pos[u, 0] - pos[v, 0]
        dy = pos[u, 1] - pos[v, 1]
        de = float(math.hypot(dx, dy))
        if de < band[0] or de > band[1]:
            continue
        # hops constraint
        try:
            hops = nx.shortest_path_length(G, u, v)  # unweighted hops
        except nx.NetworkXNoPath:
            continue
        if hops < min_hops:
            continue
        pairs.append((u, v, de, hops))
    return pairs

# --------------------------
# Error & plotting
# --------------------------
def fit_loglog(h, e):
    """Fit slope of log10(e) vs log10(h), ignoring non-positive e."""
    h = np.asarray(h, dtype=float)
    e = np.asarray(e, dtype=float)
    mask = np.isfinite(h) & np.isfinite(e) & (h > 0) & (e > 0)
    if mask.sum() < 2:
        return float("nan")
    x = np.log10(h[mask]); y = np.log10(e[mask])
    A = np.vstack([x, np.ones_like(x)]).T
    slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope)

def plot_scatter(figpath, all_euclid, all_graph, cfg):
    plt.figure(figsize=cfg["figsize"], dpi=cfg["dpi"])
    plt.plot(all_euclid, all_graph, '.', ms=2)
    mmin = min(all_euclid.min(), all_graph.min())
    mmax = max(all_euclid.max(), all_graph.max())
    plt.plot([mmin, mmax], [mmin, mmax], lw=1)
    plt.xlabel("Euclidean distance")
    plt.ylabel("Graph shortest-path distance")
    plt.title("Distance scatter")
    plt.tight_layout()
    plt.savefig(figpath)
    plt.close()

def plot_convergence(figpath, h_list, L2_list, cfg):
    plt.figure(figsize=cfg["figsize"], dpi=cfg["dpi"])
    h = np.array(h_list, dtype=float); e = np.array(L2_list, dtype=float)
    mask = np.isfinite(h) & np.isfinite(e) & (h > 0) & (e > 0)
    if mask.sum() >= 2:
        x = h[mask]; y = e[mask]
        plt.loglog(x, y, 'o-')
        sl = fit_loglog(h, e)
        plt.title(f"L2 error vs h (slope ≈ {sl:.3f})")
    else:
        plt.loglog(h_list, L2_list, 'o-')
        plt.title("L2 error vs h")
    plt.xlabel("h")
    plt.ylabel("L2 error")
    plt.tight_layout()
    plt.savefig(figpath)
    plt.close()

# --------------------------
# Main
# --------------------------
def main():
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    # Resolve output paths
    here = Path(__file__).resolve().parent
    out_root = Path(CONFIG["out_root"]) if CONFIG["out_root"] else here
    img_dir = ensure_versioned_images_dir(out_root, CONFIG["images_dirname"])
    results_path = here / "distance_convergence_RESULTS.txt"

    # Header
    print(f"=== distance_convergence | images -> {img_dir} ===")

    refinements = CONFIG["refinements"]
    band = tuple(CONFIG["band"])
    pairs_per_level = int(CONFIG["pairs_per_level"])
    radius_phys = float(CONFIG["radius_phys"])
    rh_factor = float(CONFIG["radius_h_factor"])
    min_hops = int(CONFIG["min_hops"])

    # Accumulators
    h_vals = []
    L1_vals = []
    L2_vals = []
    Linf_vals = []
    avg_degrees = []

    all_euclid = []
    all_graph = []

    # Run levels
    for li, (nx_n, ny_n) in enumerate(refinements, start=1):
        h = max(1.0/(nx_n-1), 1.0/(ny_n-1))  # grid spacing (worst axis)
        # radius: fixed (since rh_factor==0) or max(radius_phys, rh_factor*h)
        radius = max(radius_phys, rh_factor*h)

        # Build graph
        G, pos, avg_deg, builder_used = build_radius_graph(nx_n, ny_n, radius)
        print(f"level {nx_n}x{ny_n}: radius={radius:.6f}, avg_deg={avg_deg:.2f}, builder={builder_used}")
        avg_degrees.append(avg_deg)

        # Guard 1: degrees should go up with refinement (radius fixed)
        if len(avg_degrees) >= 2:
            if avg_degrees[-1] <= 0.99*avg_degrees[-2]:
                raise RuntimeError("Average degree did not increase with refinement — radius-graph likely not applied.")

        # Guard 2: edge weights must be Euclidean
        edges_sample = list(G.edges(data=True))
        if edges_sample:
            k = min(500, len(edges_sample))
            sel_idx = np.random.default_rng(0).choice(len(edges_sample), size=k, replace=False)
            bad = 0
            for idx in sel_idx:
                u,v,d = edges_sample[idx]
                (x1,y1) = pos[u]; (x2,y2) = pos[v]
                w = float(d.get("weight", 1.0))
                w_ref = float(math.hypot(x1-x2, y1-y2))
                if abs(w - w_ref) > 1e-9:
                    bad += 1
            if bad > 0:
                raise RuntimeError("Edge weights are not Euclidean — fix builder.")

        # Choose pairs in band with hops constraint
        pairs = sample_pairs_band(G, pos, band, pairs_per_level, min_hops, seed=CONFIG["seed"])
        if len(pairs) == 0:
            print("WARNING: no valid pairs found at this level; marking errors NaN.")
            h_vals.append(h); L1_vals.append(float("nan")); L2_vals.append(float("nan")); Linf_vals.append(float("nan"))
            continue

        # Guard 3: not too many direct connections (trivial)
        direct_hits = sum(1 for (u,v,_,_) in pairs if G.has_edge(u, v))
        ratio = direct_hits / max(1, len(pairs))
        if ratio > 0.05:
            raise RuntimeError(f"Too many directly connected pairs at level {nx_n}x{ny_n} "
                               f"({ratio*100:.1f}%). Reduce radius_phys or narrow band.")

        # Compute distances
        # Efficient trick: compute SSSP for unique sources only
        unique_src = sorted({u for (u,_,_,_) in pairs})
        dmap = sssp_lengths(G, unique_src)

        e_abs = []
        eu = []
        gr = []

        for (u, v, d_e, _) in pairs:
            # Weighted shortest-path distance between u,v:
            if u in dmap:
                d_uv = dmap[u].get(v, float("inf"))
            else:
                # If u not in source set (shouldn't happen), fallback single dijkstra
                d_uv = nx.shortest_path_length(G, u, v, weight="weight")
            if not np.isfinite(d_uv):
                # skip disconnected (shouldn't happen if radius chosen well)
                continue
            eu.append(d_e)
            gr.append(float(d_uv))
            e_abs.append(abs(d_uv - d_e))

        eu = np.array(eu, dtype=float)
        gr = np.array(gr, dtype=float)
        e_abs = np.array(e_abs, dtype=float)
        if e_abs.size == 0:
            print("WARNING: all pairs filtered; marking errors NaN.")
            h_vals.append(h); L1_vals.append(float("nan")); L2_vals.append(float("nan")); Linf_vals.append(float("nan"))
            continue

        L1 = float(np.mean(e_abs))
        L2 = float(np.sqrt(np.mean(e_abs**2)))
        Linf = float(np.max(e_abs))

        h_vals.append(h); L1_vals.append(L1); L2_vals.append(L2); Linf_vals.append(Linf)
        all_euclid.append(eu); all_graph.append(gr)
        print(f"  pairs={len(e_abs)}, L1={L1:.6g}, L2={L2:.6g}, Linf={Linf:.6g}")

    # Flatten scatter arrays
    if len(all_euclid) > 0:
        all_euclid = np.concatenate(all_euclid)
        all_graph  = np.concatenate(all_graph)
    else:
        all_euclid = np.array([0.0])
        all_graph  = np.array([0.0])

    # Plot & compute slope
    fig_scatter = img_dir / "fig_dist_scatter_001.png"
    fig_conv    = img_dir / "fig_convergence_001.png"
    plot_scatter(fig_scatter, all_euclid, all_graph, CONFIG)
    plot_convergence(fig_conv, h_vals, L2_vals, CONFIG)
    slope = fit_loglog(h_vals, L2_vals)

    # Write results block
    cfg_snapshot = dict(CONFIG)
    cfg_snapshot["refinements"] = CONFIG["refinements"]
    metrics = {
        "levels": len(CONFIG["refinements"]),
        "pairs_total": int(sum([CONFIG["pairs_per_level"] for _ in CONFIG["refinements"]])),
        "h": [round(float(x), 6) if np.isfinite(x) else "nan" for x in h_vals],
        "L1": [float(x) if np.isfinite(x) else float("nan") for x in L1_vals],
        "L2": [float(x) if np.isfinite(x) else float("nan") for x in L2_vals],
        "Linf": [float(x) if np.isfinite(x) else float("nan") for x in Linf_vals],
        "L2_fit_slope": float(slope) if np.isfinite(slope) else float("nan"),
        "band": [float(band[0]), float(band[1])],
        "radius_used": [float(max(CONFIG["radius_phys"], CONFIG["radius_h_factor"]*max(1.0/(nx-1), 1.0/(ny-1))))
                        for (nx, ny) in CONFIG["refinements"]],
        "images": [str(fig_scatter), str(fig_conv)],
    }
    write_results_block(results_path, img_dir, metrics, cfg_snapshot)

    # Console summary
    print("\nSummary:")
    print(f"  h      : {metrics['h']}")
    print(f"  L2     : {metrics['L2']}")
    print(f"  slope  : {metrics['L2_fit_slope']:.3f}")
    print(f"  images : {metrics['images']}")
    print("Done.")

if __name__ == "__main__":
    main()
