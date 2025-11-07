# -*- coding: utf-8 -*-
"""
EFD — Stage 5 Continuum Limit: Grid Distance Convergence (single-file)
- No external project imports (self-contained).
- Auto-scales target radius per level: radius_lvl = max(radius_phys, radius_h_factor * h)
- Enforces min graph hops between sampled pairs
- Uses SciPy CSR Dijkstra for speed; falls back to NetworkX if SciPy not present
- Simple text progress bar (no deps)
- Writes RESULTS block + two figures in images/vNNN
"""

import os, sys, math, time, json, hashlib, random, pathlib, itertools
from datetime import datetime
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# ---- Try SciPy (faster CSR Dijkstra + KDTree). Fallback gracefully.
try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra as cs_dijkstra
    from scipy.spatial import KDTree
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ---- Try NetworkX for graph construction / fallback distances.
try:
    import networkx as nx
    NX_OK = True
except Exception as e:
    print("ERROR: networkx is required for this script.")
    raise

# ---------------------------
# Config (tweak here)
# ---------------------------
CONFIG = {
    "refinements": [(16, 16), (32, 32), (64, 64), (128, 128), (200, 200)],  # (nx, ny)
    "pairs_per_level": 256,
    "band": (0.3, 0.9),             # wider valid range → more stable stats at coarse levels
    "diag_8nbr": True,               # Ignored with radius; here for legacy, but radius takes precedence
    "radius_phys": 0.05,             # fixed physical radius for fine levels
    "radius_h_factor": 2.5,          # scales with h for coarse levels
    "min_hops": 2,                   # avoid trivial neighbors
    "seed": 12345,
    "out_root": None,                # auto: this file's folder
    "images_dirname": "images",      # versioned subfolder images/vNNN
    "figsize": (6.5, 4.2),
    "dpi": 120,
}


# ---------------------------
# Tiny utils
# ---------------------------
def sha12_of_file(path: str) -> str:
    if not os.path.exists(path):
        return "n/a"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def next_versioned_dir(images_root: str) -> str:
    ensure_dir(images_root)
    # images/vNNN
    existing = [d for d in os.listdir(images_root) if d.lower().startswith("v")]
    nums = []
    for name in existing:
        num = ''.join([c for c in name if c.isdigit()])
        if num:
            nums.append(int(num))
    nxt = (max(nums) + 1) if nums else 1
    vdir = os.path.join(images_root, f"v{nxt:03d}")
    ensure_dir(vdir)
    return vdir

def nice_bar(prefix, i, n, width=40):
    filled = int(width * i / max(1, n))
    return f"{prefix} [{'·'*filled}{' '*(width-filled)}] {i*100//max(1,n)}% {i}/{n}"

def radius_for_level(h, base_radius, factor):
    # ensure radius >= factor*h
    if base_radius is None or base_radius <= 0:
        return factor * h
    return max(base_radius, factor * h)

# ---------------------------
# Grid + weights
# ---------------------------
def build_grid_graph(nx_n, ny_n, radius, diag_8=True):
    """Rect grid on [0,1]x[0,1], nodes are (i,j) with i in [0..nx-1], j in [0..ny-1].
       Connects all pairs within physical radius, with weight = Euclidean length.
       Returns (G, pos, h) where pos[(i,j)] = (x,y), h ~ spacing."""
    G = nx.Graph()
    nx_n = int(nx_n); ny_n = int(ny_n)
    xs = np.linspace(0.0, 1.0, nx_n)
    ys = np.linspace(0.0, 1.0, ny_n)
    # nominal spacing: use average of dx, dy
    dx = xs[1] - xs[0] if nx_n > 1 else 1.0
    dy = ys[1] - ys[0] if ny_n > 1 else 1.0
    h = 0.5 * (dx + dy)

    pos = {}
    points = []  # for KDTree
    for i in range(nx_n):
        for j in range(ny_n):
            p = (float(xs[i]), float(ys[j]))
            pos[(i, j)] = p
            G.add_node((i, j))
            points.append(p)

    # Use KDTree for efficient neighbor queries if SciPy available
    if SCIPY_OK:
        tree = KDTree(points)
        node_list = list(G.nodes())
        for idx, u in enumerate(node_list):
            dists, indices = tree.query(pos[u], k=len(points), distance_upper_bound=radius)
            for dist, neighbor_idx in zip(dists, indices):
                if dist <= radius and neighbor_idx != idx and math.isfinite(dist):
                    v = node_list[neighbor_idx]
                    G.add_edge(u, v, weight=dist)
    else:
        # Fallback O(n^2) for small grids
        print("WARNING: SciPy not available; using O(n^2) neighbor search.")
        for u in G.nodes():
            for v in G.nodes():
                if u != v:
                    x1, y1 = pos[u]
                    x2, y2 = pos[v]
                    d = math.hypot(x2 - x1, y2 - y1)
                    if d <= radius:
                        G.add_edge(u, v, weight=d)

    return G, pos, h

# ---------------------------
# Distances
# ---------------------------
def scipy_csr_from_nx(G):
    """Build CSR adjacency (weights) from nx Graph; also node<->idx mappings."""
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    rows, cols, data = [], [], []
    for u, v, d in G.edges(data=True):
        i, j = node_to_idx[u], node_to_idx[v]
        w = float(d.get("weight", 1.0))
        rows.append(i); cols.append(j); data.append(w)
        rows.append(j); cols.append(i); data.append(w)
    M = csr_matrix((data, (rows, cols)), shape=(len(nodes), len(nodes)))
    return M, nodes, node_to_idx

def shortest_path_lengths_all_sources(G, sources, use_scipy=True):
    """Return dict: src -> dict(node->dist). Uses SciPy CSR if available."""
    out = {}
    if use_scipy and SCIPY_OK:
        M, nodes, node_to_idx = scipy_csr_from_nx(G)
        src_indices = [node_to_idx[s] for s in sources if s in node_to_idx]
        D = cs_dijkstra(M, directed=False, indices=src_indices)
        for idx, s in enumerate(src_indices):
            row = D[idx]
            src_node = nodes[s]
            out[src_node] = {nodes[j]: float(row[j]) if math.isfinite(row[j]) else np.inf for j in range(len(nodes))}
    else:
        for s in sources:
            dists = nx.single_source_dijkstra_path_length(G, s, weight='weight')
            out[s] = {v: float(dists.get(v, np.inf)) for v in G.nodes()}
    return out

# ---------------------------
# Sampling pairs
# ---------------------------
def sample_pairs_band(G, pos, rng, k, radius, band=(0.6, 0.8), min_hops=2, precomputed_hops=None):
    a, b = band
    nodes = list(G.nodes())
    n = len(nodes)
    out = []
    trials = 0
    max_trials = 200_000
    while len(out) < k and trials < max_trials:
        trials += 1
        u_idx = int(rng.random() * n)
        v_idx = int(rng.random() * n)
        u, v = nodes[u_idx], nodes[v_idx]
        if u == v:
            continue
        d = math.hypot(pos[u][0] - pos[v][0], pos[u][1] - pos[v][1])
        if a <= d <= b:
            # Check min hops
            hops = get_hops(u, v, precomputed_hops)
            if hops >= min_hops:
                out.append((u, v))
    if len(out) < k:
        print(f"Warning: Only found {len(out)} pairs in band after {max_trials} trials.")
    return out

def get_hops(u, v, precomputed_hops):
    # Simple unweighted shortest path length (hops)
    if precomputed_hops and u in precomputed_hops:
        return precomputed_hops[u].get(v, np.inf)
    return np.inf  # Fallback if not precomputed

# ---------------------------
# Results writer
# ---------------------------
def write_results_block(results_path, images_dir, config, metrics, env_dict=None):
    with open(results_path, 'a') as f:
        utc = datetime.utcnow().isoformat() + 'Z'
        header = f"=== RUN {utc} | {images_dir} | seed={config['seed']} | commit=unknown | code_sha=n/a ===\n"
        f.write(header)
        f.write("STAGES: S1 v3.4 | S2 v3.3 | S3 v3.1 | S4 v3.2 | S5 v5.0\n")
        f.write("DOC SHAS: S1=n/a S2=n/a S3=n/a S4=n/a S5=n/a\n")
        f.write("CONFIG:\n")
        f.write(json.dumps(config, indent=2) + "\n")
        f.write("METRICS:\n")
        for m in metrics:
            f.write(f"  - {m}\n")
        f.write("IMAGES:\n")
        for img in metrics.get("images", []):
            f.write(f"  {img}\n")
        if env_dict:
            f.write("ENV:\n")
            for k, v in env_dict.items():
                f.write(f"  {k}={v}\n")
        f.write("=== END RUN ===\n")

# ---------------------------
# Main
# ---------------------------
def main():
    rng = np.random.default_rng(CONFIG["seed"])
    root = pathlib.Path(__file__).parent.resolve() if CONFIG["out_root"] is None else pathlib.Path(CONFIG["out_root"])
    images_root = root / CONFIG["images_dirname"]
    images_dir = next_versioned_dir(str(images_root))

    h_list = []
    L1_list, L2_list, Linf_list = [], [], []
    per_level_radii = []
    total_levels = len(CONFIG["refinements"])

    for level_idx, (nx_n, ny_n) in enumerate(CONFIG["refinements"], 1):
        h = 0.5 * (1.0 / (nx_n - 1) + 1.0 / (ny_n - 1)) if nx_n > 1 and ny_n > 1 else 1.0
        h_list.append(h)
        radius = radius_for_level(h, CONFIG["radius_phys"], CONFIG["radius_h_factor"])
        per_level_radii.append(radius)

        G, pos, _ = build_grid_graph(nx_n, ny_n, radius, diag_8=CONFIG["diag_8nbr"])

        # Precompute hops for a subset of sources to enforce min_hops efficiently
        nodes = list(G.nodes())
        num_src = min(500, len(nodes))  # Balance computation
        src_idx = rng.choice(len(nodes), size=num_src, replace=False)
        src_nodes = [nodes[i] for i in src_idx]
        hop_cache = {s: nx.single_source_shortest_path_length(G, s) for s in src_nodes}

        pairs = sample_pairs_band(
            G, pos, rng, CONFIG["pairs_per_level"],
            radius, CONFIG["band"], CONFIG["min_hops"], precomputed_hops=hop_cache
        )

        by_src = defaultdict(list)
        for u, v in pairs:
            by_src[u].append(v)
        sources = list(by_src.keys())

        dmap = shortest_path_lengths_all_sources(G, sources, use_scipy=SCIPY_OK)

        errs = []
        for u, targets in by_src.items():
            row = dmap.get(u, {})
            for v in targets:
                d_graph = row.get(v, np.inf)
                (x1, y1) = pos[u]; (x2, y2) = pos[v]
                d_true = math.hypot(x2 - x1, y2 - y1)
                e = abs(d_graph - d_true)
                if np.isfinite(e):
                    errs.append(e)
        errs = np.asarray(errs, dtype=float)
        if errs.size == 0:
            L1, L2, LINF = float("nan"), float("nan"), float("nan")
        else:
            L1 = float(np.mean(np.abs(errs)))
            L2 = float(math.sqrt(np.mean(errs**2)))
            LINF = float(np.max(np.abs(errs)))

        L1_list.append(L1); L2_list.append(L2); Linf_list.append(LINF)

        # Update progress bar
        print(nice_bar("refinements", level_idx, total_levels), end=("\n" if level_idx==total_levels else "\r"), flush=True)

    # ---------- PLOTS ----------
    # A) Scatter (distance vs. error) for the finest level only (visual sanity)
    # Recompute pairs and errs for finest level to show a scatter
    nx_n, ny_n = CONFIG["refinements"][-1]
    radius_fin = per_level_radii[-1]
    G, pos, h_fin = build_grid_graph(nx_n, ny_n, radius_fin, diag_8=CONFIG["diag_8nbr"])
    nodes = list(G.nodes())
    # small hops cache again (finest)
    num_src = min(500, len(nodes))
    src_idx = np.random.default_rng(CONFIG["seed"]).choice(len(nodes), size=num_src, replace=False)
    src_nodes = [nodes[i] for i in src_idx]
    hop_cache = {s: nx.single_source_shortest_path_length(G, s) for s in src_nodes}

    pairs_fin = sample_pairs_band(
        G, pos, np.random.default_rng(CONFIG["seed"]), CONFIG["pairs_per_level"],
        radius_fin, CONFIG["band"], CONFIG["min_hops"], precomputed_hops=hop_cache
    )
    sources_fin = sorted({u for (u, _) in pairs_fin})
    dmap_fin = shortest_path_lengths_all_sources(G, sources_fin, use_scipy=SCIPY_OK)

    dtrue_list, err_list = [], []
    for (u, v) in pairs_fin:
        (x1,y1) = pos[u]; (x2,y2) = pos[v]
        d_true = math.hypot(x2-x1, y2-y1)
        d_graph = float(dmap_fin.get(u, {}).get(v, np.inf))
        if np.isfinite(d_graph):
            dtrue_list.append(d_true)
            err_list.append(abs(d_graph - d_true))
    dtrue_arr = np.asarray(dtrue_list); err_arr = np.asarray(err_list)

    # Scatter
    fig1 = plt.figure(figsize=CONFIG["figsize"], dpi=CONFIG["dpi"])
    ax1 = fig1.add_subplot(111)
    ax1.scatter(dtrue_arr, err_arr, s=8, alpha=0.6)
    ax1.set_xlabel("True Euclidean distance")
    ax1.set_ylabel("|d_graph - d_true|")
    ax1.set_title(f"Distance error (finest {nx_n}×{ny_n})")
    fig1.tight_layout()
    fig_scatter_path = os.path.join(images_dir, "fig_dist_scatter_001.png")
    fig1.savefig(fig_scatter_path)
    plt.close(fig1)

    # B) Convergence plot: log-log of error vs h
    h_arr  = np.asarray(h_list, dtype=float)
    L1_arr = np.asarray(L1_list, dtype=float)
    L2_arr = np.asarray(L2_list, dtype=float)
    Li_arr = np.asarray(Linf_list, dtype=float)

    # zero-aware slope fit (use L2)
    nz = np.nonzero(np.isfinite(L2_arr) & (L2_arr > 0))[0]
    if nz.size >= 2:
        slope, _ = np.polyfit(np.log(h_arr[nz]), np.log(L2_arr[nz]), 1)
        slope_val = float(np.round(slope, 3))
    else:
        slope_val = float("nan")

    fig2 = plt.figure(figsize=CONFIG["figsize"], dpi=CONFIG["dpi"])
    ax2 = fig2.add_subplot(111)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.plot(h_arr, np.clip(L1_arr, 1e-16, None), "o-", label="L1")
    ax2.plot(h_arr, np.clip(L2_arr, 1e-16, None), "o-", label="L2")
    ax2.plot(h_arr, np.clip(Li_arr, 1e-16, None), "o-", label="Linf")
    ax2.set_xlabel("h (grid spacing)")
    ax2.set_ylabel("error")
    ax2.set_title(f"Convergence (slope L2 ≈ {slope_val})")
    ax2.legend()
    fig2.tight_layout()
    fig_conv_path = os.path.join(images_dir, "fig_convergence_001.png")
    fig2.savefig(fig_conv_path)
    plt.close(fig2)

    # ---------- RESULTS BLOCK ----------
    results_txt_path = os.path.join(os.path.dirname(str(images_root)), "distance_convergence_RESULTS.txt")
    metrics = [
        f"levels = {len(CONFIG['refinements'])}",
        f"pairs_total = {CONFIG['pairs_per_level'] * len(CONFIG['refinements'])}",
        f"h = {', '.join(f'{v:.6f}' for v in h_list)}",
        f"L1 = {', '.join(f'{v:.7g}' if np.isfinite(v) else 'nan' for v in L1_list)}",
        f"L2 = {', '.join(f'{v:.7g}' if np.isfinite(v) else 'nan' for v in L2_list)}",
        f"Linf = {', '.join(f'{v:.7g}' if np.isfinite(v) else 'nan' for v in Linf_list)}",
        f"L2_fit_slope = {slope_val if np.isfinite(slope_val) else 'nan'}",
        f"band = [{CONFIG['band'][0]:.3f}, {CONFIG['band'][1]:.3f}]",
        f"radius_used = {per_level_radii}",
        f"diag_8nbr = {bool(CONFIG['diag_8nbr'])}",
        f"images = [{fig_scatter_path}, {fig_conv_path}]"
    ]
    write_results_block(results_txt_path, images_dir, CONFIG, metrics, env_dict=None)

    # Console summary
    print("\n--- SUMMARY ---")
    print("h        :", h_list)
    print("L1       :", L1_list)
    print("L2       :", L2_list)
    print("Linf     :", Linf_list)
    print("L2 slope :", slope_val)
    print("images   :", [fig_scatter_path, fig_conv_path])
    print("results  :", results_txt_path)

if __name__ == "__main__":
    main()