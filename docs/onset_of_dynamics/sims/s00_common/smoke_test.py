# docs/onset_of_dynamics/sims/06_stage5_continuum_limit/00_smoke_test/smoke_test_CODE.py
from __future__ import annotations
from pathlib import Path
import sys
import math
import datetime as dt

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# --- import helpers (add 00_common to sys.path) ---
HERE = Path(__file__).resolve()
SIMS_DIR = HERE.parents[2]            # .../sims
COMMON_DIR = SIMS_DIR / "00_common"
sys.path.append(str(COMMON_DIR))

from grid_refine import GridSpec, build_grid, sample_pairs_by_spacing
from meshes import sphere_graph, cylinder_graph, saddle_graph
from loops import simple_cycles_local, polygon_area_from_pos
from rng import make_rng

from sim_paths import sim_root, images_root, results_path
from sim_plots import start_fig_env, save_fig
from sim_results import write_run_block  # assumes your version takes (results_path, header, stages_meta, config_dump, metrics_lines, image_paths)

# ---- basic doc metadata (optional/empty here) ----
STAGES_META = {
    "S1": {"Version": "v3.4", "DocSHA": "n/a"},
    "S2": {"Version": "v3.3", "DocSHA": "n/a"},
    "S3": {"Version": "v3.1", "DocSHA": "n/a"},
    "S4": {"Version": "v3.2", "DocSHA": "n/a"},
    "S5": {"Version": "v5.0", "DocSHA": "n/a"},
}

def _header(seed: int, commit: str = "unknown") -> str:
    utc = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return f"=== RUN {utc} | {{images_dir}} | seed={seed} | commit={commit} | code_sha=n/a ==="

def main() -> None:
    seed = 12345
    rng = make_rng(seed)

    # paths & versioned images dir
    root = sim_root(__file__)
    vdir = start_fig_env(images_root(root))      # creates images/vNNN
    results_txt = results_path(root, "smoke_test")

    # ---------- Figure 1: Grid + sampled pairs ----------
    G = build_grid(GridSpec(nx=32, ny=32, diag=True, tag_edge_len=True))
    pos = nx.get_node_attributes(G, "pos")
    pairs = sample_pairs_by_spacing(G, k=20, rng=rng, min_sep=0.25)

    fig1 = plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(G, pos, node_size=8)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.4)
    # draw sampled pairs as straight segments
    for (u, v) in pairs:
        x1, y1 = pos[u]; x2, y2 = pos[v]
        plt.plot([x1, x2], [y1, y2])
    plt.title("Smoke: Grid (32×32) + sampled pairs")
    plt.axis("equal"); plt.tight_layout()
    img1 = save_fig(fig1, vdir, prefix="fig_grid_pairs")

    # pick a center and show a couple of local cycles with area
    center = (16, 16)
    cycles = simple_cycles_local(G, center, max_len=6)[:3]  # first 3
    fig2 = plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(G, pos, node_size=6)
    nx.draw_networkx_edges(G, pos, width=0.3, alpha=0.25)
    cx, cy = pos[center]
    plt.scatter([cx], [cy], s=40)
    for k, cyc in enumerate(cycles, start=1):
        xs = [pos[n][0] for n in cyc]
        ys = [pos[n][1] for n in cyc]
        plt.plot(xs, ys)
        area = polygon_area_from_pos(G, cyc)
        mid = len(cyc) // 2
        mx, my = pos[cyc[mid]]
        plt.text(mx, my, f"A{k}={area:.3g}", fontsize=8)
    plt.title("Smoke: local cycles & areas around center")
    plt.axis("equal"); plt.tight_layout()
    img2 = save_fig(fig2, vdir, prefix="fig_cycles_areas")

    # ---------- Figure 2: Mesh previews (UV projection) ----------
    sph, _ = sphere_graph(Nu=24, Nv=12, R=1.0)
    cyl, _ = cylinder_graph(Nu=32, Nv=12, R=1.0, H=1.0)
    sad, _ = saddle_graph(Nx=30, Ny=30, a=1.0, b=1.0, extent=1.0)

    def draw_uv_graph(Gm: nx.Graph, title: str, fname: str) -> str:
        uv = nx.get_node_attributes(Gm, "pos") or nx.get_node_attributes(Gm, "uv")
        fig = plt.figure(figsize=(6, 4))
        nx.draw_networkx_nodes(Gm, uv, node_size=6)
        nx.draw_networkx_edges(Gm, uv, width=0.4, alpha=0.35)
        plt.title(title)
        plt.axis("equal"); plt.tight_layout()
        return save_fig(fig, vdir, prefix=fname)

    img3 = draw_uv_graph(sph, "Smoke: Sphere mesh (UV)", "fig_mesh_sphere")
    img4 = draw_uv_graph(cyl, "Smoke: Cylinder mesh (UV)", "fig_mesh_cylinder")
    img5 = draw_uv_graph(sad, "Smoke: Saddle mesh (UV)", "fig_mesh_saddle")

    # ---------- Results block ----------
    metrics = [
        f"grid_nodes = {G.number_of_nodes()}",
        f"grid_edges = {G.number_of_edges()}",
        f"pairs_sampled = {len(pairs)}",
        f"sphere_nodes = {sph.number_of_nodes()}",
        f"cylinder_nodes = {cyl.number_of_nodes()}",
        f"saddle_nodes = {sad.number_of_nodes()}",
    ]

    header_line = _header(seed).replace("{images_dir}", str(vdir).replace("\\", "/"))
    config_dump = "smoke:\n  grid: {nx:32, ny:32, diag:true}\n  sphere:{Nu:24,Nv:12}\n  cylinder:{Nu:32,Nv:12}\n  saddle:{Nx:30,Ny:30}\n"

    write_run_block(
        results_txt,
        header_line,
        STAGES_META,
        config_dump,
        metrics,
        [img1, img2, img3, img4, img5],
    )

if __name__ == "__main__":
    main()
