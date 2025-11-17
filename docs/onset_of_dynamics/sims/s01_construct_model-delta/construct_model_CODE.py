# Onset_Of_Dynamics/sims/01_construct_model/construct_model_CODE.py
from __future__ import annotations
from pathlib import Path
import json, random
import hashlib

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from onset_of_dynamics.sims.s00_common.sim_paths import sim_root, results_path, images_root
from onset_of_dynamics.sims.s00_common.sim_plots import start_fig_env, save_fig
from onset_of_dynamics.sims.s00_common.sim_results import write_run_block, parse_doc_meta
from lib.efd.io_utils import run_header, sha256_of_file

# ---------- CONFIG ----------
SEED = 12345
N = 20          # nodes
P = 0.12        # edge probability (undirected for baseline picture)

def main():
    root = sim_root(__file__)
    rng = random.Random(SEED)
    np.random.seed(SEED)

    # Build a simple undirected graph just to visualize base structure
    G = nx.gnp_random_graph(N, P, seed=SEED, directed=False)

    # --- Save a quick layout plot ---
    vdir = start_fig_env(images_root(root))
    pos = nx.spring_layout(G, seed=SEED)
    fig, ax = plt.subplots(figsize=(6, 4))
    nx.draw_networkx(G, pos=pos, ax=ax, with_labels=False, node_size=80, width=0.8)
    ax.set_title("Constructed Model: Base Graph")
    ax.set_axis_off()
    figpath = save_fig(fig, vdir, prefix="construct_model_base")
    plt.close(fig)

    # --- Stage doc meta (parsed from docs headers) ---
    docs_dir = Path("docs") / "onset_of_dynamics"
    stages_meta = {
        "S1": parse_doc_meta(docs_dir / "Stage_1.txt"),
        "S2": parse_doc_meta(docs_dir / "Stage_2.txt"),
        "S3": parse_doc_meta(docs_dir / "Stage_3.txt"),
        "S4": parse_doc_meta(docs_dir / "Stage_4.txt"),
    }

    # --- Config dump (pretty) ---
    config_dump = json.dumps({"seed": SEED, "N": N, "P": P}, indent=2)

    # --- Metrics example ---
    metrics = [
        f"nodes = {G.number_of_nodes()}",
        f"edges = {G.number_of_edges()}",
        "note = baseline connectivity image only; math sims follow in next stages",
    ]

    # --- Results block ---
    code_sha = sha256_of_file(Path(__file__))
    header = run_header(vdir, seed=SEED, commit=None, code_sha=code_sha[:12])
    write_run_block(
        results_txt=results_path(root, "construct_model"),
        header_line=header,
        stages_meta=stages_meta,
        config_dump=config_dump,
        metrics_lines=metrics,
        image_paths=[figpath],
    )

if __name__ == "__main__":
    main()
