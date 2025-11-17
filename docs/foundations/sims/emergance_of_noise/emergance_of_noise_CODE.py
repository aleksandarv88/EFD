#!/usr/bin/env python3
# EFD — FOUNDATIONS — SIM S03 (Noise Emergence, observers-only)
# Neutral dynamics: unbiased random walks on K_N (no self-loops).
# Observers (no reweighting, no new rules):
#   A) Temporal persistence across walker blocks (Jaccard of top-p% edge sets)
#   B) Novelty rate over time (new bigrams per block)
#   C) Multi-scale fragility across percentiles
#   D) Cross-lens consistency: edge-intensity vs co-membership
#   E) Participation entropy per node
#
# IO:
# - Reads config ONLY from /config/defaults.yaml (searched upward)
# - Prints pretty JSON to stdout
# - Appends RUN block to a file named like this script, replacing "_CODE.py" with "_RESUTLS.txt"

import json, time, random, math, math as _math
from pathlib import Path
from collections import Counter, defaultdict, deque
from itertools import combinations

# ---------------------------
# Config loading
# ---------------------------
def load_yaml(p: Path) -> dict:
    try:
        import yaml  # pip install pyyaml
    except Exception:
        raise SystemExit("PyYAML required. Install with:  pip install pyyaml")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}

def find_defaults_yaml(start: Path) -> Path:
    cur = start
    for _ in range(12):
        cand = cur / "config" / "defaults.yaml"
        if cand.exists():
            return cand
        if cur.parent == cur:
            break
        cur = cur.parent
    raise SystemExit("Could not find /config/defaults.yaml")

def load_config(script_path: Path):
    p = find_defaults_yaml(script_path.parent)
    cfg = load_yaml(p)
    if not isinstance(cfg, dict):
        raise SystemExit(f"Config at {p} must be a mapping/dict.")
    return cfg, p

def results_filename_from_script(script_path: Path) -> Path:
    name = script_path.name
    if name.endswith("_CODE.py"):
        fname = name.replace("_CODE.py", "_RESUTLS.txt")  # spelling per your convention
    else:
        fname = f"{script_path.stem}_RESUTLS.txt"
    return script_path.parent / fname

# ---------------------------
# Neutral substrate dynamics (same as S01/S02)
# ---------------------------
def next_node_uniform(cur: int, N: int, rng: random.Random) -> int:
    r = rng.randrange(N - 1)
    return r if r < cur else r + 1

def normalize_cycle(cyc):
    if len(cyc) >= 2 and cyc[0] == cyc[-1]:
        cyc = cyc[:-1]
    if not cyc:
        return cyc
    rots = [tuple(cyc[i:] + cyc[:i]) for i in range(len(cyc))]
    return list(min(rots))

def walk_detect_loops(N: int, max_steps: int, rng: random.Random, max_loops: int):
    """Return loops (normalized) and a segment path (time-ordered) used to find them."""
    cur = rng.randrange(N)
    path = [cur]
    first_idx = {cur: 0}
    loops, steps = [], 0
    segments = []
    seg_start = 0

    while steps < max_steps and len(loops) < max_loops:
        nxt = next_node_uniform(cur, N, rng)
        path.append(nxt); steps += 1
        if nxt in first_idx:
            i0 = first_idx[nxt]
            cyc = normalize_cycle(path[i0:])
            loops.append(cyc)
            segments.append(path[seg_start:])
            # rebase
            path = [nxt]
            first_idx = {nxt: 0}
            seg_start = 0
        else:
            first_idx[nxt] = len(path) - 1
        cur = nxt

    if path:
        segments.append(path[seg_start:])
    return loops, steps, segments

# ---------------------------
# Observers utilities
# ---------------------------
def gini(xs):
    xs = sorted(x for x in xs if x > 0)
    if not xs:
        return 0.0
    n = len(xs); s = sum(xs)
    running = 0.0; weighted = 0.0
    for x in xs:
        running += x
        weighted += running
    B = weighted / (n * s)
    return 1 + 1/n - 2*B

def build_loop_intensity(N, loops):
    """node_load[u], edge_load[(i,j)] with i<j as undirected weight counts from loops."""
    node_load = [0]*N
    edge_load = defaultdict(int)
    bigrams = Counter()  # directed pairs for novelty/entropy
    for cyc in loops:
        for u in cyc:
            node_load[u] += 1
        seq = cyc + [cyc[0]] if cyc else []
        for a, b in zip(seq, seq[1:]):
            i, j = (a, b) if a < b else (b, a)
            edge_load[(i, j)] += 1
            bigrams[(a, b)] += 1
    return node_load, edge_load, bigrams

def build_co_membership(N, loops):
    """Weighted node-node graph by co-appearance in the same loop (unique per loop)."""
    W = defaultdict(int)   # (i,j) with i<j -> weight
    for cyc in loops:
        uniq = sorted(set(cyc))
        for a, b in combinations(uniq, 2):
            i, j = (a, b) if a < b else (b, a)
            W[(i, j)] += 1
    return W

def percentile_threshold(values, q):
    v = sorted([v for v in values if v > 0])
    if not v:
        return 0
    idx = int(q * (len(v) - 1))
    return v[idx]

def top_percentile_edge_set(weights, q):
    """Return set of edges with weight >= thr(q)."""
    vals = [w for w in weights.values() if w > 0]
    if not vals:
        return set(), 0
    thr = percentile_threshold(vals, q)
    S = {e for e, w in weights.items() if w >= thr}
    return S, thr

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    i = len(a & b)
    u = len(a | b)
    return i / u if u else 1.0

def entropy(probs):
    return -sum(p * math.log(p + 1e-12, 2) for p in probs if p > 0)

# ---------------------------
# Main
# ---------------------------
def main():
    script_path = Path(__file__).resolve()
    cfg, cfg_path = load_config(script_path)

    sim_cfg = cfg.get("sim", {})
    N                  = int(sim_cfg.get("N", 256))
    walkers            = int(sim_cfg.get("walkers", 10000))
    max_steps          = int(sim_cfg.get("max_steps", 4000))
    max_loops_per_walk = int(sim_cfg.get("max_loops_per_walk", 5))
    seed               = int(sim_cfg.get("seed", 12345))

    obs_cfg = cfg.get("observer", {})
    # Blocks for temporal stability
    blocks             = int(obs_cfg.get("noise_blocks", 5))
    top_edge_percentile= float(obs_cfg.get("noise_top_edge_percentile", 0.995))  # for block Jaccard/persistence
    # Multi-scale sweep
    frag_percentiles   = obs_cfg.get("noise_frag_percentiles", [0.97, 0.985, 0.99, 0.995, 0.999])
    # Co-membership top set
    top_co_percentile  = float(obs_cfg.get("noise_top_co_percentile", 0.995))
    # Participation entropy report quantiles
    entropy_quants     = obs_cfg.get("noise_entropy_quantiles", [0.0, 0.25, 0.5, 0.75, 0.9, 1.0])

    rng = random.Random(seed)

    # Partition walkers into blocks (roughly equal)
    per_block = [walkers // blocks] * blocks
    for i in range(walkers % blocks):
        per_block[i] += 1

    # Generate per-block loops
    all_loops = []
    all_segments = []
    total_steps = 0

    blocks_loops = []
    blocks_bigrams = []
    blk_rng = random.Random(seed)  # deterministic per block
    for bi in range(blocks):
        blk_loops = []
        blk_bigrams = Counter()
        for _ in range(per_block[bi]):
            loops, steps, segments = walk_detect_loops(N, max_steps, blk_rng, max_loops_per_walk)
            total_steps += steps
            all_loops.extend(loops)
            all_segments.extend(segments)
            blk_loops.extend(loops)
        # build per-block intensity just for novelty/edge-set later
        _, _, blk_big = build_loop_intensity(N, blk_loops)
        blk_bigrams.update(blk_big)
        blocks_loops.append(blk_loops)
        blocks_bigrams.append(blk_bigrams)

    # Global intensity/co-membership
    node_load, edge_load, bigrams = build_loop_intensity(N, all_loops)
    coW = build_co_membership(N, all_loops)

    # --- A) Temporal persistence across blocks (Jaccard of top-p% edge sets) ---
    block_edge_sets = []
    thr_per_block = []
    for bi in range(blocks):
        # build weights for this block only
        _, blk_edges, _ = build_loop_intensity(N, blocks_loops[bi])
        S, thr = top_percentile_edge_set(blk_edges, top_edge_percentile)
        block_edge_sets.append(S)
        thr_per_block.append(thr)

    # pairwise Jaccards, and per-edge persistence across blocks
    pairwise = []
    for i in range(blocks):
        for j in range(i+1, blocks):
            pairwise.append(jaccard(block_edge_sets[i], block_edge_sets[j]))
    # edge persistence fraction
    union_all = set().union(*block_edge_sets) if block_edge_sets else set()
    appear_counts = Counter()
    for S in block_edge_sets:
        for e in S:
            appear_counts[e] += 1
    persistence_hist = Counter(appear_counts.values())
    persistence_summary = {
        "num_edges_in_union": len(union_all),
        "appear_count_hist": dict(sorted(persistence_hist.items())),
        "stable_edges_count": sum(1 for e,c in appear_counts.items() if c >= max(2, int(math.ceil(0.6*blocks)))),
        "unstable_edges_count": sum(1 for e,c in appear_counts.items() if c <= 1),
        "pairwise_jaccard_mean": (sum(pairwise)/len(pairwise)) if pairwise else None,
        "pairwise_jaccard_min": (min(pairwise) if pairwise else None),
        "pairwise_jaccard_max": (max(pairwise) if pairwise else None),
    }

    # --- B) Novelty rate over time (new bigrams per block) ---
    seen_bigrams = set()
    novelty = []
    for bi in range(blocks):
        blk_pairs = set(blocks_bigrams[bi].keys())
        new_pairs = blk_pairs - seen_bigrams
        total_pairs = len(blk_pairs) if blk_pairs else 1
        novelty.append({
            "block": bi,
            "new_bigrams": len(new_pairs),
            "total_bigrams": len(blk_pairs),
            "new_fraction": len(new_pairs)/total_pairs
        })
        seen_bigrams |= blk_pairs

    # --- C) Multi-scale fragility (edge-intensity) ---
    vals = [w for w in edge_load.values() if w > 0]
    thr_map = {q: percentile_threshold(vals, q) for q in frag_percentiles} if vals else {}
    # For each edge, find max percentile q where it survives
    edge_frag_scores = []
    for e, w in edge_load.items():
        if w <= 0:
            continue
        max_q = 0.0
        for q in sorted(thr_map.keys()):
            if w >= thr_map[q]:
                max_q = q
        edge_frag_scores.append(max_q)
    frag_hist = Counter(edge_frag_scores)

    # --- D) Cross-lens consistency (edge-intensity vs co-membership) ---
    S_edge, thr_edge = top_percentile_edge_set(edge_load, top_edge_percentile)
    S_co,   thr_co   = top_percentile_edge_set(coW,       top_co_percentile)
    # unify key spaces (both are undirected pairs (i,j) with i<j)
    inter = len(S_edge & S_co)
    union = len(S_edge | S_co) if (S_edge or S_co) else 1
    cross_consistency = {
        "edge_top_percentile": top_edge_percentile, "edge_threshold": thr_edge, "edge_top_count": len(S_edge),
        "co_top_percentile": top_co_percentile,     "co_threshold": thr_co,     "co_top_count": len(S_co),
        "intersection_count": inter,
        "union_count": union,
        "jaccard": inter / union if union else 1.0
    }

    # --- E) Participation entropy per node (based on outgoing bigram distribution) ---
    # For each node u, consider its outgoing bigrams (u->v) frequencies
    neighbors_by_u = defaultdict(Counter)
    for (a, b), c in bigrams.items():
        neighbors_by_u[a][b] += c
    entropies = []
    for u in range(N):
        total = sum(neighbors_by_u[u].values())
        if total <= 0:
            entropies.append(0.0)
        else:
            probs = [cnt/total for cnt in neighbors_by_u[u].values()]
            entropies.append(entropy(probs))

    def quantiles(xs, qs):
        if not xs:
            return {str(q): None for q in qs}
        ys = sorted(xs)
        out = {}
        for q in qs:
            if q <= 0:
                out[str(q)] = ys[0]; continue
            if q >= 1:
                out[str(q)] = ys[-1]; continue
            pos = q * (len(ys) - 1)
            lo = int(pos); hi = min(lo + 1, len(ys) - 1)
            frac = pos - lo
            out[str(q)] = ys[lo] * (1 - frac) + ys[hi] * frac
        return out

    entropy_summary = {
        "quantiles": quantiles(entropies, entropy_quants),
        "mean": (sum(entropies)/len(entropies)) if entropies else 0.0
    }

    # Global loop/edge heterogeneity (context)
    node_g = gini(node_load)
    edge_g = gini([w for w in edge_load.values()])

    # Assemble results
    results = {
        "meta": {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config_path": str(cfg_path),
        },
        "sim": {
            "seed": seed,
            "N": N,
            "walkers": walkers,
            "max_steps_per_walker": max_steps,
            "max_loops_per_walk": max_loops_per_walk,
            "total_steps": total_steps,
            "blocks": blocks
        },
        "heterogeneity": {
            "node_load_gini": node_g,
            "edge_load_gini": edge_g
        },
        "temporal_persistence": {
            "top_edge_percentile": top_edge_percentile,
            "pairwise_jaccard": {
                "mean": persistence_summary["pairwise_jaccard_mean"],
                "min": persistence_summary["pairwise_jaccard_min"],
                "max": persistence_summary["pairwise_jaccard_max"]
            },
            "edge_persistence": {
                "num_edges_in_union": persistence_summary["num_edges_in_union"],
                "appear_count_hist": persistence_summary["appear_count_hist"],
                "stable_edges_count": persistence_summary["stable_edges_count"],
                "unstable_edges_count": persistence_summary["unstable_edges_count"]
            }
        },
        "novelty_rate": novelty,
        "fragility": {
            "percentiles": frag_percentiles,
            "edge_max_survival_percentile_hist": dict(sorted(frag_hist.items()))
        },
        "cross_lens_consistency": cross_consistency,
        "participation_entropy": entropy_summary
    }

    # Print JSON
    print(json.dumps(results, indent=2))

    # Append RUN block
    results_path = results_filename_from_script(script_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    run_header = (
        f"=== RUN {results['meta']['timestamp_utc']} | "
        f"seed={seed} | N={N} | walkers={walkers} | steps_per={max_steps} ==="
    )
    block = [
        run_header,
        "RESULTS:",
        json.dumps({
            "heterogeneity": results["heterogeneity"],
            "temporal_persistence": results["temporal_persistence"],
            "cross_lens_consistency": results["cross_lens_consistency"],
            "participation_entropy": results["participation_entropy"],
            # keep novelty & fragility compact in the file to avoid huge dumps
            "novelty_rate_brief": {
                "blocks": len(novelty),
                "mean_new_fraction": (sum(x["new_fraction"] for x in novelty)/len(novelty)) if novelty else None
            },
            "fragility_brief": {
                "percentiles": frag_percentiles,
                "unique_edges_scored": sum(results["fragility"]["edge_max_survival_percentile_hist"].values())
            }
        }, indent=2),
        "",
    ]
    with results_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(block) + "\n")

if __name__ == "__main__":
    main()
