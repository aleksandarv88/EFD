#!/usr/bin/env python3
# EFD — FOUNDATIONS — SIM S02 (observers only, two-tier boundaries)
# Neutral dynamics: unbiased random walks on K_N (no self-loops).
# We OBSERVE (no reweighting, no new rules):
#   1) Loop-intensity structure (nodes/edges) across percentiles → components
#   2) Co-membership structure (nodes co-appearing in same loop) across percentiles
#   3) Boundaries via TWO-TIER observer (components at q_hi; boundaries from edges ≥ q_lo across those components)
#   4) Boundary strength via weighted conductance per component
#   5) Sequentiality: component run-lengths (at configurable percentile)
#
# IO:
# - Reads config ONLY from /config/defaults.yaml (searched upward)
# - Prints pretty JSON to stdout
# - Appends RUN block to a file named like this script, replacing "_CODE.py" with "_RESUTLS.txt"

import json, time, random
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
# Neutral substrate dynamics
# ---------------------------
def next_node_uniform(cur: int, N: int, rng: random.Random) -> int:
    # sample any node != cur uniformly (complete graph without self-loops)
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
    """Return loops (normalized node lists) and time-ordered segments used to find them."""
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
# Observers: intensity, structure, boundaries, co-membership, sequentiality
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
    for cyc in loops:
        for u in cyc:
            node_load[u] += 1
        for a, b in zip(cyc, cyc[1:]+[cyc[0]]):
            i, j = (a, b) if a < b else (b, a)
            edge_load[(i, j)] += 1
    return node_load, edge_load

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
    """Return threshold t such that fraction q of NONZERO values are <= t."""
    v = sorted([v for v in values if v > 0])
    if not v:
        return 0
    idx = int(q * (len(v) - 1))
    return v[idx]

def threshold_unweighted_adj_from_weighted(N, weights, thr):
    """Unweighted adjacency where weights[(i,j)] >= thr."""
    adj = [[] for _ in range(N)]
    for (i, j), w in weights.items():
        if w >= thr:
            adj[i].append(j)
            adj[j].append(i)
    return adj

def connected_components(adj):
    N = len(adj)
    seen = [False]*N
    comps = []
    for s in range(N):
        if seen[s]:
            continue
        if not adj[s]:
            seen[s] = True
            comps.append([s])
            continue
        q = deque([s]); seen[s] = True; cur = [s]
        while q:
            u = q.popleft()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    q.append(v)
                    cur.append(v)
        comps.append(cur)
    return comps

def component_map(comps, N):
    cid = [-1]*N
    for k, nodes in enumerate(comps):
        for u in nodes:
            cid[u] = k
    return cid

def conductance_for_component(nodes, weights, N):
    """
    Weighted conductance φ(A) = cut(A,¬A) / min(vol(A), vol(¬A)),
    where vol(*) is sum of incident weights (each undirected edge counts once per endpoint).
    """
    in_set = set(nodes)
    volume_A = 0.0
    cut_A = 0.0
    total_volume = 0.0
    for (i, j), w in weights.items():
        if i in in_set: volume_A += w
        if j in in_set: volume_A += w
        total_volume += 2*w
        if (i in in_set) ^ (j in in_set):
            cut_A += w
    volume_notA = total_volume - volume_A
    denom = min(volume_A, volume_notA) if min(volume_A, volume_notA) > 0 else 1.0
    return cut_A / denom

def multires_structure(weights, N, percentiles, top_k):
    """Per-percentile components + same-threshold 'bridges' (for context) + conductance."""
    vals = [w for w in weights.values() if w > 0]
    out = []
    if not vals:
        return out
    for q in percentiles:
        thr = percentile_threshold(vals, q)
        adj = threshold_unweighted_adj_from_weighted(N, weights, thr)
        comps = connected_components(adj)
        cid = component_map(comps, N)

        # NOTE: same-threshold "bridges" will often be zero (edge is in the graph).
        bridges = {}
        for (i, j), w in weights.items():
            if w >= thr and cid[i] != cid[j]:
                bridges[(i, j)] = w

        thr_weights = {(i, j): w for (i, j), w in weights.items() if w >= thr}
        conds = [conductance_for_component(nodes, thr_weights, N) for nodes in comps]

        out.append({
            "percentile": q,
            "threshold": thr,
            "num_components": len(comps),
            "component_sizes_sorted": sorted([len(c) for c in comps], reverse=True)[:top_k],
            "num_boundary_edges": len(bridges),
            "top_boundary_edges": sorted([[i, j, w] for (i, j), w in bridges.items()],
                                         key=lambda x: -x[2])[:top_k],
            "conductance_summary": {
                "mean": (sum(conds)/len(conds)) if conds else 0.0,
                "min": min(conds) if conds else 0.0,
                "max": max(conds) if conds else 0.0
            }
        })
    return out

def boundaries_two_tier(weights, N, q_hi, q_lo, top_k):
    """Two-tier boundary observer:
       - components from edges >= thr(q_hi)
       - boundaries are edges >= thr(q_lo) whose endpoints lie in different hi-components
       Conductance for hi-components is measured on the lo-thresholded weights.
    """
    vals = [w for w in weights.values() if w > 0]
    if not vals:
        return {
            "percentile_hi": q_hi, "percentile_lo": q_lo,
            "threshold_hi": 0, "threshold_lo": 0,
            "num_components": N, "component_sizes_sorted": [1]*N,
            "num_boundary_edges": 0, "top_boundary_edges": [],
            "conductance_summary": {"mean": 0.0, "min": 0.0, "max": 0.0}
        }

    thr_hi = percentile_threshold(vals, q_hi)
    thr_lo = percentile_threshold(vals, q_lo)

    # Components at hi threshold
    adj_hi = threshold_unweighted_adj_from_weighted(N, weights, thr_hi)
    comps_hi = connected_components(adj_hi)
    cid_hi = component_map(comps_hi, N)

    # Boundaries from edges at lo threshold across hi-components
    bridges = {}
    for (i, j), w in weights.items():
        if w >= thr_lo and cid_hi[i] != cid_hi[j]:
            bridges[(i, j)] = w

    # Conductance of hi-components using lo-thresholded weights
    lo_weights = {(i, j): w for (i, j), w in weights.items() if w >= thr_lo}
    conds = [conductance_for_component(nodes, lo_weights, N) for nodes in comps_hi]

    return {
        "percentile_hi": q_hi,
        "percentile_lo": q_lo,
        "threshold_hi": thr_hi,
        "threshold_lo": thr_lo,
        "num_components": len(comps_hi),
        "component_sizes_sorted": sorted([len(c) for c in comps_hi], reverse=True)[:top_k],
        "num_boundary_edges": len(bridges),
        "top_boundary_edges": sorted([[i, j, w] for (i, j), w in bridges.items()],
                                     key=lambda x: -x[2])[:top_k],
        "conductance_summary": {
            "mean": (sum(conds)/len(conds)) if conds else 0.0,
            "min": min(conds) if conds else 0.0,
            "max": max(conds) if conds else 0.0
        }
    }

def run_length_stats(seq):
    if not seq:
        return {"count": 0}
    runs = []
    last = seq[0]; r = 1
    for x in seq[1:]:
        if x == last:
            r += 1
        else:
            runs.append(r); last = x; r = 1
    runs.append(r)
    return {
        "count": len(runs),
        "mean": sum(runs)/len(runs),
        "hist": dict(sorted(Counter(runs).items()))
    }

def loop_phase_order_stats(loops, top_k):
    pair_counts = Counter()
    for cyc in loops:
        for a, b in zip(cyc, cyc[1:]+[cyc[0]]):
            pair_counts[(a, b)] += 1
    top = pair_counts.most_common(top_k)
    return {"top_bigram_edges": [[a, b, c] for ((a, b), c) in top]}

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
    rng = random.Random(seed)

    obs_cfg = cfg.get("observer", {})
    edge_percentiles   = obs_cfg.get("edge_intensity_percentiles", [0.97, 0.985, 0.99, 0.995, 0.999, 0.9995])
    co_percentiles     = obs_cfg.get("co_membership_percentiles", [0.99, 0.995, 0.999, 0.9995])
    seq_percentile     = float(obs_cfg.get("sequential_component_percentile", 0.985))
    q_hi               = float(obs_cfg.get("component_percentile_hi", 0.995))
    q_lo               = float(obs_cfg.get("boundary_percentile_lo", 0.985))
    top_k              = int(obs_cfg.get("top_k", 20))

    # 1) Generate loops with neutral dynamics
    all_loops = []
    segments_all = []
    total_steps = 0
    for _ in range(walkers):
        loops, steps, segments = walk_detect_loops(N, max_steps, rng, max_loops_per_walk)
        total_steps += steps
        all_loops.extend(loops)
        segments_all.extend(segments)

    # 2) Loop-intensity field
    node_load, edge_load = build_loop_intensity(N, all_loops)
    node_gini = gini(node_load)
    edge_vals = list(edge_load.values())
    edge_gini = gini(edge_vals) if edge_vals else 0.0

    # 3) Multi-resolution structure/boundaries on edge-intensity graph
    structure_multires = multires_structure(edge_load, N, edge_percentiles, top_k)

    # 4) Co-membership structure (multi-resolution)
    coW = build_co_membership(N, all_loops)
    co_vals = [w for w in coW.values() if w > 0]
    co_multires = multires_structure(coW, N, co_percentiles, top_k) if co_vals else []

    # 5) Two-tier boundaries (edge-intensity & co-membership)
    edge_two_tier = boundaries_two_tier(edge_load, N, q_hi, q_lo, top_k)
    co_two_tier   = boundaries_two_tier(coW,       N, q_hi, q_lo, top_k) if co_vals else None

    # 6) Sequentiality at chosen edge percentile
    if edge_vals:
        thr_seq = percentile_threshold([v for v in edge_vals if v > 0], seq_percentile)
        adj_seq = threshold_unweighted_adj_from_weighted(N, edge_load, thr_seq)
        comps_seq = connected_components(adj_seq)
        cid_seq = component_map(comps_seq, N)
    else:
        cid_seq = list(range(N))

    comp_seq = []
    for seg in segments_all:
        comp_seq.extend([cid_seq[u] if 0 <= u < N else -1 for u in seg])
    run_stats = run_length_stats(comp_seq)

    # phase order motifs
    phase_stats = loop_phase_order_stats(all_loops, top_k)

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
        },
        "loops_summary": {
            "count_total": len(all_loops),
            "node_load_gini": node_gini,
            "edge_load_gini": edge_gini,
            "node_load_top": sorted([[i, c] for i, c in enumerate(node_load)], key=lambda x: -x[1])[:top_k],
            "edge_load_top": sorted([[i, j, w] for (i, j), w in edge_load.items()], key=lambda x: -x[2])[:top_k],
        },
        "structure_multires": structure_multires,
        "co_membership_multires": co_multires,
        "two_tier_boundaries": {
            "edge_intensity": edge_two_tier,
            "co_membership": co_two_tier
        },
        "sequentiality": {
            "using_edge_percentile": seq_percentile,
            "component_run_lengths": run_stats,
            "loop_phase_order": phase_stats,
        }
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
            "loops_summary": {
                "count_total": results["loops_summary"]["count_total"],
                "node_load_gini": results["loops_summary"]["node_load_gini"],
                "edge_load_gini": results["loops_summary"]["edge_load_gini"],
            },
            "structure_multires": results["structure_multires"],
            "co_membership_multires": results["co_membership_multires"],
            "two_tier_boundaries": results["two_tier_boundaries"],
            "sequentiality": results["sequentiality"],
        }, indent=2),
        "",
    ]
    with results_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(block) + "\n")

if __name__ == "__main__":
    main()
