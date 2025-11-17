#!/usr/bin/env python3
# EFD — FOUNDATIONS — SIM S05 (Core-conditioned metrics with frozen-core, q-sweep, and core Jaccard)
# Neutral dynamics (unbiased random walk on K_N). We DO NOT change step probabilities.
# We "use noise as given" via the observer lens.
#
# Adds:
#  (a) Frozen-core tracking: pick core once at the smallest budget cut (and first q_hi in the sweep),
#      then track (i) weight share on frozen-core nodes, (ii) gini on edges within those nodes,
#      (iii) persistence of the original frozen-core edges above the current hi-threshold.
#  (b) q-sweep: evaluate multiple q_hi values in one run, given in YAML.
#  (c) Core-restricted cross-lens Jaccard: on the core nodes, compare
#      A = top-q_hi fraction of edges by weight (within core nodes) vs
#      B = edges that pass the hi-threshold (>= thr_hi) within core nodes.
#
# IO:
# - Reads config only from /config/defaults.yaml (searched upward)
# - Prints full JSON to stdout
# - Appends a compact RUN block to a file named like this script, replacing "_CODE.py" with "_RESUTLS.txt"

import json, time, random, math
from pathlib import Path
from collections import defaultdict, deque

# ---------------------------
# Config helpers
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
        fname = name.replace("_CODE.py", "_RESUTLS.txt")  # keep your exact spelling
    else:
        fname = f"{script_path.stem}_RESUTLS.txt"
    return script_path.parent / fname

# ---------------------------
# Neutral substrate dynamics
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
    cur = rng.randrange(N)
    path = [cur]
    first_idx = {cur: 0}
    loops, steps = [], 0
    while steps < max_steps and len(loops) < max_loops:
        nxt = next_node_uniform(cur, N, rng)
        path.append(nxt); steps += 1
        if nxt in first_idx:
            i0 = first_idx[nxt]
            cyc = normalize_cycle(path[i0:])
            loops.append(cyc)
            path = [nxt]
            first_idx = {nxt: 0}
        else:
            first_idx[nxt] = len(path) - 1
        cur = nxt
    return loops, steps

# ---------------------------
# Observers & utils
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

def accumulate_loops_into_loads(N, loops, node_load, edge_load):
    for cyc in loops:
        for u in cyc:
            node_load[u] += 1
        for a, b in zip(cyc, cyc[1:]+[cyc[0]]):
            i, j = (a, b) if a < b else (b, a)
            edge_load[(i, j)] += 1

def percentile_threshold(values, q):
    v = sorted([v for v in values if v > 0])
    if not v:
        return 0
    idx = int(q * (len(v) - 1))
    return v[idx]

def threshold_unweighted_adj_from_weighted(N, weights, thr):
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

def conductance_for_component(nodes, weights):
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

def head_mass(edge_weights_list, eps):
    xs = sorted([x for x in edge_weights_list if x > 0], reverse=True)
    if not xs:
        return 0.0
    k = max(1, int(math.ceil(eps * len(xs))))
    top_sum = sum(xs[:k])
    tot = sum(xs)
    return top_sum / tot if tot > 0 else 0.0

def top_q_edge_set_within_nodes(edge_load, nodes_set, q_hi):
    # Return set of edges among nodes_set that are in the top q_hi by weight (ties broken by cutoff)
    pairs = [((i, j), w) for (i, j), w in edge_load.items() if i in nodes_set and j in nodes_set and w > 0]
    if not pairs:
        return set()
    pairs.sort(key=lambda t: t[1], reverse=True)
    k = max(1, int(math.ceil(q_hi * len(pairs))))
    return set(p for p, _ in pairs[:k])

# ---------------------------
# Aggregation helpers
# ---------------------------
def mean_std_at_index(arrs, i):
    vals = [arr[i] for arr in arrs]
    mu = sum(vals) / len(vals)
    var = sum((x - mu)**2 for x in vals) / (len(vals)-1 if len(vals) > 1 else 1)
    return mu, math.sqrt(var)

def agg_series(per_seed, getter, K):
    means, stds = [], []
    for i in range(K):
        mu, sd = mean_std_at_index([getter(e) for e in per_seed], i)
        means.append(mu); stds.append(sd)
    return means, stds

# ---------------------------
# Main
# ---------------------------
def main():
    script_path = Path(__file__).resolve()
    cfg, cfg_path = load_config(script_path)

    sim_cfg = cfg.get("sim", {})
    N                  = int(sim_cfg.get("N", 256))
    max_steps          = int(sim_cfg.get("max_steps", 4000))
    max_loops_per_walk = int(sim_cfg.get("max_loops_per_walk", 5))
    base_seed          = int(sim_cfg.get("seed", 12345))

    obs_cfg = cfg.get("observer", {})
    budgets = obs_cfg.get("gini_budgets_walkers", [2000, 5000, 10000, 20000])
    budgets = sorted(int(x) for x in budgets if int(x) > 0)
    if not budgets:
        raise SystemExit("observer.gini_budgets_walkers must contain positive integers.")

    # q-sweep (hi percentiles). Ensure ascending and within (0,1).
    q_hi_sweep = obs_cfg.get("q_sweep_hi", [0.99, 0.995, 0.997, 0.999])
    q_hi_sweep = sorted(float(q) for q in q_hi_sweep if 0.0 < float(q) < 1.0)
    if not q_hi_sweep:
        q_hi_sweep = [0.99, 0.995, 0.997, 0.999]

    q_lo = float(obs_cfg.get("boundary_percentile_lo", 0.985))
    head_eps_list = obs_cfg.get("head_mass_epsilons", [0.01, 0.05])
    head_eps_list = [float(e) for e in head_eps_list if 0 < float(e) <= 1.0]

    # Frozen-core: pick at first budget and first q_hi in sweep.
    frozen_pick_budget = int(obs_cfg.get("frozen_core_pick_budget", budgets[0]))

    seeds  = obs_cfg.get("gini_seeds", None)
    if seeds:
        seeds = [int(s) for s in seeds]
    else:
        repeats = int(obs_cfg.get("gini_repeats", 3))
        seeds = [base_seed + k for k in range(repeats)]

    max_budget = max(budgets)
    total_steps = 0
    K = len(budgets)

    # Prepare containers (per q_hi, per seed)
    per_q = {q: [] for q in q_hi_sweep}

    for s in seeds:
        rng = random.Random(s)
        node_load = [0]*N
        edge_load = defaultdict(int)

        walkers_done = 0
        cut_idx = 0

        # Frozen-core state (filled when we hit frozen_pick_budget for the first q_hi)
        frozen_nodes = None
        frozen_edges_initial = set()
        frozen_picked = False

        # Per-q trackers over budgets
        per_q_seed = {q: {
            "global_gini": [],
            "core_gini": [],
            "outside_gini": [],
            "core_over_global_gini": [],
            "core_share": [],
            "boundary_load_share": [],
            "boundary_conductance_mean": [],
            "head_mass_global": {str(eps): [] for eps in head_eps_list},
            "head_mass_core":   {str(eps): [] for eps in head_eps_list},
            "head_mass_out":    {str(eps): [] for eps in head_eps_list},
            # Jaccard (core-restricted, top-q vs hi-threshold)
            "core_jaccard_edgeint_vs_comembership": [],
            # Frozen-core tracking (same across q_hi values once picked; we record per q for convenience)
            "frozen_core_weight_share": [],
            "frozen_core_gini": [],
            "frozen_core_edge_persistence": []  # fraction of frozen-core initial edges that remain >= current thr_hi
        } for q in q_hi_sweep}

        while walkers_done < max_budget:
            loops, steps = walk_detect_loops(N, max_steps, rng, max_loops_per_walk)
            total_steps += steps
            accumulate_loops_into_loads(N, loops, node_load, edge_load)
            walkers_done += 1

            while cut_idx < K and walkers_done >= budgets[cut_idx]:
                # Common values
                edge_vals_all = list(edge_load.values())
                vals_pos_all = [w for w in edge_vals_all if w > 0]
                total_w_all = sum(edge_vals_all) if edge_vals_all else 0.0

                for q_hi in q_hi_sweep:
                    # Defaults when nothing positive yet
                    g_global = gini(edge_vals_all) if edge_vals_all else 0.0
                    g_core = g_out = ratio = cshare = bshare = cond_mean = 0.0
                    jacc_core = 0.0
                    hm_global = {str(eps): 0.0 for eps in head_eps_list}
                    hm_core   = {str(eps): 0.0 for eps in head_eps_list}
                    hm_out    = {str(eps): 0.0 for eps in head_eps_list}
                    frozen_share = frozen_gini = frozen_persist = 0.0

                    if vals_pos_all:
                        thr_hi = percentile_threshold(vals_pos_all, q_hi)
                        thr_lo = percentile_threshold(vals_pos_all, q_lo)

                        # Components at hi
                        adj_hi = threshold_unweighted_adj_from_weighted(N, edge_load, thr_hi)
                        comps_hi = connected_components(adj_hi)
                        if comps_hi:
                            # Choose largest component as "the" core at this q_hi/budget
                            comps_hi_sorted = sorted(comps_hi, key=lambda x: len(x), reverse=True)
                            core_nodes_current = set(comps_hi_sorted[0])
                        else:
                            core_nodes_current = set()

                        cid_hi = component_map(comps_hi, N)

                        core_weights = []
                        boundary_weights = []
                        outside_weights = []

                        for (i, j), w in edge_load.items():
                            same_comp = (cid_hi[i] == cid_hi[j])
                            if w >= thr_hi and same_comp:
                                core_weights.append(w)
                            elif w >= thr_lo and not same_comp:
                                boundary_weights.append(w)
                            if w > 0 and not (w >= thr_hi and same_comp):
                                outside_weights.append(w)

                        g_core = gini(core_weights) if core_weights else 0.0
                        g_out  = gini(outside_weights) if outside_weights else 0.0
                        ratio  = (g_core / g_global) if g_global > 0 else 0.0

                        cshare = (sum(core_weights) / total_w_all) if total_w_all > 0 else 0.0
                        bshare = (sum(boundary_weights) / total_w_all) if total_w_all > 0 else 0.0

                        lo_weights = {(i, j): w for (i, j), w in edge_load.items() if w >= thr_lo}
                        conds = []
                        if comps_hi:
                            for nodes in comps_hi:
                                conds.append(conductance_for_component(nodes, lo_weights))
                        cond_mean = (sum(conds)/len(conds)) if conds else 0.0

                        # Head mass metrics
                        for eps in head_eps_list:
                            hm_global[str(eps)] = head_mass(edge_vals_all, eps)
                            hm_core[str(eps)]   = head_mass(core_weights, eps)
                            hm_out[str(eps)]    = head_mass(outside_weights, eps)

                        # (c) Core-restricted cross-lens Jaccard
                        #   A = top-q_hi fraction (by count) of edges by weight within core nodes
                        #   B = edges >= thr_hi whose endpoints both in core nodes
                        if core_nodes_current:
                            A = top_q_edge_set_within_nodes(edge_load, core_nodes_current, q_hi)
                            B = set()
                            for (i, j), w in edge_load.items():
                                if i in core_nodes_current and j in core_nodes_current and w >= thr_hi:
                                    B.add((i, j))
                            inter = len(A & B)
                            union = len(A | B) if (A or B) else 1
                            jacc_core = inter / union

                        # (a) Frozen-core tracking: pick once when budget==frozen_pick_budget and q_hi is first in sweep
                        if not frozen_picked and budgets[cut_idx] >= frozen_pick_budget and q_hi == q_hi_sweep[0]:
                            # Freeze the largest hi-component at this moment
                            frozen_nodes = set(core_nodes_current)
                            # The initial frozen-core edges = those >= thr_hi within frozen_nodes now
                            for (i, j), w in edge_load.items():
                                if i in frozen_nodes and j in frozen_nodes and w >= thr_hi:
                                    frozen_edges_initial.add((i, j))
                            frozen_picked = True

                        # If frozen picked, compute its tracking stats (independent of current q_hi lens)
                        if frozen_picked and frozen_nodes:
                            # Weight share on edges among frozen_nodes (all positive edges among them)
                            frozen_edge_weights = [w for (i, j), w in edge_load.items()
                                                   if i in frozen_nodes and j in frozen_nodes and w > 0]
                            tot_w_frozen = sum(frozen_edge_weights)
                            frozen_share = (tot_w_frozen / total_w_all) if total_w_all > 0 else 0.0
                            frozen_gini  = gini(frozen_edge_weights) if frozen_edge_weights else 0.0

                            # Persistence: fraction of initial frozen-core edges that remain >= current thr_hi
                            alive = 0
                            for (i, j) in frozen_edges_initial:
                                if edge_load.get((i, j), 0) >= thr_hi:
                                    alive += 1
                            frozen_persist = (alive / len(frozen_edges_initial)) if frozen_edges_initial else 0.0

                    # Record per-q series
                    srec = per_q_seed[q_hi]
                    srec["global_gini"].append(g_global)
                    srec["core_gini"].append(g_core)
                    srec["outside_gini"].append(g_out)
                    srec["core_over_global_gini"].append(ratio)
                    srec["core_share"].append(cshare)
                    srec["boundary_load_share"].append(bshare)
                    srec["boundary_conductance_mean"].append(cond_mean)
                    srec["core_jaccard_edgeint_vs_comembership"].append(jacc_core)
                    for eps in head_eps_list:
                        srec["head_mass_global"][str(eps)].append(hm_global[str(eps)])
                        srec["head_mass_core"][str(eps)].append(hm_core[str(eps)])
                        srec["head_mass_out"][str(eps)].append(hm_out[str(eps)])
                    srec["frozen_core_weight_share"].append(frozen_share)
                    srec["frozen_core_gini"].append(frozen_gini)
                    srec["frozen_core_edge_persistence"].append(frozen_persist)

                cut_idx += 1  # advance to next budget

        # Store per seed for each q
        for q_hi in q_hi_sweep:
            per_q[q_hi].append({
                "seed": s,
                "budgets": budgets,
                **per_q_seed[q_hi]
            })

    # Aggregate across seeds for each q
    aggregate = {}
    for q_hi in q_hi_sweep:
        arrs = per_q[q_hi]
        if not arrs:
            continue
        def agg(get_key):
            return agg_series(arrs, lambda e: e[get_key], K)

        mean_g_global, std_g_global = agg("global_gini")
        mean_g_core,   std_g_core   = agg("core_gini")
        mean_g_out,    std_g_out    = agg("outside_gini")
        mean_ratio,    std_ratio    = agg("core_over_global_gini")
        mean_cshare,   std_cshare   = agg("core_share")
        mean_bshare,   std_bshare   = agg("boundary_load_share")
        mean_cond,     std_cond     = agg("boundary_conductance_mean")
        mean_jacc,     std_jacc     = agg("core_jaccard_edgeint_vs_comembership")
        mean_fshare,   std_fshare   = agg("frozen_core_weight_share")
        mean_fgini,    std_fgini    = agg("frozen_core_gini")
        mean_fpersist, std_fpersist = agg("frozen_core_edge_persistence")

        # Head-mass per eps
        hm_global = {}; hm_core = {}; hm_out = {}
        for eps in head_eps_list:
            key = str(eps)
            hm_global[key], _ = agg_series(arrs, lambda e: e["head_mass_global"][key], K)
            hm_core[key],   _ = agg_series(arrs, lambda e: e["head_mass_core"][key],   K)
            hm_out[key],    _ = agg_series(arrs, lambda e: e["head_mass_out"][key],    K)

        aggregate[str(q_hi)] = {
            "global_gini": {"mean": mean_g_global, "std": std_g_global},
            "core_gini":   {"mean": mean_g_core,   "std": std_g_core},
            "outside_gini":{"mean": mean_g_out,    "std": std_g_out},
            "core_over_global_gini": {"mean": mean_ratio, "std": std_ratio},
            "core_share": {"mean": mean_cshare, "std": std_cshare},
            "boundary_load_share": {"mean": mean_bshare, "std": std_bshare},
            "boundary_conductance_mean": {"mean": mean_cond, "std": std_cond},
            "core_jaccard_edgeint_vs_comembership": {"mean": mean_jacc, "std": std_jacc},
            "frozen_core_weight_share": {"mean": mean_fshare, "std": std_fshare},
            "frozen_core_gini": {"mean": mean_fgini, "std": std_fgini},
            "frozen_core_edge_persistence": {"mean": mean_fpersist, "std": std_fpersist},
            "head_mass_global": hm_global,
            "head_mass_core":   hm_core,
            "head_mass_out":    hm_out
        }

    results = {
        "meta": {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config_path": str(cfg_path),
            "script": str(script_path)
        },
        "sim": {
            "N": N,
            "max_steps_per_walker": max_steps,
            "max_loops_per_walk": max_loops_per_walk,
            "seeds": seeds,
            "total_steps": total_steps
        },
        "budgets": budgets,
        "observer": {
            "q_sweep_hi": q_hi_sweep,
            "boundary_percentile_lo": q_lo,
            "frozen_core_pick_budget": frozen_pick_budget,
            "head_mass_epsilons": head_eps_list
        },
        "per_q_per_seed": per_q,     # full per-seed series (can be large)
        "aggregate": aggregate       # means/stds per q_hi
    }

    print(json.dumps(results, indent=2))

    # Append a compact RUN block per q_hi
    results_path = results_filename_from_script(script_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    hdr = f"=== RUN {results['meta']['timestamp_utc']} | N={N} | budgets={budgets} | seeds={seeds} | q_sweep_hi={q_hi_sweep} | q_lo={q_lo} ==="
    # For the brief, show just a few key series for each q
    brief = {"budgets": budgets, "q": {}}
    for q_hi in q_hi_sweep:
        aq = results["aggregate"][str(q_hi)]
        brief["q"][str(q_hi)] = {
            "global_gini_mean": aq["global_gini"]["mean"],
            "core_over_global_gini_mean": aq["core_over_global_gini"]["mean"],
            "core_jaccard_core_mean": aq["core_jaccard_edgeint_vs_comembership"]["mean"],
            "frozen_core_weight_share_mean": aq["frozen_core_weight_share"]["mean"],
            "frozen_core_edge_persistence_mean": aq["frozen_core_edge_persistence"]["mean"]
        }
    block = [hdr, "RESULTS:", json.dumps(brief, indent=2), ""]
    with results_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(block) + "\n")

if __name__ == "__main__":
    main()
