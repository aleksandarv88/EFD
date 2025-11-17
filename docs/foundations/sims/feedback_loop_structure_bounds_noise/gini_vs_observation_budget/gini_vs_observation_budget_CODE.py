#!/usr/bin/env python3
# EFD — FOUNDATIONS — SIM S05 (Core-conditioned Gini vs Observation Budget, extended)
# Neutral dynamics; we DO NOT change step probabilities.
# We "use noise as given" by conditioning measurements on the boundary-defined core:
#   - At each observation budget cut, we:
#       1) build edge-intensity weights from all loops so far
#       2) form high-percentile (q_hi) components (structure)
#       3) CORE edges = edges >= thr(q_hi) with endpoints in SAME hi-component
#       4) BOUNDARY edges (two-tier) = edges >= thr(q_lo) with endpoints in DIFFERENT hi-components
#   - Report (per budget, per seed + aggregates):
#       - global_gini, core_gini, outside_gini
#       - core/global gini ratio
#       - head mass (top ε% edges) for ε in YAML
#       - core share (weight in core / total weight)
#       - boundary load share & mean conductance (context)
#
# IO:
# - Reads config ONLY from /config/defaults.yaml (searched upward)
# - Prints pretty JSON to stdout
# - Appends RUN block to a file named like this script, replacing "_CODE.py" with "_RESUTLS.txt"

import json, time, random, math
from pathlib import Path
from collections import defaultdict, deque

# ---------------------------
# Config helpers
# ---------------------------
def load_yaml(p: Path) -> dict:
    try:
        import yaml
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
        fname = name.replace("_CODE.py", "_RESUTLS.txt")  # (your spelling)
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
    """Fraction of total weight in top eps% of edges (eps in (0,1])."""
    xs = sorted([x for x in edge_weights_list if x > 0], reverse=True)
    if not xs:
        return 0.0
    k = max(1, int(math.ceil(eps * len(xs))))
    top_sum = sum(xs[:k])
    tot = sum(xs)
    return top_sum / tot if tot > 0 else 0.0

def mean_std_at_index(arrs, i):
    vals = [arr[i] for arr in arrs]
    mu = sum(vals) / len(vals)
    var = sum((x - mu)**2 for x in vals) / (len(vals)-1 if len(vals) > 1 else 1)
    return mu, math.sqrt(var)

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
    q_hi   = float(obs_cfg.get("component_percentile_hi", 0.995))
    q_lo   = float(obs_cfg.get("boundary_percentile_lo", 0.985))
    head_eps_list = obs_cfg.get("head_mass_epsilons", [0.01, 0.05])  # top 1%, 5% by default
    head_eps_list = [float(e) for e in head_eps_list if 0 < float(e) <= 1.0]

    seeds  = obs_cfg.get("gini_seeds", None)
    if seeds:
        seeds = [int(s) for s in seeds]
    else:
        repeats = int(obs_cfg.get("gini_repeats", 3))
        seeds = [base_seed + k for k in range(repeats)]

    max_budget = max(budgets)
    total_steps = 0

    per_seed = []  # each entry: dict of arrays per budget
    for s in seeds:
        rng = random.Random(s)
        node_load = [0]*N
        edge_load = defaultdict(int)

        global_ginis, core_ginis, outside_ginis = [], [], []
        ratio_core_global = []
        core_shares = []
        bshare_means, cond_means = [], []

        headmass_global = {eps: [] for eps in head_eps_list}
        headmass_core   = {eps: [] for eps in head_eps_list}
        headmass_out    = {eps: [] for eps in head_eps_list}

        cut_idx = 0
        walkers_done = 0

        while walkers_done < max_budget:
            loops, steps = walk_detect_loops(N, max_steps, rng, max_loops_per_walk)
            total_steps += steps
            accumulate_loops_into_loads(N, loops, node_load, edge_load)
            walkers_done += 1

            while cut_idx < len(budgets) and walkers_done >= budgets[cut_idx]:
                edge_vals = list(edge_load.values())
                g_global = gini(edge_vals) if edge_vals else 0.0

                vals_pos = [w for w in edge_load.values() if w > 0]
                if vals_pos:
                    thr_hi = percentile_threshold(vals_pos, q_hi)
                    thr_lo = percentile_threshold(vals_pos, q_lo)

                    # components at hi
                    adj_hi = threshold_unweighted_adj_from_weighted(N, edge_load, thr_hi)
                    comps_hi = connected_components(adj_hi)
                    cid_hi = component_map(comps_hi, N)

                    core_weights = []
                    boundary_weights = []
                    outside_weights = []

                    # classify edges
                    for (i, j), w in edge_load.items():
                        same_comp = (cid_hi[i] == cid_hi[j])
                        if w >= thr_hi and same_comp:
                            core_weights.append(w)
                        elif w >= thr_lo and not same_comp:
                            boundary_weights.append(w)
                        # outside = everything not in core set (but positive weight); define as:
                        if w > 0 and not (w >= thr_hi and same_comp):
                            outside_weights.append(w)

                    g_core = gini(core_weights) if core_weights else 0.0
                    g_out  = gini(outside_weights) if outside_weights else 0.0
                    ratio  = (g_core / g_global) if g_global > 0 else 0.0

                    total_w = sum(edge_vals) if edge_vals else 0.0
                    core_w  = sum(core_weights) if core_weights else 0.0
                    bshare  = (sum(boundary_weights) / total_w) if total_w > 0 else 0.0
                    cshare  = (core_w / total_w) if total_w > 0 else 0.0

                    lo_weights = {(i, j): w for (i, j), w in edge_load.items() if w >= thr_lo}
                    conds = [conductance_for_component(nodes, lo_weights) for nodes in comps_hi] if comps_hi else []
                    cond_mean = (sum(conds)/len(conds)) if conds else 0.0

                    # head mass metrics
                    for eps in head_eps_list:
                        hm_g = head_mass(edge_vals, eps)
                        hm_c = head_mass(core_weights, eps)
                        hm_o = head_mass(outside_weights, eps)
                        headmass_global[eps].append(hm_g)
                        headmass_core[eps].append(hm_c)
                        headmass_out[eps].append(hm_o)
                else:
                    g_core = g_out = ratio = bshare = cshare = cond_mean = 0.0
                    for eps in head_eps_list:
                        headmass_global[eps].append(0.0)
                        headmass_core[eps].append(0.0)
                        headmass_out[eps].append(0.0)

                global_ginis.append(g_global)
                core_ginis.append(g_core)
                outside_ginis.append(g_out)
                ratio_core_global.append(ratio)
                core_shares.append(cshare)
                bshare_means.append(bshare)
                cond_means.append(cond_mean)

                cut_idx += 1

        per_seed.append({
            "seed": s,
            "budgets": budgets,
            "global_gini": global_ginis,
            "core_gini": core_ginis,
            "outside_gini": outside_ginis,
            "core_over_global_gini": ratio_core_global,
            "core_share": core_shares,
            "boundary_load_share": bshare_means,
            "boundary_conductance_mean": cond_means,
            "head_mass_global": {str(eps): headmass_global[eps] for eps in head_eps_list},
            "head_mass_core":   {str(eps): headmass_core[eps]   for eps in head_eps_list},
            "head_mass_out":    {str(eps): headmass_out[eps]    for eps in head_eps_list},
        })

    # aggregate across seeds
    K = len(budgets)
    def agg_series(getter):
        means, stds = [], []
        for i in range(K):
            mu, sd = mean_std_at_index([getter(e) for e in per_seed], i)
            means.append(mu); stds.append(sd)
        return means, stds

    mean_g_global, std_g_global = agg_series(lambda e: e["global_gini"])
    mean_g_core,   std_g_core   = agg_series(lambda e: e["core_gini"])
    mean_g_out,    std_g_out    = agg_series(lambda e: e["outside_gini"])
    mean_ratio,    std_ratio    = agg_series(lambda e: e["core_over_global_gini"])
    mean_cshare,   std_cshare   = agg_series(lambda e: e["core_share"])
    mean_bshare,   std_bshare   = agg_series(lambda e: e["boundary_load_share"])
    mean_cond,     std_cond     = agg_series(lambda e: e["boundary_conductance_mean"])

    hm_global = {}
    hm_core   = {}
    hm_out    = {}
    for eps in head_eps_list:
        k = str(eps)
        hm_global[k], _ = agg_series(lambda e: e["head_mass_global"][k])
        hm_core[k],   _ = agg_series(lambda e: e["head_mass_core"][k])
        hm_out[k],    _ = agg_series(lambda e: e["head_mass_out"][k])

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
            "component_percentile_hi": q_hi,
            "boundary_percentile_lo": q_lo,
            "head_mass_epsilons": head_eps_list
        },
        "per_seed": per_seed,
        "aggregate": {
            "global_gini": {"mean": mean_g_global, "std": std_g_global},
            "core_gini":   {"mean": mean_g_core,   "std": std_g_core},
            "outside_gini":{"mean": mean_g_out,    "std": std_g_out},
            "core_over_global_gini": {"mean": mean_ratio, "std": std_ratio},
            "core_share": {"mean": mean_cshare, "std": std_cshare},
            "boundary_load_share": {"mean": mean_bshare, "std": std_bshare},
            "boundary_conductance_mean": {"mean": mean_cond, "std": std_cond},
            "head_mass_global": hm_global,
            "head_mass_core":   hm_core,
            "head_mass_out":    hm_out
        }
    }

    print(json.dumps(results, indent=2))

    # Append compact RUN block
    results_path = results_filename_from_script(script_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    run_header = (
        f"=== RUN {results['meta']['timestamp_utc']} | "
        f"N={N} | budgets={budgets} | seeds={seeds} | q_hi={q_hi} | q_lo={q_lo} ==="
    )
    brief = {
        "budgets": budgets,
        "global_gini_mean": results["aggregate"]["global_gini"]["mean"],
        "core_gini_mean":   results["aggregate"]["core_gini"]["mean"],
        "outside_gini_mean":results["aggregate"]["outside_gini"]["mean"],
        "core_over_global_gini_mean": results["aggregate"]["core_over_global_gini"]["mean"],
        "core_share_mean": results["aggregate"]["core_share"]["mean"],
        "boundary_load_share_mean": results["aggregate"]["boundary_load_share"]["mean"],
        "boundary_conductance_mean": results["aggregate"]["boundary_conductance_mean"]["mean"],
        "head_mass_global": results["aggregate"]["head_mass_global"],
        "head_mass_core":   results["aggregate"]["head_mass_core"],
        "head_mass_out":    results["aggregate"]["head_mass_out"],
    }
    block = [run_header, "RESULTS:", json.dumps(brief, indent=2), ""]
    with results_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(block) + "\n")

if __name__ == "__main__":
    main()
