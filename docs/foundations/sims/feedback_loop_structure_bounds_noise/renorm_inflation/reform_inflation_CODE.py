#!/usr/bin/env python3
# EFD — FOUNDATIONS — SIM S07 (Observer-only Renormalization: contraction of hi-q components)
# Neutral dynamics on K_N. No ad-hoc rules. We only change DESCRIPTION:
#   Level 0: edge_load on base nodes from loop reuse.
#   Build hi-q components -> contract each component into a super-node.
#   Aggregate inter-component edge weights -> super-edge_load.
#   Repeat for L levels; at each level compute graph stats on the hi-q supergraph.
#
# I/O:
# - Reads only /config/defaults.yaml (searched upward from this script).
# - Prints full JSON to stdout.
# - Appends a compact RUN block to a file named like this script with "_RESUTLS.txt".
#
# Metrics per level (for each budget, each seed):
#   - num_nodes, num_edges, total_weight
#   - gini(super_edge_weights)
#   - conductance_mean (on hi-q components measured with lo-threshold)
#   - mean_shortest_path, diameter (on unweighted super-adj >= thr_hi)
#   - core_persistence: fraction of level-0 largest-core nodes still within the mapped
#     largest-core at this level (via contraction map)

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
        fname = name.replace("_CODE.py", "_RESUTLS.txt")  # keep user's spelling
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
# Graph helpers
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

def threshold_unweighted_adj_from_weighted(num_nodes, weights, thr, node_index_map=None):
    # weights keyed by tuple of node ids (consistent with index_map mapping)
    adj = [[] for _ in range(num_nodes)]
    for (i, j), w in weights.items():
        if w >= thr:
            a = i if node_index_map is None else node_index_map[i]
            b = j if node_index_map is None else node_index_map[j]
            if a == b:
                continue
            adj[a].append(b)
            adj[b].append(a)
    return adj

def connected_components(adj):
    N = len(adj)
    seen = [False]*N
    comps = []
    for s in range(N):
        if seen[s]:
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

def mean_shortest_path_and_diameter(adj):
    # On unweighted graph (adj). If disconnected, compute on giant component only.
    N = len(adj)
    if N == 0:
        return 0.0, 0
    # find giant component
    comps = connected_components(adj)
    giant = max(comps, key=len) if comps else []
    if len(giant) <= 1:
        return 0.0, 0
    index = {node: idx for idx, node in enumerate(giant)}
    M = len(giant)
    # BFS from each node in giant (O(M*(M+E))) — fine for small super-graphs
    dsum = 0
    pairs = 0
    dia = 0
    for s in giant:
        dist = [-1]*N
        dist[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        # accumulate within giant only
        for t in giant:
            if t == s: continue
            d = dist[t]
            if d >= 0:
                dsum += d
                pairs += 1
                if d > dia:
                    dia = d
    mean_sp = (dsum / pairs) if pairs > 0 else 0.0
    return mean_sp, dia

# ---------------------------
# Renormalization (observer-only contraction)
# ---------------------------
def build_hi_components_and_thresholds(edge_load, q_hi, q_lo):
    vals_pos = [w for w in edge_load.values() if w > 0]
    if not vals_pos:
        return [], {}, 0, 0
    thr_hi = percentile_threshold(vals_pos, q_hi)
    thr_lo = percentile_threshold(vals_pos, q_lo)
    # Build adjacency on original node ids for hi threshold
    # Need N; infer from edge keys
    max_node = 0
    for (i, j) in edge_load.keys():
        if i > max_node: max_node = i
        if j > max_node: max_node = j
    N = max_node + 1
    adj_hi = threshold_unweighted_adj_from_weighted(N, edge_load, thr_hi, node_index_map=None)
    comps_hi = connected_components(adj_hi)
    return comps_hi, {"thr_hi": thr_hi, "thr_lo": thr_lo}, N, adj_hi

def contract_components(edge_load, comps_hi):
    """
    Contract each component into a super-node. Return:
      - super_edge_load: dict[(A,B)] -> weight (sum of inter-component weights)
      - node_map: original node -> super-node id
      - num_super_nodes
    """
    # Map original node -> component id
    max_node = 0
    for (i, j) in edge_load.keys():
        if i > max_node: max_node = i
        if j > max_node: max_node = j
    N = max_node + 1
    cid = [-1]*N
    for k, comp in enumerate(comps_hi):
        for u in comp:
            cid[u] = k
    # It's possible some isolated nodes didn't appear in comps_hi (if no hi-edges).
    # Assign them as singleton components to preserve mapping.
    for u in range(N):
        if cid[u] == -1:
            cid[u] = len(comps_hi)
            comps_hi.append([u])

    M = len(comps_hi)
    super_edge_load = defaultdict(int)
    for (i, j), w in edge_load.items():
        a = cid[i]; b = cid[j]
        if a == b:
            continue  # intra-component becomes self-loop; ignore for super-graph edges
        A, B = (a, b) if a < b else (b, a)
        super_edge_load[(A, B)] += w
    return super_edge_load, cid, M

def measure_level_stats(edge_load, comps_hi, thr_hi, thr_lo):
    # Build adjacency at hi on current node index space
    # Infer node count
    max_node = 0
    for (i, j) in edge_load.keys():
        if i > max_node: max_node = i
        if j > max_node: max_node = j
    N = max_node + 1
    adj_hi = threshold_unweighted_adj_from_weighted(N, edge_load, thr_hi)
    # basic counts
    num_nodes = N
    num_edges = len(edge_load)
    total_w = sum(edge_load.values())
    # gini on super-edge weights
    g = gini(list(edge_load.values()))
    # conductance (mean over hi components, flow measured with lo-threshold)
    lo_weights = {(i, j): w for (i, j), w in edge_load.items() if w >= thr_lo}
    conds = []
    for nodes in comps_hi:
        conds.append(conductance_for_component(nodes, lo_weights))
    conductance_mean = (sum(conds)/len(conds)) if conds else 0.0
    # graph distances on hi-adj
    msp, dia = mean_shortest_path_and_diameter(adj_hi)
    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "total_weight": total_w,
        "edge_gini": g,
        "conductance_mean": conductance_mean,
        "mean_shortest_path": msp,
        "diameter": dia
    }

# ---------------------------
# Aggregation helpers
# ---------------------------
def mean_std(arr):
    if not arr:
        return 0.0, 0.0
    mu = sum(arr)/len(arr)
    var = sum((x-mu)**2 for x in arr)/(len(arr)-1 if len(arr)>1 else 1)
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
    budgets   = obs_cfg.get("gini_budgets_walkers", [2000, 5000, 10000, 20000])
    budgets   = sorted(int(x) for x in budgets if int(x) > 0)
    q_hi      = float(obs_cfg.get("component_percentile_hi", 0.995))
    q_lo      = float(obs_cfg.get("boundary_percentile_lo", 0.985))

    ren_cfg   = obs_cfg.get("renorm", {})
    levels    = int(ren_cfg.get("levels", 4))            # number of contraction levels (including level 0 as original)
    assert levels >= 1
    seeds     = obs_cfg.get("gini_seeds", None)
    if seeds:
        seeds = [int(s) for s in seeds]
    else:
        repeats = int(obs_cfg.get("gini_repeats", 3))
        seeds = [base_seed + k for k in range(repeats)]

    total_steps = 0
    per_seed = []

    for s in seeds:
        rng = random.Random(s)
        node_load = [0]*N
        edge_load = defaultdict(int)

        walkers_done = 0
        budget_idx = 0

        # For core persistence across levels: capture level-0 largest-core nodes at each budget
        lvl0_core_nodes_by_budget = {}

        # Storage per budget: list of level dicts
        per_budget_levels = []

        while budget_idx < len(budgets):
            # advance walkers until reaching this budget
            while walkers_done < budgets[budget_idx]:
                loops, steps = walk_detect_loops(N, max_steps, rng, max_loops_per_walk)
                total_steps += steps
                accumulate_loops_into_loads(N, loops, node_load, edge_load)
                walkers_done += 1

            # ----- Level 0 (original graph) hi-components & thresholds
            comps0, thr, N0, adj0_hi = build_hi_components_and_thresholds(edge_load, q_hi, q_lo)
            thr_hi = thr["thr_hi"]; thr_lo = thr["thr_lo"]
            # largest core at level 0
            comps0_sorted = sorted(comps0, key=len, reverse=True) if comps0 else []
            lvl0_core_nodes = set(comps0_sorted[0]) if comps0_sorted else set()
            lvl0_core_nodes_by_budget[budgets[budget_idx]] = lvl0_core_nodes

            # measure level 0 stats
            lvl_stats = []
            stats0 = measure_level_stats(edge_load, comps0, thr_hi, thr_lo)
            lvl_stats.append({"level": 0, **stats0})

            # ----- Contract iteratively
            current_edge_load = edge_load
            current_comps_hi  = comps0
            # map from original nodes to current super-nodes (update progressively)
            # Start with identity map
            max_node = 0
            for (i, j) in current_edge_load.keys():
                if i > max_node: max_node = i
                if j > max_node: max_node = j
            node_to_super = list(range(max_node+1))

            for lev in range(1, levels):
                super_edge_load, cid_map, M = contract_components(current_edge_load, current_comps_hi)
                # update mapping to super nodes
                for u, comp_id in enumerate(cid_map):
                    # node u (original space at this stage) maps to comp_id
                    node_to_super[u] = comp_id

                # Build hi-components at this super-level
                compsL, thrL, ML, adjL_hi = build_hi_components_and_thresholds(super_edge_load, q_hi, q_lo)
                thr_hi_L = thrL["thr_hi"]; thr_lo_L = thrL["thr_lo"]

                # measure stats at this level
                statsL = measure_level_stats(super_edge_load, compsL, thr_hi_L, thr_lo_L)
                lvl_stats.append({"level": lev, **statsL})

                # prepare next iteration
                current_edge_load = super_edge_load
                current_comps_hi  = compsL

            # ----- Core persistence across levels
            # For each level, estimate fraction of level-0 core nodes that land inside the level's largest core after mapping.
            core_persistence = []
            # Rebuild the contraction chain to compute mapping of original nodes to each level
            # (Small overhead; keeps clarity.)
            current_edge_load = edge_load
            current_comps_hi  = comps0
            # mapping from original nodes to current level super-nodes
            max_node = 0
            for (i, j) in current_edge_load.keys():
                if i > max_node: max_node = i
                if j > max_node: max_node = j
            map_orig_to_level = list(range(max_node+1))  # level 0: identity

            # level 0 persistence: trivially 1.0 if non-empty (we'll still compute properly)
            if lvl0_core_nodes:
                # build largest core id at level 0
                if comps0:
                    comps0_sorted = sorted(comps0, key=len, reverse=True)
                    core0_id = None
                    # find which comp equals lvl0_core_nodes
                    comp_sets = [set(c) for c in comps0_sorted]
                    for idx, nodeset in enumerate(comp_sets):
                        if nodeset == lvl0_core_nodes:
                            core0_id = idx  # index in sorted order; we don't need id beyond set itself
                            break
                # at level 0, persistence = 1.0
                core_persistence.append(1.0)
            else:
                core_persistence.append(0.0)

            # march levels and compute persistence
            # we need at each level the largest core node set (in that level's node labels),
            # and a mapping of original nodes -> that level's node labels to test membership.
            # We'll reconstruct components and cid maps again (cheap at super-level).
            current_edge_load = edge_load
            comps_hi = comps0

            # Build mapping from original -> level labels progressively
            # Start: original label space
            for lev in range(1, levels):
                # Contract this level
                super_edge_load, cid_map, M = contract_components(current_edge_load, comps_hi)
                # Map original->new level: apply cid_map to previous label space
                map_orig_to_level = [cid_map[u] if u < len(cid_map) else 0 for u in map_orig_to_level]
                # Build hi comps on super graph
                comps_hi, thrL, ML, adjL_hi = build_hi_components_and_thresholds(super_edge_load, q_hi, q_lo)
                if comps_hi:
                    comps_sorted = sorted(comps_hi, key=len, reverse=True)
                    largest_core_nodes = set(comps_sorted[0])
                    # Count how many original lvl0_core_nodes map inside largest_core_nodes at this level
                    count_in = 0
                    for u in lvl0_core_nodes:
                        lab = map_orig_to_level[u]
                        if lab in largest_core_nodes:
                            count_in += 1
                    frac = count_in / len(lvl0_core_nodes) if lvl0_core_nodes else 0.0
                else:
                    frac = 0.0
                core_persistence.append(frac)
                current_edge_load = super_edge_load

            per_budget_levels.append({
                "budget": budgets[budget_idx],
                "levels": lvl_stats,
                "core_persistence": core_persistence
            })

            budget_idx += 1

        per_seed.append({
            "seed": s,
            "budgets": budgets,
            "per_budget_levels": per_budget_levels
        })

    # -------- Aggregate across seeds
    # For each budget and level, aggregate stats.
    budgets_list = budgets
    if not per_seed:
        print("{}", flush=True); return

    # Initialize aggregate containers
    # dict[budget][level][metric] -> list of values to aggregate
    agg = {b: {lv: {
        "num_nodes": [], "num_edges": [], "total_weight": [],
        "edge_gini": [], "conductance_mean": [],
        "mean_shortest_path": [], "diameter": [],
        "core_persistence": []  # special: same size as levels
    } for lv in range(len(per_seed[0]["per_budget_levels"][0]["levels"]))} for b in budgets_list}

    for entry in per_seed:
        for bpack in entry["per_budget_levels"]:
            b = bpack["budget"]
            levels_list = bpack["levels"]
            cpers = bpack["core_persistence"]
            for lvl_obj in levels_list:
                lv = lvl_obj["level"]
                agg[b][lv]["num_nodes"].append(lvl_obj["num_nodes"])
                agg[b][lv]["num_edges"].append(lvl_obj["num_edges"])
                agg[b][lv]["total_weight"].append(lvl_obj["total_weight"])
                agg[b][lv]["edge_gini"].append(lvl_obj["edge_gini"])
                agg[b][lv]["conductance_mean"].append(lvl_obj["conductance_mean"])
                agg[b][lv]["mean_shortest_path"].append(lvl_obj["mean_shortest_path"])
                agg[b][lv]["diameter"].append(lvl_obj["diameter"])
            # core persistence aligned by level index
            for lv_idx, frac in enumerate(cpers):
                agg[b][lv_idx]["core_persistence"].append(frac)

    aggregate = {b: {lv: {
        "num_nodes":          {"mean": mean_std(agg[b][lv]["num_nodes"])[0],          "std": mean_std(agg[b][lv]["num_nodes"])[1]},
        "num_edges":          {"mean": mean_std(agg[b][lv]["num_edges"])[0],          "std": mean_std(agg[b][lv]["num_edges"])[1]},
        "total_weight":       {"mean": mean_std(agg[b][lv]["total_weight"])[0],       "std": mean_std(agg[b][lv]["total_weight"])[1]},
        "edge_gini":          {"mean": mean_std(agg[b][lv]["edge_gini"])[0],          "std": mean_std(agg[b][lv]["edge_gini"])[1]},
        "conductance_mean":   {"mean": mean_std(agg[b][lv]["conductance_mean"])[0],   "std": mean_std(agg[b][lv]["conductance_mean"])[1]},
        "mean_shortest_path": {"mean": mean_std(agg[b][lv]["mean_shortest_path"])[0], "std": mean_std(agg[b][lv]["mean_shortest_path"])[1]},
        "diameter":           {"mean": mean_std(agg[b][lv]["diameter"])[0],           "std": mean_std(agg[b][lv]["diameter"])[1]},
        "core_persistence":   {"mean": mean_std(agg[b][lv]["core_persistence"])[0],   "std": mean_std(agg[b][lv]["core_persistence"])[1]}
    } for lv in agg[b]} for b in agg}

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
        "observer": {
            "component_percentile_hi": q_hi,
            "boundary_percentile_lo": q_lo,
            "renorm_levels": levels
        },
        "budgets": budgets_list,
        "per_seed": per_seed,
        "aggregate": aggregate
    }

    print(json.dumps(results, indent=2))

    # Append compact RUN block
    results_path = results_filename_from_script(script_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    hdr = (f"=== RUN {results['meta']['timestamp_utc']} | "
           f"N={N} | budgets={budgets_list} | seeds={seeds} | "
           f"q_hi={q_hi} | q_lo={q_lo} | levels={levels} ===")
    # Brief: for each budget, print per-level mean_shortest_path, diameter, gini, core_persistence
    brief = {"budgets": {}}
    for b in budgets_list:
        brief["budgets"][str(b)] = {
            "mean_shortest_path": [results["aggregate"][b][lv]["mean_shortest_path"]["mean"] for lv in sorted(results["aggregate"][b].keys())],
            "diameter":           [results["aggregate"][b][lv]["diameter"]["mean"]           for lv in sorted(results["aggregate"][b].keys())],
            "edge_gini":          [results["aggregate"][b][lv]["edge_gini"]["mean"]          for lv in sorted(results["aggregate"][b].keys())],
            "core_persistence":   [results["aggregate"][b][lv]["core_persistence"]["mean"]   for lv in sorted(results["aggregate"][b].keys())]
        }
    block = [hdr, "RESULTS:", json.dumps(brief, indent=2), ""]
    with results_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(block) + "\n")

if __name__ == "__main__":
    main()
