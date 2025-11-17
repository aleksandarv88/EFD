#!/usr/bin/env python3
# EFD — FOUNDATIONS — SIM S01 (theory-check)
# Pure substrate: unbiased random walks on a complete graph (no self-loops).
# Observers:
#   (A) First-loop hitting time distribution (steps until first repeat)
#   (B) Recurrence rate per walker (repeat-steps / total-steps)
# Adds THEORY baseline + observed/expected ratios.
#
# Behavior:
# - Reads config ONLY from /config/defaults.yaml (searched upward).
# - Prints pretty JSON to stdout.
# - Appends standardized RUN block to a results file whose name matches this script,
#   with "_CODE.py" replaced by "_RESUTLS.txt" (per your naming).

import json, time, random, math
from pathlib import Path
from collections import Counter

# ---------------------------
# Config loading
# ---------------------------
def load_yaml(p: Path) -> dict:
    try:
        import yaml  # pip install pyyaml
    except Exception:
        raise SystemExit(
            f"PyYAML is required to read {p}.\nInstall with:  pip install pyyaml"
        )
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}

def find_defaults_yaml(start: Path) -> Path:
    """
    Walk upward from 'start' to locate repo-root /config/defaults.yaml.
    Stop at filesystem root if not found.
    """
    cur = start
    for _ in range(12):
        candidate = cur / "config" / "defaults.yaml"
        if candidate.exists():
            return candidate
        nxt = cur.parent
        if nxt == cur:
            break
        cur = nxt
    raise SystemExit(
        "Could not find /config/defaults.yaml.\nExpected at: <repo_root>/config/defaults.yaml"
    )

def load_config(script_path: Path):
    defaults_path = find_defaults_yaml(script_path.parent)
    cfg = load_yaml(defaults_path)
    if not isinstance(cfg, dict):
        raise SystemExit(f"Config at {defaults_path} must be a mapping/dict.")
    return cfg, defaults_path

# ---------------------------
# Substrate & loop detection
# ---------------------------
def next_node_uniform(cur: int, N: int, rng: random.Random) -> int:
    # sample any node != cur uniformly (complete graph without self-loops)
    r = rng.randrange(N - 1)
    return r if r < cur else r + 1

def normalize_cycle(cyc: list[int]) -> list[int]:
    # drop trailing repeat if present; rotate to lexicographically minimal form
    if len(cyc) >= 2 and cyc[0] == cyc[-1]:
        cyc = cyc[:-1]
    if not cyc:
        return cyc
    rots = [tuple(cyc[i:] + cyc[:i]) for i in range(len(cyc))]
    return list(min(rots))

def walk_detect_loops_and_observers(N: int, max_steps: int, rng: random.Random, max_loops: int):
    """
    Perform a single random walk and:
      - detect up to 'max_loops' feedback loops
      - record steps-to-first-loop (first-loop hitting time)
      - record recurrence rate (repeat steps / total steps)
    We 'rebase' after a loop (keep only the repeated node) to allow multiple disjoint loop finds.
    """
    cur = rng.randrange(N)
    path = [cur]
    first_idx = {cur: 0}

    loops = []
    steps = 0

    steps_to_first_loop = None
    repeat_steps = 0  # transitions that landed on a previously seen node

    while steps < max_steps and len(loops) < max_loops:
        nxt = next_node_uniform(cur, N, rng)
        path.append(nxt)
        steps += 1

        if nxt in first_idx:
            repeat_steps += 1
            if steps_to_first_loop is None:
                steps_to_first_loop = steps  # first loop closed
            i0 = first_idx[nxt]
            cyc = normalize_cycle(path[i0:])
            loops.append(cyc)
            # rebase so we can keep finding loops
            path = [nxt]
            first_idx = {nxt: 0}
        else:
            first_idx[nxt] = len(path) - 1

        cur = nxt

    recurrence_rate = (repeat_steps / steps) if steps > 0 else 0.0
    return loops, steps, steps_to_first_loop, recurrence_rate

# ---------------------------
# Helpers
# ---------------------------
def quantiles(xs, qs=(0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)):
    if not xs:
        return {str(q): None for q in qs}
    ys = sorted(xs)
    out = {}
    for q in qs:
        if q <= 0:
            out[str(q)] = ys[0]
            continue
        if q >= 1:
            out[str(q)] = ys[-1]
            continue
        pos = q * (len(ys) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(ys) - 1)
        frac = pos - lo
        out[str(q)] = ys[lo] * (1 - frac) + ys[hi] * frac
    return out

def mean(xs):
    return (sum(xs) / len(xs)) if xs else 0.0

def results_filename_from_script(script_path: Path) -> Path:
    name = script_path.name
    if name.endswith("_CODE.py"):
        fname = name.replace("_CODE.py", "_RESUTLS.txt")  # spelling per request
    else:
        stem = script_path.stem
        fname = f"{stem}_RESUTLS.txt"
    return script_path.parent / fname

# ---------------------------
# Main
# ---------------------------
def main():
    script_path = Path(__file__).resolve()
    cfg, cfg_path = load_config(script_path)

    sim_cfg = cfg.get("sim", {})
    N                  = int(sim_cfg.get("N", 64))
    walkers            = int(sim_cfg.get("walkers", 200))
    max_steps          = int(sim_cfg.get("max_steps", 4000))
    max_loops_per_walk = int(sim_cfg.get("max_loops_per_walk", 5))
    sample             = int(sim_cfg.get("sample_loops", 8))
    seed               = int(sim_cfg.get("seed", 12345))

    rng = random.Random(seed)

    all_loops = []
    loop_lengths = []
    total_steps = 0

    # Observers
    hitting_times = []     # steps to first loop per walker
    recurrence_rates = []  # repeat-steps/total-steps per walker

    for _ in range(walkers):
        loops, steps, t_first, r_rate = walk_detect_loops_and_observers(
            N=N, max_steps=max_steps, rng=rng, max_loops=max_loops_per_walk
        )
        total_steps += steps
        recurrence_rates.append(r_rate)
        if t_first is not None:
            hitting_times.append(t_first)
        for cyc in loops:
            all_loops.append(cyc)
            loop_lengths.append(len(cyc))

    # Deduplicate canonical cycles
    seen = set()
    uniq = []
    for cyc in all_loops:
        key = tuple(cyc)
        if key not in seen:
            seen.add(key)
            uniq.append(cyc)

    # THEORY baselines
    expected_first_loop = math.sqrt(math.pi * N / 2.0)      # birthday scaling
    expected_recur_scale = math.sqrt(2.0 / (math.pi * N))   # because we rebase after each loop
   # because we rebase after each loop
           # order-of-magnitude guide

    # Observed summary
    first_loop_mean = mean(hitting_times)
    recur_mean = mean(recurrence_rates)

    ratios_block = {
        "first_loop_mean_over_expected": (
            (first_loop_mean / expected_first_loop) if expected_first_loop > 0 else None
        ),
        "recurrence_mean_over_expected_scale": (
            (recur_mean / expected_recur_scale) if expected_recur_scale > 0 else None
        ),
    }

    loops_block = {
        "count_total": len(all_loops),
        "count_unique": len(uniq),
        "length_hist": dict(sorted(Counter(loop_lengths).items())),
        "sample": uniq[:sample],
    }

    observers_block = {
        "first_loop_hitting_time": {
            "count": len(hitting_times),
            "mean": first_loop_mean,
            "quantiles": quantiles(hitting_times),
            "histogram": dict(sorted(Counter(hitting_times).items()))
        },
        "recurrence_rate": {
            "count": len(recurrence_rates),
            "mean": recur_mean,
            "quantiles": quantiles(recurrence_rates),
        }
    }

    theory_block = {
        "expected_first_loop_hitting_time": expected_first_loop,
        "expected_recurrence_rate_scale": expected_recur_scale,
        "observed_vs_expected": ratios_block
    }

    results_obj = {
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
        "theory": theory_block,
        "loops": loops_block,
        "observers": observers_block,
    }

    # PRINT to stdout (pretty JSON)
    print(json.dumps(results_obj, indent=2))

    # APPEND standardized RUN block to results file matching the script name
    results_path = results_filename_from_script(script_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    run_header = (
        f"=== RUN {results_obj['meta']['timestamp_utc']} | "
        f"seed={seed} | N={N} | walkers={walkers} | steps_per={max_steps} ==="
    )
    block = [
        run_header,
        "RESULTS:",
        json.dumps({
            "theory": theory_block,
            "loops": loops_block,
            "observers": observers_block
        }, indent=2),
        "",
    ]
    with results_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(block) + "\n")

if __name__ == "__main__":
    main()
