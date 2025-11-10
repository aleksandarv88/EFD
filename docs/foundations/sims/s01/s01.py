#!/usr/bin/env python3
# efd_foundations_harness.py
# Rigor-first simulation harness for EFD FOUNDATIONS (Δ graph) with NO ad-hoc assumptions.
# You must fill MATH-SPECIFIC sections only with constructs that exist in your math.

from __future__ import annotations
import argparse, json, random, time
from dataclasses import dataclass
from typing import List, Set, Tuple, Dict, Callable

# ----------------------------
# AXIOMS (fixed)
# ----------------------------

@dataclass
class Digraph:
    n: int
    out: List[Set[int]]

    @staticmethod
    def random(n: int, p_edge: float, seed: int | None) -> "Digraph":
        rng = random.Random(seed)
        out = [set() for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:  # Axiom: irreflexive
                    continue
                if rng.random() < p_edge:
                    out[i].add(j)
        g = Digraph(n=n, out=out)
        assert_axioms(g)
        return g

def assert_axioms(g: Digraph):
    # (G1) Irreflexive
    for i in range(g.n):
        assert i not in g.out[i], "Axiom G1 (irreflexive) violated"
    # (G2) Non-empty Δ (allowing 'partial' connectivity)
    total_edges = sum(len(g.out[i]) for i in range(g.n))
    assert total_edges >= 1, "Axiom G2 (non-empty Δ) violated"
    # (G3) Partial: no requirement enforced (by design)

def has_edge(g: Digraph, u: int, v: int) -> bool:
    return v in g.out[u]

def add_edge(g: Digraph, u: int, v: int):
    if u == v:
        return  # keep G1
    g.out[u].add(v)

def remove_edge(g: Digraph, u: int, v: int):
    g.out[u].discard(v)

def toggle_edge(g: Digraph, u: int, v: int):
    if u == v:
        return
    if v in g.out[u]:
        g.out[u].remove(v)
    else:
        g.out[u].add(v)
    assert_axioms(g)

# ----------------------------
# MATH-SPEC INTERFACE (FILL THESE FROM YOUR MATH ONLY)
# ----------------------------
#
# 1) REWRITE GENERATOR: admissible rewrites on Δ that your proofs allow.
#    Each rewrite must preserve axioms and correspond to an allowed transformation in the math.
#
# 2) FUNCTIONALS/INVARIANTS: only if your theorem claims monotonicity or invariance.
#
# 3) CLAIMS TO TEST: encode theorem statements as checkers.
#

class MathSpec:
    """
    Provide ONLY math-justified pieces. If you don't have a functional,
    set provide_functional=False and the harness will skip monotonicity tests.
    """

    def __init__(self,
                 provide_functional: bool,
                 rewrite_strategy: str = "sequential"):
        self.provide_functional = provide_functional
        self.rewrite_strategy = rewrite_strategy

    # ---- (1) REWRITES ----
    def admissible_rewrites(self, g: Digraph) -> List[Tuple[str, Tuple[int,int]]]:
        """
        Return a list of rewrites allowed by the math.
        Each is ('add'| 'remove' | 'toggle', (u,v)).
        Replace this stub with your math-defined rule set.
        """
        # >>> REPLACE WITH MATH RULES <<<
        return []

    # ---- (2) FUNCTIONAL (optional) ----
    def functional_value(self, g: Digraph) -> int:
        """
        Return value of math-defined functional (e.g., coherence cost).
        Only implement if the math defines it. Otherwise do NOT claim monotonicity.
        """
        raise NotImplementedError("No functional defined in math (leave provide_functional=False).")

    # ---- (3) CLAIMS (encode the theorem to test) ----
    def check_invariant(self, g: Digraph) -> bool:
        """
        Return True if invariant stated by the theorem holds on g.
        If no invariant is stated in the math, keep this as 'return True'.
        """
        # >>> REPLACE with invariant checks from math (if any). <<<
        return True

# ----------------------------
# HARNESS
# ----------------------------

@dataclass
class RunResult:
    seed: int
    steps: int
    rewrites_applied: int
    functional_start: int | None
    functional_end: int | None
    invariant_fail_step: int | None
    runtime_sec: float

def simulate_once(n: int,
                  p_edge: float,
                  seed: int,
                  max_steps: int,
                  patience: int,
                  mathspec: MathSpec,
                  witness_limit: int) -> RunResult:

    rng = random.Random(seed)
    g = Digraph.random(n=n, p_edge=p_edge, seed=seed)

    functional_prev = None
    if mathspec.provide_functional:
        functional_prev = mathspec.functional_value(g)

    best_seen_step = 0
    rewrites = 0
    invariant_fail_at = None
    witnesses: List[Dict] = []

    t0 = time.time()
    for step in range(1, max_steps + 1):
        # Generate math-admissible rewrites only
        moves = mathspec.admissible_rewrites(g)
        if not moves:
            # nothing to do; check invariant and exit
            if not mathspec.check_invariant(g):
                invariant_fail_at = step
            break

        # Strategy: sequential (try one by RNG) — strategy itself must be allowed by math.
        utype, payload = rng.choice(moves)
        u, v = payload

        # Apply
        if utype == "add":
            add_edge(g, u, v)
        elif utype == "remove":
            remove_edge(g, u, v)
        elif utype == "toggle":
            toggle_edge(g, u, v)
        else:
            raise ValueError(f"Unknown rewrite type: {utype}")
        rewrites += 1

        # Axioms must hold after every rewrite
        try:
            assert_axioms(g)
        except AssertionError as e:
            witnesses.append({"step": step, "reason": "axiom_violation", "msg": str(e), "u": u, "v": v})
            invariant_fail_at = step
            break

        # Invariant check (if theorem claims one)
        if not mathspec.check_invariant(g):
            witnesses.append({"step": step, "reason": "invariant_failed", "u": u, "v": v})
            invariant_fail_at = step
            break

        # Functional monotonicity (only if provided by math)
        if mathspec.provide_functional:
            curr = mathspec.functional_value(g)
            if curr > functional_prev:
                witnesses.append({"step": step, "reason": "functional_increase",
                                  "prev": functional_prev, "curr": curr, "u": u, "v": v})
                invariant_fail_at = step
                functional_prev = curr
                break
            if curr < functional_prev:
                best_seen_step = step
            functional_prev = curr

        # Optional early stop on patience (only meaningful with functional)
        if mathspec.provide_functional and (step - best_seen_step >= patience):
            break

        if len(witnesses) >= witness_limit:
            break

    t1 = time.time()

    # Save witnesses if any
    if witnesses:
        with open(f"witness_seed_{seed}.json", "w", encoding="utf-8") as f:
            json.dump(witnesses, f, indent=2)

    return RunResult(
        seed=seed,
        steps=step if 'step' in locals() else 0,
        rewrites_applied=rewrites,
        functional_start=None if not mathspec.provide_functional else witnesses[0].get("prev", functional_prev),
        functional_end=None if not mathspec.provide_functional else (None if invariant_fail_at else functional_prev),
        invariant_fail_step=invariant_fail_at,
        runtime_sec=(t1 - t0),
    )

# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="EFD FOUNDATIONS rigor harness (math-only).")
    ap.add_argument("--n", type=int, default=64, help="Number of nodes in Δ.")
    ap.add_argument("--p-edge", type=float, default=0.05, help="Initial edge probability (must respect axioms).")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=200, help="Only used if functional is provided.")
    ap.add_argument("--witness-limit", type=int, default=50, help="Max recorded counterexamples per run.")
    args = ap.parse_args()

    # >>>> USER: instantiate MathSpec with ONLY math-backed pieces <<<<
    mathspec = MathSpec(
        provide_functional=False,   # set True only if the functional exists in your math
        rewrite_strategy="sequential"
    )

    all_results = []
    for i in range(args.runs):
        rseed = args.seed + i
        res = simulate_once(
            n=args.n,
            p_edge=args.p_edge,
            seed=rseed,
            max_steps=args.max_steps,
            patience=args.patience,
            mathspec=mathspec,
            witness_limit=args.witness_limit,
        )
        all_results.append(res)

    # Print a compact summary (you can redirect to a file)
    summary = [{
        "seed": r.seed,
        "steps": r.steps,
        "rewrites": r.rewrites_applied,
        "functional_start": r.functional_start,
        "functional_end": r.functional_end,
        "invariant_fail_step": r.invariant_fail_step,
        "runtime_sec": round(r.runtime_sec, 6),
    } for r in all_results]
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
