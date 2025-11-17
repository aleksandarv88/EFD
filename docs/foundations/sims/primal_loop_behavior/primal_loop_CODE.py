# C:\Users\Asus\Desktop\MATH\EFD\docs\foundation\sims\primal_loop_behavior\primal_loop_behavior_CODE.py

"""
EFD SIM — PRIMAL LOOP BEHAVIOR
(∅ ↔ ∃) → Δ → closure → growing substrate

This sim shows:
1. "nothing" (∅) and "something" (∃) only make sense together;
   if we remove one pole, the relation is not meaningful.
2. once we have a nontrivial difference Δ, closing Δ on its own outputs
   forces the substrate to grow (monotone increase).
3. results are written to primal_loop_behavior_RESULTS.txt in this folder.

Config:
Looks for:
  ..\config\defaults.yaml
relative to this file.

Expected YAML:

    iterations: 4
    verbose: true
"""

import os
from datetime import datetime

# optional — only if you have PyYAML, otherwise we fallback
try:
    import yaml
except ImportError:
    yaml = None


# --------------------------------------------------
# CONFIG LOADING
# --------------------------------------------------
def load_config():
    """
    Try to load ../config/defaults.yaml relative to this file.
    If not found or yaml not available, return sensible defaults.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(here, "..", "config", "defaults.yaml")
    config_path = os.path.normpath(config_path)
    print(config_path)
    config = {
        "iterations": 4,
        "verbose": True,
    }

    if yaml is None:
        return config

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        config.update({k: v for k, v in data.items() if k in config})
    return config


# --------------------------------------------------
# CORE LOGIC
# --------------------------------------------------
def is_relation_meaningful(relation):
    """
    A relation is meaningful iff:
    - it's not empty
    - each pair has two endpoints
    - endpoints are distinct
    """
    if not relation:
        return False
    for a, b in relation:
        if a is None or b is None:
            return False
        if a == b:
            return False
    return True


def closure_step(substrate):
    """
    Given a list of substrate elements, create NEW elements
    representing differences between distinct elements.

    If x != y -> create "Δ(x,y)"
    """
    new_elems = []
    n = len(substrate)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            x = substrate[i]
            y = substrate[j]
            diff_elem = f"Δ({x},{y})"
            if diff_elem not in substrate and diff_elem not in new_elems:
                new_elems.append(diff_elem)
    return new_elems


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    cfg = load_config()
    iterations = int(cfg.get("iterations", 6))
    verbose = bool(cfg.get("verbose", True))

    # primal poles
    nothing = "∅"
    something = "∃"

    # minimal nontrivial difference
    Delta = {(nothing, something)}

    print("=== EFD SIM: PRIMAL LOOP BEHAVIOR ===")
    print(f"start time: {datetime.utcnow().isoformat()}Z")
    print("primal poles:", nothing, something)
    print("Δ =", Delta)
    print("Δ meaningful?", is_relation_meaningful(Delta))
    print()

    # show broken case
    broken = {("∅", None)}
    print("[check] broken relation:", broken)
    print("[check] broken relation meaningful?", is_relation_meaningful(broken))
    print()

    # substrate starts with the two poles
    substrate = [nothing, something]
    print("initial substrate:", substrate)
    print()

    # run closure iterations
    for step in range(iterations):
        new_elems = closure_step(substrate)
        substrate.extend(new_elems)
        print(f"[step {step}] substrate size = {len(substrate)}")
        if verbose:
            print("   ", substrate)

    print()
    print("final substrate size:", len(substrate))
    print("sim done.")

    # write results next to this file
    here = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(here, "primal_loop_behavior_RESULTS.txt")
    try:
        with open(results_path, "a", encoding="utf-8") as f:
            f.write("=== RUN {}Z ===\n".format(datetime.utcnow().isoformat()))
            f.write(f"iterations: {iterations}\n")
            f.write(f"final_substrate_size: {len(substrate)}\n")
            f.write("substrate_sample:\n")
            for item in substrate[:30]:
                f.write(f"  - {item}\n")
            f.write("=== END RUN ===\n\n")
    except OSError:
        # if we can't write, we just skip
        pass


if __name__ == "__main__":
    main()
