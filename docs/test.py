#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
unitary_slice_proxy_CODE.py

Bigram loop-operator proxy for ζ(s):
- States: ordered prime bigrams (p_prev, p_curr), count = K^(L-1). With L=3, states = K^2.
- Edges: (a,b) -> (b,c) with complex weight w = exp(-(σ+it) * τ), τ = log(c) + eps*log(a).
- Mixer: optional global unitary (DFT over the state space) or identity.
- Conjugation: diagonal W from equilibrium at σ=1/2 (left singular vector), Ã = W^{-1/2} A W^{1/2}.
- Neutral renorm: gamma = 1 / smax(Ã at σ=1/2), then scale all A by gamma (pins neutrality to 1).
- Output: JSON summary and writes "<thisfile>_RESUTLS.txt".

No external data; only numpy required.
"""

import os, json, math, cmath
from itertools import product
from typing import List, Tuple, Dict
import numpy as np

# ----------------------------
# Config
# ----------------------------
CONFIG = {
    # Prime alphabet (you can extend this)
    "alphabet_primes": [2, 3, 5, 7, 11, 13],
    # Word length L (states are length L-1 bigrams). With L=3, states=K^2
    "L": 3,
    # Sigma (real parts) to probe
    "sigmas": [0.50, 0.45, 0.55],
    # Imag parts t to probe
    "t_values": [0.0, 3.0, 6.0, 9.0],
    # Mixer: "DFT(full)" or "identity"
    "mixer": "DFT(full)",
    # Bigram asymmetry; set >0 to induce t-dependence without breaking neutrality
    "epsilon_bigrame": 0.0,   # try 0.05 for visible t-structure
    # Numerical safety
    "svd_tol": 1e-12,
    # Report how many singular values (debug; we only record the top in JSON)
    "report_sv_count": 1
}

# ----------------------------
# Utilities
# ----------------------------

def states_bigrams(primes: List[int], L: int) -> List[Tuple[int, ...]]:
    """
    Build state space as length-(L-1) tuples over primes.
    For L=3, states are pairs (a,b) with a,b in primes. Count = K^(L-1).
    """
    if L < 2:
        raise ValueError("L must be >= 2 (bigram requires at least pairs).")
    return list(product(primes, repeat=L-1))

def index_map(states: List[Tuple[int, ...]]) -> Dict[Tuple[int, ...], int]:
    return {st: i for i, st in enumerate(states)}

def build_adjacency(primes: List[int],
                    states: List[Tuple[int, ...]],
                    idx: Dict[Tuple[int, ...], int],
                    sigma: float,
                    t: float,
                    eps_bi: float) -> np.ndarray:
    """
    Build raw adjacency/transfer matrix A_s for (σ,t).
    Transition rule: (a,b) -> (b,c) for any c in primes.
    Weight: exp(-(σ+it) * τ), τ = log(c) + eps_bi * log(a).
    """
    n = len(states)
    A = np.zeros((n, n), dtype=np.complex128)

    for st in states:
        a = st[0]
        b = st[-1]
        src = idx[st]
        # Next states: (b,c)
        for c in primes:
            nxt = (b, c)
            j = idx[nxt]
            tau = math.log(c) + eps_bi * math.log(a)
            w = cmath.exp(-(sigma + 1j * t) * tau)
            A[j, src] += w  # column-stochastic style (src -> j)
    return A

def unitary_mixer_matrix(n: int, kind: str) -> np.ndarray:
    """
    Global unitary mixer over the state space.
    "DFT(full)": normalized DFT_n, U*U^* = I.
    "identity": identity.
    """
    if kind == "identity":
        return np.eye(n, dtype=np.complex128)
    if kind == "DFT(full)":
        # Standard DFT with 1/sqrt(n) normalization
        omega = np.exp(-2j * np.pi / n)
        j_idx = np.arange(n).reshape(-1, 1)
        k_idx = np.arange(n).reshape(1, -1)
        F = omega ** (j_idx @ k_idx)
        return F / np.sqrt(n)
    raise ValueError(f"Unknown mixer kind: {kind}")

def apply_mixer(A: np.ndarray, U: np.ndarray) -> np.ndarray:
    # Conjugation by a global unitary doesn't change singular values,
    # but we include it to match the user's experiments.
    return U @ A @ U.conj().T

def top_singular_value(M: np.ndarray, svd_tol: float) -> float:
    # economical SVD
    s = np.linalg.svd(M, compute_uv=False, hermitian=False)
    return float(s[0] if s.size else 0.0)

def equilibrium_weights(A_half: np.ndarray, svd_tol: float) -> np.ndarray:
    """
    Use leading left singular vector at the neutral slice (σ=1/2, t arbitrary)
    to construct diagonal W. Normalize to probability vector.
    """
    # SVD: A = U S V^*
    U, S, Vh = np.linalg.svd(A_half, full_matrices=False)
    u = U[:, 0]  # left singular vector for top singular value
    w = np.abs(u)**2
    s = w.sum()
    if s < svd_tol:
        w = np.ones_like(w) / w.size
    else:
        w = w / s
    return w

def conjugate_by_weights(A: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Return Ã = W^{-1/2} A W^{1/2} with W = diag(weights).
    """
    w_sqrt = np.sqrt(weights)
    # Avoid dividing by zero:
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_w_sqrt = np.where(w_sqrt > 0, 1.0 / w_sqrt, 0.0)
    D_left = np.diag(inv_w_sqrt)
    D_right = np.diag(w_sqrt)
    return D_left @ A @ D_right

def neutral_renormalize(A_half_conj: np.ndarray, svd_tol: float) -> float:
    """
    Compute gamma = 1 / smax(Ã_{1/2}), so that after scaling all A by gamma
    the neutral slice has top singular exactly 1.
    """
    smax = top_singular_value(A_half_conj, svd_tol)
    if smax < svd_tol:
        return 1.0
    return 1.0 / smax

def as_json_friendly_complex(x: complex) -> Dict[str, float]:
    return {"re": float(x.real), "im": float(x.imag)}

# ----------------------------
# Main routine
# ----------------------------

def main():
    cfg = CONFIG.copy()
    primes = cfg["alphabet_primes"]
    K = len(primes)
    L = cfg["L"]
    sigmas = cfg["sigmas"]
    t_vals = cfg["t_values"]
    mixer_kind = cfg["mixer"]
    eps_bi = cfg["epsilon_bigrame"]
    svd_tol = cfg["svd_tol"]

    # Build state space = bigrams of primes (length L-1)
    states = states_bigrams(primes, L)
    idx = index_map(states)
    n = len(states)

    # Global mixer
    U = unitary_mixer_matrix(n, mixer_kind)

    # Build A at neutral slice (σ=1/2, t=0) to compute equilibrium W
    A_half_raw = build_adjacency(primes, states, idx, sigma=0.5, t=0.0, eps_bi=eps_bi)
    A_half = apply_mixer(A_half_raw, U)

    # Equilibrium weights & conjugation at neutral slice
    w_eq = equilibrium_weights(A_half, svd_tol)
    A_half_conj = conjugate_by_weights(A_half, w_eq)

    # Neutral renormalization gamma
    gamma = neutral_renormalize(A_half_conj, svd_tol)

    # Prepare JSON result
    result = {
        "config": {
            "alphabet_primes": primes,
            "K": K,
            "L": L,
            "sigmas": sigmas,
            "t_values": t_vals,
            "state_count": n,
            "mixer": mixer_kind,
            "epsilon_bigrame": eps_bi
        },
        "equilibrium": {
            "top_singular_at_sigma_0p5_before_scale": top_singular_value(A_half_conj, svd_tol),
            "gamma_scale": gamma
        },
        "grid": []
    }

    # Sweep grid, apply gamma scaling and same conjugation by W
    for sigma in sigmas:
        for t in t_vals:
            A_raw = build_adjacency(primes, states, idx, sigma=sigma, t=t, eps_bi=eps_bi)
            A = apply_mixer(A_raw, U)
            A_conj = conjugate_by_weights(gamma * A, w_eq)
            smax = top_singular_value(A_conj, svd_tol)
            result["grid"].append({
                "sigma": sigma,
                "t": t,
                "top_singular": smax
            })

    # Pretty print JSON to stdout
    print(json.dumps(result, indent=2))

    # Save to *_RESUTLS.txt next to this script
    try:
        this = os.path.basename(__file__)
    except NameError:
        this = "unitary_slice_proxy_CODE.py"
    base, _ = os.path.splitext(this)
    out_path = f"./{base.replace('_CODE', '')}_RESUTLS.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, indent=2))
    # Also tell the user where it went (stdout already has JSON)
    print(f"\n# Saved results to: {out_path}")

if __name__ == "__main__":
    main()
