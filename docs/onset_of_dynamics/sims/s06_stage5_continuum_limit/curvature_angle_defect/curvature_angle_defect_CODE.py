#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 5 — Curvature from Coherence (operationalization via discrete Gaussian curvature)
Self-contained: no repo-internal imports required.

Places outputs under: ./images/vNNN/
Writes a RESULTS block like the other sims.
"""

import os, sys, json, math, time, hashlib, random
from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# --------------------------
# Config
# --------------------------
CONFIG = {
    "seed": 12345,
    "out_root": None,            # default = folder of this file
    "images_dirname": "images",  # versioned subfolder images/vNNN
    "figsize": (6.6, 4.2),
    "dpi": 140,

    # Mesh resolutions to test
    "plane": {"Nx": 40, "Ny": 40, "extent": 1.0},           # square [-E,E]^2 in xy
    "sphere": {"R": 1.0, "Nu": 64, "Nv": 32},               # UV sphere
    "saddle": {"Nx": 40, "Ny": 40, "extent": 0.6, "alpha": 0.6},  # z = α(x^2 - y^2)

    # Convergence sweep for sphere (increase Nu,Nv proportionally)
    "sphere_refine": [
        {"Nu": 16, "Nv": 8},
        {"Nu": 32, "Nv": 16},
        {"Nu": 64, "Nv": 32},
        {"Nu": 128, "Nv": 64},
    ],
}

# --------------------------
# Helpers: IO and versioning
# --------------------------
def this_file_dir() -> Path:
    return Path(__file__).resolve().parent

def ensure_versioned_images(root: Path) -> Path:
    img_dir = root / CONFIG["images_dirname"]
    img_dir.mkdir(parents=True, exist_ok=True)
    # Find next vNNN
    existing = sorted([p for p in img_dir.glob("v*") if p.is_dir()])
    n = 1
    if existing:
        try:
            nums = []
            for p in existing:
                s = p.name.lstrip("v")
                if s.isdigit(): nums.append(int(s))
            n = (max(nums) + 1) if nums else 1
        except Exception:
            n = len(existing) + 1
    vdir = img_dir / f"v{n:03d}"
    vdir.mkdir(parents=True, exist_ok=True)
    return vdir

def write_results_block(path_txt: Path, block: str):
    with open(path_txt, "a", encoding="utf-8") as f:
        f.write(block + "\n")

def sha_file(path: Path, n=12) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<16), b""):
            h.update(chunk)
    return h.hexdigest()[:n]

# --------------------------
# Progress helper
# --------------------------
class _ProgressBar:
    def __init__(self, total: int | None = None, desc: str = ""):
        self.total = total
        self.desc = desc
        self.count = 0
        self._last_render = 0.0

    def update(self, step: int = 1):
        self.count += step
        self._render(final=False)

    def close(self):
        self._render(final=True)

    def _render(self, final: bool):
        if self.total:
            pct = (100.0 * self.count / max(1, self.total))
            msg = f"\r{self.desc} [{self.count}/{self.total}] {pct:5.1f}%"
        else:
            msg = f"\r{self.desc} {self.count}"
        sys.stdout.write(msg)
        if final:
            sys.stdout.write("\n")
        sys.stdout.flush()


@contextmanager
def progress(total: int | None = None, desc: str = ""):
    bar = _ProgressBar(total, desc)
    try:
        yield bar
    finally:
        bar.close()

# --------------------------
# Geometry utilities
# --------------------------
def tri_area(p, q, r):
    """Area of triangle in 3D."""
    return 0.5 * np.linalg.norm(np.cross(q - p, r - p))

def angle_at(p, q, r):
    """Angle at q in triangle (p,q,r) using 3D vectors."""
    v1 = p - q
    v2 = r - q
    a = np.linalg.norm(v1)
    b = np.linalg.norm(v2)
    if a == 0 or b == 0:
        return 0.0
    c = np.dot(v1, v2) / (a * b)
    c = np.clip(c, -1.0, 1.0)
    return math.acos(c)

def vertex_mixed_area(vid, V, F):
    """
    Mixed Voronoi area around vertex vid (Meyer et al. 2003, discrete differential-geometry staple).
    For simplicity and robustness on mostly non-obtuse triangulations, we fallback to
    1/3 of incident triangle areas when obtuse is detected.
    """
    nbr_faces = [f for f in F if vid in f]
    A = 0.0
    for f in nbr_faces:
        i,j,k = f
        p, q, r = V[i], V[j], V[k]
        # Vertex mapping to local labels:
        if vid == i:
            a, b, c = i, j, k
        elif vid == j:
            a, b, c = j, k, i
        else:
            a, b, c = k, i, j
        pa, pb, pc = V[a], V[b], V[c]
        # triangle angles
        A_a = angle_at(pb, pa, pc)
        A_b = angle_at(pc, pb, pa)
        A_c = angle_at(pa, pc, pb)
        obtuse = (A_a > math.pi/2) or (A_b > math.pi/2) or (A_c > math.pi/2)
        area_f = tri_area(p, q, r)
        if obtuse:
            # obtuse triangle: assign 1/2 area to obtuse vertex and 1/4 to others,
            # but we only collect the part at 'a'
            if A_a > math.pi/2:
                A += 0.5 * area_f
            else:
                A += 0.25 * area_f
        else:
            # non-obtuse: Voronoi area split using cotangents
            # A_a = (|pb - pa|^2 * cot A_c + |pc - pa|^2 * cot A_b) / 8
            v_ab = pb - pa
            v_ac = pc - pa
            lab2 = np.dot(v_ab, v_ab)
            lac2 = np.dot(v_ac, v_ac)
            cot_b = 1.0 / math.tan(A_b) if abs(A_b) > 1e-12 else 0.0
            cot_c = 1.0 / math.tan(A_c) if abs(A_c) > 1e-12 else 0.0
            A += (lab2 * cot_c + lac2 * cot_b) / 8.0
    return A

def discrete_gaussian_curvature(V, F):
    """
    Returns per-vertex K using angle-defect / mixed area:
      K(v) = (2π - sum angles around v) / A_voronoi
    """
    n = V.shape[0]
    K = np.zeros(n, dtype=np.float64)
    # Precompute angle sums
    angle_sum = np.zeros(n, dtype=np.float64)
    for (i, j, k) in F:
        # angles at vertices i,j,k
        Ai = angle_at(V[j], V[i], V[k])
        Aj = angle_at(V[k], V[j], V[i])
        Ak = angle_at(V[i], V[k], V[j])
        angle_sum[i] += Ai
        angle_sum[j] += Aj
        angle_sum[k] += Ak
    # Mixed areas
    areas = np.zeros(n, dtype=np.float64)
    for v in range(n):
        areas[v] = vertex_mixed_area(v, V, F)
        if areas[v] <= 1e-18:
            areas[v] = 1e-18
    defect = (2.0 * math.pi) - angle_sum
    K[:] = defect / areas
    return K, areas

# --------------------------
# Mesh generators (triangulated)
# --------------------------
def plane_patch(Nx, Ny, extent=1.0):
    """Plane z=0, square [-E,E]^2, triangulated grid."""
    xs = np.linspace(-extent, extent, Nx)
    ys = np.linspace(-extent, extent, Ny)
    VV = []
    for j in range(Ny):
        for i in range(Nx):
            VV.append([xs[i], ys[j], 0.0])
    V = np.array(VV, dtype=np.float64)
    F = []
    def vid(i,j): return j*Nx + i
    for j in range(Ny-1):
        for i in range(Nx-1):
            v00 = vid(i,j); v10 = vid(i+1,j)
            v01 = vid(i,j+1); v11 = vid(i+1,j+1)
            F.append((v00, v10, v11))
            F.append((v00, v11, v01))
    return V, np.array(F, dtype=np.int64)

def sphere_uv(R=1.0, Nu=64, Nv=32):
    """
    UV-sphere triangulated:
      u in [0, 2π], v in [0, π]
    """
    us = np.linspace(0, 2*math.pi, Nu, endpoint=False)
    vs = np.linspace(0, math.pi, Nv)
    VV = []
    for j, v in enumerate(vs):
        for i, u in enumerate(us):
            x = R * math.sin(v)*math.cos(u)
            y = R * math.sin(v)*math.sin(u)
            z = R * math.cos(v)
            VV.append([x,y,z])
    V = np.array(VV, dtype=np.float64)
    F = []
    def vid(i,j):
        return j*Nu + (i % Nu)
    for j in range(Nv-1):
        for i in range(Nu):
            a = vid(i, j)
            b = vid(i+1, j)
            c = vid(i, j+1)
            d = vid(i+1, j+1)
            if j == 0:
                # north cap triangles
                F.append((a, d, c))
            elif j == Nv-2:
                # south cap triangles
                F.append((a, b, c))
            else:
                F.append((a, b, d))
                F.append((a, d, c))
    return V, np.array(F, dtype=np.int64)

def saddle_patch(Nx, Ny, extent=0.6, alpha=0.6):
    """
    z = α(x^2 - y^2), triangulated grid on [-E,E]^2
    This surface has negative Gaussian curvature near origin; magnitude varies with position.
    """
    xs = np.linspace(-extent, extent, Nx)
    ys = np.linspace(-extent, extent, Ny)
    VV = []
    for j in range(Ny):
        for i in range(Nx):
            x = xs[i]; y = ys[j]
            z = alpha * (x*x - y*y)
            VV.append([x, y, z])
    V = np.array(VV, dtype=np.float64)
    F = []
    def vid(i,j): return j*Nx + i
    for j in range(Ny-1):
        for i in range(Nx-1):
            v00 = vid(i,j); v10 = vid(i+1,j)
            v01 = vid(i,j+1); v11 = vid(i+1,j+1)
            F.append((v00, v10, v11))
            F.append((v00, v11, v01))
    return V, np.array(F, dtype=np.int64)

# --------------------------
# Evaluation / metrics
# --------------------------
@dataclass
class CurvStats:
    L1: float
    L2: float
    Linf: float
    mean: float
    std: float
    n_used: int

def mask_interior_plane(V, Nx, Ny, border=1):
    """Mask vertices away from boundary on plane (to avoid boundary artifacts)."""
    m = np.zeros(V.shape[0], dtype=bool)
    for j in range(border, Ny-border):
        for i in range(border, Nx-border):
            m[j*Nx + i] = True
    return m

def mask_interior_grid(nI, nJ, border=1):
    """Generic interior mask for grid-like indexing."""
    m = np.zeros(nI*nJ, dtype=bool)
    for j in range(border, nJ-border):
        for i in range(border, nI-border):
            m[j*nI + i] = True
    return m

def stats_vs_target(K, target, mask):
    Km = K[mask]
    if Km.size == 0:
        return CurvStats(np.nan, np.nan, np.nan, np.nan, np.nan, 0)
    err = np.abs(Km - target)
    return CurvStats(
        L1=float(np.mean(err)),
        L2=float(np.sqrt(np.mean(err*err))),
        Linf=float(np.max(err)),
        mean=float(np.mean(Km)),
        std=float(np.std(Km)),
        n_used=int(Km.size)
    )

# --------------------------
# Plots
# --------------------------
def plot_histograms(imgdir, Ks, labels):
    plt.figure(figsize=CONFIG["figsize"], dpi=CONFIG["dpi"])
    bins = 60
    for K, lab in zip(Ks, labels):
        finite = np.isfinite(K)
        plt.hist(K[finite], bins=bins, histtype='step', label=lab, density=True)
    plt.xlabel("Discrete Gaussian curvature K")
    plt.ylabel("Density")
    plt.title("Stage 5: curvature via angle-defect")
    plt.legend()
    out = imgdir / "fig_curvature_hist_001.png"
    plt.tight_layout(); plt.savefig(out); plt.close()
    return str(out)

def plot_convergence(imgdir, hs, L2s):
    if len(hs) != len(L2s) or len(hs) < 2:
        return None
    x = np.array(hs, dtype=np.float64)
    y = np.array(L2s, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x = x[mask]; y = y[mask]
    if x.size < 2: return None
    # fit slope in log-log
    p = np.polyfit(np.log(x), np.log(y), 1)
    slope = float(p[0])

    xs = np.linspace(x.min(), x.max(), 200)
    ys = np.exp(p[1]) * xs**slope

    plt.figure(figsize=CONFIG["figsize"], dpi=CONFIG["dpi"])
    plt.loglog(x, y, 'o', label="sphere L2 error")
    plt.loglog(xs, ys, '-', label=f"fit: slope={slope:.3f}")
    plt.gca().invert_xaxis()
    plt.xlabel("h (≈ R·π/Nu)")
    plt.ylabel("L2 error |K - 1/R²|")
    plt.title("Convergence on sphere (angle-defect)")
    plt.legend()
    out = imgdir / "fig_curvature_convergence_001.png"
    plt.tight_layout(); plt.savefig(out); plt.close()
    return str(out), slope

# --------------------------
# Main
# --------------------------
def main():
    random.seed(CONFIG["seed"]); np.random.seed(CONFIG["seed"])
    out_root = Path(CONFIG["out_root"]) if CONFIG["out_root"] else this_file_dir()
    imgdir = ensure_versioned_images(out_root)

    # --- Plane
    Nx_p = CONFIG["plane"]["Nx"]; Ny_p = CONFIG["plane"]["Ny"]; E = CONFIG["plane"]["extent"]
    Vp, Fp = plane_patch(Nx_p, Ny_p, extent=E)
    Kp, Ap = discrete_gaussian_curvature(Vp, Fp)
    mask_p = mask_interior_plane(Vp, Nx_p, Ny_p, border=2)
    stats_plane = stats_vs_target(Kp, 0.0, mask_p)

    # --- Sphere
    R = CONFIG["sphere"]["R"]; Nu = CONFIG["sphere"]["Nu"]; Nv = CONFIG["sphere"]["Nv"]
    Vs, Fs = sphere_uv(R=R, Nu=Nu, Nv=Nv)
    Ks, As = discrete_gaussian_curvature(Vs, Fs)
    # sphere interior: exclude poles rows
    mask_s = mask_interior_grid(Nu, Nv, border=2)
    stats_sphere = stats_vs_target(Ks, 1.0/(R*R), mask_s)

    # --- Saddle (qualitative)
    Nx_s = CONFIG["saddle"]["Nx"]; Ny_s = CONFIG["saddle"]["Ny"]
    Ext = CONFIG["saddle"]["extent"]; alpha = CONFIG["saddle"]["alpha"]
    Vh, Fh = saddle_patch(Nx_s, Ny_s, extent=Ext, alpha=alpha)
    Kh, Ah = discrete_gaussian_curvature(Vh, Fh)
    mask_h = mask_interior_plane(Vh, Nx_s, Ny_s, border=2)  # same grid logic
    # no single analytic constant; just report proportion negative
    Khm = Kh[mask_h]
    frac_neg = float(np.mean(Khm < 0.0))

    # --- Convergence sweep on sphere
    hs, L2s = [], []
    refinements = CONFIG["sphere_refine"]
    with progress(len(refinements), desc="Sphere refinements") as bar:
        for lv in refinements:
            Nu_r = lv["Nu"]; Nv_r = lv["Nv"]
            Vr, Fr = sphere_uv(R=R, Nu=Nu_r, Nv=Nv_r)
            Kr, Ar = discrete_gaussian_curvature(Vr, Fr)
            mask_r = mask_interior_grid(Nu_r, Nv_r, border=2)
            st = stats_vs_target(Kr, 1.0/(R*R), mask_r)
            # characteristic spacing h ~ great-circle spacing along u: ≈ π R / Nu
            h = math.pi * R / float(Nu_r)
            hs.append(h); L2s.append(st.L2)
            bar.update()

    # --- Plots
    f_hist = plot_histograms(imgdir, [Kp[mask_p], Ks[mask_s], Kh[mask_h]],
                             ["plane (target 0)", f"sphere R={R} (target {1.0/(R*R):.2f})", "saddle (qualitative)"])
    out_fit = plot_convergence(imgdir, hs, L2s)
    if out_fit is None:
        f_conv, slope = None, float('nan')
    else:
        f_conv, slope = out_fit

    # --- RESULTS block
    # Compute code hash
    code_sha = "n/a"
    try:
        code_sha = sha_file(Path(__file__))
    except Exception:
        pass

    images_list = [f_hist] + ([f_conv] if f_conv else [])
    cfg_dump = {
        "plane": CONFIG["plane"],
        "sphere": {**CONFIG["sphere"], "K_target": 1.0/(R*R)},
        "saddle": CONFIG["saddle"],
        "sphere_refine": CONFIG["sphere_refine"],
    }

    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    results_txt = (this_file_dir() / "curvature_angle_defect_RESULTS.txt")

    block = []
    block.append(f"=== RUN {ts} | {imgdir} | seed={CONFIG['seed']} | commit=None | code_sha={code_sha} ===")
    block.append("STAGES: S1 n/a | S2 n/a | S3 n/a | S4 n/a | S5 n/a")
    block.append("DOC SHAS: S1=n/a S2=n/a S3=n/a S4=n/a S5=n/a")
    block.append("CONFIG:")
    block.append("  " + json.dumps(cfg_dump))
    block.append("METRICS:")
    block.append(f"  - plane: L1={stats_plane.L1:.6g}, L2={stats_plane.L2:.6g}, Linf={stats_plane.Linf:.6g}, n={stats_plane.n_used}")
    block.append(f"  - sphere: L1={stats_sphere.L1:.6g}, L2={stats_sphere.L2:.6g}, Linf={stats_sphere.Linf:.6g}, n={stats_sphere.n_used}")
    block.append(f"  - saddle: frac_negative={frac_neg:.3f}, n={int(np.sum(mask_h))}")
    block.append(f"  - sphere_refine_h={list(hs)}")
    block.append(f"  - sphere_refine_L2={list(L2s)}")
    block.append(f"  - sphere_refine_slope={slope:.3f}")
    block.append("IMAGES:")
    for p in images_list:
        block.append(f"  {p}")
    block.append("ENV:")
    block.append(f"  python={sys.version.split()[0]}")
    block.append(f"  numpy={np.__version__} | matplotlib={matplotlib.__version__}")
    block.append(f"  os={os.name}")
    block.append("=== END RUN ===")
    write_results_block(results_txt, "\n".join(block))

    print("\n".join(block))

if __name__ == "__main__":
    main()
