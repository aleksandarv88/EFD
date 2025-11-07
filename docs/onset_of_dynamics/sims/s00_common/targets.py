"""
targets.py — Analytic targets for metric and curvature benchmarks.

Distance targets:
  - Euclidean on [0,1]^2
  - Constant anisotropic metric with tensor M (2x2 SPD)
  - Simple spatially varying anisotropy via diagonal field λx(x,y), λy(x,y)
    (uses midpoint rule for the line segment as an approximation)

Curvature targets:
  - Sphere of radius R: K = +1/R^2
  - Cylinder radius R:  K = 0 (Gaussian)
  - Hyperbolic paraboloid (saddle): z = a x^2 - b y^2
      K(x,y) = -4 a^2 b^2 / (1 + 4 a^2 x^2 + 4 b^2 y^2)^2   (approx.)
"""

from __future__ import annotations
from typing import Callable, Tuple
import math
import numpy as np

Vec2 = Tuple[float, float]

# ---------------------------------------------------------------------------
# Metric / distance

def euclidean_distance(p: Vec2, q: Vec2) -> float:
    (x1, y1), (x2, y2) = p, q
    return math.hypot(x2 - x1, y2 - y1)

def anisotropic_distance_constant(p: Vec2, q: Vec2, M: np.ndarray) -> float:
    """
    Distance under constant SPD metric tensor M (2x2).
    Uses straight segment: sqrt( (q-p)^T M (q-p) ).
    """
    v = np.array([q[0] - p[0], q[1] - p[1]], dtype=float)
    return float(np.sqrt(v @ (M @ v)))

def anisotropic_distance_midpoint(
    p: Vec2, q: Vec2, lam: Callable[[float, float], Tuple[float, float]]
) -> float:
    """
    Approximate distance under a diagonal spatially varying metric:
        g = diag(λx(x,y), λy(x,y))
    via midpoint evaluation along the straight segment.
    """
    mx, my = (p[0] + q[0]) * 0.5, (p[1] + q[1]) * 0.5
    lx, ly = lam(mx, my)
    dx, dy = q[0] - p[0], q[1] - p[1]
    return math.sqrt(lx * dx * dx + ly * dy * dy)

# Example λ-field factories
def lam_iso(c: float) -> Callable[[float, float], Tuple[float, float]]:
    return lambda x, y: (c, c)

def lam_x_stretch(a: float) -> Callable[[float, float], Tuple[float, float]]:
    """Stretch in x by factor 'a' (bigger a ⇒ longer in x)."""
    return lambda x, y: (a, 1.0)

def lam_gaussian_bump(cx: float, cy: float, amp: float, sigma: float) -> Callable[[float, float], Tuple[float, float]]:
    """Diagonal bump λx=λy = 1 + amp * exp(-||x-c||^2 / (2σ^2))."""
    def _lam(x: float, y: float) -> Tuple[float, float]:
        dx, dy = x - cx, y - cy
        g = amp * math.exp(-(dx*dx + dy*dy) / (2.0 * sigma * sigma))
        s = 1.0 + g
        return (s, s)
    return _lam

# ---------------------------------------------------------------------------
# Curvature targets

def K_sphere(R: float) -> float:
    """Gaussian curvature on a sphere of radius R."""
    return 1.0 / (R * R)

def K_cylinder(R: float) -> float:
    """Gaussian curvature on a cylinder — zero (ignoring end caps)."""
    return 0.0

def K_saddle(a: float, b: float, x: float, y: float) -> float:
    """
    Approximate Gaussian curvature of z = a x^2 - b y^2 surface at (x,y).
    Good for sign and qualitative magnitude; exact formula is more involved.
    """
    num = -4.0 * (a * a) * (b * b)
    den = (1.0 + 4.0 * a * a * x * x + 4.0 * b * b * y * y) ** 2
    return num / den
