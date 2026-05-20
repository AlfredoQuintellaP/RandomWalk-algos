"""
point_generators.py
-------------------
Generates point clouds in R^n using three distributions:
  1. Gaussian
  2. Uniform (in a hypercube)
  3. Dirichlet (on the simplex then projected to R^n)
     Justification: Dirichlet naturally lives on the probability simplex,
     producing a "corner-rich" geometry that yields polytopes with many
     vertices near the coordinate axes — a useful contrast to the
     round/symmetric shapes from Gaussian/Uniform.

Each distribution supports two variants:
  - normal   : isotropic, identity-scaled
  - elongated: anisotropic, axis-scaled by a diagonal transformation

Future extension: add distributions or transformations for custom random-walk experiments.
"""

import numpy as np
from typing import Literal, Optional


VariantType = Literal["normal", "elongated"]

# Default scale factors for elongated variant (per axis)
DEFAULT_ELONGATION = None  # will be set per dimension inside functions


def _elongation_scales(n: int) -> np.ndarray:
    """
    Build a diagonal scale vector for the elongated variant.
    Axis k gets scale 2^k, so the geometry grows exponentially along each axis.
    """
    return np.array([2.0 ** k for k in range(n - 1, -1, -1)])


def _apply_elongation(points: np.ndarray, scales: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply a diagonal (anisotropic) linear transformation: x_k -> scale_k * x_k.

    Parameters
    ----------
    points : (N, n) array
    scales : (n,) array of per-axis scale factors; defaults to 2^k per axis.

    Returns
    -------
    Transformed (N, n) array.
    """
    n = points.shape[1]
    if scales is None:
        scales = _elongation_scales(n)
    return points * scales[np.newaxis, :]


# ---------------------------------------------------------------------------
# 1. Gaussian
# ---------------------------------------------------------------------------

def gaussian_points(
    n: int,
    N: int,
    variant: VariantType = "normal",
    rng: Optional[np.random.Generator] = None,
    elongation_scales: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Sample N points from an isotropic standard Gaussian in R^n,
    optionally stretched by an anisotropic diagonal transform.

    Parameters
    ----------
    n       : dimension
    N       : number of points
    variant : 'normal' or 'elongated'
    rng     : numpy random Generator (for reproducibility)
    elongation_scales : custom per-axis scales for 'elongated' variant

    Returns
    -------
    (N, n) float64 array
    """
    if rng is None:
        rng = np.random.default_rng()

    points = rng.standard_normal((N, n))

    if variant == "elongated":
        points = _apply_elongation(points, elongation_scales)

    return points


# ---------------------------------------------------------------------------
# 2. Uniform (hypercube [-1, 1]^n)
# ---------------------------------------------------------------------------

def uniform_points(
    n: int,
    N: int,
    variant: VariantType = "normal",
    rng: Optional[np.random.Generator] = None,
    low: float = -1.0,
    high: float = 1.0,
    elongation_scales: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Sample N points uniformly in [low, high]^n,
    optionally stretched by an anisotropic diagonal transform.

    Parameters
    ----------
    n, N    : dimension, number of points
    variant : 'normal' or 'elongated'
    rng     : numpy random Generator
    low, high : hypercube bounds
    elongation_scales : custom per-axis scales for 'elongated' variant

    Returns
    -------
    (N, n) float64 array
    """
    if rng is None:
        rng = np.random.default_rng()

    points = rng.uniform(low, high, size=(N, n))

    if variant == "elongated":
        points = _apply_elongation(points, elongation_scales)

    return points


# ---------------------------------------------------------------------------
# 3. Dirichlet (simplex-based)
# ---------------------------------------------------------------------------

def dirichlet_points(
    n: int,
    N: int,
    variant: VariantType = "normal",
    alpha: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    elongation_scales: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Sample N points that are Dirichlet-distributed in R^n (full-dimensional).

    Strategy: sample from Dirichlet(alpha) with n+1 components (living on the
    n-simplex in R^{n+1}), then project onto the first n coordinates and add
    a small isotropic noise to ensure full-dimensional spread.  This preserves
    the corner-biased, simplex-like geometry while avoiding the degenerate
    co-planar case that arises when all coordinates sum to 1 exactly.

    Justification: Dirichlet concentrates mass near simplex vertices (corners
    of the positive orthant face), producing polytopes with a very different
    combinatorial structure from Gaussian/Uniform — useful for benchmarking
    random walks in corner-rich geometries.

    Parameters
    ----------
    n       : ambient dimension
    N       : number of points
    variant : 'normal' or 'elongated'
    alpha   : Dirichlet concentration parameter vector (length n+1).
              Defaults to [0.5, ..., 0.5] (sparse / corner-biased).
    rng     : numpy random Generator
    elongation_scales : custom per-axis scales for 'elongated' variant

    Returns
    -------
    (N, n) float64 array.
    """
    if rng is None:
        rng = np.random.default_rng()
    if alpha is None:
        # n+1 components so the simplex spans R^n after projection
        alpha = np.full(n + 1, 0.5)

    # Sample on the n-simplex embedded in R^{n+1}
    simplex_pts = rng.dirichlet(alpha, size=N)  # (N, n+1)

    # Project to first n coordinates — points no longer sum to exactly 1
    points = simplex_pts[:, :n]  # (N, n)

    # Add tiny isotropic noise to guarantee full-dimensional hull
    noise_scale = 1e-4
    points = points + rng.standard_normal((N, n)) * noise_scale

    if variant == "elongated":
        points = _apply_elongation(points, elongation_scales)

    return points


# ---------------------------------------------------------------------------
# Registry: maps string names to generator functions
# (convenient for main.py orchestration and future random-walk experiments)
# ---------------------------------------------------------------------------

GENERATORS = {
    "gaussian":  gaussian_points,
    "uniform":   uniform_points,
    "dirichlet": dirichlet_points,
}


def generate(
    distribution: str,
    n: int,
    N: int,
    variant: VariantType = "normal",
    seed: Optional[int] = None,
    **kwargs,
) -> np.ndarray:
    """
    Unified entry point for point generation.

    Parameters
    ----------
    distribution : one of 'gaussian', 'uniform', 'dirichlet'
    n            : dimension
    N            : number of points
    variant      : 'normal' or 'elongated'
    seed         : integer seed for reproducibility
    **kwargs     : forwarded to the specific generator

    Returns
    -------
    (N, n) float64 array
    """
    if distribution not in GENERATORS:
        raise ValueError(f"Unknown distribution '{distribution}'. Choose from {list(GENERATORS)}")

    rng = np.random.default_rng(seed)
    return GENERATORS[distribution](n=n, N=N, variant=variant, rng=rng, **kwargs)
