"""
point_generators.py
-------------------
Generates point clouds in R^n using three distributions:

  1. Gaussian — isotropic standard normal
  2. Uniform  — uniform in a hypercube
  3. Dirichlet — sampled from the probability simplex, giving a
                 corner-rich geometry that contrasts with the round
                 shapes of Gaussian and Uniform

Each distribution has two variants: "normal" (isotropic) and "elongated"
(anisotropic, scaled per axis by powers of 2).
"""

import numpy as np
from typing import Literal, Optional


VariantType = Literal["normal", "elongated"]


def _elongation_scales(n: int) -> np.ndarray:
    """Per-axis scale factors for the elongated variant: axis k gets scale 2^k."""
    return np.array([2.0 ** k for k in range(n - 1, -1, -1)])


def _apply_elongation(points: np.ndarray, scales: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply a diagonal anisotropic transform: x_k -> scale_k * x_k.
    Scales default to 2^k per axis if not provided.
    """
    n = points.shape[1]
    if scales is None:
        scales = _elongation_scales(n)
    return points * scales[np.newaxis, :]


def gaussian_points(
    n: int,
    N: int,
    variant: VariantType = "normal",
    rng: Optional[np.random.Generator] = None,
    elongation_scales: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Sample N points from an isotropic standard Gaussian in R^n,
    optionally stretched by a diagonal anisotropic transform.
    """
    if rng is None:
        rng = np.random.default_rng()

    points = rng.standard_normal((N, n))

    if variant == "elongated":
        points = _apply_elongation(points, elongation_scales)

    return points


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
    optionally stretched by a diagonal anisotropic transform.
    """
    if rng is None:
        rng = np.random.default_rng()

    points = rng.uniform(low, high, size=(N, n))

    if variant == "elongated":
        points = _apply_elongation(points, elongation_scales)

    return points


def dirichlet_points(
    n: int,
    N: int,
    variant: VariantType = "normal",
    alpha: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    elongation_scales: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Sample N points from a Dirichlet distribution embedded in R^n.

    Points are drawn from Dirichlet(alpha) with n+1 components (living on
    the n-simplex in R^{n+1}), then projected onto the first n coordinates.
    A small isotropic noise term ensures the resulting cloud is
    full-dimensional rather than degenerate.

    The default alpha=[0.5, ..., 0.5] produces a sparse, corner-biased
    geometry that yields polytopes with a different combinatorial structure
    than Gaussian or Uniform clouds.
    """
    if rng is None:
        rng = np.random.default_rng()
    if alpha is None:
        alpha = np.full(n + 1, 0.5)

    simplex_pts = rng.dirichlet(alpha, size=N)  # (N, n+1)
    points = simplex_pts[:, :n]

    noise_scale = 1e-4
    points = points + rng.standard_normal((N, n)) * noise_scale

    if variant == "elongated":
        points = _apply_elongation(points, elongation_scales)

    return points


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

    distribution : one of "gaussian", "uniform" or "dirichlet"
    n            : dimension
    N            : number of points
    variant      : "normal" or "elongated"
    seed         : integer seed for reproducibility

    Returns an (N, n) float64 array.
    """
    if distribution not in GENERATORS:
        raise ValueError(
            f"Unknown distribution '{distribution}'. Choose from {list(GENERATORS)}"
        )

    rng = np.random.default_rng(seed)
    return GENERATORS[distribution](n=n, N=N, variant=variant, rng=rng, **kwargs)
