"""
point_generators.py
-------------------
Generates point clouds in R^n using three distributions:

  1. Gaussian  — isotropic standard normal
  2. Uniform   — uniform in a hypercube
  3. Dirichlet — sampled from the probability simplex, giving a
                 corner-rich geometry that contrasts with the round
                 shapes of Gaussian and Uniform

Each distribution has two variants:
  - "normal"   : isotropic, same range on all axes
  - "elongated": same distribution, but each axis has a smaller range
                 than the previous one. In R^2: x in (-2,2), y in (-1,1).
                 In R^n: axis k has half-width 2/(k+1), so axis 0 is the
                 widest and the last axis is the narrowest.
"""

import numpy as np
from typing import Literal, Optional


VariantType = Literal["normal", "elongated"]


def _elongated_halfwidths(n: int) -> np.ndarray:
    """
    Per-axis half-widths for the elongated variant.
    Axis k has half-width 2/(k+1):
      axis 0 -> 2.0  (widest)
      axis 1 -> 1.0
      axis 2 -> 0.67
      ...
    """
    return np.array([2.0 / (k + 1) for k in range(n)])


def gaussian_points(
    n: int,
    N: int,
    variant: VariantType = "normal",
    rng: Optional[np.random.Generator] = None,
    halfwidths: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Sample N points from a Gaussian in R^n.

    normal   : standard normal on all axes (std=1)
    elongated: each axis k has std = halfwidth_k, so the cloud is
               narrower on higher-indexed axes
    """
    if rng is None:
        rng = np.random.default_rng()

    if variant == "normal":
        return rng.standard_normal((N, n))

    # elongated: per-axis std = half-width
    hw = halfwidths if halfwidths is not None else _elongated_halfwidths(n)
    return rng.standard_normal((N, n)) * hw[np.newaxis, :]


def uniform_points(
    n: int,
    N: int,
    variant: VariantType = "normal",
    rng: Optional[np.random.Generator] = None,
    low: float = -1.0,
    high: float = 1.0,
    halfwidths: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Sample N points uniformly in R^n.

    normal   : uniform in [low, high]^n  (same range on all axes)
    elongated: axis k is uniform in [-hw_k, hw_k] where hw_k = 2/(k+1),
               so x in (-2,2), y in (-1,1), z in (-0.67, 0.67), ...
    """
    if rng is None:
        rng = np.random.default_rng()

    if variant == "normal":
        return rng.uniform(low, high, size=(N, n))

    # elongated: per-axis bounds
    hw = halfwidths if halfwidths is not None else _elongated_halfwidths(n)
    points = np.empty((N, n))
    for k in range(n):
        points[:, k] = rng.uniform(-hw[k], hw[k], size=N)
    return points


def dirichlet_points(
    n: int,
    N: int,
    variant: VariantType = "normal",
    alpha: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    halfwidths: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Sample N points from a Dirichlet distribution embedded in R^n.

    Points are drawn from Dirichlet(alpha) with n+1 components, then
    projected onto the first n coordinates. A small noise term ensures
    the cloud is full-dimensional.

    normal   : raw projection, no axis rescaling
    elongated: each axis k is rescaled to fit within [-hw_k, hw_k],
               preserving the simplex shape but compressing higher axes
    """
    if rng is None:
        rng = np.random.default_rng()
    if alpha is None:
        alpha = np.full(n + 1, 0.5)

    simplex_pts = rng.dirichlet(alpha, size=N)   # (N, n+1)
    points = simplex_pts[:, :n].copy()
    points += rng.standard_normal((N, n)) * 1e-4  # ensure full-dimensional

    if variant == "elongated":
        hw = halfwidths if halfwidths is not None else _elongated_halfwidths(n)
        # rescale each axis from [0,1] range to [-hw_k, hw_k]
        points = (points - 0.5) * 2 * hw[np.newaxis, :]

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

    distribution : one of "gaussian", "uniform", "dirichlet"
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
