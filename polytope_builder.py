"""
polytope_builder.py
-------------------
Builds a polytope from a point cloud:
  1. Compute convex hull (using scipy)
  2. Convert to H-representation: Ax <= b
  3. Optionally wrap in volestipy HPolytope for future random-walk use
  4. Export to disk (npz + json)

The H-representation is derived from the convex hull equations:
  scipy ConvexHull stores each facet as: equations[i] = [normal | offset]
  meaning  normal · x + offset <= 0  for interior points,
  which we rewrite as  normal · x <= -offset  =>  A = normals, b = -offsets.

Future extension: pass HPolytope directly to walk samplers (Hit-and-Run, etc.).
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull, QhullError


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class HPolytope:
    """
    H-representation of a convex polytope: { x in R^n : A @ x <= b }

    Attributes
    ----------
    A        : (m, n) constraint matrix
    b        : (m,)  right-hand side vector
    n        : ambient dimension
    vertices : (k, n) array of vertex coordinates (from convex hull)
    metadata : optional dict for provenance / experiment info
    """
    A: np.ndarray
    b: np.ndarray
    n: int
    vertices: np.ndarray
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience properties (useful for future random-walk implementations)
    # ------------------------------------------------------------------

    @property
    def num_constraints(self) -> int:
        return self.A.shape[0]

    @property
    def num_vertices(self) -> int:
        return self.vertices.shape[0]

    def contains(self, x: np.ndarray, tol: float = 1e-8) -> bool:
        """Return True if point x satisfies all inequalities (up to tol)."""
        return bool(np.all(self.A @ x <= self.b + tol))

    def interior_point(self) -> np.ndarray:
        """
        Return the centroid of vertices — a robust interior point.
        Useful as a starting point for random walks.
        """
        return self.vertices.mean(axis=0)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_polytope(
    points: np.ndarray,
    metadata: Optional[dict] = None,
) -> HPolytope:
    """
    Compute the convex hull of `points` and return an HPolytope.

    Parameters
    ----------
    points   : (N, n) array of points in R^n
    metadata : optional provenance dict stored on the polytope

    Returns
    -------
    HPolytope with A, b, n, vertices set.

    Raises
    ------
    ValueError  if the point cloud is degenerate (fewer than n+1 points,
                or all points are co-planar up to numerical precision).
    """
    N, n = points.shape

    if N < n + 1:
        raise ValueError(
            f"Need at least n+1 = {n+1} points to form a full-dimensional "
            f"polytope in R^{n}, got {N}."
        )

    try:
        hull = ConvexHull(points)
    except QhullError as e:
        raise ValueError(
            f"ConvexHull failed — point cloud may be degenerate (co-planar / "
            f"co-linear). Try more points or a different distribution.\n"
            f"QhullError: {e}"
        )

    # hull.equations[i] = [a_0, ..., a_{n-1}, b_offset]
    # facet condition: a · x + b_offset <= 0  =>  a · x <= -b_offset
    normals = hull.equations[:, :n]   # (m, n)
    offsets = hull.equations[:, n]    # (m,)

    A = normals
    b = -offsets

    vertices = points[hull.vertices]  # (k, n)

    return HPolytope(
        A=A,
        b=b,
        n=n,
        vertices=vertices,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Export / Import
# ---------------------------------------------------------------------------

def export_polytope(polytope: HPolytope, path: str) -> None:
    """
    Save the polytope to two files:
      <path>.npz  — numpy arrays (A, b, vertices)
      <path>.json — metadata + scalar info

    Parameters
    ----------
    polytope : HPolytope to save
    path     : base file path (without extension)
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    np.savez_compressed(
        path + ".npz",
        A=polytope.A,
        b=polytope.b,
        vertices=polytope.vertices,
    )

    meta = {
        "n": polytope.n,
        "num_constraints": polytope.num_constraints,
        "num_vertices": polytope.num_vertices,
        **polytope.metadata,
    }
    with open(path + ".json", "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"  [export] {path}.npz  ({polytope.num_constraints} constraints, "
          f"{polytope.num_vertices} vertices)")
    print(f"  [export] {path}.json")


def export_points(points: np.ndarray, path: str, label: str = "points") -> None:
    """
    Save raw point cloud to <path>.npz.

    Parameters
    ----------
    points : (N, n) array
    path   : base file path (without extension)
    label  : key name inside the npz archive
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path + ".npz", **{label: points})
    print(f"  [export] {path}.npz  ({points.shape[0]} points, dim={points.shape[1]})")


def load_polytope(path: str) -> HPolytope:
    """
    Load an HPolytope previously saved with export_polytope().

    Parameters
    ----------
    path : base file path (without extension)

    Returns
    -------
    HPolytope
    """
    arrays = np.load(path + ".npz")
    with open(path + ".json") as f:
        meta = json.load(f)

    n = int(meta.pop("n"))
    meta.pop("num_constraints", None)
    meta.pop("num_vertices", None)

    return HPolytope(
        A=arrays["A"],
        b=arrays["b"],
        n=n,
        vertices=arrays["vertices"],
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# volestipy integration
# ---------------------------------------------------------------------------

def to_volestipy(polytope: HPolytope):
    """
    Wrap an HPolytope in a volestipy HPolytope object.

    volestipy's HPolytope(A, b) expects A (m×n) and b (m,) or (m,1).
    Returns None (with a warning) if volestipy is not installed.

    Future extension: pass the returned object directly to volesti random-walk
    samplers, e.g.:
        vp = to_volestipy(hp)
        samples = vp.generate_samples("HnR", ...)
    """
    try:
        from volestipy import HPolytope as VHPolytope  # type: ignore
    except ImportError:
        print("  [volestipy] not installed — skipping volestipy wrapping.")
        return None

    vp = VHPolytope(polytope.A, polytope.b)
    return vp
