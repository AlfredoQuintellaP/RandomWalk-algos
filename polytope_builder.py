"""
polytope_builder.py
-------------------
Builds a polytope from a point cloud by computing its convex hull and
converting the result to an H-representation (Ax <= b).

scipy's ConvexHull stores each facet as equations[i] = [normal | offset],
where normal · x + offset <= 0 for interior points. Rewriting gives
normal · x <= -offset, so A = normals and b = -offsets.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull, QhullError


@dataclass
class HPolytope:
    """
    H-representation of a convex polytope: { x in R^n : A @ x <= b }

    A        : (m, n) constraint matrix
    b        : (m,) right-hand side vector
    n        : ambient dimension
    vertices : (k, n) vertex coordinates from the convex hull
    metadata : optional provenance dict
    """
    A: np.ndarray
    b: np.ndarray
    n: int
    vertices: np.ndarray
    metadata: dict = field(default_factory=dict)

    @property
    def num_constraints(self) -> int:
        return self.A.shape[0]

    @property
    def num_vertices(self) -> int:
        return self.vertices.shape[0]

    def contains(self, x: np.ndarray, tol: float = 1e-8) -> bool:
        """Return True if x satisfies all inequalities up to the given tolerance."""
        return bool(np.all(self.A @ x <= self.b + tol))

    def interior_point(self) -> np.ndarray:
        """Return the centroid of the vertices, a reliable interior starting point."""
        return self.vertices.mean(axis=0)


def build_polytope(
    points: np.ndarray,
    metadata: Optional[dict] = None,
) -> HPolytope:
    """
    Compute the convex hull of the given points and return an HPolytope.

    Raises ValueError if the point cloud is degenerate (fewer than n+1
    points, or all points are coplanar up to numerical precision).
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
            f"ConvexHull failed — point cloud may be degenerate (coplanar or "
            f"colinear). Try more points or a different distribution.\n"
            f"QhullError: {e}"
        )

    normals = hull.equations[:, :n]
    offsets = hull.equations[:, n]

    A = normals
    b = -offsets

    vertices = points[hull.vertices]

    return HPolytope(
        A=A,
        b=b,
        n=n,
        vertices=vertices,
        metadata=metadata or {},
    )


def export_polytope(polytope: HPolytope, path: str) -> None:
    """
    Save the polytope to <path>.npz (arrays) and <path>.json (metadata).
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
    """Save a raw point cloud to <path>.npz."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez_compressed(path + ".npz", **{label: points})
    print(f"  [export] {path}.npz  ({points.shape[0]} points, dim={points.shape[1]})")


def load_polytope(path: str) -> HPolytope:
    """Load an HPolytope previously saved with export_polytope()."""
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


def to_volestipy(polytope: HPolytope):
    """
    Wrap an HPolytope in a volestipy HPolytope object.
    Returns None with a warning if volestipy is not installed.
    """
    try:
        from volestipy import HPolytope as VHPolytope  # type: ignore
    except ImportError:
        print("  [volestipy] not installed — skipping volestipy wrapping.")
        return None

    vp = VHPolytope(polytope.A, polytope.b)
    return vp
