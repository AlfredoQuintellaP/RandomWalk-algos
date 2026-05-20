"""
visualization.py
----------------
Plotting utilities for n = 2 and n = 3 only.

For 2D:
  - scatter plot of the original point cloud
  - convex hull boundary drawn as a polygon

For 3D:
  - scatter plot of points
  - convex hull facets rendered as semi-transparent polygons

All functions accept an HPolytope and the original points array.
"""

import warnings
from typing import Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull

from polytope_builder import HPolytope


def _ensure_backend(show: bool) -> None:
    """Switch to Agg (non-interactive) backend when we only need to save files."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Guard: only 2D or 3D
# ---------------------------------------------------------------------------

def _check_dim(polytope: HPolytope) -> None:
    if polytope.n not in (2, 3):
        raise ValueError(
            f"Visualization is only supported for n=2 or n=3, got n={polytope.n}."
        )


# ---------------------------------------------------------------------------
# 2D
# ---------------------------------------------------------------------------

def plot_2d(
    polytope: HPolytope,
    points: np.ndarray,
    title: str = "Polytope (2D)",
    ax=None,
    point_color: str = "steelblue",
    hull_color: str = "darkorange",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a 2D convex hull with the underlying point cloud.

    Parameters
    ----------
    polytope   : HPolytope (n must be 2)
    points     : (N, 2) original point cloud
    title      : plot title
    ax         : existing matplotlib Axes (created if None)
    point_color: colour for scatter points
    hull_color : colour for hull boundary
    save_path  : if given, save figure to this path
    """
    _check_dim(polytope)

    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import Polygon
        from matplotlib.collections import PatchCollection
    except ImportError:
        warnings.warn("matplotlib not installed — skipping plot.")
        return

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure

    # Scatter
    ax.scatter(points[:, 0], points[:, 1], s=15, alpha=0.4,
               color=point_color, label="points", zorder=2)

    # Hull polygon — order vertices by angle around centroid
    verts = polytope.vertices  # (k, 2)
    centroid = verts.mean(axis=0)
    angles = np.arctan2(verts[:, 1] - centroid[1], verts[:, 0] - centroid[0])
    order = np.argsort(angles)
    verts_ordered = verts[order]

    poly = Polygon(verts_ordered, closed=True,
                   edgecolor=hull_color, facecolor=hull_color,
                   alpha=0.15, linewidth=2, label="hull", zorder=1)
    ax.add_patch(poly)

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.legend()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [viz] saved 2D plot → {save_path}")

    return fig, ax


# ---------------------------------------------------------------------------
# 3D
# ---------------------------------------------------------------------------

def plot_3d(
    polytope: HPolytope,
    points: np.ndarray,
    title: str = "Polytope (3D)",
    ax=None,
    point_color: str = "steelblue",
    hull_color: str = "darkorange",
    hull_alpha: float = 0.20,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a 3D convex hull with the underlying point cloud.

    Parameters
    ----------
    polytope   : HPolytope (n must be 3)
    points     : (N, 3) original point cloud
    title      : plot title
    ax         : existing Axes3D (created if None)
    point_color: scatter colour
    hull_color : facet colour
    hull_alpha : facet transparency
    save_path  : if given, save figure to this path
    """
    _check_dim(polytope)

    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        warnings.warn("matplotlib not installed — skipping plot.")
        return

    if ax is None:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    # Scatter
    ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        s=10, alpha=0.3, color=point_color, label="points",
    )

    # Rebuild hull to get simplices (triangulated facets)
    hull = ConvexHull(points)
    facets = [points[simplex] for simplex in hull.simplices]
    poly3d = Poly3DCollection(
        facets,
        alpha=hull_alpha,
        facecolor=hull_color,
        edgecolor="gray",
        linewidth=0.4,
    )
    ax.add_collection3d(poly3d)

    ax.set_title(title)
    ax.set_xlabel("x₀")
    ax.set_ylabel("x₁")
    ax.set_zlabel("x₂")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [viz] saved 3D plot → {save_path}")

    return fig, ax


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def visualize(
    polytope: HPolytope,
    points: np.ndarray,
    title: str = "Polytope",
    save_path: Optional[str] = None,
    show: bool = True,
    **kwargs,
) -> None:
    """
    Auto-dispatch to plot_2d or plot_3d based on polytope.n.

    Parameters
    ----------
    polytope  : HPolytope (n = 2 or 3)
    points    : (N, n) original point cloud
    title     : plot title
    save_path : if given, save figure here (extension determines format)
    show      : call plt.show() after plotting
    **kwargs  : forwarded to plot_2d / plot_3d
    """
    _check_dim(polytope)
    _ensure_backend(show)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib not installed — skipping visualization.")
        return

    if polytope.n == 2:
        result = plot_2d(polytope, points, title=title, save_path=save_path, **kwargs)
    else:
        result = plot_3d(polytope, points, title=title, save_path=save_path, **kwargs)

    if show:
        plt.show()
    else:
        plt.close("all")  # free memory, no blocking window
