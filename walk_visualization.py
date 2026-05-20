"""
walk_visualization.py
---------------------
Visualization of random walk samples inside a polytope.
Only n=2 and n=3 are supported.

Both cases produce three panels: a scatter plot of all samples,
a color-coded path showing the first N steps, and a convergence
plot of the running mean distance to the centroid.
"""

import warnings
from typing import Optional

import numpy as np
from polytope_builder import HPolytope


def _check_dim(n: int) -> None:
    if n not in (2, 3):
        raise ValueError(f"Walk visualization only supported for n=2 or n=3, got n={n}.")


def plot_walk_2d(
    polytope: HPolytope,
    samples: np.ndarray,
    title: str = "Hit-and-Run (2D)",
    max_path: int = 200,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """Three-panel 2D walk visualization: scatter, path and convergence."""
    _check_dim(polytope.n)

    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
    except ImportError as e:
        warnings.warn(f"Missing dependency: {e}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle(title, color="white", fontsize=13)

    BG   = "#1a1d27"
    PT   = "#4a9eff"
    HULL = "#ff6b6b"
    PATH = "#ffd43b"

    def draw_hull(ax):
        ax.set_facecolor(BG)
        verts = polytope.vertices
        c = verts.mean(axis=0)
        angles = np.arctan2(verts[:,1]-c[1], verts[:,0]-c[0])
        v = verts[np.argsort(angles)]
        ax.add_patch(Polygon(v, closed=True,
                             edgecolor=HULL, facecolor=HULL,
                             alpha=0.15, linewidth=1.5))
        ax.scatter(verts[:,0], verts[:,1], s=30, color=HULL,
                   zorder=4, edgecolors="white", linewidths=0.5)
        ax.tick_params(colors="#888")
        for sp in ax.spines.values(): sp.set_edgecolor("#444")

    # Panel 1: all samples
    ax = axes[0]
    draw_hull(ax)
    ax.scatter(samples[:,0], samples[:,1], s=4, alpha=0.35, color=PT, zorder=3)
    ax.set_title("Samples", color="white")
    ax.set_aspect("equal")

    # Panel 2: walk path, colored by progress (early=cool, late=warm)
    ax = axes[1]
    draw_hull(ax)
    path = samples[:max_path]
    from matplotlib.collections import LineCollection
    points = path.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="plasma", linewidth=0.9, alpha=0.85)
    lc.set_array(np.linspace(0, 1, len(segments)))
    ax.add_collection(lc)
    ax.scatter(path[0,0],  path[0,1],  s=60, color="lime", zorder=5, label="start")
    ax.scatter(path[-1,0], path[-1,1], s=60, color="red",  zorder=5, label="end")
    ax.set_xlim(samples[:,0].min()-0.1, samples[:,0].max()+0.1)
    ax.set_ylim(samples[:,1].min()-0.1, samples[:,1].max()+0.1)
    ax.set_title(f"Path (first {max_path} steps)", color="white")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, facecolor="#333", labelcolor="white")

    # Panel 3: running mean distance to centroid
    ax = axes[2]
    ax.set_facecolor(BG)
    centroid = polytope.interior_point()
    dists = np.linalg.norm(samples - centroid, axis=1)
    running_mean = np.cumsum(dists) / (np.arange(len(dists)) + 1)
    ax.plot(running_mean, color=PT, lw=1.2)
    ax.axhline(dists.mean(), color=HULL, lw=1, linestyle="--", label="mean")
    ax.set_title("Convergence\n(running mean ‖x − centroid‖)", color="white")
    ax.set_xlabel("step", color="#888")
    ax.tick_params(colors="#888")
    ax.legend(fontsize=8, facecolor="#333", labelcolor="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  [walk viz] saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close("all")


def plot_walk_3d(
    polytope: HPolytope,
    samples: np.ndarray,
    title: str = "Hit-and-Run (3D)",
    max_path: int = 500,
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """Three-panel 3D walk visualization: scatter, path and convergence."""
    _check_dim(polytope.n)

    try:
        import matplotlib
        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D          # noqa
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
        from scipy.spatial import ConvexHull
    except ImportError as e:
        warnings.warn(f"Missing dependency: {e}")
        return

    fig = plt.figure(figsize=(18, 6))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle(title, color="white", fontsize=13)

    BG   = "#1a1d27"
    PT   = "#4a9eff"
    HULL = "#ff6b6b"

    def _add_hull(ax, alpha=0.10):
        hull = ConvexHull(polytope.vertices)
        facets = [polytope.vertices[s] for s in hull.simplices]
        ax.add_collection3d(Poly3DCollection(
            facets, alpha=alpha, facecolor=HULL, edgecolor="#555", linewidth=0.3
        ))

    def _style(ax):
        ax.set_facecolor(BG)
        ax.tick_params(colors="#888")
        ax.xaxis.label.set_color("#888")
        ax.yaxis.label.set_color("#888")
        ax.zaxis.label.set_color("#888")

    # Panel 1: all samples
    ax1 = fig.add_subplot(131, projection="3d")
    _style(ax1)
    _add_hull(ax1)
    ax1.scatter(samples[:,0], samples[:,1], samples[:,2],
                s=3, alpha=0.25, color=PT)
    ax1.set_title("Samples", color="white")

    # Panel 2: walk path, colored by progress (purple=early, yellow=late)
    ax2 = fig.add_subplot(132, projection="3d")
    _style(ax2)
    _add_hull(ax2, alpha=0.07)
    path = samples[:max_path]
    cmap = plt.get_cmap("plasma")
    n_seg = len(path) - 1
    for i in range(n_seg):
        color = cmap(i / max(n_seg - 1, 1))
        ax2.plot(path[i:i+2, 0], path[i:i+2, 1], path[i:i+2, 2],
                 color=color, lw=0.9, alpha=0.85)
    ax2.scatter(*path[0],  s=60, color="lime", zorder=5)
    ax2.scatter(*path[-1], s=60, color="red",  zorder=5)
    ax2.set_title(f"Path (first {max_path} steps)\npurple=early, yellow=late",
                  color="white")

    # Panel 3: running mean distance to centroid
    ax3 = fig.add_subplot(133)
    ax3.set_facecolor(BG)
    centroid = polytope.interior_point()
    dists = np.linalg.norm(samples - centroid, axis=1)
    running_mean = np.cumsum(dists) / (np.arange(len(dists)) + 1)
    ax3.plot(running_mean, color=PT, lw=1.2)
    ax3.axhline(dists.mean(), color=HULL, lw=1, linestyle="--", label="mean")
    ax3.set_title("Convergence\n(running mean ‖x − centroid‖)", color="white")
    ax3.set_xlabel("step", color="#888")
    ax3.tick_params(colors="#888")
    ax3.legend(fontsize=8, facecolor="#333", labelcolor="white")
    for sp in ax3.spines.values(): sp.set_edgecolor("#444")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  [walk viz] saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close("all")


def visualize_walk(
    polytope: HPolytope,
    samples: np.ndarray,
    title: str = "Hit-and-Run",
    save_path: Optional[str] = None,
    show: bool = False,
    **kwargs,
) -> None:
    """Dispatch to plot_walk_2d or plot_walk_3d based on polytope.n."""
    _check_dim(polytope.n)
    if polytope.n == 2:
        plot_walk_2d(polytope, samples, title=title,
                     save_path=save_path, show=show, **kwargs)
    else:
        plot_walk_3d(polytope, samples, title=title,
                     save_path=save_path, show=show, **kwargs)
