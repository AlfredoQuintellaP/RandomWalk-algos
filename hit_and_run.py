"""
hit_and_run.py
--------------
Hit-and-Run random walk sampler on a convex polytope in H-representation.

Given a current point x inside P = {x : Ax <= b}, each step works as follows:

  1. Sample a uniform random direction d on the unit sphere S^{n-1}.
  2. Find the maximal segment along x + t*d that stays in P:
       t_min = max over {i: a_i·d < 0} of (b_i - a_i·x) / (a_i·d)
       t_max = min over {i: a_i·d > 0} of (b_i - a_i·x) / (a_i·d)
  3. Sample t uniformly in [t_min, t_max] and move to x + t*d.

The stationary distribution of this chain is uniform over P.
A coordinate-direction variant (CDHR) is also provided.
"""

import numpy as np
from typing import Optional, Tuple
from polytope_builder import HPolytope


def _chord_limits(
    A: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    d: np.ndarray,
    tol: float = 1e-12,
) -> Tuple[float, float]:
    """
    Compute [t_min, t_max] such that x + t*d stays inside Ax <= b.

    For each constraint a_i · (x + t·d) <= b_i, let r_i = b_i - a_i·x
    (the slack) and s_i = a_i·d. When s_i > 0 we get an upper bound on t;
    when s_i < 0 a lower bound, and when s_i = 0 the constraint is inactive.
    """
    r = b - A @ x
    s = A @ d

    pos = s >  tol
    neg = s < -tol

    t_max = np.min(r[pos] / s[pos]) if np.any(pos) else  np.inf
    t_min = np.max(r[neg] / s[neg]) if np.any(neg) else -np.inf

    return t_min, t_max


def hit_and_run(
    polytope: HPolytope,
    n_steps: int,
    x0: Optional[np.ndarray] = None,
    burn_in: int = 0,
    thin: int = 1,
    seed: Optional[int] = None,
    direction: str = "spherical",
) -> np.ndarray:
    """
    Hit-and-Run random walk on a convex polytope.

    polytope  : HPolytope with A (m x n) and b (m,)
    n_steps   : number of samples to return, after burn-in and thinning
    x0        : starting point, defaults to the vertex centroid
    burn_in   : number of initial steps to discard
    thin      : keep every n-th sample to reduce autocorrelation
    seed      : random seed
    direction : "spherical" for uniform on S^{n-1}, or "coordinate" for
                CDHR (random axis — faster but higher autocorrelation)

    Returns an (n_steps, n) array of samples uniformly distributed in P.
    """
    rng = np.random.default_rng(seed)
    A, b, n = polytope.A, polytope.b, polytope.n

    x = polytope.interior_point() if x0 is None else np.array(x0, dtype=float)
    if not polytope.contains(x):
        raise ValueError("Starting point x0 is outside the polytope.")

    total_steps = burn_in + n_steps * thin
    samples = np.empty((n_steps, n))
    sample_idx = 0

    for step in range(total_steps):
        if direction == "spherical":
            d = rng.standard_normal(n)
            d /= np.linalg.norm(d)
        elif direction == "coordinate":
            d = np.zeros(n)
            d[rng.integers(n)] = 1.0
        else:
            raise ValueError(f"Unknown direction type '{direction}'.")

        t_min, t_max = _chord_limits(A, b, x, d)

        if t_max <= t_min:
            continue

        t = rng.uniform(t_min, t_max)
        x = x + t * d

        if step >= burn_in and (step - burn_in) % thin == 0:
            samples[sample_idx] = x
            sample_idx += 1

    return samples[:sample_idx]


def hit_and_run_volestipy(
    polytope: HPolytope,
    n_steps: int,
    walk_len: int = 1,
    burn_in: int = 0,
    seed: int = 0,
) -> Optional[np.ndarray]:
    """
    Hit-and-Run via volestipy, which delegates to the volesti C++ backend.

    Returns an (n_steps, n) array, or None if volestipy is not available.
    """
    try:
        from volestipy import HPolytope as VHP  # type: ignore
    except ImportError:
        print("[volestipy] not available — falling back to pure numpy sampler.")
        return None

    vp = VHP(polytope.A, polytope.b)
    samples = vp.sample(
        n_samples=n_steps,
        walk_length=walk_len,
        burn_in=burn_in,
        walk_type="rdhr",
        seed=seed,
    )
    return np.array(samples).T


def sample(
    polytope: HPolytope,
    n_steps: int,
    method: str = "numpy",
    **kwargs,
) -> np.ndarray:
    """
    Unified sampler entry point.

    method can be "numpy" (spherical Hit-and-Run), "cdhr" (coordinate
    direction Hit-and-Run) or "volestipy" (volesti C++ backend, falls
    back to numpy if not installed).

    Returns an (n_steps, n) array of samples.
    """
    if method == "numpy":
        return hit_and_run(polytope, n_steps, direction="spherical", **kwargs)
    elif method == "cdhr":
        return hit_and_run(polytope, n_steps, direction="coordinate", **kwargs)
    elif method == "volestipy":
        voles_kwargs = {k: v for k, v in kwargs.items()
                        if k in ("walk_len", "burn_in", "seed")}
        result = hit_and_run_volestipy(polytope, n_steps, **voles_kwargs)
        if result is None:
            print("[fallback] using pure numpy Hit-and-Run.")
            return hit_and_run(polytope, n_steps, **kwargs)
        return result
    else:
        raise ValueError(f"Unknown method '{method}'. Choose: numpy, cdhr, volestipy.")
