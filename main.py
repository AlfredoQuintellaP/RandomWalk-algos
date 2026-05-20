"""
main.py
-------
The main objective of the project is to produce some polytopes with differents 
distribuitons, transform from V-representation to the H-representation and then use
common random walk algorithms with that polytope. One important thing is: for 2 and 3
dimensions I should produce some type vizualization.
--------
Full pipeline orchestration:
  1. Generate point clouds (6 combinations: 3 distributions × 2 variants)
  2. Build convex hull polytope -> H-representation
  3. Export results to disk
  4. Optionally visualize (n = 2 or n = 3 only)
  5. Optionally wrap in volestipy HPolytope (ready for random walks)

Usage
-----
  python main.py                        # default: n=2, N=200
  python main.py --dim 3 --npoints 500
  python main.py --dim 5 --npoints 1000 --no-viz
  python main.py --dim 2 --distributions gaussian uniform --variants normal
  python main.py --help
"""

import argparse
import os
import sys
import time
from itertools import product as iterproduct

import numpy as np

from point_generators import generate, GENERATORS
from polytope_builder import build_polytope, export_polytope, export_points, to_volestipy


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Random walks on polytopes — generation & building pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dim", "-d", type=int, default=2,
                        help="Ambient dimension n (>= 2)")
    parser.add_argument("--npoints", "-N", type=int, default=200,
                        help="Number of points to generate per experiment")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (incremented per experiment)")
    parser.add_argument("--outdir", type=str, default="outputs",
                        help="Directory for exported files")
    parser.add_argument(
        "--distributions", nargs="+",
        choices=list(GENERATORS.keys()),
        default=list(GENERATORS.keys()),
        help="Distributions to run",
    )
    parser.add_argument(
        "--variants", nargs="+",
        choices=["normal", "elongated"],
        default=["normal", "elongated"],
        help="Variants to run",
    )
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualization even for n=2 or n=3")
    parser.add_argument("--show", action="store_true",
                        help="Display plots interactively (blocks until closed)")
    parser.add_argument("--volestipy", action="store_true",
                        help="Attempt to wrap polytopes in volestipy HPolytope")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_experiment(
    distribution: str,
    variant: str,
    n: int,
    N: int,
    seed: int,
    outdir: str,
    visualize_flag: bool,
    show: bool,
    use_volestipy: bool,
) -> None:
    tag = f"{distribution}_{variant}_n{n}"
    print(f"\n{'='*60}")
    print(f"  Experiment: {tag}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------ #
    # 1. Generate points                                                   #
    # ------------------------------------------------------------------ #
    t0 = time.perf_counter()
    points = generate(distribution, n=n, N=N, variant=variant, seed=seed)
    print(f"  Generated {N} points  (dim={n}, dist={distribution}, variant={variant})"
          f"  [{time.perf_counter() - t0:.3f}s]")

    # Export raw points
    pts_path = os.path.join(outdir, tag, "points")
    export_points(points, pts_path, label="points")

    # ------------------------------------------------------------------ #
    # 2. Build polytope                                                    #
    # ------------------------------------------------------------------ #
    t1 = time.perf_counter()
    try:
        polytope = build_polytope(
            points,
            metadata={
                "distribution": distribution,
                "variant": variant,
                "n": n,
                "N_points": N,
                "seed": seed,
            },
        )
    except ValueError as e:
        print(f"  [SKIP] Could not build polytope: {e}")
        return

    print(f"  Polytope: {polytope.num_vertices} vertices, "
          f"{polytope.num_constraints} facets"
          f"  [{time.perf_counter() - t1:.3f}s]")

    # ------------------------------------------------------------------ #
    # 3. Export polytope                                                   #
    # ------------------------------------------------------------------ #
    poly_path = os.path.join(outdir, tag, "polytope")
    export_polytope(polytope, poly_path)

    # ------------------------------------------------------------------ #
    # 4. volestipy wrapping (optional)                                     #
    # ------------------------------------------------------------------ #
    if use_volestipy:
        vp = to_volestipy(polytope)
        if vp is not None:
            print(f"  [volestipy] HPolytope created — ready for random walks.")
            # Future: vp.generate_samples("HnR", walk_len=10, ...)

    # ------------------------------------------------------------------ #
    # 5. Visualization (n = 2 or 3 only)                                  #
    # ------------------------------------------------------------------ #
    if visualize_flag and n in (2, 3):
        try:
            from visualization import visualize
            viz_path = os.path.join(outdir, tag, f"plot_{tag}.png")
            visualize(
                polytope,
                points,
                title=f"{distribution.capitalize()} ({variant}) — n={n}",
                save_path=viz_path,
                show=show,
            )
        except Exception as e:
            print(f"  [viz] failed: {e}")
    elif visualize_flag and n > 3:
        print(f"  [viz] skipped (n={n} > 3)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.dim < 2:
        print("Error: --dim must be >= 2", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    experiments = list(iterproduct(args.distributions, args.variants))
    print(f"Running {len(experiments)} experiments  "
          f"(n={args.dim}, N={args.npoints}, seed_base={args.seed})")

    for i, (dist, variant) in enumerate(experiments):
        run_experiment(
            distribution=dist,
            variant=variant,
            n=args.dim,
            N=args.npoints,
            seed=args.seed + i,        # different seed per experiment
            outdir=args.outdir,
            visualize_flag=not args.no_viz,
            show=args.show,
            use_volestipy=args.volestipy,
        )

    print(f"\n{'='*60}")
    print(f"  All experiments complete. Outputs in: {os.path.abspath(args.outdir)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
