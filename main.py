"""
main.py
-------
Full pipeline: generate point clouds, build convex hull polytopes,
export results, and optionally run a Hit-and-Run random walk.

Six combinations are run by default: 3 distributions x 2 variants.
Visualization is only available for n=2 and n=3.

Usage
-----
  python main.py
  python main.py --dim 3 --npoints 500
  python main.py --dim 2 --walk
  python main.py --dim 2 --walk --walk-steps 5000 --walk-burnin 1000
  python main.py --dim 2 --walk --walk-method volestipy
  python main.py --dim 5 --npoints 1000 --no-viz
"""

import argparse
import os
import sys
import time
from itertools import product as iterproduct

import numpy as np

from point_generators import generate, GENERATORS
from polytope_builder import build_polytope, export_polytope, export_points, to_volestipy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Random walks on polytopes — generation and building pipeline",
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
    parser.add_argument("--walk", action="store_true",
                        help="Run Hit-and-Run after building the polytope")
    parser.add_argument("--walk-steps", type=int, default=2000,
                        help="Number of walk samples to collect")
    parser.add_argument("--walk-burnin", type=int, default=500,
                        help="Burn-in steps discarded at the start")
    parser.add_argument("--walk-thin", type=int, default=1,
                        help="Thinning factor (keep every k-th sample)")
    parser.add_argument("--walk-method", type=str, default="numpy",
                        choices=["numpy", "cdhr", "volestipy"],
                        help="Hit-and-Run backend")
    return parser.parse_args()


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
    walk: bool,
    walk_steps: int,
    walk_burnin: int,
    walk_thin: int,
    walk_method: str,
) -> None:
    tag = f"{distribution}_{variant}_n{n}"
    print(f"\n{'='*60}")
    print(f"  Experiment: {tag}")
    print(f"{'='*60}")

    # Generate points
    t0 = time.perf_counter()
    points = generate(distribution, n=n, N=N, variant=variant, seed=seed)
    print(f"  Generated {N} points  (dim={n}, dist={distribution}, variant={variant})"
          f"  [{time.perf_counter() - t0:.3f}s]")

    pts_path = os.path.join(outdir, tag, "points")
    export_points(points, pts_path, label="points")

    # Build polytope
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

    # Export polytope
    poly_path = os.path.join(outdir, tag, "polytope")
    export_polytope(polytope, poly_path)

    # volestipy wrapping (optional)
    if use_volestipy:
        vp = to_volestipy(polytope)
        if vp is not None:
            print(f"  [volestipy] HPolytope created.")

    # Polytope visualization (n=2 or n=3 only)
    if visualize_flag and n in (2, 3):
        try:
            from visualization import visualize
            viz_path = os.path.join(outdir, tag, f"plot_{tag}.png")
            visualize(
                polytope, points,
                title=f"{distribution.capitalize()} ({variant}) — n={n}",
                save_path=viz_path,
                show=show,
            )
        except Exception as e:
            print(f"  [viz] failed: {e}")
    elif visualize_flag and n > 3:
        print(f"  [viz] polytope plot skipped (n={n} > 3)")

    # Hit-and-Run random walk
    if walk:
        t2 = time.perf_counter()
        try:
            from hit_and_run import sample
            samples = sample(
                polytope,
                n_steps=walk_steps,
                method=walk_method,
                burn_in=walk_burnin,
                thin=walk_thin,
                seed=seed + 1000,
            )
            print(f"  Walk: {len(samples)} samples collected "
                  f"(method={walk_method}, burn_in={walk_burnin}, thin={walk_thin})"
                  f"  [{time.perf_counter() - t2:.3f}s]")

            walk_path = os.path.join(outdir, tag, "walk_samples")
            os.makedirs(os.path.dirname(os.path.abspath(walk_path + ".npz")), exist_ok=True)
            np.savez_compressed(walk_path + ".npz", samples=samples)
            print(f"  [export] {walk_path}.npz")

            if visualize_flag and n in (2, 3):
                from walk_visualization import visualize_walk
                walk_viz_path = os.path.join(outdir, tag, f"walk_{tag}.png")
                visualize_walk(
                    polytope, samples,
                    title=f"Hit-and-Run — {distribution} ({variant}), n={n}",
                    save_path=walk_viz_path,
                    show=show,
                )
            elif visualize_flag and n > 3:
                print(f"  [viz] walk plot skipped (n={n} > 3)")

        except Exception as e:
            print(f"  [walk] failed: {e}")
            raise


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
            seed=args.seed + i,
            outdir=args.outdir,
            visualize_flag=not args.no_viz,
            show=args.show,
            use_volestipy=args.volestipy,
            walk=args.walk,
            walk_steps=args.walk_steps,
            walk_burnin=args.walk_burnin,
            walk_thin=args.walk_thin,
            walk_method=args.walk_method,
        )

    print(f"\n{'='*60}")
    print(f"  All experiments complete. Outputs in: {os.path.abspath(args.outdir)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
