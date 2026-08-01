"""cProfile the main minplascalc workloads and print the hot spots."""

import argparse
import cProfile
import pstats
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from workloads import WORKLOADS  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("workload", choices=sorted(WORKLOADS))
    p.add_argument("--n-T", type=int, default=20)
    p.add_argument("--n-mixtures", type=int, default=3)
    p.add_argument("--top", type=int, default=30)
    p.add_argument("--sort", default="tottime")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    fn = WORKLOADS[args.workload]

    # Warm up numba JIT + any caches so we profile steady-state cost.
    t0 = time.perf_counter()
    fn(n_T=2, n_mixtures=1)
    warm = time.perf_counter() - t0
    print(f"# warmup (n_T=2, 1 mixture): {warm:.3f} s", file=sys.stderr)

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    fn(n_T=args.n_T, n_mixtures=args.n_mixtures)
    pr.disable()
    wall = time.perf_counter() - t0
    print(
        f"# {args.workload}: n_T={args.n_T} n_mixtures={args.n_mixtures} "
        f"wall={wall:.3f} s (profiled)",
        file=sys.stderr,
    )

    st = pstats.Stats(pr)
    if args.out:
        st.dump_stats(args.out)
    st.sort_stats(args.sort).print_stats(args.top)


if __name__ == "__main__":
    main()
