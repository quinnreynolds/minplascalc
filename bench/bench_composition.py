"""Anatomy of the Gibbs free energy minimisation.

The premise to test: "adding gradient and hessian will speed up the
optimisation".  This script measures

1. how many Newton iterations the minimiser actually takes,
2. how much of an iteration is linear algebra (the part a Hessian helps)
   versus function evaluation (partition functions),
3. what one cold ``calculate_composition()`` costs end to end.
"""

import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np  # noqa: E402
from workloads import make_sico  # noqa: E402

import minplascalc.mixture as mx_mod  # noqa: E402

TEMPERATURES = np.linspace(1000, 25000, 20)


def count_iterations():
    """Instrument np.linalg.solve to count Newton steps per composition."""
    real_solve = np.linalg.solve
    counter = {"n": 0, "t": 0.0}

    def counting_solve(a, b):
        t0 = time.perf_counter()
        r = real_solve(a, b)
        counter["t"] += time.perf_counter() - t0
        counter["n"] += 1
        return r

    mx_mod.np.linalg.solve = counting_solve
    try:
        rows = []
        total_t = 0.0
        for T in TEMPERATURES:
            m = make_sico(0.5)
            m.T = T
            counter["n"] = 0
            counter["t"] = 0.0
            t0 = time.perf_counter()
            m.calculate_composition()
            dt = time.perf_counter() - t0
            total_t += dt
            rows.append((T, counter["n"], dt, counter["t"]))
    finally:
        mx_mod.np.linalg.solve = real_solve
    return rows, total_t


def main():
    m = make_sico(0.5)
    m.T = 10000
    m.calculate_composition()  # warm caches

    rows, total = count_iterations()

    print("# Gibbs minimisation, SiCO 15 species + e, cold solve at each T\n")
    print(
        f"  {'T (K)':>8s} {'newton its':>11s} {'total (ms)':>11s} "
        f"{'solve (ms)':>11s} {'solve %':>8s}"
    )
    n_its = 0
    t_solve = 0.0
    for T, n, dt, ts in rows:
        n_its += n
        t_solve += ts
        print(
            f"  {T:8.0f} {n:11d} {dt * 1e3:11.2f} {ts * 1e3:11.3f} "
            f"{100 * ts / dt:7.2f}%"
        )
    print(
        f"\n  {'TOTAL':>8s} {n_its:11d} {total * 1e3:11.2f} "
        f"{t_solve * 1e3:11.3f} {100 * t_solve / total:7.2f}%"
    )
    print(f"  mean Newton iterations per composition: {n_its / len(rows):.1f}")
    print(
        f"  mean cost per Newton iteration: "
        f"{total / n_its * 1e3:.3f} ms, of which linear solve "
        f"{t_solve / n_its * 1e3:.4f} ms"
    )

    # Where does the non-solve time go?  Time the partition-function work.
    m = make_sico(0.5)
    m.T = 12000
    m.calculate_composition()
    species = list(m.species)
    V = 1.0
    n_rep = 200
    t0 = time.perf_counter()
    for _ in range(n_rep):
        [sp.total_partition_function(V, 12000.0, 0.0) for sp in species]
    t_pf = (time.perf_counter() - t0) / n_rep
    print(
        f"\n  one full set of total_partition_function() calls: "
        f"{t_pf * 1e3:.3f} ms  ({len(species)} species)"
    )
    print(f"  -> ~{100 * t_pf / (total / n_its):.0f}% of a Newton iteration")

    for sp in species:
        n_lv = len(getattr(sp, "energy_levels", []) or [])
        if n_lv:
            n_rep = 2000
            t0 = time.perf_counter()
            for _ in range(n_rep):
                sp.internal_partition_function(12000.0, 0.0)
            dt = (time.perf_counter() - t0) / n_rep
            print(
                f"    {sp.name:<5s} {type(sp).__name__:<10s} "
                f"{n_lv:4d} levels  {dt * 1e6:8.2f} us"
            )


if __name__ == "__main__":
    main()
