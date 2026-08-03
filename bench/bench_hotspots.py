"""Measure the opportunities the profile actually points at.

Option 1-4 all target the q-matrix assembly, which the profile shows is
~1.3% of q().  This script sizes the two things that are *not* 1.3%:

A. ``Monatomic.internal_partition_function`` -- a Python loop over up to
   580 energy levels calling scalar ``np.exp``.
B. ``Qij_mix`` recomputing species-pair potential parameters for every one
   of the 16 (l, s) collision-integral pairs.
"""

import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np  # noqa: E402
import pf_vectorised  # noqa: E402
from workloads import make_sico  # noqa: E402

import minplascalc.functions_transport as ft  # noqa: E402

TEMPERATURES = np.linspace(1000, 25000, 20)


def sweep_all_four():
    mixtures = [make_sico(sio) for sio in (0.1, 0.5, 0.9)]
    out = []
    t0 = time.perf_counter()
    for mixture in mixtures:
        for T in TEMPERATURES:
            mixture.T = T
            out.append(
                (
                    mixture.calculate_viscosity(),
                    mixture.calculate_electrical_conductivity(),
                    mixture.calculate_total_emission_coefficient(),
                    mixture.calculate_thermal_conductivity(),
                )
            )
    return time.perf_counter() - t0, np.array(out)


def sweep_composition():
    mixtures = [make_sico(sio) for sio in (0.1, 0.5, 0.9)]
    out = []
    t0 = time.perf_counter()
    for mixture in mixtures:
        for T in TEMPERATURES:
            mixture.T = T
            out.append(mixture.calculate_composition())
    return time.perf_counter() - t0, np.array(out)


def main():
    # Warm numba.
    m = make_sico(0.5)
    m.T = 10000
    m.calculate_viscosity()
    m.calculate_thermal_conductivity()

    print("# A. Vectorising Monatomic.internal_partition_function\n")

    # Per-call microbenchmark first.
    m = make_sico(0.5)
    m.T = 12000
    m.calculate_composition()
    mono = [s for s in m.species if type(s).__name__ == "Monatomic"]

    print(
        f"  {'species':<7s} {'levels':>7s} {'summed':>7s} "
        f"{'loop (us)':>10s} {'vec (us)':>10s} {'speedup':>9s} "
        f"{'rel err':>10s}"
    )
    for sp in mono:
        loop_fn = type(sp).internal_partition_function
        ref = loop_fn(sp, 12000.0, 0.0)
        got = pf_vectorised.internal_partition_function_vec(sp, 12000.0, 0.0)
        err = abs(got - ref) / abs(ref)

        E = np.array([e for _, e in sp.energy_levels])
        over = E >= sp.ionisation_energy
        k = int(np.argmax(over)) if over.any() else len(E)
        if over.any() and k == 0:
            k = 0

        n_rep = 3000
        t0 = time.perf_counter()
        for _ in range(n_rep):
            loop_fn(sp, 12000.0, 0.0)
        t_loop = (time.perf_counter() - t0) / n_rep
        t0 = time.perf_counter()
        for _ in range(n_rep):
            pf_vectorised.internal_partition_function_vec(sp, 12000.0, 0.0)
        t_vec = (time.perf_counter() - t0) / n_rep

        print(
            f"  {sp.name:<7s} {len(E):7d} {k:7d} {t_loop * 1e6:10.2f} "
            f"{t_vec * 1e6:10.2f} {t_loop / t_vec:8.1f}x {err:10.2e}"
        )

    # End-to-end effect.
    print("\n  end-to-end (SiCO, 20 T x 3 mixtures):")
    t_comp_before, comp_before = sweep_composition()
    t_all_before, all_before = sweep_all_four()

    undo = pf_vectorised.patch()
    try:
        t_comp_after, comp_after = sweep_composition()
        t_all_after, all_after = sweep_all_four()
    finally:
        undo()

    c_err = np.abs(comp_after - comp_before).max() / np.abs(comp_before).max()
    a_err = np.abs(all_after - all_before).max() / np.abs(all_before).max()

    print(
        f"    composition sweep   {t_comp_before:7.3f} s -> "
        f"{t_comp_after:7.3f} s  ({t_comp_before / t_comp_after:.2f}x)"
    )
    print(
        f"    all-four sweep      {t_all_before:7.3f} s -> "
        f"{t_all_after:7.3f} s  ({t_all_before / t_all_after:.2f}x)"
    )
    print(f"    max rel change in composition results: {c_err:.3e}")
    print(f"    max rel change in property results:    {a_err:.3e}")

    # ---------------------------------------------------------------
    print(
        "\n\n# B. Redundant work across the 16 (l, s) collision "
        "integral pairs\n"
    )

    counts = {}
    for fname in (
        "pot_parameters_ion_neut",
        "pot_parameters_neut_neut",
        "beta",
        "x0_ion_neut",
        "x0_neut_neut",
        "cl_charged",
        "psiconst",
    ):
        real = getattr(ft, fname)
        counts[fname] = [0, real]

        def make(name, fn):
            def wrapper(*a, **kw):
                counts[name][0] += 1
                return fn(*a, **kw)

            return wrapper

        setattr(ft, fname, make(fname, real))

    m = make_sico(0.5)
    m.T = 12000
    m.calculate_composition()
    try:
        ft.q(m)
    finally:
        for fname, (_, real) in counts.items():
            setattr(ft, fname, real)

    nb = len(m.species)
    print(f"  calls made during ONE q() call ({nb} species):")
    print(
        f"  {'function':<28s} {'calls':>8s} {'per (l,s)':>10s} "
        f"{'distinct pairs':>15s}"
    )
    for fname, (n, _) in counts.items():
        print(f"  {fname:<28s} {n:8d} {n / 16:10.1f} {nb * nb:15d}")
    print(
        f"\n  16 (l, s) pairs x {nb}x{nb} species pairs = "
        f"{16 * nb * nb} Qij evaluations per q()"
    )
    print(
        "  potential parameters depend only on the species pair, not on "
        "(l, s):\n  they are recomputed 16x more often than necessary."
    )


if __name__ == "__main__":
    main()
