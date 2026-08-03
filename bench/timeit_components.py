"""Wall-clock (uninstrumented) timing of the major minplascalc components.

cProfile inflates the cost of code that makes many small calls, so this
script measures the same workloads with plain timers to get ground truth.
"""

import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np  # noqa: E402
from workloads import make_sico  # noqa: E402

from minplascalc import functions_transport as ft  # noqa: E402

TEMPERATURES = np.linspace(1000, 25000, 20)


def timed(fn, *a, **kw):
    t0 = time.perf_counter()
    r = fn(*a, **kw)
    return time.perf_counter() - t0, r


def sweep(label, per_T):
    """Run per_T(mixture) across the standard sweep, return total seconds."""
    mixtures = [make_sico(sio) for sio in (0.1, 0.5, 0.9)]
    t0 = time.perf_counter()
    for mixture in mixtures:
        for T in TEMPERATURES:
            mixture.T = T
            per_T(mixture)
    dt = time.perf_counter() - t0
    print(f"{label:<46s} {dt:8.3f} s")
    return dt


def main():
    # Warm the numba JIT.
    m = make_sico(0.5)
    m.T = 10000
    m.calculate_viscosity()
    m.calculate_thermal_conductivity()
    m.calculate_electrical_conductivity()
    m.calculate_total_emission_coefficient()

    print(f"# SiCO, 15 species, {len(TEMPERATURES)} T x 3 mixtures\n")

    t_comp = sweep(
        "composition only (Gibbs minimisation)",
        lambda mx: mx.calculate_composition(),
    )
    t_visc = sweep(
        "viscosity  (composition + q)", lambda mx: mx.calculate_viscosity()
    )
    t_ec = sweep(
        "electrical conductivity",
        lambda mx: mx.calculate_electrical_conductivity(),
    )
    t_em = sweep(
        "emission coefficient",
        lambda mx: mx.calculate_total_emission_coefficient(),
    )
    t_tc = sweep(
        "thermal conductivity",
        lambda mx: mx.calculate_thermal_conductivity(),
    )

    def all_four(mx):
        mx.calculate_viscosity()
        mx.calculate_electrical_conductivity()
        mx.calculate_total_emission_coefficient()
        mx.calculate_thermal_conductivity()

    # sweep() prints its own line; the total is not needed again below.
    sweep("ALL FOUR (tutorial 10 workload)", all_four)

    print()
    print(f"{'sum of parts':<46s} {t_visc + t_ec + t_em + t_tc:8.3f} s")
    print(f"{'  of which composition (shared, cached)':<46s} {t_comp:8.3f} s")

    # Isolate the collision-integral vs q-assembly split for one call.
    m = make_sico(0.5)
    m.T = 12000
    m.calculate_composition()
    n_rep = 20

    t0 = time.perf_counter()
    for _ in range(n_rep):
        for ls in [
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
            (1, 5),
            (1, 6),
            (1, 7),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 5),
            (2, 6),
            (3, 3),
            (3, 4),
            (3, 5),
            (4, 4),
        ]:
            ft.Qij_mix(m, *ls)
    t_Qmix = (time.perf_counter() - t0) / n_rep

    t0 = time.perf_counter()
    for _ in range(n_rep):
        ft.q(m)
    t_q = (time.perf_counter() - t0) / n_rep

    t0 = time.perf_counter()
    for _ in range(n_rep):
        ft.qhat(m)
    t_qhat = (time.perf_counter() - t0) / n_rep

    print("\n# single-call breakdown (mean of %d reps, T=12000 K)" % n_rep)
    print(
        f"{'  16x Qij_mix (collision integrals)':<46s} {t_Qmix * 1e3:8.2f} ms"
    )
    print(f"{'  q()   total':<46s} {t_q * 1e3:8.2f} ms")
    print(
        f"{'  q()   minus collision integrals (njit part)':<46s} "
        f"{(t_q - t_Qmix) * 1e3:8.2f} ms"
    )
    print(f"{'  qhat() total':<46s} {t_qhat * 1e3:8.2f} ms")

    # Isolate one composition solve (uncached).
    n_rep = 20
    t0 = time.perf_counter()
    for k in range(n_rep):
        mx = make_sico(0.5)
        mx.T = 12000
        mx.calculate_composition()
    t_one_comp = (time.perf_counter() - t0) / n_rep
    print(
        f"{'  one cold calculate_composition()':<46s} "
        f"{t_one_comp * 1e3:8.2f} ms"
    )

    # And one internal_partition_function call, to size the leaf cost.
    sp = [s for s in mx.species if s.name != "e"][0]
    n_rep = 20000
    t0 = time.perf_counter()
    for _ in range(n_rep):
        sp.internal_partition_function(12000.0, 0.0)
    t_ipf = (time.perf_counter() - t0) / n_rep
    print(
        f"{'  one internal_partition_function()':<46s} "
        f"{t_ipf * 1e6:8.2f} us  ({sp.name}, "
        f"{len(sp.energy_levels)} levels)"
    )


if __name__ == "__main__":
    main()
