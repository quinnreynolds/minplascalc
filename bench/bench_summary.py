"""Consolidated numbers for the performance report."""

import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import collision_cache  # noqa: E402
import numpy as np  # noqa: E402
import pf_vectorised  # noqa: E402
from workloads import make_sico  # noqa: E402

import minplascalc.functions_transport as ft  # noqa: E402

TEMPERATURES = np.linspace(1000, 25000, 20)
LS_PAIRS = [
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
]


def sweep(fn):
    mixtures = [make_sico(sio) for sio in (0.1, 0.5, 0.9)]
    t0 = time.perf_counter()
    for mixture in mixtures:
        for T in TEMPERATURES:
            mixture.T = T
            fn(mixture)
    return time.perf_counter() - t0


def all_four(mx):
    mx.calculate_viscosity()
    mx.calculate_electrical_conductivity()
    mx.calculate_total_emission_coefficient()
    mx.calculate_thermal_conductivity()


def q_breakdown(label):
    m = make_sico(0.5)
    m.T = 12000
    m.calculate_composition()
    real = ft.Qij_mix
    cache = {(l, s): real(m, l, s) for l, s in LS_PAIRS}

    def timed(fn, n, *a):
        fn(*a)
        t0 = time.perf_counter()
        for _ in range(n):
            fn(*a)
        return (time.perf_counter() - t0) / n

    t_full = timed(ft.q, 10, m)
    ft.Qij_mix = lambda mx, l, s: cache[(l, s)]
    try:
        t_asm = timed(ft.q, 200, m)
    finally:
        ft.Qij_mix = real
    print(
        f"  {label:<34s} q()={t_full * 1e3:7.2f} ms  "
        f"collision integrals={((t_full - t_asm) * 1e3):7.2f} ms  "
        f"assembly={t_asm * 1e3:6.3f} ms ({100 * t_asm / t_full:.1f}%)"
    )


def main():
    m = make_sico(0.5)
    m.T = 10000
    all_four(m)

    print("# Where the time goes, before and after\n")
    q_breakdown("baseline")

    print("\n# Whole-workload shares (SiCO, 20 T x 3 mixtures)\n")
    for label, patches in [
        ("baseline (main)", []),
        (
            "with partition-fn + cache fixes",
            [pf_vectorised.patch, collision_cache.patch],
        ),
    ]:
        undos = [p() for p in patches]
        try:
            t_comp = sweep(lambda mx: mx.calculate_composition())
            t_all = sweep(all_four)
            print(
                f"  {label:<34s} all four={t_all:6.2f} s   "
                f"composition={t_comp:5.2f} s "
                f"({100 * t_comp / t_all:4.1f}% of total)"
            )
            if patches:
                q_breakdown("  q() after fixes")
        finally:
            for u in reversed(undos):
                u()


if __name__ == "__main__":
    main()
