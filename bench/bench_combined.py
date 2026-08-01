"""End-to-end comparison of every candidate on the tutorial-10 workload.

Reports wall-clock for the full SiCO property sweep under each change,
verifying that results are unchanged to roundoff in every case.
"""

import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import collision_cache  # noqa: E402
import numpy as np  # noqa: E402
import pf_vectorised  # noqa: E402
from workloads import make_sico  # noqa: E402

TEMPERATURES = np.linspace(1000, 25000, 20)


def sweep():
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


def main():
    m = make_sico(0.5)
    m.T = 10000
    m.calculate_viscosity()
    m.calculate_thermal_conductivity()

    print(
        "# tutorial-10 workload: SiCO, 20 T x 3 mixtures, all four "
        "properties\n"
    )

    t_base, ref = sweep()
    print(
        f"  {'variant':<44s} {'time (s)':>9s} {'speedup':>9s} "
        f"{'max rel err':>12s}"
    )
    print(
        f"  {'baseline (main, numba njit)':<44s} {t_base:9.2f} "
        f"{1.0:8.2f}x {0.0:12.1e}"
    )

    variants = [
        ("+ vectorised partition function", [pf_vectorised.patch]),
        ("+ cached collision-integral parameters", [collision_cache.patch]),
        (
            "+ both",
            [pf_vectorised.patch, collision_cache.patch],
        ),
    ]

    for label, patches in variants:
        undos = [p() for p in patches]
        try:
            t, got = sweep()
        finally:
            for u in reversed(undos):
                u()
        err = np.abs(got - ref).max() / np.abs(ref).max()
        print(f"  {label:<44s} {t:9.2f} {t_base / t:8.2f}x {err:12.1e}")


if __name__ == "__main__":
    main()
