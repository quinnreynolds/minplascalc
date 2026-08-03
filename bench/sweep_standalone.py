"""Self-contained tutorial-10 sweep, runnable against any minplascalc rev."""

import sys
import time

import numpy as np

import minplascalc as mpc

SPECIES = [
    "O2",
    "O2+",
    "O",
    "O+",
    "O++",
    "CO",
    "CO+",
    "C",
    "C+",
    "C++",
    "SiO",
    "SiO+",
    "Si",
    "Si+",
    "Si++",
]
TEMPERATURES = np.linspace(1000, 25000, 20)


def x0(sio):
    return [0, 0, 0, 0, 0, 1 - sio, 0, 0, 0, 0, sio, 0, 0, 0, 0]


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "current"

    # warm up (JIT compile, if any)
    w = mpc.mixture.lte_from_names(SPECIES, x0(0.5), 10000, 101325)
    w.calculate_viscosity()
    w.calculate_electrical_conductivity()
    w.calculate_total_emission_coefficient()
    w.calculate_thermal_conductivity()

    mixtures = [
        mpc.mixture.lte_from_names(SPECIES, x0(s), 1000, 101325)
        for s in (0.1, 0.5, 0.9)
    ]
    out = []
    t0 = time.perf_counter()
    for m in mixtures:
        for T in TEMPERATURES:
            m.T = T
            out.append(
                (
                    m.calculate_viscosity(),
                    m.calculate_electrical_conductivity(),
                    m.calculate_total_emission_coefficient(),
                    m.calculate_thermal_conductivity(),
                )
            )
    dt = time.perf_counter() - t0
    a = np.array(out)
    print(f"{label}: {dt:.2f} s   checksum={np.abs(a).sum():.12e}")


if __name__ == "__main__":
    main()
