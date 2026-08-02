"""Compare step controls and convergence tests for the Gibbs minimiser.

Separates the two axes that issue #16 conflates: the Newton step (sound),
the step control (the "governor"), and the convergence test.
"""

import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np  # noqa: E402
from minimiser_variants import solve  # noqa: E402

import minplascalc as mpc  # noqa: E402
import minplascalc.mixture as mx  # noqa: E402

SICO = [
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


def sico(sio, T, P, rtol=1e-10, max_iter=1000):
    sp = [mpc.species.from_name(n) for n in SICO]
    x0 = [0, 0, 0, 0, 0, 1 - sio, 0, 0, 0, 0, sio, 0, 0, 0, 0]
    return mx.LTE(sp, x0, T, P, 1e20, rtol, max_iter)


def main():
    print("# Step rule, shipped convergence test held fixed\n")
    print(f"  {'T (K)':>7s} {'governor':>10s} {'ftb':>7s} {'ratio':>7s}")
    for T in (1000.0, 2000.0, 3000.0, 5000.0, 8000.0, 12000.0, 25000.0):
        _, ia, _ = solve(sico(0.5, T, 101325.0), rule="governor")
        _, ib, _ = solve(sico(0.5, T, 101325.0), rule="ftb")
        print(f"  {T:>7.0f} {ia:>10d} {ib:>7d} {ia / ib:>6.1f}x")

    print("\n# Both axes, over 168 (composition, pressure, temperature)\n")
    print(
        f"  {'rule':>10s} {'convergence':>12s} {'converged':>12s} "
        f"{'iters':>10s} {'underflow':>10s}"
    )
    for rule in ("governor", "ftb"):
        for conv in ("shipped", "all"):
            ok = tot = iters = uf = 0
            for sio in (0.05, 0.5, 0.95):
                for P in (1013.25, 10132.5, 101325.0, 1013250.0):
                    for T in np.linspace(500, 30000, 14):
                        tot += 1
                        try:
                            _, it, c = solve(
                                sico(sio, T, P),
                                rule=rule,
                                convergence=conv,
                            )
                            ok += c
                            iters += it
                        except (FloatingPointError, np.linalg.LinAlgError):
                            uf += 1
            print(
                f"  {rule:>10s} {conv:>12s} {ok:>5d} / {tot:<4d} "
                f"{iters:>10d} {uf:>10d}"
            )
    print("\n  shipped solver, for reference: 143 / 168 converged")


if __name__ == "__main__":
    main()
