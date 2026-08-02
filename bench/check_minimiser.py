"""Robustness probe for the hand-rolled Gibbs minimiser (issue #16).

Issue #16 asks whether scipy could replace the custom minimiser, on
robustness grounds.  This measures where the current one actually stands:

1. Does it converge across a wide range of conditions, and how often does
   the relaxation ("governor") ladder have to step down?
2. Are the hard constraints -- element balance and charge neutrality --
   satisfied at the returned composition?
3. The convergence test looks at only the most abundant species
   (``mixture.py`` has a TODO about this).  Are the minor species converged
   when it declares success?  Measured by re-solving at a much tighter
   tolerance and comparing per species.
4. Would warm-starting from the previous temperature help?
"""

import sys
import warnings

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np  # noqa: E402

import minplascalc as mpc  # noqa: E402
import minplascalc.mixture as mxmod  # noqa: E402

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
    species = [mpc.species.from_name(n) for n in SICO]
    x0 = [0, 0, 0, 0, 0, 1 - sio, 0, 0, 0, 0, sio, 0, 0, 0, 0]
    return mxmod.LTE(species, x0, T, P, 1e20, rtol, max_iter)


def charge_imbalance(mixture, nd):
    """Net charge relative to the electron density.

    Normalising by n_e rather than by the total charge magnitude keeps the
    number meaningful: at low temperature both are near zero and a ratio of
    the two is noise.
    """
    charge = np.array([sp.charge_number for sp in mixture.species])
    n_e = nd[-1]
    return abs(charge @ nd) / n_e if n_e > 0 else 0.0


def main():
    print("# 1. Convergence across conditions\n")
    n_fail = 0
    n_total = 0
    governor_used = []
    real_solve = np.linalg.solve
    counter = {"n": 0}

    def counting(a, b):
        counter["n"] += 1
        return real_solve(a, b)

    mxmod.np.linalg.solve = counting
    try:
        for sio in (0.05, 0.5, 0.95):
            for P in (1013.25, 10132.5, 101325.0, 1013250.0):
                for T in np.linspace(500, 30000, 14):
                    m = sico(sio, T, P)
                    counter["n"] = 0
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        m.calculate_composition()
                    n_total += 1
                    if any("converged" in str(x.message) for x in w):
                        n_fail += 1
                        print(
                            f"    NOT CONVERGED: sio={sio} P={P:g} T={T:.0f}"
                        )
                    governor_used.append(counter["n"])
    finally:
        mxmod.np.linalg.solve = real_solve

    it = np.array(governor_used)
    print(f"  {n_total} states, {n_fail} failed to converge")
    print(
        f"  Newton iterations: min {it.min()}, median "
        f"{int(np.median(it))}, max {it.max()}"
    )
    print(
        f"  states needing > 1000 iterations (governor step-down): "
        f"{(it > 1000).sum()}"
    )

    print("\n# 2. Charge neutrality at the returned composition\n")
    print(f"  {'T (K)':>7s} {'n_e/n_tot':>11s} {'|sum z_i n_i| / n_e':>21s}")
    for T in (2000.0, 5000.0, 8000.0, 12000.0, 18000.0, 25000.0):
        m = sico(0.5, T, 101325.0)
        nd = m.calculate_composition()
        print(
            f"  {T:>7.0f} {nd[-1] / nd.sum():>11.2e} "
            f"{charge_imbalance(m, nd):>21.3e}"
        )

    print("\n# 3. Are the minor species converged when it stops?\n")
    print(
        f"  {'T (K)':>7s} {'species':>7s} {'x_i':>11s} "
        f"{'rel err vs tight solve':>24s}"
    )
    for T in (1000.0, 3000.0, 12000.0, 25000.0):
        loose = sico(0.5, T, 101325.0, rtol=1e-10)
        tight = sico(0.5, T, 101325.0, rtol=1e-14, max_iter=20000)
        a = loose.calculate_composition()
        b = tight.calculate_composition()
        rel = np.abs(a - b) / np.maximum(b, 1e-300)
        x = b / b.sum()
        order = np.argsort(-rel)[:3]
        for k in order:
            name = loose.species[k].name
            print(f"  {T:>7.0f} {name:>7s} {x[k]:>11.3e} {rel[k]:>24.3e}")
        print(f"  {'':>7s} {'(worst over all species)':<32s} {rel.max():.3e}")

    print("\n# 4. Warm start from the previous temperature\n")
    temps = np.linspace(1000, 25000, 25)

    cold = []
    for T in temps:
        m = sico(0.5, T, 101325.0)
        counter["n"] = 0
        mxmod.np.linalg.solve = counting
        try:
            m.calculate_composition()
        finally:
            mxmod.np.linalg.solve = real_solve
        cold.append(counter["n"])

    print(
        f"  cold start at every T (current behaviour): "
        f"{sum(cold)} Newton iterations total"
    )
    print(
        f"    per T: min {min(cold)}, median "
        f"{int(np.median(cold))}, max {max(cold)}"
    )
    print("  A sweep re-solves from gfe_initial_particles at each T, so the")
    print("  previous temperature's answer -- an excellent guess -- is")
    print("  discarded. The low-T end is where it costs most.")


if __name__ == "__main__":
    main()
