"""Show the analytic derivative and the recursive finite difference agree.

Three checks:

1. For (l, s) needing no recursion, analytic == current, to roundoff.
2. For (l, s) needing recursion, shrinking the finite-difference step drives
   the current scheme onto the analytic value at the expected O(h^2) rate --
   demonstrating they are the same quantity, and that the analytic form is
   the h -> 0 limit rather than a different formula.
3. End to end, plasma properties are unchanged to the size of the finite
   difference error.
"""

import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np  # noqa: E402
import q_analytic_dropin as ad  # noqa: E402
from workloads import make_sico  # noqa: E402

import minplascalc.functions_transport as ft  # noqa: E402

LS = [
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


def pick_pairs(mixture):
    """One representative neutral-neutral and one ion-neutral pair."""
    sp = [s for s in mixture.species if s.name != "e"]
    nn = next(
        (i, j)
        for i in sp
        for j in sp
        if i.charge_number == 0 and j.charge_number == 0
    )
    inn = next(
        (i, j)
        for i in sp
        for j in sp
        if i.charge_number != 0
        and j.charge_number == 0
        and i.stoichiometry != j.stoichiometry
    )
    return nn, inn


def main():
    m = make_sico(0.5)
    m.T = 12000
    m.calculate_composition()
    (nn_i, nn_j), (in_i, in_j) = pick_pairs(m)
    print(f"# neutral-neutral pair: {nn_i.name}-{nn_j.name}")
    print(f"# ion-neutral pair:     {in_i.name}-{in_j.name}\n")

    T = 12000.0

    print("## 1. Analytic vs the current implementation, at T = 12000 K\n")
    print(
        f"  {'(l,s)':>7s} {'derivs':>7s} {'current':>14s} "
        f"{'analytic':>14s} {'rel diff':>11s}"
    )
    for l, s in LS:
        k = max(0, s - ad.BASE_S[l])
        cur = ft.Qnn(nn_i, nn_j, l, s, T)
        ana = ad.Qnn_analytic(nn_i, nn_j, l, s, T)
        print(
            f"  {f'({l},{s})':>7s} {k:>7d} {cur:14.7e} {ana:14.7e} "
            f"{abs(cur - ana) / abs(ana):11.3e}"
        )

    print("\n## 2. Finite difference -> analytic as the step shrinks\n")
    for l, s in [(1, 6), (1, 7), (2, 6), (3, 5)]:
        ana = ad.Qnn_analytic(nn_i, nn_j, l, s, T)
        print(
            f"  Omega*({l},{s}), {max(0, s - ad.BASE_S[l])} derivative(s), "
            f"analytic = {ana:.10e}"
        )
        prev = None
        for h in (0.5, 0.25, 0.125, 0.0625, 0.03125):
            fd = ad.omega_fd("nn", nn_i, nn_j, l, s, T, h=h)
            rel = abs(fd - ana) / abs(ana)
            ratio = "" if prev is None else f"  ratio {prev / rel:5.2f}"
            tag = "  <- current" if h == 0.5 else ""
            print(f"    h={h:<8g} rel err {rel:10.3e}{ratio}{tag}")
            prev = rel
        print()

    print("## 3. Effect on the ion-neutral branch (same check, Qin)\n")
    print(
        f"  {'(l,s)':>7s} {'current':>14s} {'analytic':>14s} {'rel diff':>11s}"
    )
    for l, s in [(1, 5), (1, 6), (1, 7), (2, 6), (3, 5)]:
        cur = ft.Qin(in_i, in_j, l, s, T)
        ana = ad.Qin_analytic(in_i, in_j, l, s, T)
        print(
            f"  {f'({l},{s})':>7s} {cur:14.7e} {ana:14.7e} "
            f"{abs(cur - ana) / abs(ana):11.3e}"
        )

    print("\n## 4. End-to-end effect on plasma properties\n")
    temps = np.linspace(1000, 25000, 12)

    def sweep():
        mixtures = [make_sico(s) for s in (0.1, 0.9)]
        out = []
        t0 = time.perf_counter()
        for mx in mixtures:
            for Tv in temps:
                mx.T = Tv
                out.append(
                    (
                        mx.calculate_viscosity(),
                        mx.calculate_electrical_conductivity(),
                        mx.calculate_thermal_conductivity(),
                    )
                )
        return time.perf_counter() - t0, np.array(out)

    mw = make_sico(0.5)
    mw.T = 10000
    mw.calculate_viscosity()
    mw.calculate_thermal_conductivity()

    t_cur, ref = sweep()
    undo = ad.patch()
    try:
        t_ana, got = sweep()
    finally:
        undo()

    rel = np.abs(got - ref) / np.abs(ref).max()
    print(f"  current  {t_cur:6.2f} s")
    print(f"  analytic {t_ana:6.2f} s   ({t_cur / t_ana:.2f}x)")
    print(f"  max rel change in properties: {rel.max():.3e}")
    per = [
        np.abs(got[:, k] - ref[:, k]).max() / np.abs(ref[:, k]).max()
        for k in range(3)
    ]
    print(
        f"  (viscosity {per[0]:.2e}, sigma {per[1]:.2e}, kappa {per[2]:.2e})"
    )


if __name__ == "__main__":
    main()
