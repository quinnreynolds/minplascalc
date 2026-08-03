"""Micro-benchmark of the q()/qhat() matrix assembly in isolation.

The collision integrals (Qij_mix) dominate q(), so this script measures the
assembly step on its own by pre-computing the Q integrals once and timing
only the q-element construction. That is the code issue #82 proposes to
vectorise, so this is the honest measurement of that option's ceiling.
"""

import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np  # noqa: E402
from workloads import make_sico  # noqa: E402

from minplascalc import functions_transport as ft  # noqa: E402
from minplascalc import units as u  # noqa: E402

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


def collect_inputs(mixture):
    """Pre-compute everything q() needs, so we can time assembly alone."""
    mixture.calculate_composition()
    Q = {f"Q{l}{s}": ft.Qij_mix(mixture, l, s) for l, s in LS_PAIRS}
    masses = np.array([sp.molar_mass / u.N_a for sp in mixture.species])
    n = mixture.calculate_composition()
    return Q, masses, len(mixture.species), n


def time_it(fn, n_rep, *args):
    fn(*args)  # warm / JIT
    t0 = time.perf_counter()
    for _ in range(n_rep):
        fn(*args)
    return (time.perf_counter() - t0) / n_rep


def main():
    n_rep = 2000
    mixture = make_sico(0.5)
    mixture.T = 12000
    Q, masses, ns, nd = collect_inputs(mixture)

    print(f"# q-matrix assembly only, {ns} species, mean of {n_rep} reps\n")

    results = {}
    for name, fn, args in [
        ("q00", ft._q00_jit, (Q["Q11"], masses, ns, nd)),
        ("q01", ft._q01_jit, (Q["Q11"], Q["Q12"], masses, ns, nd)),
        ("q02", ft._q02_jit, (Q["Q11"], Q["Q12"], Q["Q13"], masses, ns, nd)),
        (
            "q03",
            ft._q03_jit,
            (Q["Q11"], Q["Q12"], Q["Q13"], Q["Q14"], masses, ns, nd),
        ),
        (
            "q11",
            ft._q11_jit,
            (Q["Q11"], Q["Q12"], Q["Q13"], Q["Q22"], masses, ns, nd),
        ),
        (
            "q12",
            ft._q12_jit,
            (
                Q["Q11"],
                Q["Q12"],
                Q["Q13"],
                Q["Q14"],
                Q["Q22"],
                Q["Q23"],
                masses,
                ns,
                nd,
            ),
        ),
        (
            "q13",
            ft._q13_jit,
            (
                Q["Q11"],
                Q["Q12"],
                Q["Q13"],
                Q["Q14"],
                Q["Q15"],
                Q["Q22"],
                Q["Q23"],
                Q["Q24"],
                masses,
                ns,
                nd,
            ),
        ),
        (
            "q22",
            ft._q22_jit,
            (
                Q["Q11"],
                Q["Q12"],
                Q["Q13"],
                Q["Q14"],
                Q["Q15"],
                Q["Q22"],
                Q["Q23"],
                Q["Q24"],
                Q["Q33"],
                masses,
                ns,
                nd,
            ),
        ),
        (
            "q23",
            ft._q23_jit,
            (
                Q["Q11"],
                Q["Q12"],
                Q["Q13"],
                Q["Q14"],
                Q["Q15"],
                Q["Q16"],
                Q["Q22"],
                Q["Q23"],
                Q["Q24"],
                Q["Q25"],
                Q["Q33"],
                Q["Q34"],
                masses,
                ns,
                nd,
            ),
        ),
        (
            "q33",
            ft._q33_jit,
            (
                Q["Q11"],
                Q["Q12"],
                Q["Q13"],
                Q["Q14"],
                Q["Q15"],
                Q["Q16"],
                Q["Q17"],
                Q["Q22"],
                Q["Q23"],
                Q["Q24"],
                Q["Q25"],
                Q["Q26"],
                Q["Q33"],
                Q["Q34"],
                Q["Q35"],
                Q["Q44"],
                masses,
                ns,
                nd,
            ),
        ),
    ]:
        dt = time_it(fn, n_rep, *args)
        results[name] = dt
        print(f"  {name:<6s} {dt * 1e6:9.2f} us")

    print(f"\n  {'TOTAL':<6s} {sum(results.values()) * 1e6:9.2f} us")

    # Full q() including collision integrals, for context.
    t_full = time_it(ft.q, 20, mixture)
    t_qmix = 0.0
    for ls in LS_PAIRS:
        t_qmix += time_it(ft.Qij_mix, 20, mixture, *ls)
    print(f"\n  {'q() full':<22s} {t_full * 1e3:9.3f} ms")
    print(
        f"  {'  16x Qij_mix':<22s} {t_qmix * 1e3:9.3f} ms "
        f"({100 * t_qmix / t_full:.1f}%)"
    )
    print(
        f"  {'  assembly':<22s} {sum(results.values()) * 1e3:9.3f} ms "
        f"({100 * sum(results.values()) / t_full:.1f}%)"
    )


if __name__ == "__main__":
    main()
