"""Head-to-head benchmark of q-assembly implementations.

Compares, on identical inputs:

* ``loops``  -- the original pure-Python triple loops (issue #82 baseline)
* ``jit``    -- the ``@njit`` versions currently on main (option 4)
* ``vec``    -- pure-numpy broadcast vectorisation (option 1)
* ``pytensor`` / ``pytensor_grad`` -- compiled graphs (options 2 and 3),
  when pytensor is importable.

Correctness is checked against the jit implementation before timing.
"""

import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np  # noqa: E402
from q_vectorised import VEC_FUNCS  # noqa: E402
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

# Which Q matrices each q-element needs, in call order.
SIGNATURES = {
    "q00": ["Q11"],
    "q01": ["Q11", "Q12"],
    "q02": ["Q11", "Q12", "Q13"],
    "q03": ["Q11", "Q12", "Q13", "Q14"],
    "q11": ["Q11", "Q12", "Q13", "Q22"],
    "q12": ["Q11", "Q12", "Q13", "Q14", "Q22", "Q23"],
    "q13": ["Q11", "Q12", "Q13", "Q14", "Q15", "Q22", "Q23", "Q24"],
    "q22": ["Q11", "Q12", "Q13", "Q14", "Q15", "Q22", "Q23", "Q24", "Q33"],
    "q23": [
        "Q11",
        "Q12",
        "Q13",
        "Q14",
        "Q15",
        "Q16",
        "Q22",
        "Q23",
        "Q24",
        "Q25",
        "Q33",
        "Q34",
    ],
    "q33": [
        "Q11",
        "Q12",
        "Q13",
        "Q14",
        "Q15",
        "Q16",
        "Q17",
        "Q22",
        "Q23",
        "Q24",
        "Q25",
        "Q26",
        "Q33",
        "Q34",
        "Q35",
        "Q44",
    ],
}

JIT_FUNCS = {name: getattr(ft, f"_{name}_jit") for name in SIGNATURES}


def make_inputs(n_species_target=None, T=12000.0):
    mixture = make_sico(0.5)
    mixture.T = T
    nd = mixture.calculate_composition()
    Q = {f"Q{l}{s}": ft.Qij_mix(mixture, l, s) for l, s in LS_PAIRS}
    masses = np.array([sp.molar_mass / u.N_a for sp in mixture.species])
    return Q, masses, len(mixture.species), nd


def synth_inputs(nb):
    """Synthetic but well-conditioned inputs for a scaling study."""
    rng = np.random.default_rng(0)
    masses = rng.uniform(1e-27, 5e-26, nb)
    nd = rng.uniform(1e18, 1e24, nb)
    Q = {f"Q{l}{s}": rng.uniform(1e-20, 1e-18, (nb, nb)) for l, s in LS_PAIRS}
    for M in Q.values():
        M[:] = (M + M.T) / 2
    return Q, masses, nb, nd


def args_for(name, Q, masses, nb, nd):
    return tuple(Q[k] for k in SIGNATURES[name]) + (masses, nb, nd)


def bench(fn, args, n_rep):
    fn(*args)
    t0 = time.perf_counter()
    for _ in range(n_rep):
        fn(*args)
    return (time.perf_counter() - t0) / n_rep


def main():
    n_rep = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    Q, masses, nb, nd = make_inputs()

    print(
        f"# q-assembly: {nb} species (SiCO + e), T=12000 K, "
        f"mean of {n_rep} reps\n"
    )

    # --- correctness -------------------------------------------------
    print("# correctness (vec vs jit, max relative error)")
    worst = 0.0
    for name in SIGNATURES:
        a = args_for(name, Q, masses, nb, nd)
        ref = JIT_FUNCS[name](*a)
        got = VEC_FUNCS[name](*a)
        scale = np.abs(ref).max()
        err = np.abs(got - ref).max() / scale
        worst = max(worst, err)
        flag = "ok " if err < 1e-12 else "BAD"
        print(f"  {name}  {err:10.3e}  {flag}")
    print(f"  worst: {worst:.3e}\n")

    # --- timing ------------------------------------------------------
    print(
        f"  {'element':<8s} {'jit (us)':>10s} {'vec (us)':>10s} "
        f"{'vec/jit':>9s}"
    )
    tot_jit = tot_vec = 0.0
    for name in SIGNATURES:
        a = args_for(name, Q, masses, nb, nd)
        t_jit = bench(JIT_FUNCS[name], a, n_rep)
        t_vec = bench(VEC_FUNCS[name], a, n_rep)
        tot_jit += t_jit
        tot_vec += t_vec
        print(
            f"  {name:<8s} {t_jit * 1e6:10.2f} {t_vec * 1e6:10.2f} "
            f"{t_vec / t_jit:8.2f}x"
        )
    print(
        f"  {'TOTAL':<8s} {tot_jit * 1e6:10.2f} {tot_vec * 1e6:10.2f} "
        f"{tot_vec / tot_jit:8.2f}x"
    )

    # --- scaling with species count ----------------------------------
    print("\n# total assembly time vs number of species")
    print(f"  {'nb':>4s} {'jit (us)':>10s} {'vec (us)':>10s} {'vec/jit':>9s}")
    for nb_s in (6, 10, 16, 24, 40, 64):
        Qs, ms, _, nds = synth_inputs(nb_s)
        tj = tv = 0.0
        reps = max(50, n_rep // max(1, (nb_s // 8) ** 3))
        for name in SIGNATURES:
            a = args_for(name, Qs, ms, nb_s, nds)
            tj += bench(JIT_FUNCS[name], a, reps)
            tv += bench(VEC_FUNCS[name], a, reps)
        print(f"  {nb_s:4d} {tj * 1e6:10.2f} {tv * 1e6:10.2f} {tv / tj:8.2f}x")


if __name__ == "__main__":
    main()
