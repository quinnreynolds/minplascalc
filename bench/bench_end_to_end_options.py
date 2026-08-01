"""End-to-end effect of options 1-4 on the tutorial-10 workload.

Swaps the ``_qXX_jit`` implementations for the numpy-vectorised (option 1)
and pytensor-compiled (option 2/3) versions and re-runs the full property
sweep, so each option is judged on whole-application wall clock rather than
on the microbenchmark of the kernel it replaces.
"""

import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np  # noqa: E402
from bench_options import SIGNATURES  # noqa: E402
from q_pytensor import LS_PAIRS, ORDER, compile_forward  # noqa: E402
from q_vectorised import VEC_FUNCS  # noqa: E402
from workloads import make_sico  # noqa: E402

import minplascalc.functions_transport as ft  # noqa: E402

TEMPERATURES = np.linspace(1000, 25000, 20)
NAMES = list(SIGNATURES)


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


def patch_vec():
    orig = {n: getattr(ft, f"_{n}_jit") for n in NAMES}
    for n in NAMES:
        setattr(ft, f"_{n}_jit", VEC_FUNCS[n])

    def undo():
        for n, f in orig.items():
            setattr(ft, f"_{n}_jit", f)

    return undo


def patch_pytensor(nb):
    """Route all ten q-elements through one compiled pytensor call.

    ``q()`` calls the ten kernels separately, so we evaluate the compiled
    graph once on the first call of a given input set and serve the rest
    from that result -- this is what a real integration would do (one
    compiled function returning the whole block).
    """
    fn, compile_s = compile_forward(nb)
    orig = {n: getattr(ft, f"_{n}_jit") for n in NAMES}
    state = {"key": None, "vals": None}

    def evaluate(Q_by_name, masses, nd):
        key = (id(masses), masses[0], nd[0], nd[-1])
        if state["key"] != key:
            args = [masses, nd] + [Q_by_name[f"Q{l}{s}"] for l, s in LS_PAIRS]
            state["vals"] = dict(zip(ORDER, fn(*args)))
            state["key"] = key
        return state["vals"]

    def make(name):
        sig = SIGNATURES[name]

        def wrapper(*args):
            *Qs, masses, nb_species, nd = args
            Q_by_name = dict(zip(sig, Qs))
            # Fill any Q matrices this element does not take from the
            # caller's frame -- q() computes all 16 before calling.
            frame = sys._getframe(1).f_locals
            for l, s in LS_PAIRS:
                k = f"Q{l}{s}"
                if k not in Q_by_name:
                    Q_by_name[k] = frame[k]
            return evaluate(Q_by_name, masses, nd)[name]

        return wrapper

    for n in NAMES:
        setattr(ft, f"_{n}_jit", make(n))

    def undo():
        for n, f in orig.items():
            setattr(ft, f"_{n}_jit", f)

    return undo, compile_s


def main():
    m = make_sico(0.5)
    m.T = 10000
    m.calculate_viscosity()
    m.calculate_thermal_conductivity()
    nb = len(m.species)

    print(
        "# tutorial-10 workload: SiCO, 20 T x 3 mixtures, all four "
        "properties\n"
    )
    t_base, ref = sweep()
    print(
        f"  {'option':<44s} {'time (s)':>9s} {'speedup':>9s} "
        f"{'max rel err':>12s}"
    )
    print(
        f"  {'4  numba njit (current main)':<44s} {t_base:9.2f} "
        f"{1.0:8.2f}x {0.0:12.1e}"
    )

    undo = patch_vec()
    try:
        t_vec, got = sweep()
    finally:
        undo()
    print(
        f"  {'1  numpy vectorised q-assembly':<44s} {t_vec:9.2f} "
        f"{t_base / t_vec:8.2f}x "
        f"{np.abs(got - ref).max() / np.abs(ref).max():12.1e}"
    )

    undo, compile_s = patch_pytensor(nb)
    try:
        t_pt, got = sweep()
    finally:
        undo()
    print(
        f"  {'2  pytensor compiled q-assembly':<44s} {t_pt:9.2f} "
        f"{t_base / t_pt:8.2f}x "
        f"{np.abs(got - ref).max() / np.abs(ref).max():12.1e}"
    )
    print(
        f"     (+ {compile_s:.1f} s one-off graph compile, per species count)"
    )


if __name__ == "__main__":
    main()
