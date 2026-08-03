"""Benchmark options 2 and 3 (pytensor) against options 1 and 4."""

import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np  # noqa: E402
from bench_options import (  # noqa: E402
    JIT_FUNCS,
    SIGNATURES,
    args_for,
    bench,
    make_inputs,
    synth_inputs,
)
from q_pytensor import (  # noqa: E402
    LS_PAIRS,
    ORDER,
    compile_forward,
    compile_with_grad,
)
from q_vectorised import VEC_FUNCS  # noqa: E402


def pt_args(Q, masses, nd):
    return [masses, nd] + [Q[f"Q{l}{s}"] for l, s in LS_PAIRS]


def run_for(nb, Q, masses, nd, n_rep, label):
    print(f"\n=== {label}: {nb} species ===")

    # Reference from the jit implementation.
    ref = {
        name: JIT_FUNCS[name](*args_for(name, Q, masses, nb, nd))
        for name in SIGNATURES
    }

    t_jit = sum(
        bench(JIT_FUNCS[n], args_for(n, Q, masses, nb, nd), n_rep)
        for n in SIGNATURES
    )
    t_vec = sum(
        bench(VEC_FUNCS[n], args_for(n, Q, masses, nb, nd), n_rep)
        for n in SIGNATURES
    )

    fwd, c_fwd = compile_forward(nb)
    grd, c_grd = compile_with_grad(nb)
    a = pt_args(Q, masses, nd)

    # correctness of the compiled forward pass
    got = fwd(*a)
    worst = max(
        np.abs(g - ref[k]).max() / max(np.abs(ref[k]).max(), 1e-300)
        for g, k in zip(got, ORDER)
    )

    t_pt = bench(fwd, a, n_rep)
    t_ptg = bench(grd, a, n_rep)

    print(
        f"  compile time  forward {c_fwd:6.2f} s   forward+grad {c_grd:6.2f} s"
    )
    print(f"  forward max rel err vs jit: {worst:.3e}")
    print(f"  {'option':<34s} {'time (us)':>11s} {'vs jit':>9s}")
    print(
        f"  {'4  numba njit (current main)':<34s} "
        f"{t_jit * 1e6:11.2f} {1.0:8.2f}x"
    )
    print(
        f"  {'1  numpy vectorised':<34s} "
        f"{t_vec * 1e6:11.2f} {t_vec / t_jit:8.2f}x"
    )
    print(
        f"  {'2  pytensor compiled':<34s} "
        f"{t_pt * 1e6:11.2f} {t_pt / t_jit:8.2f}x"
    )
    print(
        f"  {'3  pytensor + grad':<34s} "
        f"{t_ptg * 1e6:11.2f} {t_ptg / t_jit:8.2f}x"
    )
    print(
        f"  {'   (grad overhead alone)':<34s} "
        f"{(t_ptg - t_pt) * 1e6:11.2f} {'':>9s}"
    )
    return {
        "nb": nb,
        "jit": t_jit,
        "vec": t_vec,
        "pt": t_pt,
        "ptg": t_ptg,
        "compile_fwd": c_fwd,
        "compile_grad": c_grd,
        "err": worst,
    }


def main():
    n_rep = int(sys.argv[1]) if len(sys.argv) > 1 else 500

    Q, masses, nb, nd = make_inputs()
    rows = [run_for(nb, Q, masses, nd, n_rep, "real SiCO mixture")]

    for nb_s in (6, 24, 40, 64):
        Qs, ms, _, nds = synth_inputs(nb_s)
        reps = max(50, n_rep // max(1, (nb_s // 10) ** 2))
        rows.append(run_for(nb_s, Qs, ms, nds, reps, "synthetic"))

    print("\n\n# summary: total q-assembly time (us)")
    print(
        f"  {'nb':>4s} {'jit':>10s} {'vec':>10s} {'pytensor':>10s} "
        f"{'pt+grad':>10s} {'compile(s)':>11s}"
    )
    for r in rows:
        print(
            f"  {r['nb']:4d} {r['jit'] * 1e6:10.1f} {r['vec'] * 1e6:10.1f} "
            f"{r['pt'] * 1e6:10.1f} {r['ptg'] * 1e6:10.1f} "
            f"{r['compile_fwd']:11.1f}"
        )


if __name__ == "__main__":
    main()
