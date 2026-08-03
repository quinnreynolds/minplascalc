"""Version-agnostic benchmark of q()/qhat() assembly cost.

Works on any revision of minplascalc by monkey-patching ``Qij_mix`` to
return pre-computed collision integrals, so that ``q(mixture)`` measures
only the matrix-assembly arithmetic -- the code issue #82 targets.

Run the same script from a git worktree of an older revision to compare
loop / jit / vectorised implementations on equal terms.
"""

import json
import sys
import time

import numpy as np

from minplascalc import functions_transport as ft
from minplascalc import mixture as mpc_mixture

SICO_SPECIES = [
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
SICO_X0 = [0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0]

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


def bench(fn, n_rep, *args):
    fn(*args)  # warm up (JIT compile etc.)
    t0 = time.perf_counter()
    for _ in range(n_rep):
        fn(*args)
    return (time.perf_counter() - t0) / n_rep


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "current"
    n_rep = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    mixture = mpc_mixture.lte_from_names(SICO_SPECIES, SICO_X0, 12000, 101325)
    mixture.calculate_composition()

    real_Qij_mix = ft.Qij_mix

    # --- full q() and qhat(), including collision integrals -------------
    t_q_full = bench(ft.q, 5, mixture)
    t_qhat_full = bench(ft.qhat, 5, mixture)
    t_Qmix = sum(bench(real_Qij_mix, 5, mixture, l, s) for l, s in LS_PAIRS)

    # --- assembly only: freeze the collision integrals ------------------
    cache = {(l, s): real_Qij_mix(mixture, l, s) for l, s in LS_PAIRS}
    ft.Qij_mix = lambda mx, l, s: cache[(l, s)]
    try:
        t_q_asm = bench(ft.q, n_rep, mixture)
        t_qhat_asm = bench(ft.qhat, n_rep, mixture)
    finally:
        ft.Qij_mix = real_Qij_mix

    out = {
        "label": label,
        "n_species": len(mixture.species),
        "q_full_ms": t_q_full * 1e3,
        "qhat_full_ms": t_qhat_full * 1e3,
        "Qij_mix_x16_ms": t_Qmix * 1e3,
        "q_assembly_ms": t_q_asm * 1e3,
        "qhat_assembly_ms": t_qhat_asm * 1e3,
        "assembly_pct_of_q": 100 * t_q_asm / t_q_full,
    }

    print(f"\n=== {label} ({out['n_species']} species) ===")
    print(f"  q() full                 {out['q_full_ms']:9.3f} ms")
    print(f"    16x Qij_mix            {out['Qij_mix_x16_ms']:9.3f} ms")
    print(
        f"    q assembly only        {out['q_assembly_ms']:9.3f} ms  "
        f"({out['assembly_pct_of_q']:.1f}% of q())"
    )
    print(f"  qhat() full              {out['qhat_full_ms']:9.3f} ms")
    print(f"    qhat assembly only     {out['qhat_assembly_ms']:9.3f} ms")

    # Checksum so we can verify numerical equivalence across revisions.
    ft.Qij_mix = lambda mx, l, s: cache[(l, s)]
    try:
        qq = ft.q(mixture)
        qh = ft.qhat(mixture)
    finally:
        ft.Qij_mix = real_Qij_mix
    out["q_checksum"] = float(np.abs(qq).sum())
    out["qhat_checksum"] = float(np.abs(qh).sum())
    print(
        f"  checksum q={out['q_checksum']:.10e} "
        f"qhat={out['qhat_checksum']:.10e}"
    )

    with open(f"/tmp/asm_{label}.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
