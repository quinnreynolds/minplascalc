import sys
import time

sys.path.insert(0, "bench")
import numpy as np
import pf_vectorised
import q_analytic_dropin as ad
import qij_memo
from q_analytic_derivative import BASE_S
from workloads import make_sico

QHAT_LS = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (2, 4), (3, 3)]
rec = [(l, s) for l, s in QHAT_LS if s > BASE_S[l]]
print(
    f"  qhat() (l,s) needing recursion: {rec or 'none'} "
    f"-> viscosity is unaffected by the derivative change\n"
)

TEMPS = np.linspace(1000, 25000, 20)


def sweep():
    ms = [make_sico(s) for s in (0.1, 0.5, 0.9)]
    out = []
    t0 = time.perf_counter()
    for m in ms:
        for T in TEMPS:
            m.T = T
            out.append(
                (
                    m.calculate_viscosity(),
                    m.calculate_electrical_conductivity(),
                    m.calculate_total_emission_coefficient(),
                    m.calculate_thermal_conductivity(),
                )
            )
    return time.perf_counter() - t0, np.array(out)


w = make_sico(0.5)
w.T = 10000
w.calculate_viscosity()
w.calculate_thermal_conductivity()

t0, ref = sweep()
print(
    f"  {'variant':<46s} {'time (s)':>9s} {'speedup':>9s} {'max rel err':>12s}"
)
print(f"  {'baseline (main)':<46s} {t0:9.2f} {1.0:8.2f}x {0.0:12.1e}")
for label, ps in [
    ("analytic (l,s) derivative", [ad.patch]),
    ("+ Qij_mix memoised", [ad.patch, qij_memo.patch]),
    (
        "+ vectorised partition function",
        [ad.patch, qij_memo.patch, pf_vectorised.patch],
    ),
]:
    undos = [p() for p in ps]
    try:
        t, got = sweep()
    finally:
        for u in reversed(undos):
            u()
    err = np.abs(got - ref).max() / np.abs(ref).max()
    print(f"  {label:<46s} {t:9.2f} {t0 / t:8.2f}x {err:12.1e}")
