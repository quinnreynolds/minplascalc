import sys
import time

sys.path.insert(0, "bench")
import numpy as np
import pf_vectorised
import qij_memo
from workloads import make_sico

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


m = make_sico(0.5)
m.T = 10000
m.calculate_viscosity()
m.calculate_thermal_conductivity()

t0, ref = sweep()
print(
    f"  {'variant':<44s} {'time (s)':>9s} {'speedup':>9s} {'max rel err':>12s}"
)
print(f"  {'baseline (main)':<44s} {t0:9.2f} {1.0:8.2f}x {0.0:12.1e}")

for label, patches in [
    ("+ Qij_mix memoised on mixture state", [qij_memo.patch]),
    (
        "+ that and vectorised partition function",
        [qij_memo.patch, pf_vectorised.patch],
    ),
]:
    undos = [p() for p in patches]
    try:
        t, got = sweep()
        st = qij_memo.stats()
    finally:
        for u in reversed(undos):
            u()
    err = np.abs(got - ref).max() / np.abs(ref).max()
    print(f"  {label:<44s} {t:9.2f} {t0 / t:8.2f}x {err:12.1e}")
    print(
        f"       Qij_mix: {st['miss']} computed, "
        f"{st['hit']} from cache "
        f"({100 * st['hit'] / (st['hit'] + st['miss']):.0f}% redundant)"
    )
