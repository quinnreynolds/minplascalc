import sys
import time

sys.path.insert(0, "bench")
import numpy as np
from q_analytic_compact import BASE_S, omega_star
from q_analytic_derivative import omega_star_fn

rng = np.random.default_rng(0)
print("  compact vs sympy-lambdified (random coefficients & x)\n")
print(f"  {'l':>3s} {'s':>3s} {'k':>3s} {'max rel err':>13s}")
worst = 0.0
for l, s0 in BASE_S.items():
    for k in (0, 1, 2):
        errs = []
        for _ in range(300):
            a = np.concatenate(
                [
                    rng.uniform(-4, 4, 3),
                    rng.uniform(0.2, 1.5, 1),
                    rng.uniform(-1, 1, 1),
                    rng.uniform(-4, 4, 1),
                    rng.uniform(0.2, 1.5, 1),
                ]
            )
            x = rng.uniform(-3, 4)
            ref = omega_star_fn(l, s0 + k)(x, *a)
            got = omega_star(x, a, k, s0)
            errs.append(abs(got - ref) / abs(ref))
        e = max(errs)
        worst = max(worst, e)
        print(f"  {l:>3d} {s0 + k:>3d} {k:>3d} {e:13.3e}")
print(f"\n  worst: {worst:.3e}")

# speed of the two forms
a = np.array([0.78, -0.024, 0.5, 0.9, -0.34, 0.42, 0.32])
n = 20000
for label, fn in [
    ("sympy lambdified (cse)", lambda: omega_star_fn(1, 7)(1.5, *a)),
    ("compact hand form   ", lambda: omega_star(1.5, a, 2, 5)),
]:
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    print(f"  {label}: {(time.perf_counter() - t0) / n * 1e6:6.2f} us/call")
