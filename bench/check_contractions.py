import sys

sys.path.insert(0, "bench")
import numpy as np

rng = np.random.default_rng(0)
n = 16
masses = rng.random(n)
hv = rng.random(n)
D = rng.random((n, n))
dxdT = rng.random(n)

# thermal_conductivity krxn_enth double loop
loop = 0.0
for j in range(n):
    for i in range(n):
        loop += masses[j] * masses[i] * hv[i] * D[i, j] * dxdT[j]
vec = (masses * hv) @ D @ (masses * dxdT)
print(
    f"  krxn_enth   loop={loop:.15e}  contraction={vec:.15e}  "
    f"rel={abs(loop - vec) / abs(loop):.2e}"
)

# electrical_conductivity sum loop
q = rng.integers(-1, 3, n).astype(float)
nd = rng.random(n)
D1 = rng.random(n)
loop = 0.0
for cn, D1j, mj, nj in zip(q, D1, masses, nd):
    loop += nj * mj * cn * D1j
vec = float(nd @ (masses * q * D1))
print(
    f"  sigma sum   loop={loop:.15e}  contraction={vec:.15e}  "
    f"rel={abs(loop - vec) / abs(loop):.2e}"
)

# Dij's dij vector built with njit delta in a python listcomp
from minplascalc.functions_transport import delta  # noqa: E402

I = np.eye(n)
ok = all(
    np.array_equal(
        np.array([delta(h, i) - delta(h, j) for h in range(n)]),
        I[:, i] - I[:, j],
    )
    for i in range(n)
    for j in range(n)
)
print(f"  delta listcomp == np.eye columns: {ok}")
