"""Check Dij's nb_species**2 solves collapse to one multi-RHS solve.

``Dij`` LU-factorises the q-matrix once, then runs a solve inside a double
loop over (i, j) -- 256 solves for a 16-species mixture.  The right-hand
sides are ``3 sqrt(pi) (e_i - e_j)``, which span only an nb_species-
dimensional space, and the solution is linear in the RHS.  So one solve
against the nb_species unit vectors suffices, and every c^{ji} follows by
subtraction.
"""

import sys
import time

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import numpy as np  # noqa: E402
import scipy.linalg as scl  # noqa: E402
from workloads import make_sico  # noqa: E402

from minplascalc import units as u  # noqa: E402
from minplascalc.functions_transport import delta, q  # noqa: E402


def dij_current(mixture):
    nb = len(mixture.species)
    nd = mixture.calculate_composition()
    masses = np.array([sp.molar_mass / u.N_a for sp in mixture.species])
    rho = mixture.calculate_density()
    n_tot = np.sum(nd)

    out = np.zeros((nb, nb))
    qq = q(mixture)
    lu_piv_q = scl.lu_factor(qq)
    b_vec = np.zeros(4 * nb)
    for i in range(nb):
        for j in range(nb):
            dij = np.array([delta(h, i) - delta(h, j) for h in range(nb)])
            b_vec[:nb] = 3 * np.sqrt(np.pi) * dij
            cflat = scl.lu_solve(lu_piv_q, b_vec)
            cip = cflat.reshape(4, nb)
            out[i, j] = (
                rho
                * nd[i]
                / (2 * n_tot * masses[j])
                * np.sqrt(2 * u.k_b * mixture.T / masses[i])
                * cip[0, i]
            )
    return out


def dij_multirhs(mixture, qq=None):
    nb = len(mixture.species)
    nd = mixture.calculate_composition()
    masses = np.array([sp.molar_mass / u.N_a for sp in mixture.species])
    rho = mixture.calculate_density()
    n_tot = np.sum(nd)

    if qq is None:
        qq = q(mixture)
    lu_piv_q = scl.lu_factor(qq)

    # One solve against all nb unit vectors at once.
    B = np.zeros((4 * nb, nb))
    B[:nb, :] = 3 * np.sqrt(np.pi) * np.eye(nb)
    X = scl.lu_solve(lu_piv_q, B)  # (4 nb, nb)

    # c^{ji}_{i0} for the (i, j) pair is X[i, i] - X[i, j].
    Xtop = X[:nb, :]
    c0 = np.diag(Xtop)[:, None] - Xtop  # (i, j)

    pref = (
        rho
        * nd[:, None]
        / (2 * n_tot * masses[None, :])
        * np.sqrt(2 * u.k_b * mixture.T / masses)[:, None]
    )
    return pref * c0


def main():
    m = make_sico(0.5)
    for T in (2000.0, 8000.0, 12000.0, 20000.0):
        m.T = T
        m.calculate_composition()
        a = dij_current(m)
        b = dij_multirhs(m)
        scale = np.abs(a).max()
        err = np.abs(a - b).max() / scale
        print(
            f"  T={T:7.0f}  max rel err = {err:9.3e}  "
            f"{'ok' if err < 1e-10 else 'MISMATCH'}"
        )

    m.T = 12000.0
    m.calculate_composition()
    qq = q(m)  # exclude q() cost from both sides

    n_rep = 20
    t0 = time.perf_counter()
    for _ in range(n_rep):
        dij_current(m)
    t_cur = (time.perf_counter() - t0) / n_rep

    t0 = time.perf_counter()
    for _ in range(n_rep):
        dij_multirhs(m, qq=qq)
    t_new = (time.perf_counter() - t0) / n_rep

    nb = len(m.species)
    print(f"\n  {nb} species -> {nb * nb} solves currently, 1 multi-RHS solve")
    print(f"  current   {t_cur * 1e3:8.2f} ms  (includes one q() call)")
    print(f"  multi-RHS {t_new * 1e3:8.2f} ms  (q() excluded)")

    # Solve cost alone, q() excluded from both.
    lu = scl.lu_factor(qq)
    b_vec = np.zeros(4 * nb)
    t0 = time.perf_counter()
    for _ in range(n_rep):
        for i in range(nb):
            for j in range(nb):
                dij = np.array([delta(h, i) - delta(h, j) for h in range(nb)])
                b_vec[:nb] = 3 * np.sqrt(np.pi) * dij
                scl.lu_solve(lu, b_vec).reshape(4, nb)
    t_solves = (time.perf_counter() - t0) / n_rep

    B = np.zeros((4 * nb, nb))
    B[:nb, :] = 3 * np.sqrt(np.pi) * np.eye(nb)
    t0 = time.perf_counter()
    for _ in range(n_rep):
        scl.lu_solve(lu, B)
    t_one = (time.perf_counter() - t0) / n_rep
    print(
        f"\n  solve stage only: {t_solves * 1e3:7.2f} ms -> "
        f"{t_one * 1e3:6.3f} ms  ({t_solves / t_one:.0f}x)"
    )


if __name__ == "__main__":
    main()
