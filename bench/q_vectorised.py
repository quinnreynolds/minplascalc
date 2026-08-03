r"""Option 1: pure-numpy vectorisation of the q-matrix elements.

This is a direct, mechanical transcription of the ``_qXX_jit`` triple loops
in ``functions_transport.py`` into broadcast numpy expressions, following
the recipe proposed in issue #82.

Every element has the shape

.. math::

    q_{ij} = 8 n_i (m_i / m_j)^p \sum_l t_1(i, l) \, t_2(i, j, l)

so the transcription builds ``(i, j, l)``-shaped arrays by broadcasting and
reduces over ``axis=2``.  Arrays are ``nb_species**3`` elements, which for a
realistic 16-species mixture is 4096 doubles -- trivial memory.
"""

import numpy as np

_n = np.newaxis


class _Ctx:
    """Broadcast views shared by every q-element expression."""

    def __init__(self, masses, number_densities):
        nb = masses.shape[0]
        idx = np.arange(nb)
        self.m_i = masses[:, _n, _n]
        self.m_j = masses[_n, :, _n]
        self.m_l = masses[_n, _n, :]
        self.n_i = number_densities[:, _n, _n]
        self.n_j = number_densities[_n, :, _n]
        self.n_l = number_densities[_n, _n, :]
        self.d_ij = (idx[:, _n, _n] == idx[_n, :, _n]).astype(np.float64)
        self.d_jl = (idx[_n, :, _n] == idx[_n, _n, :]).astype(np.float64)
        self.d_il = (idx[:, _n, _n] == idx[_n, _n, :]).astype(np.float64)
        self.d_sub = self.d_ij - self.d_jl
        self.d_add = self.d_ij + self.d_jl
        # 8 * n_i, and m_i / m_j, both as (nb, nb) for the outer prefactor.
        self.pref_base = 8 * number_densities[:, _n]
        self.mratio = masses[:, _n] / masses[_n, :]

    def pref(self, p):
        return self.pref_base * self.mratio**p


def _il(Q):
    """View a (nb, nb) collision-integral matrix as Q[i, l] over (i, j, l)."""
    return Q[:, _n, :]


def q00_vec(Q11, masses, nb_species, number_densities):
    c = _Ctx(masses, number_densities)
    term1 = c.n_l * np.sqrt(c.m_i / (c.m_i + c.m_l))
    term2 = c.n_i * np.sqrt(c.m_l / c.m_j) * c.d_sub - c.n_j * np.sqrt(
        c.m_l * c.m_j
    ) / c.m_i * (1 - c.d_il)
    return 8 * np.sum(term1 * _il(Q11) * term2, axis=2)


def q01_vec(Q11, Q12, masses, nb_species, number_densities):
    c = _Ctx(masses, number_densities)
    term1 = c.n_l * c.m_l**1.5 / (c.m_i + c.m_l) ** 1.5
    term2 = c.d_sub * (5 / 2 * _il(Q11) - 3 * _il(Q12))
    return c.pref(1.5) * np.sum(term1 * term2, axis=2)


def q02_vec(Q11, Q12, Q13, masses, nb_species, number_densities):
    c = _Ctx(masses, number_densities)
    term1 = c.n_l * c.m_l**2.5 / (c.m_i + c.m_l) ** 2.5
    term2 = c.d_sub * (35 / 8 * _il(Q11) - 21 / 2 * _il(Q12) + 6 * _il(Q13))
    return c.pref(2.5) * np.sum(term1 * term2, axis=2)


def q03_vec(Q11, Q12, Q13, Q14, masses, nb_species, number_densities):
    c = _Ctx(masses, number_densities)
    term1 = c.n_l * c.m_l**3.5 / (c.m_i + c.m_l) ** 3.5
    term2 = c.d_sub * (
        105 / 16 * _il(Q11)
        - 189 / 8 * _il(Q12)
        + 27 * _il(Q13)
        - 10 * _il(Q14)
    )
    return c.pref(3.5) * np.sum(term1 * term2, axis=2)


def q11_vec(Q11, Q12, Q13, Q22, masses, nb_species, number_densities):
    c = _Ctx(masses, number_densities)
    mj2, ml2 = c.m_j**2, c.m_l**2
    term1 = c.n_l * c.m_l**0.5 / (c.m_i + c.m_l) ** 2.5
    term2 = c.d_sub * (
        5 / 4 * (6 * mj2 + 5 * ml2) * _il(Q11)
        - 15 * ml2 * _il(Q12)
        + 12 * ml2 * _il(Q13)
    ) + c.d_add * 4 * c.m_j * c.m_l * _il(Q22)
    return c.pref(1.5) * np.sum(term1 * term2, axis=2)


def q12_vec(
    Q11, Q12, Q13, Q14, Q22, Q23, masses, nb_species, number_densities
):
    c = _Ctx(masses, number_densities)
    mj2, ml2 = c.m_j**2, c.m_l**2
    mjml = c.m_j * c.m_l
    term1 = c.n_l * c.m_l**1.5 / (c.m_i + c.m_l) ** 3.5
    term2 = c.d_sub * (
        35 / 16 * (12 * mj2 + 5 * ml2) * _il(Q11)
        - 63 / 2 * (mj2 + 5 / 4 * ml2) * _il(Q12)
        + 57 * ml2 * _il(Q13)
        - 30 * ml2 * _il(Q14)
    ) + c.d_add * (14 * mjml * _il(Q22) - 16 * mjml * _il(Q23))
    return c.pref(2.5) * np.sum(term1 * term2, axis=2)


def q13_vec(
    Q11,
    Q12,
    Q13,
    Q14,
    Q15,
    Q22,
    Q23,
    Q24,
    masses,
    nb_species,
    number_densities,
):
    c = _Ctx(masses, number_densities)
    mj2, ml2 = c.m_j**2, c.m_l**2
    term1 = c.n_l * c.m_l**2.5 / (c.m_i + c.m_l) ** 4.5
    term2 = c.d_sub * (
        105 / 32 * (18 * mj2 + 5 * ml2) * _il(Q11)
        - 63 / 4 * (9 * mj2 + 5 * ml2) * _il(Q12)
        + 81 * (mj2 + 2 * ml2) * _il(Q13)
        - 160 * ml2 * _il(Q14)
        + 60 * ml2 * _il(Q15)
    ) + c.d_add * c.m_j * c.m_l * (
        63 / 2 * _il(Q22) - 72 * _il(Q23) + 40 * _il(Q24)
    )
    return c.pref(3.5) * np.sum(term1 * term2, axis=2)


def q22_vec(
    Q11,
    Q12,
    Q13,
    Q14,
    Q15,
    Q22,
    Q23,
    Q24,
    Q33,
    masses,
    nb_species,
    number_densities,
):
    c = _Ctx(masses, number_densities)
    mj2, ml2 = c.m_j**2, c.m_l**2
    mj4, ml4 = mj2**2, ml2**2
    mjml2 = (c.m_j * c.m_l) ** 2
    term1 = c.n_l * c.m_l**0.5 / (c.m_i + c.m_l) ** 4.5
    term2 = c.d_sub * (
        35 / 64 * (40 * mj4 + 168 * mjml2 + 35 * ml4) * _il(Q11)
        - 21 / 8 * ml2 * (84 * mj2 + 35 * ml2) * _il(Q12)
        + 3 / 2 * ml2 * (108 * mj2 + 133 * ml2) * _il(Q13)
        - 210 * ml4 * _il(Q14)
        + 90 * ml4 * _il(Q15)
        + 24 * mjml2 * _il(Q33)
    ) + c.d_add * (
        7 * c.m_j * c.m_l * (4 * (mj2 + 7 * ml2)) * _il(Q22)
        - 112 * c.m_j * c.m_l**3 * _il(Q23)
        + 80 * c.m_j * c.m_l**3 * _il(Q24)
    )
    return c.pref(2.5) * np.sum(term1 * term2, axis=2)


def q23_vec(
    Q11,
    Q12,
    Q13,
    Q14,
    Q15,
    Q16,
    Q22,
    Q23,
    Q24,
    Q25,
    Q33,
    Q34,
    masses,
    nb_species,
    number_densities,
):
    c = _Ctx(masses, number_densities)
    mj2, ml2 = c.m_j**2, c.m_l**2
    mj4, ml4 = mj2**2, ml2**2
    mjml2 = (c.m_j * c.m_l) ** 2
    term1 = c.n_l * c.m_l**1.5 / (c.m_i + c.m_l) ** 5.5
    term2 = c.d_sub * (
        105 / 128 * (120 * mj4 + 252 * mjml2 + 35 * ml4) * _il(Q11)
        - 63 / 64 * (120 * mj4 + 756 * mjml2 + 175 * ml4) * _il(Q12)
        + 9 / 4 * ml2 * (450 * mj2 + 217 * ml2) * _il(Q13)
        - 5 / 2 * ml2 * (198 * mj2 + 301 * ml2) * _il(Q14)
        + 615 * ml4 * _il(Q15)
        - 210 * ml4 * _il(Q16)
        + 108 * mjml2 * _il(Q33)
        - 120 * mjml2 * _il(Q34)
    ) + c.d_add * (
        63 / 4 * c.m_j * c.m_l * (8 * (mj2 + 7 * ml2)) * _il(Q22)
        - 18 * c.m_j * c.m_l * (8 * mj2 + 21 * ml2) * _il(Q23)
        + 500 * c.m_j * c.m_l**3 * _il(Q24)
        - 240 * c.m_j * c.m_l**3 * _il(Q25)
    )
    return c.pref(3.5) * np.sum(term1 * term2, axis=2)


def q33_vec(
    Q11,
    Q12,
    Q13,
    Q14,
    Q15,
    Q16,
    Q17,
    Q22,
    Q23,
    Q24,
    Q25,
    Q26,
    Q33,
    Q34,
    Q35,
    Q44,
    masses,
    nb_species,
    number_densities,
):
    c = _Ctx(masses, number_densities)
    mj2, ml2 = c.m_j**2, c.m_l**2
    mj4, ml4 = mj2**2, ml2**2
    mj6, ml6 = mj2**3, ml2**3
    mjml2 = (c.m_j * c.m_l) ** 2
    mjml3 = (c.m_j * c.m_l) ** 3
    term1 = c.n_l * c.m_l**0.5 / (c.m_i + c.m_l) ** 6.5
    term2 = c.d_sub * (
        105
        / 256
        * (112 * mj6 + 1080 * mj4 * ml2 + 1134 * mj2 * ml4 + 105 * ml6)
        * _il(Q11)
        - 567 / 64 * ml2 * (120 * mj4 + 252 * mjml2 + 35 * ml4) * _il(Q12)
        + 27 / 16 * ml2 * (440 * mj4 + 2700 * mjml2 + 651 * ml4) * _il(Q13)
        - 15 / 2 * ml4 * (594 * mj2 + 301 * ml2) * _il(Q14)
        + 135 / 2 * ml4 * (26 * mj2 + 41 * ml2) * _il(Q15)
        - 1890 * ml6 * _il(Q16)
        + 560 * ml6 * _il(Q17)
        + 18 * mjml2 * (10 * mj2 + 27 * ml2) * _il(Q33)
        - 1080 * mj2 * ml4 * _il(Q34)
        + 720 * mj2 * ml4 * _il(Q35)
    ) + c.d_add * (
        189 / 16 * c.m_j * c.m_l * (8 * mj4 + 48 * mjml2 + 21 * ml4) * _il(Q22)
        - 162 * c.m_j * c.m_l**3 * (8 * mj2 + 7 * ml2) * _il(Q23)
        + 10 * c.m_j * c.m_l**3 * (88 * mj2 + 225 * ml2) * _il(Q24)
        - 2160 * c.m_j * c.m_l**5 * _il(Q25)
        + 840 * c.m_j * c.m_l**5 * _il(Q26)
        + 64 * mjml3 * _il(Q44)
    )
    return c.pref(3.5) * np.sum(term1 * term2, axis=2)


VEC_FUNCS = {
    "q00": q00_vec,
    "q01": q01_vec,
    "q02": q02_vec,
    "q03": q03_vec,
    "q11": q11_vec,
    "q12": q12_vec,
    "q13": q13_vec,
    "q22": q22_vec,
    "q23": q23_vec,
    "q33": q33_vec,
}
