"""Options 2 and 3: compile the q-assembly with pytensor.

Option 2 builds a symbolic graph of the same broadcast expressions used in
``q_vectorised`` and compiles it once with ``pytensor.function``.  Option 3
additionally asks pytensor for derivative information (the Jacobian of a
scalar reduction of the q-matrix with respect to the number densities), to
measure what automatic differentiation of this code path costs.

The graph is built for a *fixed* number of species, because the shapes and
the Kronecker deltas are baked in as constants; that is the compile-time /
run-time tradeoff this option carries.
"""

import time

import numpy as np
import pytensor
import pytensor.tensor as pt

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

_n = np.newaxis


def build_graph(nb):
    """Symbolic q00..q33 for a fixed species count ``nb``."""
    masses = pt.dvector("masses")
    nd = pt.dvector("nd")
    Qs = {f"Q{l}{s}": pt.dmatrix(f"Q{l}{s}") for l, s in LS_PAIRS}

    m_i = masses[:, _n, _n]
    m_j = masses[_n, :, _n]
    m_l = masses[_n, _n, :]
    n_i = nd[:, _n, _n]
    n_j = nd[_n, :, _n]
    n_l = nd[_n, _n, :]

    eye = np.eye(nb)
    d_ij = pt.constant(eye[:, :, _n])
    d_jl = pt.constant(eye[_n, :, :])
    d_il = pt.constant(eye[:, _n, :])
    d_sub = d_ij - d_jl
    d_add = d_ij + d_jl

    def il(name):
        return Qs[name][:, _n, :]

    def pref(p):
        return 8 * nd[:, _n] * (masses[:, _n] / masses[_n, :]) ** p

    mj2, ml2 = m_j**2, m_l**2
    mj4, ml4 = mj2**2, ml2**2
    mj6, ml6 = mj2**3, ml2**3
    mjml = m_j * m_l
    mjml2 = mjml**2
    mjml3 = mjml**3

    out = {}

    t1 = n_l * pt.sqrt(m_i / (m_i + m_l))
    t2 = n_i * pt.sqrt(m_l / m_j) * d_sub - n_j * pt.sqrt(m_l * m_j) / m_i * (
        1 - d_il
    )
    out["q00"] = 8 * pt.sum(t1 * il("Q11") * t2, axis=2)

    t1 = n_l * m_l**1.5 / (m_i + m_l) ** 1.5
    t2 = d_sub * (5 / 2 * il("Q11") - 3 * il("Q12"))
    out["q01"] = pref(1.5) * pt.sum(t1 * t2, axis=2)

    t1 = n_l * m_l**2.5 / (m_i + m_l) ** 2.5
    t2 = d_sub * (35 / 8 * il("Q11") - 21 / 2 * il("Q12") + 6 * il("Q13"))
    out["q02"] = pref(2.5) * pt.sum(t1 * t2, axis=2)

    t1 = n_l * m_l**3.5 / (m_i + m_l) ** 3.5
    t2 = d_sub * (
        105 / 16 * il("Q11")
        - 189 / 8 * il("Q12")
        + 27 * il("Q13")
        - 10 * il("Q14")
    )
    out["q03"] = pref(3.5) * pt.sum(t1 * t2, axis=2)

    t1 = n_l * m_l**0.5 / (m_i + m_l) ** 2.5
    t2 = d_sub * (
        5 / 4 * (6 * mj2 + 5 * ml2) * il("Q11")
        - 15 * ml2 * il("Q12")
        + 12 * ml2 * il("Q13")
    ) + d_add * 4 * mjml * il("Q22")
    out["q11"] = pref(1.5) * pt.sum(t1 * t2, axis=2)

    t1 = n_l * m_l**1.5 / (m_i + m_l) ** 3.5
    t2 = d_sub * (
        35 / 16 * (12 * mj2 + 5 * ml2) * il("Q11")
        - 63 / 2 * (mj2 + 5 / 4 * ml2) * il("Q12")
        + 57 * ml2 * il("Q13")
        - 30 * ml2 * il("Q14")
    ) + d_add * (14 * mjml * il("Q22") - 16 * mjml * il("Q23"))
    out["q12"] = pref(2.5) * pt.sum(t1 * t2, axis=2)

    t1 = n_l * m_l**2.5 / (m_i + m_l) ** 4.5
    t2 = d_sub * (
        105 / 32 * (18 * mj2 + 5 * ml2) * il("Q11")
        - 63 / 4 * (9 * mj2 + 5 * ml2) * il("Q12")
        + 81 * (mj2 + 2 * ml2) * il("Q13")
        - 160 * ml2 * il("Q14")
        + 60 * ml2 * il("Q15")
    ) + d_add * mjml * (63 / 2 * il("Q22") - 72 * il("Q23") + 40 * il("Q24"))
    out["q13"] = pref(3.5) * pt.sum(t1 * t2, axis=2)

    t1 = n_l * m_l**0.5 / (m_i + m_l) ** 4.5
    t2 = d_sub * (
        35 / 64 * (40 * mj4 + 168 * mjml2 + 35 * ml4) * il("Q11")
        - 21 / 8 * ml2 * (84 * mj2 + 35 * ml2) * il("Q12")
        + 3 / 2 * ml2 * (108 * mj2 + 133 * ml2) * il("Q13")
        - 210 * ml4 * il("Q14")
        + 90 * ml4 * il("Q15")
        + 24 * mjml2 * il("Q33")
    ) + d_add * (
        7 * mjml * (4 * (mj2 + 7 * ml2)) * il("Q22")
        - 112 * m_j * m_l**3 * il("Q23")
        + 80 * m_j * m_l**3 * il("Q24")
    )
    out["q22"] = pref(2.5) * pt.sum(t1 * t2, axis=2)

    t1 = n_l * m_l**1.5 / (m_i + m_l) ** 5.5
    t2 = d_sub * (
        105 / 128 * (120 * mj4 + 252 * mjml2 + 35 * ml4) * il("Q11")
        - 63 / 64 * (120 * mj4 + 756 * mjml2 + 175 * ml4) * il("Q12")
        + 9 / 4 * ml2 * (450 * mj2 + 217 * ml2) * il("Q13")
        - 5 / 2 * ml2 * (198 * mj2 + 301 * ml2) * il("Q14")
        + 615 * ml4 * il("Q15")
        - 210 * ml4 * il("Q16")
        + 108 * mjml2 * il("Q33")
        - 120 * mjml2 * il("Q34")
    ) + d_add * (
        63 / 4 * mjml * (8 * (mj2 + 7 * ml2)) * il("Q22")
        - 18 * mjml * (8 * mj2 + 21 * ml2) * il("Q23")
        + 500 * m_j * m_l**3 * il("Q24")
        - 240 * m_j * m_l**3 * il("Q25")
    )
    out["q23"] = pref(3.5) * pt.sum(t1 * t2, axis=2)

    t1 = n_l * m_l**0.5 / (m_i + m_l) ** 6.5
    t2 = d_sub * (
        105
        / 256
        * (112 * mj6 + 1080 * mj4 * ml2 + 1134 * mj2 * ml4 + 105 * ml6)
        * il("Q11")
        - 567 / 64 * ml2 * (120 * mj4 + 252 * mjml2 + 35 * ml4) * il("Q12")
        + 27 / 16 * ml2 * (440 * mj4 + 2700 * mjml2 + 651 * ml4) * il("Q13")
        - 15 / 2 * ml4 * (594 * mj2 + 301 * ml2) * il("Q14")
        + 135 / 2 * ml4 * (26 * mj2 + 41 * ml2) * il("Q15")
        - 1890 * ml6 * il("Q16")
        + 560 * ml6 * il("Q17")
        + 18 * mjml2 * (10 * mj2 + 27 * ml2) * il("Q33")
        - 1080 * mj2 * ml4 * il("Q34")
        + 720 * mj2 * ml4 * il("Q35")
    ) + d_add * (
        189 / 16 * mjml * (8 * mj4 + 48 * mjml2 + 21 * ml4) * il("Q22")
        - 162 * m_j * m_l**3 * (8 * mj2 + 7 * ml2) * il("Q23")
        + 10 * m_j * m_l**3 * (88 * mj2 + 225 * ml2) * il("Q24")
        - 2160 * m_j * m_l**5 * il("Q25")
        + 840 * m_j * m_l**5 * il("Q26")
        + 64 * mjml3 * il("Q44")
    )
    out["q33"] = pref(3.5) * pt.sum(t1 * t2, axis=2)

    inputs = [masses, nd] + [Qs[f"Q{l}{s}"] for l, s in LS_PAIRS]
    return inputs, out, nd


ORDER = [
    "q00",
    "q01",
    "q02",
    "q03",
    "q11",
    "q12",
    "q13",
    "q22",
    "q23",
    "q33",
]


def compile_forward(nb, mode=None):
    """Option 2: compiled forward evaluation of all ten q-elements."""
    inputs, out, _ = build_graph(nb)
    t0 = time.perf_counter()
    fn = pytensor.function(
        inputs,
        [out[k] for k in ORDER],
        mode=mode,
        on_unused_input="ignore",
    )
    return fn, time.perf_counter() - t0


def compile_with_grad(nb, mode=None):
    """Option 3: forward evaluation plus d(sum q)/d(number densities).

    A scalar reduction is used because the full Jacobian of a
    (4 nb x 4 nb) output w.r.t. nb inputs is what an optimiser would need
    row by row; ``pt.grad`` of a scalar is the cheapest honest probe of
    what reverse-mode AD costs on this graph.
    """
    inputs, out, nd = build_graph(nb)
    scalar = sum(pt.sum(out[k]) for k in ORDER)
    g = pytensor.grad(scalar, nd)
    t0 = time.perf_counter()
    fn = pytensor.function(
        inputs,
        [out[k] for k in ORDER] + [g],
        mode=mode,
        on_unused_input="ignore",
    )
    return fn, time.perf_counter() - t0
