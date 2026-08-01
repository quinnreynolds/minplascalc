r"""Compact symbolic derivation of the (l, s) recursion, with LaTeX output.

``q_analytic_derivative`` differentiates eq. 15 as written -- in terms of
``exp`` -- giving a 601-operation tree for two derivatives that
``sympy.simplify`` cannot untangle.  That is an artefact of the basis.

Eq. 15 is built from two logistic terms.  Using

.. math::

    \frac{e^{u}}{e^{u} + e^{-u}} = \frac{1}{1 + e^{-2u}} = \sigma(2u)

the fit becomes

.. math::

    \ln \Omega^{\star} = g(x) = c(x)\,\sigma_1 + a_4\,\sigma_2,
    \qquad c(x) = a_0 + a_1 x

with :math:`\sigma_1 = \sigma(k_1(x-a_2))`, :math:`k_1 = 2/a_3`, and
:math:`\sigma_2 = \sigma(k_2(x-a_5))`, :math:`k_2 = 2/a_6`.

Sigmoids are closed under differentiation, so with
:math:`d_i = \sigma_i(1-\sigma_i)`:

.. math::

    g'  &= a_1 \sigma_1 + c k_1 d_1 + a_4 k_2 d_2 \\
    g'' &= 2 a_1 k_1 d_1 + c k_1^2 d_1 (1 - 2\sigma_1)
           + a_4 k_2^2 d_2 (1 - 2\sigma_2)

Factoring :math:`e^{g}` out of :math:`\Omega^{\star}_{(k)} = e^{g} P_k`
turns the recursion into three lines:

.. math::

    P_0 = 1, \qquad
    P_1 = 1 + \frac{g'}{s_0+2}, \qquad
    P_2 = P_1 + \frac{g' P_1 + g''/(s_0+2)}{s_0+3}

That is the whole implementation.  Expanding these into polynomials in
:math:`\sigma_1,\sigma_2` is possible but pointless -- it inflates
:math:`P_2` from a handful of operations to 190.
"""

import sympy as sp

x = sp.Symbol("x", real=True)
a0, a1, a2, a3, a4, a5, a6 = sp.symbols("a0 a1 a2 a3 a4 a5 a6", real=True)

# Sigmoids carried as opaque symbols so derivatives stay rational.
S1, S2 = sp.symbols("sigma_1 sigma_2", real=True)

K1 = 2 / a3
K2 = 2 / a6
C = a0 + a1 * x

G = C * S1 + a4 * S2


def D(expr):
    r"""Total derivative in x, using sigma' = k sigma (1 - sigma)."""
    return (
        sp.diff(expr, x)
        + sp.diff(expr, S1) * K1 * S1 * (1 - S1)
        + sp.diff(expr, S2) * K2 * S2 * (1 - S2)
    )


def compact_derivatives():
    r"""Return (g', g'') both as sympy derives them and in hand form.

    The hand forms are what gets implemented; this checks they agree.
    """
    d1 = S1 * (1 - S1)
    d2 = S2 * (1 - S2)

    g1_auto = D(G)
    g2_auto = D(g1_auto)

    g1_hand = a1 * S1 + C * K1 * d1 + a4 * K2 * d2
    g2_hand = (
        2 * a1 * K1 * d1
        + C * K1**2 * d1 * (1 - 2 * S1)
        + a4 * K2**2 * d2 * (1 - 2 * S2)
    )

    ok1 = sp.simplify(sp.expand(g1_auto - g1_hand)) == 0
    ok2 = sp.simplify(sp.expand(g2_auto - g2_hand)) == 0
    return (g1_hand, g2_hand), (ok1, ok2)


def structured_polys(s0, n_steps=2):
    r"""P_0..P_n kept structured (not expanded), plus their op counts."""
    (g1, g2), _ = compact_derivatives()
    polys = [sp.Integer(1)]
    derivs = [sp.Integer(0)]  # P_k' ; P_0' = 0
    for k in range(n_steps):
        P, Pp = polys[-1], derivs[-1]
        nxt = P + (g1 * P + Pp) / (s0 + k + 2)
        polys.append(nxt)
        derivs.append(D(nxt))
    return polys


def latex_derivation(s0=5):
    """Return a LaTeX fragment giving the full derivation."""
    (g1, g2), (ok1, ok2) = compact_derivatives()
    assert ok1 and ok2
    d1 = r"d_1 = \sigma_1(1-\sigma_1)"
    d2 = r"d_2 = \sigma_2(1-\sigma_2)"
    return "\n".join(
        [
            r"\section*{Analytic form of the $(l,s)$ recursion}",
            "",
            r"\subsection*{1. The recursion is exact}",
            r"From the definition of the collision integral, with",
            r"$\gamma^2 = E/kT$,",
            r"\begin{equation}",
            r"  \Omega^{(l,s)} \propto \frac{T^{1/2}}{(s+1)!\,(kT)^{s+2}}",
            r"  \int_0^\infty e^{-E/kT} E^{s+1} Q^{(l)}(E)\,dE",
            r"\end{equation}",
            r"Differentiating in $T$ and re-identifying the $s{+}1$ integral,",
            r"\begin{equation}",
            r"  T\,\partial_T \Omega^{(l,s)}",
            r"   = -(s+\tfrac32)\,\Omega^{(l,s)} + (s+2)\,\Omega^{(l,s+1)}",
            r"\end{equation}",
            r"Dividing by the rigid-sphere reference "
            r"$\Omega_{rs}\propto T^{1/2}$, which is independent of $s$, "
            r"absorbs the $\tfrac12$ and leaves eq.~18:",
            r"\begin{equation}",
            r"  \boxed{\;\Omega^{\star}_{s+1} = \Omega^{\star}_{s}"
            r"   + \frac{T}{s+2}\,\partial_T \Omega^{\star}_{s}\;}",
            r"\end{equation}",
            "",
            r"\subsection*{2. The fit, in the sigmoid basis}",
            r"With $x=\ln T^{\star}$ and "
            r"$e^{u}/(e^{u}+e^{-u}) = \sigma(2u)$, eq.~15 is",
            r"\begin{align}",
            r"  \ln\Omega^{\star} = g(x) &= " + sp.latex(G) + r", \qquad"
            r" c(x) = a_0 + a_1 x \\",
            r"  \sigma_1 &= \sigma\!\left(k_1 (x-a_2)\right),"
            r"\; k_1 = 2/a_3 \\",
            r"  \sigma_2 &= \sigma\!\left(k_2 (x-a_5)\right),\; k_2 = 2/a_6",
            r"\end{align}",
            "",
            r"\subsection*{3. Derivatives stay closed}",
            r"Since $\sigma' = k\,\sigma(1-\sigma)$, with $"
            + d1
            + r"$ and $"
            + d2
            + r"$:",
            r"\begin{align}",
            r"  g'  &= "
            + sp.latex(a1 * S1)
            + r" + c\,k_1 d_1 + a_4 k_2 d_2 \\",
            r"  g'' &= 2 a_1 k_1 d_1 + c\,k_1^2 d_1 (1-2\sigma_1)"
            r"        + a_4 k_2^2 d_2 (1-2\sigma_2)",
            r"\end{align}",
            r"(both verified against \texttt{sympy.diff} in "
            r"\texttt{compact\_derivatives()}).",
            "",
            r"\subsection*{4. Three lines of recursion}",
            r"Writing $\Omega^{\star}_{(k)} = e^{g} P_k$ and using "
            r"$T\partial_T = \partial_x$,",
            r"\begin{equation}",
            r"  P_{k+1} = P_k + \frac{g' P_k + P_k'}{s_0+k+2},\qquad P_0 = 1",
            r"\end{equation}",
            r"so for the two levels the code actually needs "
            r"($s_0 = " + str(s0) + r"$ for $l=1$):",
            r"\begin{align}",
            r"  P_0 &= 1 \\",
            r"  P_1 &= 1 + \frac{g'}{s_0+2} \\",
            r"  P_2 &= P_1 + \frac{g' P_1 + g''/(s_0+2)}{s_0+3}",
            r"\end{align}",
            r"and $\Omega^{\star}_{(k)} = e^{g} P_k$.",
        ]
    )


GENERATED_NUMPY = '''
def omega_star(x, a, k, s0):
    """Reduced collision integral at moment order s0 + k.

    Analytic evaluation of eq. 18 -- no finite differences.

    Parameters
    ----------
    x : float
        ln(T*), the reduced temperature logarithm.
    a : np.ndarray
        Length-7 fit coefficients of eq. 16.
    k : int
        Number of recursion steps above the tabulated order s0.
    s0 : int
        Largest tabulated moment order for this l.
    """
    a0, a1, a2, a3, a4, a5, a6 = a
    k1, k2 = 2 / a3, 2 / a6
    c = a0 + a1 * x

    s1 = 1 / (1 + np.exp(-k1 * (x - a2)))
    s2 = 1 / (1 + np.exp(-k2 * (x - a5)))
    d1, d2 = s1 * (1 - s1), s2 * (1 - s2)

    g = c * s1 + a4 * s2
    if k == 0:
        return np.exp(g)

    # g' and g'' in the sigmoid basis.
    g1 = a1 * s1 + c * k1 * d1 + a4 * k2 * d2
    P = 1 + g1 / (s0 + 2)
    if k == 1:
        return np.exp(g) * P

    g2 = (
        2 * a1 * k1 * d1
        + c * k1**2 * d1 * (1 - 2 * s1)
        + a4 * k2**2 * d2 * (1 - 2 * s2)
    )
    P = P + (g1 * P + g2 / (s0 + 2)) / (s0 + 3)
    return np.exp(g) * P
'''


if __name__ == "__main__":
    (g1, g2), (ok1, ok2) = compact_derivatives()
    print(f"hand-derived g'  matches sympy.diff: {ok1}")
    print(f"hand-derived g'' matches sympy.diff: {ok2}\n")

    from q_analytic_derivative import omega_star_expr

    polys = structured_polys(5, 2)
    print("# operation counts\n")
    print(
        f"  {'k':>3s} {'raw exp basis':>14s} {'P_k structured':>15s} "
        f"{'P_k expanded':>13s}"
    )
    for k in range(3):
        raw = sp.count_ops(omega_star_expr(1, 5 + k))
        struct = sp.count_ops(polys[k])
        expanded = sp.count_ops(sp.expand(polys[k]))
        print(f"  {k:>3d} {raw:>14d} {struct:>15d} {expanded:>13d}")

    print("\n# g' =")
    sp.pprint(g1)
    print("\n# g'' =")
    sp.pprint(g2)
    print("\n# generated implementation:")
    print(GENERATED_NUMPY)
