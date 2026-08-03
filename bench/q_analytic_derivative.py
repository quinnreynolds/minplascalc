r"""Analytic form of the (l, s) recursion in ``Qnn`` / ``Qin``, via sympy.

Background
----------
``Qnn`` and ``Qin`` return the *reduced* collision integral
:math:`\Omega^{(l,s)\star}` scaled by the T-independent factor
:math:`\pi \sigma^2 \times 10^{-20}`.  Eq. 18 of [Laricchiuta2007]_ relates
successive moment orders by a temperature derivative:

.. math::

    \Omega^{(l,s+1)\star} = \Omega^{(l,s)\star}
        + \frac{T}{s+2} \frac{\partial \Omega^{(l,s)\star}}{\partial T}

This is an *identity*, not an approximation -- see ``derive_recursion()``
for the derivation from the definition of the collision integral.  What the
current code approximates is the derivative, using a central difference:

.. code:: python

    negT, posT = T - 0.5, T + 0.5
    d = Q(l, s - 1, posT) - Q(l, s - 1, negT)
    return Q(l, s - 1, T) + T / (s + 1) * d

The step is chosen as +/- 0.5 K precisely so the divisor 2h equals 1 and can
be dropped -- so this is a unit-step central difference, not an arbitrary
tolerance.  It costs a factor of three in evaluations per recursion level,
and it is the only source of error, since the recursion itself is exact.

Because the fitted expression for :math:`\Omega^{\star}` is a closed-form
function of :math:`x = \ln T^{\star}`, and :math:`T \partial_T = \partial_x`,
the derivative is available in closed form.  This module builds it with
sympy and lambdifies the result.
"""

import sympy as sp

# Symbols: x = ln(T*), and the seven fit coefficients a0..a6 of eq. 16.
X = sp.Symbol("x", real=True)
A = sp.symbols("a0:7", real=True)

# The largest s for which each l has its own fit coefficients.  Above this
# the code recurses; below or equal, it evaluates the fit directly.
BASE_S = {1: 5, 2: 4, 3: 3, 4: 4}


def base_omega_star():
    r"""Symbolic :math:`\Omega^{\star}(x)` -- eq. 15 of [Laricchiuta2007]_.

    Written exactly as ``functions_transport`` writes it, so the two agree
    term by term rather than only numerically.
    """
    u1 = (X - A[2]) / A[3]
    u2 = (X - A[5]) / A[6]
    lnS1 = (A[0] + A[1] * X) * sp.exp(u1) / (sp.exp(u1) + sp.exp(-u1))
    lnS2 = A[4] * sp.exp(u2) / (sp.exp(u2) + sp.exp(-u2))
    return sp.exp(lnS1 + lnS2)


def step(expr, s_from):
    r"""One exact recursion step, from moment order ``s_from`` to +1.

    Uses :math:`T \partial_T = \partial_x` with :math:`x = \ln T^{\star}`,
    so the ``T/(s+2)`` prefactor becomes a plain ``1/(s+2)``.
    """
    return expr + sp.diff(expr, X) / (s_from + 2)


def omega_star_expr(l, s):
    """Symbolic Omega* for (l, s), recursing analytically where needed.

    No ``simplify`` here: the second-derivative tree is large and
    ``simplify`` on it is intractable.  Common subexpressions are handled
    at lambdify time instead, which is what actually matters for speed.
    """
    s0 = BASE_S[l]
    expr = base_omega_star()
    for p in range(s0, s):
        expr = step(expr, p)
    return expr


_LAMBDIFIED: dict = {}


def omega_star_fn(l, s):
    """Lambdified f(x, a0..a6) -> Omega*, cached per (l, s)."""
    key = (l, s)
    fn = _LAMBDIFIED.get(key)
    if fn is None:
        fn = sp.lambdify((X,) + A, omega_star_expr(l, s), "numpy", cse=True)
        _LAMBDIFIED[key] = fn
    return fn


def derive_recursion():
    r"""Derive eq. 18 from the definition, to show the recursion is exact.

    With :math:`\gamma^2 = E/(k T)`, the collision integral is

    .. math::

        \Omega^{(l,s)} \propto \frac{T^{1/2}}{(s+1)!\,(kT)^{s+2}}
            \int_0^\infty e^{-E/kT} E^{s+1} Q^{(l)}(E)\, dE

    Differentiating in T and re-identifying the s+1 integral gives

    .. math::

        T \partial_T \Omega^{(l,s)}
            = -(s + 3/2)\,\Omega^{(l,s)} + (s+2)\,\Omega^{(l,s+1)}

    Dividing through by the rigid-sphere reference
    :math:`\Omega_{rs} \propto T^{1/2}` (independent of s) to form
    :math:`\Omega^{\star}` removes the 1/2, leaving

    .. math::

        T \partial_T \Omega^{(l,s)\star} = (s+2)\left(
            \Omega^{(l,s+1)\star} - \Omega^{(l,s)\star}\right)

    which rearranges to eq. 18.  Returns the symbolic check that the
    reduction step is consistent.
    """
    T, k, s = sp.symbols("T k s", positive=True)
    E = sp.Symbol("E", positive=True)
    Ql = sp.Function("Q")(E)

    # The only step that needs checking: differentiating the integrand of
    # I_s(T) in T reproduces the integrand of I_{s+1}(T) / (k T^2).
    integrand_s = sp.exp(-E / (k * T)) * E ** (s + 1) * Ql
    integrand_s1 = sp.exp(-E / (k * T)) * E ** (s + 2) * Ql
    residual = sp.diff(integrand_s, T) - integrand_s1 / (k * T**2)
    return sp.cancel(sp.expand(residual)) == 0


def omega_reduced_analytic(l, s, a, x):
    """Numeric Omega* for (l, s) from the analytic recursion."""
    return omega_star_fn(l, s)(x, *a)


def derivative_orders_needed(l, s):
    """How many symbolic derivatives the (l, s) value requires."""
    return max(0, s - BASE_S[l])


if __name__ == "__main__":
    print("recursion identity consistent:", derive_recursion())
    for l, s in [(1, 5), (1, 6), (1, 7), (2, 4), (2, 6), (3, 3), (3, 5)]:
        k = derivative_orders_needed(l, s)
        print(
            f"  Omega*({l},{s}): base s0={BASE_S[l]}, "
            f"{k} analytic derivative(s), "
            f"{3**k} base evaluations under the current recursion"
        )
    e = omega_star_expr(1, 6)
    print("\nOmega*(1,6) leading structure:")
    print(sp.count_ops(e), "operations in the unsimplified graph")
    print("sympy", sp.__version__)
