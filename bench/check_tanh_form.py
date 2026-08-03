r"""Evaluate whether a hyperbolic rewrite of eq. 15 is worthwhile.

There is an exact identity connecting the three ways of writing the logistic
factor in eq. 15:

.. math::

    \frac{e^{u}}{e^{u}+e^{-u}}
      = \frac{1}{1+e^{-2u}}
      = \frac{1 + \tanh u}{2}

The third form is the only one that keeps the code's own ``u = (x-a2)/a3``
without a factor-of-two relabelling, and ``tanh`` is a single libm call.

Derivatives in the tanh basis, with :math:`t_i = \tanh u_i` and
:math:`w_i = 1 - t_i^2 = \operatorname{sech}^2 u_i`:

.. math::

    g   &= \tfrac12\left[c(1+t_1) + a_4(1+t_2)\right] \\
    g'  &= \tfrac12 a_1 (1+t_1) + \frac{c\,w_1}{2a_3}
           + \frac{a_4 w_2}{2a_6} \\
    g'' &= \frac{a_1 w_1}{a_3} - \frac{c\,t_1 w_1}{a_3^2}
           - \frac{a_4 t_2 w_2}{a_6^2}

This script checks those against sympy, then compares all three bases
against a 50-digit mpmath reference -- over the range the code actually
uses, and out into the extremes.
"""

import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402

# Range actually reached by the shipped data (measured): |u1| <= 5.85,
# |u2| <= 3.39.
REAL_U_MAX = 5.85


def verify_symbolic():
    """Check the tanh-basis derivatives against sympy.diff."""
    x, a0, a1, a2, a3, a4, a5, a6 = sp.symbols(
        "x a0 a1 a2 a3 a4 a5 a6", real=True
    )
    u1, u2 = (x - a2) / a3, (x - a5) / a6
    c = a0 + a1 * x
    g = c * (1 + sp.tanh(u1)) / 2 + a4 * (1 + sp.tanh(u2)) / 2

    t1, t2 = sp.tanh(u1), sp.tanh(u2)
    w1, w2 = 1 - t1**2, 1 - t2**2

    g1_hand = a1 * (1 + t1) / 2 + c * w1 / (2 * a3) + a4 * w2 / (2 * a6)
    g2_hand = a1 * w1 / a3 - c * t1 * w1 / a3**2 - a4 * t2 * w2 / a6**2

    ok1 = sp.simplify(sp.diff(g, x) - g1_hand) == 0
    ok2 = sp.simplify(sp.diff(g, x, 2) - g2_hand) == 0

    # And that the three writings of the logistic agree symbolically.
    u = sp.Symbol("u", real=True)
    idA = sp.exp(u) / (sp.exp(u) + sp.exp(-u))
    idB = 1 / (1 + sp.exp(-2 * u))
    idC = (1 + sp.tanh(u)) / 2
    # simplify() cannot close the tanh case on its own; rewrite first.
    okAB = sp.simplify(idA - idB.rewrite(sp.exp)) == 0
    okAC = sp.simplify(idA - idC.rewrite(sp.exp)) == 0
    return ok1, ok2, okAB, okAC


def omega_ref(x, a, k, s0, dps=50):
    """High-precision Omega* via mpmath, used as ground truth."""
    mp.mp.dps = dps
    a0, a1, a2, a3, a4, a5, a6 = [mp.mpf(float(v)) for v in a]
    X = mp.mpf(float(x))

    def g_of(xx):
        c = a0 + a1 * xx
        s1 = 1 / (1 + mp.e ** (-2 * (xx - a2) / a3))
        s2 = 1 / (1 + mp.e ** (-2 * (xx - a5) / a6))
        return c * s1 + a4 * s2

    om = lambda xx: mp.e ** g_of(xx)  # noqa: E731
    if k == 0:
        return om(X)
    d1 = mp.diff(om, X)
    P1 = om(X) + d1 / (s0 + 2)
    if k == 1:
        return P1
    d_om = lambda xx: om(xx) + mp.diff(om, xx) / (s0 + 2)  # noqa: E731
    return P1 + mp.diff(d_om, X) / (s0 + 3)


def _omega(x, a, k, s0, basis):
    a0, a1, a2, a3, a4, a5, a6 = a
    c = a0 + a1 * x
    u1, u2 = (x - a2) / a3, (x - a5) / a6

    if basis == "exp":  # as functions_transport writes it
        s1 = np.exp(u1) / (np.exp(u1) + np.exp(-u1))
        s2 = np.exp(u2) / (np.exp(u2) + np.exp(-u2))
        d1, d2 = s1 * (1 - s1), s2 * (1 - s2)
    elif basis == "sigmoid":  # what q_analytic_compact ships
        s1 = 1 / (1 + np.exp(-2 * u1))
        s2 = 1 / (1 + np.exp(-2 * u2))
        d1, d2 = s1 * (1 - s1), s2 * (1 - s2)
    elif basis == "tanh":  # hyperbolic, sech^2 for the derivative factor
        t1, t2 = np.tanh(u1), np.tanh(u2)
        s1, s2 = (1 + t1) / 2, (1 + t2) / 2
        d1 = 1 / (4 * np.cosh(u1) ** 2)  # = s1 (1 - s1), no cancellation
        d2 = 1 / (4 * np.cosh(u2) ** 2)
    else:
        raise ValueError(basis)

    k1, k2 = 2 / a3, 2 / a6
    g = c * s1 + a4 * s2
    if k == 0:
        return np.exp(g)
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


def main():
    ok1, ok2, okAB, okAC = verify_symbolic()
    print("## Symbolic checks\n")
    print(f"  exp form == sigmoid form                 : {okAB}")
    print(f"  exp form == (1 + tanh(u))/2              : {okAC}")
    print(f"  tanh-basis g'  matches sympy.diff        : {ok1}")
    print(f"  tanh-basis g'' matches sympy.diff        : {ok2}")

    rng = np.random.default_rng(0)
    print("\n## Accuracy vs 50-digit mpmath, k=2 (worst case)\n")
    print(
        f"  {'|u| range':>14s} {'exp (current)':>15s} "
        f"{'sigmoid':>12s} {'tanh':>12s}"
    )

    for umax, label in [
        (REAL_U_MAX, "<= 5.85 (real)"),
        (18.0, "<= 18"),
        (40.0, "<= 40"),
        (400.0, "<= 400"),
        (800.0, "<= 800"),
    ]:
        errs = {"exp": [], "sigmoid": [], "tanh": []}
        for _ in range(120):
            a3v, a6v = rng.uniform(0.4, 1.2, 2)
            a2v, a5v = rng.uniform(-1, 1, 2)
            scale = umax * min(a3v, a6v)
            xv = rng.uniform(-scale, scale)
            a = np.array(
                [
                    rng.uniform(-1, 1),
                    rng.uniform(-0.1, 0.1),
                    a2v,
                    a3v,
                    rng.uniform(-0.5, 0.5),
                    a5v,
                    a6v,
                ]
            )
            ref = omega_ref(xv, a, 2, 5)
            if not mp.isfinite(ref) or ref == 0:
                continue
            for b in errs:
                with np.errstate(all="ignore"):
                    got = _omega(xv, a, 2, 5, b)
                if not np.isfinite(got):
                    errs[b].append(np.inf)
                else:
                    errs[b].append(
                        float(abs(mp.mpf(float(got)) - ref) / abs(ref))
                    )
        row = []
        for b in ("exp", "sigmoid", "tanh"):
            e = np.array(errs[b])
            row.append(
                "nan/inf" if not np.isfinite(e).all() else f"{e.max():.2e}"
            )
        print(f"  {label:>14s} {row[0]:>15s} {row[1]:>12s} {row[2]:>12s}")

    print("\n## Tail behaviour of the logistic factor itself\n")
    print(
        f"  {'u':>7s} {'exact (mpmath)':>18s} {'exp':>13s} "
        f"{'sigmoid':>13s} {'tanh':>13s}"
    )
    mp.mp.dps = 50
    for uv in (-800.0, -40.0, -20.0, -5.85, 0.0, 5.85, 20.0, 40.0, 800.0):
        ex = 1 / (1 + mp.e ** (-2 * mp.mpf(uv)))
        with np.errstate(all="ignore"):
            fe = np.exp(uv) / (np.exp(uv) + np.exp(-uv))
            fs = 1 / (1 + np.exp(-2 * uv))
            ft_ = (1 + np.tanh(uv)) / 2

        def rel(v):
            if not np.isfinite(v):
                return "nan"
            if ex == 0:
                return "0"
            return f"{float(abs(mp.mpf(float(v)) - ex) / abs(ex)):.1e}"

        print(
            f"  {uv:>7.2f} {mp.nstr(ex, 6):>18s} {rel(fe):>13s} "
            f"{rel(fs):>13s} {rel(ft_):>13s}"
        )

    print("\n## Cost per call (k=2)\n")
    import time

    a = np.array([0.78, -0.024, 0.5, 0.9, -0.34, 0.42, 0.32])
    n = 40000
    for b in ("exp", "sigmoid", "tanh"):
        _omega(1.5, a, 2, 5, b)
        t0 = time.perf_counter()
        for _ in range(n):
            _omega(1.5, a, 2, 5, b)
        print(f"  {b:>10s}: {(time.perf_counter() - t0) / n * 1e6:5.2f} us")


if __name__ == "__main__":
    main()
