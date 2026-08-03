"""Compare the two algebraically-identical writings of the eq. 15 sigmoid.

``functions_transport`` writes each logistic factor as
``exp(u)/(exp(u)+exp(-u))``.  The analytic derivation uses the equivalent
``1/(1+exp(-2u))``, which is what makes the derivatives collapse to
polynomials in sigma.

The two agree to within a bit or two, which is the source of the ~2e-16
differences in otherwise-unchanged (l, s) values, and the rewrite stays
finite where the current form overflows to nan.
"""

import numpy as np

np.seterr(all="ignore")


def current(u):
    return np.exp(u) / (np.exp(u) + np.exp(-u))


def rewrite(u):
    return 1 / (1 + np.exp(-2 * u))


def main():
    print("  current:  exp(u)/(exp(u)+exp(-u))")
    print("  rewrite:  1/(1+exp(-2u))\n")
    print(f"  {'u':>8s} {'current':>16s} {'rewrite':>16s} {'ulps':>6s}")
    for u in (0.25, 0.5, 1.0, 2.0, 5.0, 50.0, 200.0, 400.0, -400.0):
        o, n = current(u), rewrite(u)
        if np.isnan(o) or np.isnan(n):
            ulps = "n/a"
        else:
            ulps = str(
                abs(
                    np.frombuffer(np.float64(o).tobytes(), dtype=np.int64)[0]
                    - np.frombuffer(np.float64(n).tobytes(), dtype=np.int64)[0]
                )
            )
        print(f"  {u:>8.2f} {o:>16.12g} {n:>16.12g} {ulps:>6s}")

    print("\n  Agreement across the range actually used (u in [-20, 20]):")
    us = np.linspace(-20, 20, 100001)
    o, n = current(us), rewrite(us)
    good = np.isfinite(o) & np.isfinite(n)
    rel = np.abs(o[good] - n[good]) / np.maximum(np.abs(n[good]), 1e-300)
    print(f"    max relative difference: {rel.max():.3e}")

    print("\n  Overflow threshold:")
    for u in (350.0, 355.0, 400.0, 710.0, 800.0):
        o, n = current(u), rewrite(u)
        tag = "  <- current returns nan" if np.isnan(o) else ""
        print(f"    u={u:>6.0f}  current={o!s:>8s}  rewrite={n:g}{tag}")


if __name__ == "__main__":
    main()
