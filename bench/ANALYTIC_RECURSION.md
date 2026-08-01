# The (l, s) recursion, analytically — and what l and s actually mean

Resolves the two `TODO` questions in the `Qnn` / `Qin` / `Qij` docstrings,
and replaces the recursive finite difference in eq. 18 with the closed-form
derivative, derived with sympy.

Wall-clock note: absolute times differ between sections because they were
measured in different sessions (machine state moves the baseline between
24 s and 35 s for the same work). Within any one table every variant is
measured back to back in a single process, where run-to-run spread is 1%.

## 1. What l and s are

The docstrings currently ask:

```
l : int
    TODO: Angular momentum quantum number? Or integer moment?
s : int
    TODO: Principal quantum number? Or integer moment?
```

**Neither is a quantum number.** Both are moment orders in the
Chapman-Enskog expansion, but of different variables. The collision integral
is

```text
                     /  kT   \ 1/2  /oo
   Omega^(l,s)(T)  ~ | ------ |     |    exp(-g^2) g^(2s+3) Q^(l)(v) dg
                     \ 2 pi u /     /0

   with   g^2 = u v^2 / (2 k T)
```

with the transport cross section

```text
                     /oo
   Q^(l)(v) = 2 pi   |   ( 1 - cos^l X(b,v) ) b db
                     /0
```

So:

- **`l` is the order of the *angular* moment.** It is the power in the
  `1 - cos^l X` weighting of the deflection angle `X` -- the Legendre order
  of the transport cross section. `l=1` is momentum transfer (drives
  diffusion and electrical conductivity), `l=2` is the viscosity-type
  weighting.
- **`s` is the order of the *energy* moment.** It is the power in the
  `g^(2s+3) exp(-g^2)` weighting of the Maxwellian average over relative
  speeds -- i.e. which moment of the relative kinetic energy is taken.

The "integer moment" half of each TODO is the right guess; "angular
momentum quantum number" and "principal quantum number" are both wrong.

### Is `s` an iteration number?

Not quite — and the distinction matters. `s` is a moment order that *can be
generated iteratively*, because raising it by one multiplies the integrand
by `g^2 = E/kT`, and that is equivalent to differentiating in
temperature. The recursion is a **consequence** of what `s` means, not its
definition.

The asymmetry is the giveaway: **there is a recursion in `s` but none in
`l`**, and there cannot be. Changing `l` changes the angular weighting
inside the cross-section integral itself, which is not a derivative of
anything already computed. Each `l` needs its own fit. That is exactly what
the coefficient tables show — `c_nn[l-1]` has independent entries per `l`,
and the recursion only ever walks along the `s` axis.

So: `l` selects *which* transport cross section; `s` selects *which moment*
of it, and consecutive moments are linked by a temperature derivative.

## 2. The recursion is exact; only the derivative is approximated

Differentiating the definition with respect to `T` and re-identifying the
`s+1` integral gives

```text
   T d/dT Omega^(l,s)  =  -(s + 3/2) Omega^(l,s)  +  (s + 2) Omega^(l,s+1)
```

Dividing by the rigid-sphere reference `Omega_rs ~ T^(1/2)` -- which is
**independent of `s`** -- to form the reduced integral `Omega*` absorbs the
`1/2`, leaving

```text
   T d/dT Omega*^(l,s)  =  (s + 2) ( Omega*^(l,s+1) - Omega*^(l,s) )

   =>   Omega*^(l,s+1)  =  Omega*^(l,s) + T/(s+2) d/dT Omega*^(l,s)
```

which is eq. 18, and matches the code's `T / (s + 1)` once you account for
its indexing from `s-1`. `q_analytic_derivative.derive_recursion()` checks
the one non-trivial step symbolically.

**This identity is exact.** The approximation in the current code is only in
how `d/dT` is evaluated:

```python
negT, posT = T - 0.5, T + 0.5
return Q(..., s - 1, T) + T / (s + 1) * (Q(..., s - 1, posT) - Q(..., s - 1, negT))
```

Note there is no division by the step: the +/-0.5 is chosen so that
`2h = 1` exactly and the divisor can be omitted. This is a **unit-step
central difference**, not an arbitrary tolerance — I mischaracterised it as
a magic number in `CODE_REVIEW.md` §3.2; that is corrected below.

Because `Omega*` is a closed-form function of `x = ln T*` and
`T d/dT = d/dx`, the derivative is available exactly. Applying
the recursion symbolically `k` times needs the `k`-th derivative of the fit,
which sympy produces directly.

## 3. Correspondence with the current implementation

`bench/check_analytic_recursion.py`, O2–O2 at T = 12000 K:

| (l, s) | derivatives needed | current | analytic | rel diff |
|---|---:|---|---|---:|
| (1,1)–(1,5) | 0 | — | — | **0.0** |
| (1,6) | 1 | 1.1195069e-19 | 1.1195069e-19 | 2.9e-11 |
| (1,7) | 2 | 1.0798479e-19 | 1.0798478e-19 | 2.4e-09 |
| (2,2)–(2,4) | 0 | — | — | **0.0** |
| (2,5) | 1 | 1.5032584e-19 | 1.5032584e-19 | 3.1e-11 |
| (2,6) | 2 | 1.4451020e-19 | 1.4451020e-19 | 5.2e-10 |
| (3,3) | 0 | — | — | **0.0** |
| (3,4) | 1 | 1.4317563e-19 | 1.4317563e-19 | 3.8e-11 |
| (3,5) | 2 | 1.3686013e-19 | 1.3686013e-19 | 1.6e-09 |
| (4,4) | 0 | — | — | **0.0** |

Every non-recursed value is bit-identical, which confirms the drop-in reuses
the same fit and potential parameters. The recursed values differ by the
finite-difference truncation error, 3e-11 at one derivative and ~1e-9 at
two.

### The analytic value is the h → 0 limit

Shrinking the step on the single-derivative case:

```
  Omega*(1,6), analytic = 1.1195068855e-19
    h=0.5      rel err  2.949e-11   <- current
    h=0.25     rel err  7.008e-12   ratio 4.21     <- O(h^2), as expected
    h=0.125    rel err  3.321e-12   ratio 2.11
    h=0.0625   rel err  3.321e-12   ratio 1.00     <- round-off floor
    h=0.03125  rel err  8.475e-12   ratio 0.39     <- round-off dominates
```

The first halving reduces the error by **4.21x**, the `O(h^2)` rate for a
central difference. That is the correspondence: the two formulations are the
same quantity, and the analytic form is the exact limit the finite
difference is converging to.

Below `h ~ 0.1` subtractive cancellation takes over and the error stops
improving. **For the doubly-recursed values it never improves at all** — the
error grows monotonically as `h` shrinks (2.4e-09 at `h=0.5`, 6.7e-07 at
`h=0.03125` for `Omega*(1,7)`), because a nested difference amplifies
the round-off of the inner one.

So the practical conclusion is the opposite of "tune the step": **±0.5 K is
already at or near the optimum**, and the ~1e-9 floor cannot be improved by
any choice of `h`. Only the analytic derivative removes it.

## 3a. How hairy is the expression? (Not very, in the right basis)

`sympy.simplify` failing was a symptom, not a verdict. Differentiating eq. 15
as written -- in terms of `exp` -- produces a tree that `simplify` cannot
untangle:

| derivatives | raw `exp` basis | structured `P_k` | expanded `P_k` |
|---:|---:|---:|---:|
| 0 | 30 | 0 (`P_0 = 1`) | 0 |
| 1 | 162 | **19** | 38 |
| 2 | 601 | **100** | 311 |

The fix is a change of basis. Each factor in eq. 15 is a logistic:

```text
   exp(u) / (exp(u) + exp(-u))  ==  1 / (1 + exp(-2u))  ==  sigma(2u)
```

so with `c = a0 + a1 x`, `k1 = 2/a3`, `k2 = 2/a6`:

```text
   g = ln Omega*  =  c * sigma_1  +  a4 * sigma_2
```

Sigmoids are closed under differentiation (`sigma' = k sigma (1 - sigma)`),
so with `d_i = sigma_i (1 - sigma_i)` the derivatives stay finite and
readable -- both verified against `sympy.diff`:

```text
   g'  =  a1 sigma_1  +  c k1 d1  +  a4 k2 d2

   g'' =  2 a1 k1 d1
          +  c k1^2 d1 (1 - 2 sigma_1)
          +  a4 k2^2 d2 (1 - 2 sigma_2)
```

Factoring the common `exp(g)` out of `Omega*_(k) = exp(g) P_k`, the whole
recursion is three lines:

```text
   P_0 = 1
   P_1 = 1 + g' / (s0 + 2)
   P_2 = P_1 + (g' P_1 + g'' / (s0 + 2)) / (s0 + 3)
```

**That is the entire implementation** -- `bench/q_analytic_compact.py`, about
twenty lines of numpy with no symbolic dependency at runtime. It agrees with
the sympy-lambdified version to **2.2e-14** across all (l, s, k) with
randomised coefficients, and is 1.7x faster (2.94 us vs 4.93 us per call).

Expanding `P_2` into an explicit polynomial in `sigma_1, sigma_2` is possible
but counterproductive: 100 operations becomes 311.

The derivation is emitted as LaTeX by
`q_analytic_symbolic.latex_derivation()` -- see
[`analytic_recursion.tex`](analytic_recursion.tex), which compiles to a
two-page note.

### One deliberate numerical change

The sigmoid rewrite is algebraically identical but not bit-identical: the two
forms differ by **1 ULP** (4.5e-16 max relative over the range used). That is
why non-recursed (l, s) values now differ by ~2e-16 rather than exactly zero.

The trade is favourable: `exp(u)/(exp(u)+exp(-u))` overflows to `nan` for
`u > ~710`, while `1/(1+exp(-2u))` saturates correctly to 1. The current
inputs stay far from that, so this is robustness rather than a live bug fix
(`bench/check_sigmoid_form.py`).

## 4. Cost and end-to-end effect

Evaluations of the base fit per collision integral:

| | current | analytic |
|---|---:|---:|
| 0 derivatives (e.g. Omega(1,5)) | 1 | 1 |
| 1 derivative (Omega(1,6)) | 3 | 1 |
| 2 derivatives (Omega(1,7)) | 9 | 1 |

Measured on the SiCO property sweep (single process, 20 T × 3 mixtures):

| variant | time | speedup | max rel change |
|---|---:|---:|---:|
| baseline | 24.23 s | 1.00x | — |
| analytic (l, s) derivative | 17.86 s | **1.36x** | 1.5e-21 |
| + `Qij_mix` memoised | 12.72 s | 1.91x | 1.5e-21 |
| + vectorised partition function | 3.47 s | **6.99x** | 1.0e-15 |

All 55 tests pass with the analytic derivative alone and with all three
combined (`MPC_PATCH=pf,memo,analytic`).

The `1.5e-21` is worth reading carefully: despite the collision integrals
themselves moving by up to 2e-9, the **plasma properties are unchanged to
roundoff**. Viscosity is *exactly* unchanged, because `qhat`'s seven (l, s)
pairs `{11,12,13,22,23,24,33}` contain none that trigger the recursion — so
the viscosity path never touched the finite difference in the first place.

## 5. Corrections to `CODE_REVIEW.md` §3.2

- The ±0.5 K step is **not** an untuned magic number. It is a unit-step
  central difference, chosen so the `2h` divisor is 1 and can be omitted.
- "Truncation error is temperature-dependent" — true but not actionable, and
  I framed it as if a better step existed. It does not: the nested
  difference is round-off limited, and shrinking `h` makes the
  doubly-recursed values monotonically worse.
- The recommendation stands and strengthens: the analytic derivative removes
  both the 3x evaluation cost and the ~1e-9 error floor, and is exact.

## Reproducing

```bash
uv run python bench/q_analytic_derivative.py       # derivation + orders needed
uv run python bench/check_analytic_recursion.py    # sections 3 and 4
PYTHONPATH=bench MPC_PATCH=analytic uv run pytest tests -q -p pf_plugin
```

Requires `sympy` (used only to build the derivative; the lambdified result
is plain numpy and could be pasted in as literal code if adding a build-time
dependency is unwelcome).
