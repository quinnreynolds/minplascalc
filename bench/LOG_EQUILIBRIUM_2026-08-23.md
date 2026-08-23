# Coupled log-space equilibrium opportunity

Snapshot: `feature/consolidate-computation-state` after the piecewise
analytical transport derivative. This is an isolated test/benchmark prototype;
it does not replace the `LTE` API or production composition solver.

## Formulation

The unknown particle numbers are represented by `u_i = log(N_i)`. Elemental
and charge multipliers are scaled by `k_B T`. The complete dimensionless
residual is

```text
chemical equilibrium:  E0/(kT) - log(Z/N) + A lambda = 0
element conservation:   log((A_element^T N) / b) = 0
charge neutrality:      (z^T N) / sum(N) = 0
```

All species and conservation laws are solved together. The analytical Jacobian
includes the composition dependence of ionisation lowering. Log particle
numbers guarantee positivity without the production solver's particle-change
governor.

A generic `1e20`-particle state converges directly between roughly 8000 and
25000 K. An independent 12000 K bootstrap followed by temperature continuation
covers 1000–25000 K in either direction. It does not use a production-solver
composition as an initial value.

The analytical Jacobian agrees with a central numerical Jacobian to
`1.5e-10` in relative norm at the 16-species, 12000 K SiCO state.

## Benchmark

Command:

```console
PYTHONPATH=src:. .venv/bin/python bench/bench_log_equilibrium.py
```

AC-powered result for 20 temperatures and three SiCO mixture ratios, seven
alternating repetitions:

| Solver | Median | Nonlinear iterations | Iterations per solve |
|---|---:|---:|---:|
| Production governed solver | 0.300629 s | 2574 | 42.90 per requested state |
| Log Newton plus continuation | 0.234832 s | 610 | 4.07 per continuation solve |

The current Python prototype is **1.28x faster** despite performing 150 solves:
60 requested states plus the independent bootstrap/intermediate continuation
states. It used 1397 residual evaluations. The largest absolute mole-fraction
difference on this 20-point grid was `1.24e-8`.

The 10.5x iteration reduction becoming only a 1.28x wall-clock improvement is
the main implementation finding. A log iteration currently:

1. evaluates reference energies and every partition function for the Newton
   residual;
1. independently recalculates lowering derivatives for the Jacobian;
1. evaluates the full thermodynamics again at the line-search candidate;
1. repeats part of the accepted candidate work when constructing the next
   Jacobian.

Only 27 evaluations beyond the expected one-Jacobian plus one-candidate pattern
were caused by actual backtracking. The overhead is repeated thermodynamic
evaluation, not poor globalization.

This makes a fused evaluator unusually well motivated: one packed pass should
return reference energies, lowering and its derivatives, partition moments,
the residual, and the Jacobian. The accepted candidate can retain its residual
and thermodynamic state for the next iteration.

## Active-set finding

The hard ionisation-lowered electronic-level cutoff permits more than one
self-consistent piecewise root near a level crossing. In the 30-point SiCO
sweep at 20862 K, production and continuation converged to roots differing by
`6.9e-6` in maximum mole fraction. The continuation root included one extra
Si+ level and had a Gibbs objective about 9.77 J lower on the solver's arbitrary
`1e24`-particle constraint scale.

This does not yet establish which branch should be selected by the public API,
but it shows that continuation direction and active-set policy need explicit
tests. It is not ordinary Newton error: both roots satisfy their respective
piecewise residuals to roundoff.

## Assessment

The log formulation is worth continuing:

- positivity is structural;
- the full coupled Jacobian is accurate;
- cold midrange convergence is 5–9 iterations;
- warm continuation generally takes 2–4 iterations;
- ascending and descending sweeps pass from 1000 to 25000 K;
- even the deliberately unfused Python prototype is already faster.

The next experiment should fuse residual and Jacobian thermodynamics before
attempting API compatibility. That will establish whether the iteration-count
gain translates into a roughly 2–3x composition-solver gain. Active-set branch
selection should be investigated in parallel, not hidden behind loose
comparison tolerances.
