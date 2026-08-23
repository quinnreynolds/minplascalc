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
alternating repetitions after packed thermodynamics and accepted-state reuse:

| Solver | Median | Nonlinear iterations | Iterations per solve |
|---|---:|---:|---:|
| Production governed solver | 0.300781 s | 2574 | 42.90 per requested state |
| Log Newton plus continuation | 0.136857 s | 610 | 4.07 per continuation solve |

The current Python prototype is **2.20x faster** despite performing 150 solves:
60 requested states plus the independent bootstrap/intermediate continuation
states. It used 787 residual evaluations. The largest absolute mole-fraction
difference on this 20-point grid was `1.24e-8`.

The 10.5x iteration reduction becoming only a 2.20x wall-clock improvement is
still the main implementation finding. The three central eliminations were:

1. packing immutable level data so monatomic partition functions use one
   vectorised exponential and reductions rather than per-species calls;
1. retaining the accepted line-search residual and thermodynamic state for the
   next Newton step, including its lowering derivative;
1. caching electronic Boltzmann weights and other factors that depend on
   temperature but not composition.

The evaluation count is now close to one accepted candidate per iteration; the
remaining excess is actual backtracking. Profiling shows the dense coupled
linear solve is only about 4% of prototype runtime. Packed thermodynamics and
ionisation lowering remain dominant, followed by species construction and JSON
loading in a benchmark that includes mixture setup.

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

The prototype now probes across the nearest cutoff in the direction calculated
from the analytical lowering gradient and selects the locally reachable root
with the lower dimensionless Gibbs objective. At 20862 K it deterministically
recovers both roots: they differ only by the 28th active Si+ level, and the
28-level branch is lower. A 0.05 K scan from 20860 to 20864 K also finds the
adjacent 27/28 and 28/29 transition pairs. The detected coexistence is narrow
and failed optional probes are harmless because the original root remains
valid.

This is a local selection rule, not a global search over all electronic active
sets. It is suitable evidence for the rewrite but should remain outside the
public API until the physical choice between a hard cutoff and a smooth
occupation model is settled.

## Assessment

The log formulation is worth continuing:

- positivity is structural;
- the full coupled Jacobian is accurate;
- cold midrange convergence is 5–9 iterations;
- warm continuation generally takes 2–4 iterations;
- ascending and descending sweeps pass from 1000 to 25000 K;
- even the deliberately unfused Python prototype is already faster.

The next performance experiment should vectorise the remaining ionisation-
lowering loop; restructuring the dense linear algebra is unlikely to pay at 16
species. A separate useful direction is a smooth electronic cutoff: it would
remove the local branch search and make the Jacobian globally continuous, at
the cost of changing the thermodynamic model rather than merely the solver.
