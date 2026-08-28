# Reduced equilibrium research benchmark — 2026-08-28

This records the Stage 5 decision benchmark for the research-only reduced
equilibrium formulation in `bench/reduced_equilibrium.py`. It is not a
production speed claim.

## Reproduction

The measurement used commit `39ecda2` plus the uncommitted Stage 5 harness on
an arm64 Mac running macOS 26.6.2, Python 3.11.6, NumPy 2.1.3, and SciPy
1.15.2. The machine was connected to mains power.

```console
PYTHONPATH=. uv run python bench/bench_reduced_equilibrium.py \
    --warmup 1 --repeats 7
```

Each workload sweeps 20 temperatures from 1,000 K to 25,000 K. The SiCO
workload repeats the sweep for SiO feed fractions 0.1, 0.5, and 0.9. Prototype
solvers bootstrap at 12,000 K and limit continuation steps to 1,000 K.
End-to-end equilibrium time includes mixture/system construction and the
equilibrium sweep. Tangent time is measured separately.

## Timing and convergence

| Workload | Solver | Dimension | Setup (s) | Equilibrium (s) | End-to-end (s) | Iterations | Residual evaluations | Continuation solves | Tangent (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Oxygen, 20 states | production | 7 | 0.002404 | 0.044522 | 0.047078 | 802 | — | 20 | — |
| | full log | 9 | 0.002530 | 0.014825 | 0.017316 | 158 | 208 | 50 | 0.002699 |
| | reduced | 4 | 0.002557 | 0.061350 | 0.063869 | 213 | 439 | 39 | 0.006969 |
| SiCO, 60 states | production | 16 | 0.025777 | 0.292786 | 0.318562 | 2,574 | — | 60 | — |
| | full log | 20 | 0.035355 | 0.082781 | 0.118104 | 610 | 787 | 150 | 0.014873 |
| | reduced | 6 | 0.026404 | 0.385151 | 0.412530 | 727 | 1,531 | 117 | 0.043061 |

All recorded sweeps completed without failures and requested-state residuals
were below `1e-9`. For SiCO, the full log solver is 2.697 times faster than
production end to end. The reduced prototype is 0.772 times production speed
(1.295 times slower) and 3.493 times slower than full log. For oxygen, reduced
is 3.688 times slower than full log.

The smaller dense system therefore does not currently translate into lower
wall time. The reduced trust-region path performs more residual evaluations,
and each evaluation reconstructs lowering, reference energies, and partition
data through Python-level species loops. SciPy's internal trust-region linear
algebra is not visible through `numpy.linalg.solve`, so the harness deliberately
does not present its observed NumPy solve time as a complete dense-solve cost.

Representative warm-state component probes explain the result. For SiCO, a
full-log packed-thermodynamics call took 69.11 microseconds and its combined
residual/Jacobian call took 91.65 microseconds. Reduced lowering alone took
12.64 microseconds, but reconstruction took 157.32 microseconds and the
combined reduced residual/Jacobian call took 219.36 microseconds. Direct NumPy
dense solves took 5.04 microseconds for the 20-dimensional full system and
3.24 microseconds for the 6-dimensional reduced system. The linear algebra
saving is real but is overwhelmed by reconstruction and the larger number of
trust-region evaluations.

## Accuracy and roots

Against the full log continuation:

| Workload | Production max absolute mole-fraction error | Reduced max absolute mole-fraction error | Reduced max absolute log-density error |
|---|---:|---:|---:|
| Oxygen | `4.959e-12` | `4.810e-11` | `4.267e+01` |
| SiCO | `1.241e-08` | `3.405e-10` | `8.536e+01` |

The large log-density errors are expected cold trace-root differences, not
bulk-composition errors. Tests independently map the full-log cold root into
the reduced coordinates and recover a reduced residual below `1e-8`. Thus the
full root is present, while continuation can select a different nearly
unobservable lowering-closure root.

## Conditioning

For oxygen and SiCO at 12,000 K and 20,000 K over 1,013.25 Pa, 101,325 Pa,
and 10.1325 MPa, one-pass row/column equilibration gives:

| Workload | Formulation | Raw condition min / median / max | Equilibrated condition min / median / max |
|---|---|---:|---:|
| Oxygen | full log | 18.62 / 23.94 / 529.34 | 7.03 / 8.62 / 11.33 |
| | reduced | 3.88 / 4.85 / 105.16 | 3.00 / 3.01 / 3.01 |
| SiCO | full log | 29.82 / 41.48 / 154.95 | 16.65 / 20.96 / 34.14 |
| | reduced | 6.40 / 8.47 / 22.25 | 3.13 / 3.73 / 6.85 |

The reduced warm/hot systems are materially better conditioned under the same
simple equilibration diagnostic. This advantage does not persist uniformly at
1,000 K: trace-ionisation closure degeneracy produces very large raw condition
numbers, and the equilibrated SiCO reduced Jacobian remains between about
`2.8e11` and `3.2e14` over the three pressures. Continuation is consequently a
required part of the present formulation, not an optional optimization.

## Decision

Keep the reduced formulation on the research branch. It demonstrates exact
root reconstruction, fewer nonlinear variables, useful fixed-active-set
tangents, improved warm/hot conditioning, and precise cutoff diagnostics. It
does not yet justify production integration: it is slower than both existing
formulations, has a narrower independent-start basin, and retains cold
trace-root multiplicity.

The next performance experiment, if pursued, should fuse and vectorize the
reconstruction/thermodynamic kernel and reduce duplicate residual/Jacobian
work inside the trust-region solver. Only after that should the full benchmark
be repeated. A production proposal should remain separate from this research
study.
