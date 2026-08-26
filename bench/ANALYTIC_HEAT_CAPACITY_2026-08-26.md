# Analytical equilibrium heat capacity

This experiment addresses temperature-derivative outliers in heat capacity
and thermal conductivity. It builds on the piecewise analytical equilibrium
tangent introduced earlier on `feature/consolidate-computation-state`.

## Root cause

Thermal conductivity already consumes the analytical mole-fraction derivative
on this branch. Heat capacity was the remaining property that changed the
mixture temperature to `T(1 +/- 0.001)`, performed two independent equilibrium
solves, and differenced their enthalpies.

That calculation has three distinct error sources:

1. ordinary second-order truncation error;
1. convergence or branch differences between the two equilibrium solves;
1. a finite-difference interval straddling a hard electronic-level cutoff,
   which turns a discrete partition-function change into a step-dependent
   local spike.

The third effect is inherent to differencing a discontinuous model. Reducing
the temperature step narrows and increases such an excursion rather than
providing a globally convergent derivative.

## Formulation

For the enthalpy per unit mass written in mole fractions as

```text
H = sum_i(x_i A_i) / sum_i(x_i m_i),
```

the implementation applies the quotient rule using the implicit equilibrium
`dx/dT`. Here `A_i` contains species internal energy, reference energy, the
`k_B T` pressure term, and the existing zero-temperature reference shift.

The full reference-energy tangent includes both explicit temperature
dependence and the composition-dependent ionisation lowering:

```text
dE0/dT = partial(E0)/partial(T) + partial(E0)/partial(N) @ dN/dT.
```

Species internal-energy derivatives are analytical. Monatomic electronic heat
capacity is the fixed-active-set energy variance divided by `k_B T^2`;
diatomic and polyatomic vibrational terms use stable hyperbolic-function
forms. The result is piecewise analytical: the active electronic levels and
the minimum-reference-energy species remain fixed at a discrete crossing.

## Verification

At 10000 K and atmospheric pressure:

| Mixture | Analytical Cp | Error of old `1e-3` difference | Error at `1e-5` |
|---|---:|---:|---:|
| Oxygen | 3248.946038 | +0.012634 | +0.0000013 |
| SiCO | 5504.183153 | +0.010732 | +0.0000013 |

The shrinking-step result converges quadratically to the analytical value.
Species energy derivatives independently agree with central differences to a
relative tolerance of `2e-8` from 1000 to 25000 K.

A 25 K SiCO scan found a maximum `43.5 J/(kg K)` change in the old result when
its relative step was changed from `1e-3` to `3e-4`. The largest discrepancies
occur in paired points around electronic cutoffs. The analytical result stays
finite and follows the local branch without averaging across the jump.

## Performance

Command:

```console
PYTHONPATH=src:. .venv/bin/python bench/bench_heat_capacity_derivative.py
```

For 20 temperatures and three SiCO ratios, with seven alternating repetitions:

| Method | Median |
|---|---:|
| Two-solve finite difference | 1.005774 s |
| Analytical equilibrium tangent | 0.375863 s |

The analytical implementation is **2.68x faster**. It also ignores the legacy
`rel_delta_T` argument, which is retained for API compatibility.

## Remaining discontinuity

This removes numerical derivative excursions, but it cannot make the hard
electronic cutoff itself differentiable. At a level crossing the model still
has distinct one-sided values. The local Gibbs branch selection explored by
the log solver, or a physically justified mollified occupation model, remains
the route to defining behaviour exactly at those crossings.
