# minplascalc: redundancy, library-routine and low-hanging-fruit review

A whole-repository pass over `src/minplascalc/` looking for three things:
mathematical redundancy that numpy contractions express better, manual
implementations that a library routine already does, and cheap wins.

Every claim below is either measured on this machine or verified numerically
against the current implementation. Verification scripts are in `bench/`.

## Headline

Two changes, neither of which alters any result, are worth **4.77x** together
on the tutorial-10 workload:

| change | workload | speedup | max rel err |
|---|---|---|---|
| baseline (`main`) | 34.41 s | 1.00x | — |
| memoise `Qij_mix` on mixture state | 20.33 s | **1.69x** | **0.0** (bit-identical) |
| that + vectorised partition function ([#91](https://github.com/quinnreynolds/minplascalc/issues/91)) | 7.21 s | **4.77x** | 1.0e-15 |

The `Qij_mix` result is the one I did not expect: **77% of all collision-integral
evaluations in a property sweep are recomputations of a value already in hand**
(3300 of 4260 across the sweep).

______________________________________________________________________

## 1. Redundancy and expressions that want to be contractions

### 1.1 `Dij` does nb_species² linear solves where one multi-RHS solve suffices

`functions_transport.py:2536-2560`. `Dij` LU-factorises `q` once, then solves
inside a double loop — 256 solves for a 16-species mixture:

```python
for i in range(nb_species):
    for j in range(nb_species):
        dij = np.array([delta(h, i) - delta(h, j) for h in range(0, nb_species)])
        b_vec[:nb_species] = 3 * np.sqrt(np.pi) * dij
        cflat = scl.lu_solve(lu_piv_q, b_vec)
```

The right-hand sides are `3√π (e_i − e_j)`. They span an nb_species-dimensional
space, and the solution is linear in the RHS, so solving once against the
nb_species unit vectors gives everything:

```python
B = np.zeros((4 * nb, nb))
B[:nb, :] = 3 * np.sqrt(np.pi) * np.eye(nb)
X = scl.lu_solve(lu_piv_q, B)          # one call, nb right-hand sides
c0 = np.diag(X[:nb])[:, None] - X[:nb] # c^{ji}_{i0} for every (i, j)
```

Verified equal to 8e-16 at T = 2000/8000/12000/20000 K
(`bench/check_dij.py`). The solve stage drops from 3.70 ms to 0.010 ms —
**366x** — though `Dij` is dominated by its `q()` call, so the end-to-end
effect is small. Take this one for clarity: it removes a double loop, an
inner `np.array` allocation, and 255 redundant solves.

This is also the direct answer to "did I find all the inverse-vs-solve
places": the remaining issue is not `inv` anywhere, it is *repeated* solves
against one factorisation.

### 1.2 `electrical_conductivity` computes the whole diffusion matrix for one row

`functions_transport.py:2754-2770` calls `Dij(mixture)` — all nb² entries,
currently 256 solves — and then uses `Dij(mixture)[-1, :]`. With 1.1 it needs
one column of `X`.

Its accumulation loop is a dot product:

```python
sum_val = 0.0
for charge_number, D1j, mj, nj in zip(charge_numbers, D1, masses, number_densities):
    sum_val += nj * mj * charge_number * D1j
```

→ `number_densities @ (masses * charge_numbers * D1)` (verified exactly equal).

### 1.3 `thermal_conductivity`'s reaction-enthalpy double loop is a bilinear form

`functions_transport.py:2951-2957`:

```python
for j in range(nb_species):
    for i in range(nb_species):
        krxn_enth += masses[j] * masses[i] * hv[i] * locDij[i, j] * dxdT[j]
```

→ `(masses * hv) @ locDij @ (masses * dxdT)` (verified, 7e-16).

### 1.4 The Monatomic level sums

`species.py:339-346` and `species.py:402-409` — both are `g @ exp(-beta E)`
style contractions written as Python loops with a scalar `np.exp` per level.
Covered in detail in [#91](https://github.com/quinnreynolds/minplascalc/issues/91);
worth 1.63x on its own. Note the `break` semantics caveat there.

### 1.5 `total_emission_coefficient` nested loop

`functions_radiation.py:116-134` loops over species and then over each
species' emission lines, doing scalar `np.exp` per line. `emission_lines` is
already an array per species, so the inner loop is
`line_pre_constant * nv / Qi * (gA / wavele * np.exp(-Ek / kbT)).sum()`.

Separately: the slice `zip(nd[:-1], mix.species[:-1])` hard-codes "the
electron is the last species". Given the electrons-optional work on
`mixture-without-electrons-object`, that assumption is worth making explicit
rather than positional.

### 1.6 The q/qhat assembly

Already characterised in `bench/PERFORMANCE_REPORT.md`: exact numpy
vectorisations of all ten q-elements exist in `bench/q_vectorised.py` and
agree to 1.4e-15, but they are **4% slower than the njit loops at 16 species**
and 8.7x slower at 6. Adopt them if you prefer how they read — not for speed.

______________________________________________________________________

## 2. Manual implementations with a library equivalent

### 2.1 Repeated factorisation of the same matrix

`np.linalg.solve` is called on `qq` in three places — `DTi`
(`functions_transport.py:2631`), `thermal_conductivity`
(`functions_transport.py:2912`), and `Dij` via `lu_factor`
(`functions_transport.py:2540`) — each re-factorising a matrix that is the
same within a single `thermal_conductivity` call. One `scl.lu_factor` shared
across them replaces three O(n³) factorisations with one. (See 3.1: the `q()`
call producing that matrix is itself repeated three times, which is the
larger cost.)

### 2.2 `delta()` in a Python list comprehension

`functions_transport.py:2546`. `delta` is `@njit`-compiled, then called
nb_species times per iteration from interpreted code — the dispatch overhead
per call exceeds the work. `np.eye(nb)[:, i] - np.eye(nb)[:, j]` is
equivalent (verified for all 256 pairs) and disappears entirely under 1.1.

### 2.3 `sum1`, `sum2`, `psiconst` rebuild arrays to sum ≤ 8 terms

`functions_transport.py:1031`, `1057`, `859`. Each is a pure function of a
small integer `s ∈ {1..7}` and each allocates via `np.arange` then `np.sum`.
They are called 1600 times per `q()`. A module-level lookup table indexed by
`s`, or `functools.cache`, removes them from the profile entirely.

Likewise `A(ionisation_energy)` and `B(ionisation_energy)`
(`functions_transport.py:889`, `957`) are pure functions of a per-species
constant, recomputed 1600 times per `q()` between them, and both recompute
`np.sqrt(np.pi)` each call.

### 2.4 `np.dot(..., out=np.zeros(...))`

`functions_transport.py:1265` and `1380`:

```python
a = np.dot(c_nn[l - 1, s - 1], beta_array, out=np.zeros((7,), dtype=np.float64))
```

The `out=` argument is handed a freshly allocated array, so it saves nothing
over `c_nn[l - 1, s - 1] @ beta_array`. Harmless, but it reads as though an
allocation is being avoided when it is not.

### 2.5 Static species properties rebuilt on every call

```python
masses = np.array([sp.molar_mass / u.N_a for sp in mixture.species])
```

appears **seven times** (`functions_transport.py:1700, 2350, 2532, 2623, 2697, 2756, 2894`), plus `charge_numbers` at 2754 and equivalents in
`mixture.py`. These depend only on the species list. A cached property on
`LTE` — invalidated by the `species` setter, which already exists — removes
them. The profile shows 1.6M `numpy.array` calls in a 20-point sweep.

______________________________________________________________________

## 3. Low-hanging fruit, largest first

### 3.1 The same collision integrals are computed 4.4x over — 1.69x, bit-identical

Counting calls for one temperature point (`bench/count_calls.py`):

| property | `q()` | `qhat()` | `Qij_mix` |
|---|---:|---:|---:|
| viscosity | 0 | 1 | 7 |
| electrical_conductivity | 1 | 0 | 16 |
| total_emission_coefficient | 0 | 0 | 0 |
| thermal_conductivity | **3** | 0 | **48** |
| **total** | | | **71** |

`thermal_conductivity` reaches `q()` three times at the same temperature and
composition — directly at `functions_transport.py:2906`, via `DTi` at 2930,
and via `Dij` at 2949. And `qhat`'s seven (l, s) pairs
`{11, 12, 13, 22, 23, 24, 33}` are a strict subset of `q`'s sixteen.

So all four properties at one temperature need **16** `Qij_mix` evaluations
and perform **71**. Memoising on mixture state gives **1.69x end-to-end with
bit-identical results** (`bench/qij_memo.py`), and confirms 77% of the calls
were redundant.

The cache is the measurement, not the fix. The fix is for the property
functions to accept a precomputed `q`/`qhat` (or for the mixture to cache
them alongside the composition, invalidated by the same `T`/`P` setters that
already clear `__isLTE`).

### 3.2 The `Qnn`/`Qin` recursion triples the base evaluations

`functions_transport.py:1243-1255` and `1358-1370`. For higher `s`, eq. 18 of
Laricchiuta is applied as a *recursive central finite difference in T*:

```python
negT, posT = T - 0.5, T + 0.5
return Qin(..., s - 1, T) + T / (s + 1) * (Qin(..., s - 1, posT) - Qin(..., s - 1, negT))
```

Each level triples the call count. Measured over one `q()` call:

| | base evaluations | (l, s) values needed | amplification |
|---|---:|---:|---:|
| `Qin` | 4608 | 1536 | **3.0x** |
| `Qnn` | 1656 | 576 | **2.9x** |

Three observations:

1. The leaves evaluate the same base function at `T + k/2` for small integer
   `k` — at most 5 distinct temperatures. Memoising on
   `(pair, l, s, T)` recovers roughly half of the amplification.
1. Better, the base collision integral is a closed-form differentiable
   expression, so the temperature derivative in eq. 18 can be taken
   analytically instead of by finite difference — removing the recursion
   entirely *and* being more accurate.
1. ~~The step is a hardcoded ±0.5 K magic number.~~ **Corrected:** it is a
   unit-step central difference — ±0.5 is chosen so the `2h` divisor equals
   1 and can be omitted. It is at or near optimal; the nested difference is
   round-off limited at ~1e-9 regardless of step. See
   [ANALYTIC_RECURSION.md](ANALYTIC_RECURSION.md), which derives the
   recursion, shows it is exact, and implements the analytic derivative
   (1.36x on its own, all 55 tests passing).

### 3.3 `__get_reference_energies` redoes structural work every Newton iteration

`mixture.py:537` calls it inside the minimiser loop, which runs ~42 times per
composition. The ionisation-energy lowering genuinely depends on the current
`n_e` and `z*`, but the surrounding work — grouping species by stoichiometry,
building and sorting the charged-species ladders (`mixture.py:275-310`) — is
a pure function of the species list. Computing that once per species set and
reusing the ordering removes 0.5 s of the 48 s profiled sweep (~2%).

### 3.4 Cheap items

- `Polyatomic.internal_partition_function` (`species.py:921-927`) builds a
  Python list inside `np.prod`; `wi_e` is short so this is cosmetic.
- ~~The `governor_factors` ladder restarts the minimisation from
  `gfe_initial_particles` on each failure.~~ **Corrected:** it does not.
  `__Ni` is initialised once at `mixture.py:511`, *before* the governor
  loop, so a step-down warm-starts from the last iterate with a tighter
  step cap. What *is* discarded is the previous temperature's answer: every
  `calculate_composition()` call restarts from `gfe_initial_particles`, so a
  sweep never warm-starts across T. That is why low-temperature compositions
  cost 182 Newton iterations against 24-32 at high temperature. See
  `bench/check_minimiser.py` and the notes on issue #16.

______________________________________________________________________

## Suggested order

1. **`Qij_mix` recomputation (3.1)** — 1.69x, bit-identical, no numerical
   decisions to make. Best return in the codebase.
1. **Partition function ([#91](https://github.com/quinnreynolds/minplascalc/issues/91))** —
   1.63x, but settle the `break` semantics first.
1. **`Qnn`/`Qin` recursion (3.2)** — 3x amplification on the dominant
   remaining cost. Memoisation is safe; the analytic derivative is better and
   changes results slightly (for the better), so it needs a decision.
1. **`Dij` multi-RHS (1.1)** and the contractions in 1.2/1.3 — small
   performance effect, meaningful clarity gain, all verified exact.
1. **Static-property caching (2.5)** and the pure-function tables (2.3) —
   trivial, mechanical.

Items 1, 4 and 5 change no results at all. Items 2 and 3 involve a semantic
decision each, and should not be done silently.

## Reproducing

```bash
uv run python bench/count_calls.py     # 3.1 call counts
uv run python bench/check_dij.py       # 1.1 equivalence and timing
uv run python bench/bench_hotspots.py  # partition function, parameter reuse
```
