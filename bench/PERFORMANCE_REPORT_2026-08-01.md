# minplascalc performance: evidence for issue #82

> Snapshot taken **2026-08-01**, against `main` at `43bff40`. Findings
> describe the code as it stood then; several were acted on in the same
> branch, so check `git log` before treating any item here as outstanding.

Investigation of four proposed approaches to speeding up minplascalc, with
measurements. All numbers are from this machine (Darwin 25.5.0, Python
3.11.6, numpy 2.1.3, numba 0.61.0, scipy 1.15.2, pytensor 3.0.7). Wall-clock
figures repeat to within about ±3%.

## TL;DR

**Issue #82's premise no longer holds.** When the issue was written, the
q-matrix assembly was 68% of `q()`. The `@njit` work already merged in
PR #83 cut that to **1.4%**. All four options under investigation target
that 1.4%, so none of them can produce a meaningful end-to-end win — and
measured end-to-end, none of them does.

| Option | q-assembly kernel | Tutorial-10 workload | Verdict |
|---|---|---|---|
| 1. Manual vectorisation | 1.04× **slower** than njit | **0.99×** | Not worth it for speed |
| 2. pytensor compiled | 4.25× faster than njit | **1.00×** | Real kernel win, invisible end-to-end |
| 3. pytensor + derivatives | 0.52× (grad costs 2.2× fwd) | n/a | No target exists — see §5 |
| 4. numba `@njit` (already merged) | 149× faster than loops | **2.10×** | Already banked the win |

The time is somewhere else. Two changes found by profiling deliver
**2.05× on top of the njit work** — 4.15× against the pre-optimisation
baseline — with all 55 tests passing and results identical to roundoff:

| Change | Tutorial-10 workload | Speedup |
|---|---|---|
| baseline (`main` today) | 35.4 s | 1.00× |
| vectorise `Monatomic.internal_partition_function` | 22.3 s | **1.63×** |
| cache (l,s)-independent collision parameters | 31.2 s | 1.16× |
| both | 17.7 s | **2.05×** |

______________________________________________________________________

## 1. Method

Workload: the `examples/plot_tutorial_10_SiCO_plasma_transport_properties.py`
path — a 15-species SiCO plasma (16 with electrons), 20 temperatures from
1000–25000 K, 3 mixture ratios, computing viscosity, electrical
conductivity, emission coefficient and thermal conductivity at each of the
60 states.

The test suite was measured too but is **not** a useful performance target:
it runs in 8.6 s, of which 2.25 s is the one-off numba compile in the first
thermal-conductivity test. The examples are where the real compute is.

Every comparison below checks numerical equivalence. Historical revisions
were benchmarked in git worktrees with the same scripts (`bench/`), and the
q-matrix checksum is byte-identical across all of them
(`1.1477534513e+33`), as is the end-to-end property checksum
(`4.882221572370e+12`).

Harness is in `bench/`; each script is standalone.

## 2. Where the time actually goes

`cProfile` on the tutorial-10 workload, top leaf costs:

```
   ncalls  tottime  cumtime  function
    93816   16.813   16.813  species.py:292  Monatomic.internal_partition_function
1508400/…   10.989   14.761  functions_transport.py:1293  Qin
 542160/…    3.947    5.345  functions_transport.py:1178  Qnn
  1148400    2.274    2.274  functions_transport.py:587   pot_parameters_ion_neut
   426000    1.130    4.351  functions_transport.py:1483  Qc
     4260    0.842   27.025  functions_transport.py:1638  Qij_mix
     5340    0.239   18.684  mixture.py:346               calculate_composition
    10484    0.135    0.240  numpy/linalg  solve          ← the Gibbs minimisation
```

The `_qXX_jit` functions — the code issue #82 proposes to rewrite — do not
appear in the top 35 entries at all.

Uninstrumented breakdown of one `q()` call (16 species, T = 12000 K):

```
  q() full                     74.57 ms
    16x Qij_mix (collision integrals)  73.53 ms   98.6%
    q-matrix assembly                   1.04 ms    1.4%   ← options 1-4 target this
```

**This is the central finding.** Amdahl's law caps every option in this
investigation at 1.4% of `q()`, and `q()` is itself ~65% of the workload.

## 3. What the njit work already did (option 4)

Benchmarking the same assembly at three points in history, with collision
integrals frozen so only assembly is timed:

| Revision | q-assembly | `q()` total | assembly share |
|---|---|---|---|
| `8cbf07e` before PR #83 | 159.91 ms | 233.67 ms | 68.4% |
| `0074213` after manual vectorisation of simple terms, before njit | 159.91 ms | 240.75 ms | 66.4% |
| `43aead0` `main` today (njit) | **1.07 ms** | 76.91 ms | **1.4%** |

`@njit` gave **149× on the assembly** and 3.1× on `q()`. End-to-end on the
tutorial-10 workload: **73.65 s → 35.00 s, a 2.10× speedup**, with an
identical result checksum.

So option 4 is not just "also worth evaluating" — it already captured
essentially the entire win that was available in this code, and it did so
before the manual vectorisation in the issue was attempted.

## 4. Options 1 and 2 measured head-to-head

All ten q-elements were transcribed into (a) broadcast numpy
(`bench/q_vectorised.py`, following the recipe in the issue verbatim) and
(b) a compiled pytensor graph (`bench/q_pytensor.py`). Both agree with the
njit versions to **1.4e-15 relative** — i.e. roundoff.

Total assembly time for all ten elements (µs):

| species | njit (opt 4) | numpy vec (opt 1) | pytensor (opt 2) | pytensor+grad (opt 3) |
|---:|---:|---:|---:|---:|
| 6 | 65.9 | 572.9 (8.7× slower) | 50.0 | 68.2 |
| **16 (realistic)** | **1003.1** | **1050.9 (1.05×)** | **235.8 (0.24×)** | 520.8 |
| 24 | 3358.5 | 2160.0 (0.64×) | 665.4 (0.20×) | 1532.3 |
| 40 | 15466.1 | 6680.6 (0.43×) | 2984.3 (0.19×) | 7206.6 |
| 64 | 62887.1 | 26897.1 (0.43×) | 28682.9 (0.46×) | 48290.0 |

Two things stand out:

- **Manual vectorisation is a wash at realistic sizes.** At 16 species it is
  4% *slower* than the njit loops. It only starts winning above ~24 species,
  and it is catastrophically worse (8.7×) for small mixtures, where numpy's
  per-operation overhead dominates 6³ = 216-element arrays.
- **pytensor genuinely is 4.25× faster than numba** on this kernel at 16
  species, by fusing the broadcast expressions and avoiding the temporaries
  numpy materialises.

But when each is dropped into the real code path and the whole workload is
re-run:

```
  option                                  time (s)   speedup   max rel err
  4  numba njit (current main)               35.44     1.00x       0.0e+00
  1  numpy vectorised q-assembly             35.96     0.99x       2.2e-23
  2  pytensor compiled q-assembly            35.40     1.00x       1.0e-22
     (+ 0.8 s one-off graph compile, per species count)
```

pytensor's 4.25× on the kernel converts to **no measurable end-to-end
change**, because it is 4.25× on 1.4% of `q()`. That is the whole story of
this investigation in one line.

Against that, option 2 costs: a new dependency, ~0.8 s of graph compilation
per distinct species count (the Kronecker deltas and shapes are baked in as
constants), and a second implementation of the Devoto appendix expressions
to keep in sync with the reference.

## 5. Option 3: the gradient/Hessian hypothesis

The intuition in the brief was that supplying gradients and a Hessian would
speed up the Gibbs energy minimisation. The measurements say this target
does not exist, for two independent reasons.

**The minimiser is already exact Newton with an analytic Hessian.**
`calculate_composition()` is not a scipy optimiser call — it is a
hand-rolled Lagrange-multiplier Newton iteration
(`mixture.py:525-607`). Its `gfe_matrix[:nb, :nb]` block is built as
`-kT/N_tot + diag(kT/N_i)` (`mixture.py:551-553`), and since
`μ_i = E⁰_i - kT[ln N_tot + ln(z_i kT/P) - ln N_i]`,

```
∂μ_i/∂N_j = -kT/N_tot + δ_ij · kT/N_i
```

which is exactly that block. The bordered system with the stoichiometry
constraints `A`/`Aᵀ` is the KKT system for the equality-constrained Newton
step. The exact gradient and Hessian are already there, in closed form.
Automatic differentiation would at best rediscover them, more slowly.

**The linear algebra is negligible anyway.** Instrumenting every Newton step
across the temperature sweep:

```
  mean Newton iterations per composition: 42.5
  mean cost per iteration:  1.448 ms, of which linear solve 0.0119 ms
  linear solve as share of composition time:  0.82%
  one full set of total_partition_function() calls: 1.293 ms  (~89% of an iteration)
```

**99% of the minimisation cost is function evaluation, not linear algebra.**

The steelman of option 3 is "use better derivatives to need fewer than 42.5
iterations". Even that has almost no headroom: once the partition-function
fix in §6 lands, the *entire* Gibbs minimisation drops to **2.2% of the
workload** (0.38 s of 17.0 s). A perfect optimiser converging in one step
would save under 2%.

For completeness, reverse-mode AD through the q-graph was measured anyway:
`pt.grad` roughly doubles the cost of the forward pass (236 → 521 µs at 16
species; 2.2× at 16, 2.3× at 24, 2.4× at 40). It is not free.

## 6. What the data says to do instead

### A. Vectorise `Monatomic.internal_partition_function` — 1.63×

This single function is **35% of the profiled workload**. It is a Python
loop over up to 580 energy levels calling scalar `np.exp` once per level
(`species.py:333-347`), costing ~0.5 µs per level.

Replacing it with two cached arrays, one vectorised `exp` and one dot
product (`bench/pf_vectorised.py`):

```
  species  levels  summed  loop (us)   vec (us)   speedup    rel err
  O           580     234     157.30       5.21     30.2x   0.00e+00
  C           433     401     264.96       5.92     44.7x   9.97e-16
  Si          540     452     298.26       5.93     50.3x   2.77e-16
  …
  composition sweep     3.72 s ->  0.40 s   (9.39x)
  all-four sweep       36.15 s -> 22.31 s   (1.63x)
```

⚠️ **One subtlety that must be preserved.** The loop `break`s at the first
level at or above the ionisation cutoff, and `energy_levels` is **not sorted
by energy** (verified for O, O+, C, Si). So the sum runs over a *prefix* of
the list, not over all levels below the cutoff — for O, 234 of 580 levels.
A natural mask-based rewrite (`E < cutoff`) would silently change every
number in the package. The prototype reproduces the prefix semantics with
`argmax` on the predicate and matches to 1e-15.

Whether that early `break` is *intended* is a separate question worth
raising — it looks like it assumes sorted input. It should be settled before
or alongside this change, but as a correctness question, not a performance
one.

### B. Stop recomputing (l,s)-independent collision parameters — 1.16×

`Qij_mix` is called 16 times per `q()`, once per (l,s) pair, and each call
re-derives the interaction-potential parameters for every species pair. Those
depend only on the species pair, never on (l,s). Counting calls in a single
`q()` (16 species → 256 distinct pairs):

```
  function                        calls   per (l,s)   distinct pairs
  beta                             6264       391.5              256
  pot_parameters_ion_neut          4608       288.0              256
  x0_ion_neut                      4608       288.0              256
  pot_parameters_neut_neut         1656       103.5              256
  cl_charged                       1600       100.0              256
```

An `lru_cache` prototype (`bench/collision_cache.py`) gives **1.16×**
end-to-end with bit-identical results. That is a lower bound and a
measurement device, not the proposed patch — the clean fix is to restructure
`Qij_mix` to loop over species pairs outermost and (l,s) innermost, which
removes the redundancy structurally and avoids any cache-invalidation
question. That restructuring is also what would make a genuinely vectorised
collision-integral path possible, which is where the remaining ~75% of the
runtime lives.

### Combined

```
  variant                                    time (s)   speedup   max rel err
  baseline (main, numba njit)                   36.38     1.00x       0.0e+00
  + vectorised partition function               22.31     1.63x       1.0e-15
  + cached collision-integral parameters        31.24     1.16x       0.0e+00
  + both                                        17.74     2.05x       1.0e-15
```

Full test suite: **55 passed** with both changes applied, suite time
8.56 s → 4.84 s.

## 7. Recommendation

1. **Close issue #82 as overtaken by events**, referencing this report. The
   njit work in PR #83 solved the problem the issue describes; the remaining
   assembly cost is 1.4% of `q()`. If the vectorised forms are wanted, want
   them for readability — they are exact, and `bench/q_vectorised.py` is
   ready — but they are 4% *slower* at realistic species counts and 8.7×
   slower for small mixtures, so this is a readability trade, not a
   performance one.
1. **Do not adopt pytensor.** Its 4.25× on the kernel is real and it was the
   best-performing option measured, but it buys 0% end-to-end while adding a
   dependency, per-species-count compile latency, and a duplicate
   implementation of the Devoto expressions.
1. **Drop the gradient/Hessian idea for the composition solver.** The
   analytic Hessian is already there and exact; the solve is 0.8% of
   composition, and composition falls to 2.2% of the workload after (A).
1. **Do (A) — vectorise the partition function.** Biggest single win in the
   codebase, 1.63× end-to-end, small and self-contained. Settle the
   sorted-levels/`break` question as part of it.
1. **Then do (B) properly** — restructure `Qij_mix` so (l,s) is the inner
   loop. Worth 1.16× on its own, and it is the prerequisite for attacking
   `Qin`/`Qnn`/`Qc`, which are ~75% of what remains and the only place a
   large further win is available.

## 8. Reproducing

```bash
uv run python bench/profile_baseline.py transport --n-T 20   # §2 profile
uv run python bench/bench_assembly.py current                # §2, §3 split
uv run python bench/bench_options.py                         # §4 opt 1 vs 4
uv run python bench/bench_pytensor.py                        # §4 opt 2, 3
uv run python bench/bench_composition.py                     # §5
uv run python bench/bench_hotspots.py                        # §6 A and B
uv run python bench/bench_combined.py                        # §6 combined
uv run python bench/bench_end_to_end_options.py              # §4 end-to-end
PYTHONPATH=bench MPC_PATCH=pf,cache uv run pytest tests -q -p pf_plugin
```

`bench/bench_assembly.py` and `bench/sweep_standalone.py` are dependency-free
and can be copied into a git worktree of any revision to reproduce the
historical comparisons in §3.
