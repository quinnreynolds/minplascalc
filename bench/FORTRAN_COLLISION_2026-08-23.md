# Fortran collision-kernel floor

Snapshot: `feature/consolidate-computation-state` at `77f3192`, measured on
AC power on 2026-08-23. The machine is Apple arm64 with Python 3.11.6,
NumPy 2.1.3, Numba 0.61.0, SciPy 1.15.2, and Homebrew gfortran 16.2.0.

## Method

`fortran_collision_kernel.f90` is a direct translation of the production
Numba collision-pair kernel. Both implementations receive the same numeric,
temperature-independent pair descriptors. Species-object traversal and
potential-parameter derivation therefore sit outside both timed kernels.

`bench_collision_fortran.py` builds temporary shared libraries and removes
them on exit. It compares:

- the retained scalar Python pair loop;
- the production Numba kernel;
- gfortran `-O3 -march=native`;
- gfortran with `-ffast-math -funroll-loops -flto` as an aggressive floor.

Each isolated result is the median per call from seven batches. Fortran was
checked against Numba for the simple and SiCO mixtures at 2000, 12000, and
25000 K before timing.

Run with:

```console
PYTHONPATH=src .venv/bin/python bench/bench_collision_fortran.py
```

## Results

Sixteen species and all sixteen collision moments at 12000 K:

| Implementation | Time per evaluation | Relative to Numba |
|---|---:|---:|
| Scalar pair loop | 7132.1 us | 25.7x slower |
| Numba | 277.3 us | 1.00x |
| Fortran `-O3` | 63.2 us | 4.39x faster |
| Fortran fast-math | 61.5 us | 4.51x faster |
| Fortran fast-math, reused output | 59.7 us | 4.64x faster |

The conservative Fortran result differs from Numba by at most `1.67e-15`
relative across the validation sweep. Fast-math differs by at most `2.11e-15`.
Reusing the output array barely changes the result, so allocation and the
`ctypes` call boundary are not material here. Fast-math also buys very little;
ordinary `-O3` captures almost all the Fortran advantage.

The alternating five-run tutorial-10 workload was:

| Implementation | 60-state median | Property checksum |
|---|---:|---:|
| Numba | 1.010407 s | `4.882201205872014e+12` |
| Fortran fast-math | 1.008370 s | `4.882201205872014e+12` |

This is a 1.002x measured end-to-end change: indistinguishable from timing
noise. Direct instrumentation found 120 collision evaluations taking 19.41 ms
of a 1.0290 s sweep, or 1.89%. Even a zero-cost collision kernel is therefore
limited by Amdahl's law to 1.019x on this workload. A 4.5x kernel improvement
predicts only about 1.015x overall.

## Conclusion

Fortran establishes a credible collision-kernel floor around 60 us, roughly
4.5x below the current Numba call. It does not justify a production language
boundary by itself: the current Numba kernel has already reduced collision
evaluation to under 2% of the complete transport sweep. The useful implication
is that further end-to-end work should move to the new dominant costs rather
than optimize this kernel again.
