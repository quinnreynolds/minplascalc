# Symbolic Devoto collision-bracket audit

`devoto_brackets_symbolic.py` independently derives the explicit collision
brackets used by the Devoto transport matrices. It is a development tool, not
a production dependency.

The derivation begins with elastic two-body collision kinematics and the
Sonine/Laguerre trial functions. It evaluates the centre-of-mass Maxwellian
integrals as exact Gaussian moments, reduces the scattering-angle polynomial
to transport cross sections, and applies the normalization of equation 11 in
Devoto (1966). Coefficients remain exact SymPy rationals throughout.

The vector basis reproduces every upper block in equations A3-A18. For A3 it
first derives the raw `(0, 0)` collision bracket `R`. The mass-flux subsidiary
condition has weights `w_i = n_i sqrt(m_i)`, so adding an outer product with
`w` does not alter the bracket on an admissible coefficient vector. Devoto's
zero-diagonal form follows directly as

```text
q00 = R - outer(diag(R) / w, w).
```

The symbolic order-zero pair moments are particularly small:

```text
self =  2 B^2 Q^(1,1)
cross = -2 A B Q^(1,1),
A^2 = m_i / (m_i + m_l),  B^2 = m_l / (m_i + m_l).
```

After the Maxwellian prefactor is applied, they give

```text
R_ii =  8 n_i sum_(l != i) n_l sqrt(m_l / (m_i + m_l)) Q_il^(1,1)
R_ij = -8 n_i n_j sqrt(m_i / (m_i + m_j)) Q_ij^(1,1),  i != j.
```

Substitution into the constraint transformation gives
`q00_ij = R_ij - R_ii n_j sqrt(m_j) / (n_i sqrt(m_i))`, whose expanded form
is A3. The diagonal cancels identically. This derivation is useful because it
checks both the collision algebra and the otherwise easy-to-miss constraint
transformation.

The traceless-tensor basis reproduces all viscosity blocks in A19-A22.

## Checks

The representative automated test derives all viscosity blocks and vector
blocks A3, A4, A7, A12, A11, and A16:

```console
PYTHONPATH=. .venv/bin/pytest -q tests/test_devoto_symbolic_brackets.py
```

The complete fourth-order check, including the expensive `(3, 3)` block A18,
is available explicitly:

```console
PYTHONPATH=. .venv/bin/python bench/devoto_brackets_symbolic.py --check
```

On a typical development machine the representative test takes roughly 25
seconds and the complete derivation roughly one minute. SymPy is intentionally
restricted to real variables, with the two positive mass coefficients marked
positive as well. The implementation avoids unrestricted `simplify()` calls;
it operates on polynomial domains and applies targeted factoring only after
the Gaussian and angular moments have been collected.

Run the module without `--check` to print every exact pair-bracket expansion:

```console
PYTHONPATH=. .venv/bin/python bench/devoto_brackets_symbolic.py
```
