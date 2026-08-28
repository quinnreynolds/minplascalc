# Gibbs free-energy equilibrium solver

This page describes the equilibrium calculation implemented by
`LTE.calculate_composition`, then compares it with the experimental coupled
log-space formulation in `bench/log_equilibrium.py`. The log-space solver is a
research prototype on the development branch; it is not yet part of the public
API.

The two formulations solve the same constrained thermodynamic problem. They
differ in their nonlinear variables, residual scaling, globalization, and
treatment of positivity.

## Problem definition

The participating species are fixed when an `LTE` mixture is constructed. The
solver changes their amounts; it does not create or remove species from that
set. An electron species is appended automatically for electron-containing
mixtures.

For $s$ species and $c$ conservation laws, define:

- $N_i > 0$: the particle number of species $i$;
- $N\_{\\mathrm{tot}}=\\sum_i N_i$;
- $V=N\_{\\mathrm{tot}}k_BT/P$: the ideal-gas volume at the requested pressure;
- $A\\in\\mathbb{R}^{s\\times c}$: the constraint matrix;
- $b\\in\\mathbb{R}^{c}$: the conserved elemental totals, with zero as the
  charge-neutrality target;
- $E_i^0$: the species reference energy, including ionisation lowering; and
- $Z_i(V,T,\\Delta E)$: the total single-species partition function.

Each element contributes one column to $A$. Its entries are the species
stoichiometric coefficients. For example, the row for $\\mathrm{CO_2}$ has one
in the carbon column and two in the oxygen column. Molecular species therefore
require no special solver rule: their multiple non-zero coefficients couple
them to multiple elemental constraints. For electron-containing mixtures, the
final column contains the species charge numbers.

The chemical potential used by minplascalc is

```{math}
\mu_i = E_i^0-k_BT\log\left(\frac{Z_i}{N_i}\right).
```

At a constrained stationary point, there is a multiplier vector $\\lambda$ such
that

```{math}
\boldsymbol{\mu}+A\boldsymbol{\lambda}=0,
\qquad
A^T\mathbf{N}=\mathbf{b}.
```

These are the Karush--Kuhn--Tucker (KKT) conditions for the Gibbs free-energy
minimum. Chemical potentials do not individually vanish when elemental and
charge constraints are present.

The elemental target scale is arbitrary. The implementation uses $10^{24}$
particles when constructing $b$. Scaling every $N_i$ and every elemental total
by the same factor also scales $V$, leaving $n_i=N_i/V$ unchanged.

## State-dependent thermodynamics

Both solvers update the thermodynamics at every nonlinear candidate:

1. Compute $N\_{\\mathrm{tot}}$, $V$, and all number densities.

1. Compute the effective positive-ion charge $z^\*$ and electron density $n_e$.

1. Evaluate Stewart--Pyatt ionisation lowering $\\Delta E_i$.

1. Build each ionisation chain's reference energies $E_i^0$.

1. Include only monatomic electronic levels satisfying

   ```{math}
   E_{ij} < I_i-\Delta E_i,
   ```

   where $I_i$ is the ionisation energy and $E\_{ij}$ is level $j$'s energy.

1. Evaluate the internal and translational partition functions.

Step 5 makes the thermodynamic model piecewise smooth. Within one electronic
active set, all derivatives below are analytical. At a level crossing the
partition sum changes discretely, so left- and right-hand derivatives can
differ. This small step is retained as a physical model feature rather than
smoothed numerically.

## Production particle-number formulation

`LTE.calculate_composition` solves in the particle numbers $N_i$ and unscaled
constraint multipliers.

### 1. Construct the constraints

Element names are collected from every species and sorted. The elemental part
of $A$ is filled from the stoichiometry dictionaries. If electrons are present,
the charge-number column is appended. The elemental targets are

```{math}
b_k=10^{24}\sum_i A_{ik}x_i^0,
```

where $x_i^0$ is the feed-composition constraint supplied to the mixture.

### 2. Initialise a positive particle vector

Every species starts at `gfe_initial_particles`, normally $10^{20}$. This is a
numerical initial guess, not the equilibrium composition.

### 3. Refresh the nonlinear thermodynamics

At the current $\\mathbf N$, the solver evaluates $V$, ionisation lowering,
reference energies, active electronic levels, partition functions, and
$\\boldsymbol\\mu$.

### 4. Assemble the bordered linear system

The ideal-mixture curvature block is

```{math}
H_{ij}=k_BT\left(\frac{\delta_{ij}}{N_i}
-\frac{1}{N_{\mathrm{tot}}}\right).
```

The solver forms

```{math}
\begin{bmatrix}
H & A\\
A^T & 0
\end{bmatrix}
\begin{bmatrix}
\mathbf N_{\mathrm{trial}}\\
\boldsymbol\lambda_{\mathrm{trial}}
\end{bmatrix}
=
\begin{bmatrix}
-\boldsymbol\mu\\
\mathbf b
\end{bmatrix}.
```

Although this contains absolute trial particle numbers rather than an
explicitly named increment, it is equivalent to the usual Newton form for the
ideal block because $H\\mathbf N=0$. Ionisation lowering and reference energies
are refreshed between iterations, but their particle derivatives are not
included in this production solve matrix. The iteration is therefore best
viewed as safeguarded quasi-Newton when lowering is active.

### 5. Govern the update

The raw bordered solve can propose a large or negative change for a trace
species. A governor chooses one common relaxation factor so that no species
changes by more than a fixed fraction of its current value:

```{math}
\mathbf N\leftarrow(1-\alpha)\mathbf N
+\alpha\mathbf N_{\mathrm{trial}}.
```

The first attempt permits a 90% per-iteration change. If the iteration limit is
reached, progressively smaller governor factors down to 10% are tried. Because
the factor is below one, a finite current positive amount cannot cross zero in
one accepted update.

### 6. Test convergence and return number densities

Convergence uses the relative change of the species with the largest proposed
particle number. If all governor attempts exceed `gfe_max_iter`, the method
warns and returns the last iterate. Otherwise, the converged particle numbers
are divided by $V$ and cached with the other equilibrium state quantities.

### Consequences

The production method is mature and supports the existing mixture API, but:

- positivity is protected by damping rather than by the variables;
- particle numbers can span many orders of magnitude;
- residual components retain their physical units and scales;
- the convergence test observes only one species; and
- difficult states can reach non-positive trial values or the iteration limit.

## Experimental coupled log-space formulation

The prototype replaces $N_i$ with $u_i=\\log N_i$ and scales the multipliers by
$k_BT$. Its unknown vector is $\\mathbf y=(\\mathbf u,\\boldsymbol\\ell)$, where
$\\boldsymbol\\ell=\\boldsymbol\\lambda/(k_BT)$. Positivity is structural because
$N_i=\\exp u_i$.

### Dimensionless residual

The chemical-equilibrium residual for every species is

```{math}
F_i^\mu = \frac{E_i^0}{k_BT}-\log Z_i+u_i
+(A\boldsymbol\ell)_i.
```

Each elemental balance uses a relative logarithmic residual,

```{math}
F_k^e=\log\left(\frac{(A^T\mathbf N)_k}{b_k}\right),
```

and charge neutrality uses

```{math}
F^z=\frac{\mathbf z^T\mathbf N}{N_{\mathrm{tot}}}.
```

All residuals are dimensionless and normally of order one away from the root.
The logarithmic elemental residual requires a positive target for every element
included in the prototype.

### Packed thermodynamic evaluation

The prototype packs monatomic level ownership, energies, degeneracies,
ionisation chains, translational prefactors, and molecular constants into
NumPy arrays. At fixed temperature it caches Boltzmann factors and the
temperature-only partition terms. One candidate evaluation then:

1. exponentiates $\\mathbf u$ once;
1. evaluates ionisation lowering and, when requested, its particle-number
   Jacobian;
1. propagates lowering through each reference-energy chain;
1. forms all monatomic active masks and partition sums together; and
1. assembles the residual and analytical Jacobian from the same state.

Timing showed that the dense linear solve is only about 4% of the log
prototype's runtime. This describes linear algebra inside the prototype, not
the entire GFE calculation relative to all minplascalc property work.

### Analytical Jacobian

Let $C_k=(A^T\\mathbf N)\_k$. The chemical block with respect to log particle
numbers is

```{math}
\frac{\partial F_i^\mu}{\partial u_j}
=\delta_{ij}-\frac{N_j}{N_{\mathrm{tot}}}
+\frac{1}{k_BT}\frac{\partial E_i^0}{\partial N_j}N_j.
```

The other non-zero blocks are

```{math}
\frac{\partial\mathbf F^\mu}{\partial\boldsymbol\ell}=A,
\qquad
\frac{\partial F_k^e}{\partial u_j}=\frac{A_{jk}N_j}{C_k},
```

and

```{math}
\frac{\partial F^z}{\partial u_j}
=\frac{z_jN_j}{N_{\mathrm{tot}}}
-\frac{(\mathbf z^T\mathbf N)N_j}{N_{\mathrm{tot}}^2}.
```

The reference-energy term includes the analytical derivative of ionisation
lowering. Active electronic masks are held fixed during Jacobian evaluation.

### Initialisation and damped Newton iteration

The default initial $u_i$ corresponds to $N_i=10^{20}$. With those particle
numbers fixed, the initial scaled multipliers are fitted by least squares to
reduce the chemical residual. Each iteration then:

1. evaluates $\\mathbf F$ and $J=\\partial\\mathbf F/\\partial\\mathbf y$;
1. solves $J\\Delta\\mathbf y=-\\mathbf F$;
1. starts with a full Newton step;
1. backtracks until the squared-residual merit function satisfies an Armijo
   decrease; and
1. accepts the candidate residual and thermodynamic state together, avoiding a
   duplicate evaluation at the next iteration.

The solve stops when $|\\mathbf F|\_\\infty$ reaches the requested tolerance. A
stalled line search or exhausted iteration limit raises an error instead of
silently returning the last state.

### Temperature continuation

For a temperature sweep, the prototype first solves a mid-range bootstrap
state, currently 12,000 K. It advances to each requested temperature in steps
no larger than 1,000 K and reuses the preceding $(\\mathbf u,\\boldsymbol\\ell)$ as
the next initial state. This makes both sweep directions less dependent on a
cold generic initial guess.

### Competing cutoff branches

Near the closest electronic cutoff, the piecewise equations can admit roots
with different active masks. The exploratory selector perturbs the log state to
either side of the nearest cutoff, resolves locally, deduplicates candidates by
active-level counts, and selects the lower Gibbs objective. It is a local probe,
not a global search across every combination of electronic levels.

## Side-by-side summary

| Stage | Production solver | Log-space prototype |
|---|---|---|
| Species set | Fixed by constructor | Same fixed set |
| Composition variables | $N_i$ | $u_i=\\log N_i$ |
| Positivity | Governed relaxation | Guaranteed by exponentiation |
| Constraints | Dimensional bordered system | Dimensionless coupled residual |
| Lowering derivative in solve | Omitted from solve matrix | Included in Jacobian |
| Globalization | Relative-change governor | Armijo backtracking |
| Convergence | Change of largest proposed species | Full residual infinity norm |
| Sweeps | Public method resolves invalidated states | Explicit continuation |
| Failure | Warning and last iterate | Exception with residual information |
| Status | Public implementation | Isolated research prototype |

## Analytical temperature tangent

Once either equilibrium system has converged, its constrained equations can be
differentiated implicitly at fixed pressure. In production variables:

```{math}
\begin{bmatrix}
H+E_N^0 & A\\
A^T & 0
\end{bmatrix}
\begin{bmatrix}
d\mathbf N/dT\\
d\boldsymbol\lambda/dT
\end{bmatrix}
=
\begin{bmatrix}
-\partial\boldsymbol\mu/\partial T\\
0
\end{bmatrix}.
```

Here $E_N^0$ is the full particle-number Jacobian of the reference energies. In
log variables the same operation is

```{math}
J\frac{d\mathbf y}{dT}=-\frac{\partial\mathbf F}{\partial T}.
```

The mole-fraction derivative follows from

```{math}
\frac{dx_i}{dT}=\frac{1}{N_{\mathrm{tot}}}\frac{dN_i}{dT}
-\frac{N_i}{N_{\mathrm{tot}}^2}\sum_j\frac{dN_j}{dT}.
```

This single tangent solve supplies the composition and reference-energy terms
used by analytical equilibrium heat capacity and reactional thermal
conductivity. It replaces separate equilibrium solves above and below the
requested temperature.

## Active-level diagnostics

`LTE.calculate_active_level_fingerprint` is an opt-in diagnostic. It reports:

- a deterministic SHA-256 of every monatomic active/inactive bit mask in
  species order;
- active and total level counts and a separate hash per monatomic species; and
- the species, level index, signed energy margin, and margin divided by $k_BT$
  for the closest cutoff.

A positive margin means that the nearest level is included; a negative margin
means it is excluded. A changed whole-system hash plus one changed per-species
hash identifies which species crossed a level.

```python
fingerprint = mixture.calculate_active_level_fingerprint()
print(fingerprint.fingerprint)
print(
    fingerprint.nearest_cutoff_species_name,
    fingerprint.nearest_cutoff_margin_over_kbt,
)
for state in fingerprint.species:
    print(
        state.species_name,
        state.active_level_count,
        state.total_level_count,
        state.fingerprint,
    )
```

The fingerprint describes model state, not numerical closeness: it changes only
when an active bit changes. Record the cutoff margin alongside it when comparing
neighbouring temperatures.

## Verification envelope

Development tests compare the production and log formulations for simple
oxygen and multi-element SiCO mixtures, 1,000--25,000 K, 1,013.25 Pa--10.1325
MPa, and both sweep directions. They compare the analytical and numerical
Jacobians, compare both temperature tangents, exercise difficult production
states, and verify competing Si+ cutoff branches and their fingerprints.

Agreement on this envelope is evidence for the formulation, not a general
equivalence proof. The [reduced-equilibrium research
note](Reduced_Equilibrium_Research_Note.md) derives the proposed smaller system
and defines the additional proof and validation obligations that must be met
before production consideration.
