# Radically reduced equilibrium system

**Research note · 28 August 2026 · Issue
[#100](https://github.com/quinnreynolds/minplascalc/issues/100)**

```{admonition} Status
:class: warning

This is a derivation and experiment design for a development branch. The
radically reduced solver has not been implemented, validated, benchmarked, or
approved as a production replacement. The measured speedups elsewhere on the
branch belong to the full log-equilibrium prototype, not to this proposal.
```

## Executive summary

The production Gibbs free-energy (GFE) solver and the experimental log-space
solver both treat every species amount as an independent nonlinear variable.
The proposed reduction instead solves for elemental and charge potentials and
reconstructs the amount of every participating species from chemical
stationarity.

The participating species set does **not** change. It remains an input to the
mixture. The reduction eliminates species amounts as independent Newton
unknowns; it does not discover species, select reactions, or remove molecular
species. A molecule with several elements is reconstructed through all of its
stoichiometric coefficients. For example, CO depends on both the carbon and
oxygen potentials, and SiO depends on both the silicon and oxygen potentials.

For $s$ species, $e$ elements, and an optional charge constraint, the full log
prototype solves $s+e+1$ unknowns in an electron-containing mixture. The
fixed-lowering reduced system solves $e+1$. Keeping electron density and
effective charge as explicit ionisation-lowering variables increases this to
$e+3$. The branch's SiCO workload therefore changes from a $20$-unknown system
to a candidate $6$-unknown system.

The smaller system is interesting chiefly as a robustness and conditioning
experiment. It may remove poorly useful species directions, make continuation
easier, and yield a compact temperature-tangent system. It does not guarantee
a wall-clock improvement: profiling attributed only about 4% of the packed
log-equilibrium prototype to its dense linear solve. That percentage does not
describe the complete GFE calculation or downstream property evaluation.

## Relationship to the current branch

The companion [GFE solver page](GFE_Solver.md) documents the implemented
production particle-number solver and the isolated full log-space prototype.
The relevant experimental implementation and evidence are retained in:

- `bench/log_equilibrium.py`;
- `tests/test_log_equilibrium.py`;
- `bench/LOG_EQUILIBRIUM_2026-08-23.md`; and
- `interactive-report/index.html`.

This note starts from their common stationarity equations, derives the reduced
system, states the limits of the equivalence claim, and defines the evidence
needed before any production proposal.

## Scope and non-goals

This study is intended to produce a derivation, a research-only prototype, and
a comparison with the existing solvers. It does not, by itself, authorize:

- replacement of `LTE.calculate_composition`;
- changes to the public mixture API;
- automatic selection or discovery of participating species;
- changes to species data, reaction chemistry, or feed interpretation;
- smoothing of the physical electronic-level cutoff;
- a global combinatorial search over electronic active sets;
- broad property or transport refactoring; or
- a claim of production speedup from the reduced matrix dimension alone.

## Notation and assumptions

Let:

- $s$ be the number of participating species, including the automatically
  appended electron where applicable;
- $e$ be the number of elements present in their stoichiometry dictionaries;
- $B\\in\\mathbb R^{s\\times e}$ be the elemental stoichiometry matrix;
- $B\_{ia}=\\nu\_{ia}$ be the number of atoms of element $a$ in species $i$;
- $\\mathbf z\\in\\mathbb R^s$ contain the species charge numbers;
- $\\chi\\in\\lbrace 0,1\\rbrace$ indicate whether charge neutrality is present;
- $\\mathbf b\\in\\mathbb R^e$ contain the elemental feed totals on any common
  positive scale;
- $N_i$, $n_i=N_i/V$, and $N\_{\\mathrm{tot}}=\\sum_iN_i$ be particle number,
  number density, and total particle number;
- $T>0$ and $P>0$ be fixed during one equilibrium solve;
- $E_i^0$ be the reference energy, including ionisation-lowering effects; and
- $Z_i$ be the single-species total partition function.

The translational partition factor is proportional to volume, so define the
partition function per unit volume

```{math}
q_i(T,\Delta E,\mathcal A)=\frac{Z_i(V,T,\Delta E,\mathcal A)}{V}.
```

Here $\\mathcal A$ is the electronic active set. Within one active set, $q_i$
is differentiable in the model variables. At a level crossing it changes
discretely.

Unless a section explicitly discusses a boundary case, the derivation assumes:

1. every retained elemental target is positive;
1. every reconstructed density is finite and positive;
1. $q_i$ is finite and positive;
1. the retained elemental columns and optional charge column have the required
   rank;
1. electron density and positive-ion moments lie in the domain of the
   ionisation-lowering model; and
1. the electronic active set is fixed.

These are material assumptions, not implementation details. Zero-feed
elements, exact zero species amounts, rank deficiency, and level crossings are
treated separately below.

## The three algorithms side by side

| Feature | Production governed GFE | Full log prototype | Radically reduced proposal |
|---|---|---|---|
| Status | Public implementation | Isolated branch prototype | Derivation only |
| Species set | Supplied up front | Same supplied set | Same supplied set |
| Nonlinear species variables | $N_i$ | $u_i=\\log N_i$ | None |
| Other variables | dimensional multipliers | scaled multipliers | element/charge potentials and optional lowering auxiliaries |
| Molecular coupling | rows of $B$ in bordered system | rows of $B$ in residual | rows of $B$ in reconstruction exponent |
| Positivity | common governed update | exponentiation | exponential reconstruction |
| Scale | arbitrary particle target and derived volume | same particle scale | density directly; pressure closes scale |
| Lowering derivative | omitted from iteration matrix | included | must be included through explicit closures |
| Globalisation | particle-change governor | Armijo backtracking | to be established experimentally |
| Tangent | full bordered solve | reuse full log Jacobian | reuse reduced Jacobian, then lift to species |

The essential iterations can be summarized as follows.

### Production governed GFE

```text
choose the fixed species set and build element/charge constraints
initialise every N_i to the configured positive particle scale
repeat:
    evaluate V, lowering, active levels, E0, partitions, and chemical potentials
    assemble the bordered particle-number/multiplier system
    solve for trial particle numbers and multipliers
    govern the common update so no species changes too far or crosses zero
    test the production relative-change criterion
return n_i = N_i / V
```

### Full log-equilibrium prototype

```text
choose the same fixed species set and build the same constraints
initialise every u_i = log(N_i) and fit scaled multipliers
repeat:
    evaluate packed thermodynamics, the full residual, and its Jacobian
    solve for updates to every u_i and every multiplier
    backtrack on the full dimensionless residual norm
return n_i = exp(u_i) / V
```

### Radically reduced proposal

```text
choose the same fixed species set and build its stoichiometric and charge rows
initialise element/charge potentials and, when needed, log(n_e) and log(z*)
repeat:
    evaluate lowering from the explicit auxiliary variables
    reconstruct every specified species density from stationarity
    assemble pressure, independent element-ratio, charge, and lowering closures
    solve and globalise only in the reduced variables
return the reconstructed n_i and the reduced state/Jacobian diagnostics
```

## From stationarity to species reconstruction

When charge neutrality is present, form $A$ by appending $\\mathbf z$ as the
final column of $B$; otherwise, $A=B$. Let $\\boldsymbol\\ell$ denote the
elemental multipliers scaled by $k_BT$, and let $\\ell_z$ be the scaled charge
multiplier. The full log-space chemical residual is

```{math}
0=\frac{E_i^0}{k_BT}-\log Z_i+\log N_i
  +\sum_a\nu_{ia}\ell_a+z_i\ell_z.
```

Because $Z_i=Vq_i$ and $N_i=Vn_i$,

```{math}
\log\left(\frac{Z_i}{N_i}\right)
=\log\left(\frac{q_i}{n_i}\right).
```

The volume and arbitrary particle-number scale therefore cancel from the
stationarity equation. Solving it for density gives

```{math}
\boxed{
n_i=q_i(T,\Delta E,\mathcal A)
\exp\left[
-\frac{E_i^0(\Delta E)}{k_BT}
-\sum_a\nu_{ia}\ell_a-z_i\ell_z
\right].
}
```

This equation eliminates $n_i$ as an independent nonlinear unknown. It does
not eliminate species $i$ from the mixture: its data, stoichiometry, charge,
reference chain, partition function, and reconstructed density all remain in
the calculation.

### Multi-element molecular coupling

For the SiCO species set, order the elemental potentials as
$(\\ell_C,\\ell_O,\\ell\_{\\mathrm{Si}})$. Ignoring lowering arguments only to keep
the notation short,

```{math}
n_{\mathrm{CO}}
=q_{\mathrm{CO}}
\exp\left[-\frac{E_{\mathrm{CO}}^0}{k_BT}
-\ell_C-\ell_O\right],
```

```{math}
n_{\mathrm{CO}^+}
=q_{\mathrm{CO}^+}
\exp\left[-\frac{E_{\mathrm{CO}^+}^0}{k_BT}
-\ell_C-\ell_O-\ell_z\right],
```

and

```{math}
n_{\mathrm{SiO}}
=q_{\mathrm{SiO}}
\exp\left[-\frac{E_{\mathrm{SiO}}^0}{k_BT}
-\ell_{\mathrm{Si}}-\ell_O\right].
```

The molecular equilibrium is coupled through the sum of all applicable
element potentials. No reaction basis and no special molecular elimination
rule are required.

## Reduced closures without composition-dependent lowering

First freeze or disable composition-dependent ionisation lowering. At fixed
$T$ and active set, $q_i$ and $E_i^0$ are then independent of the potentials.
Define the reconstructed vector $\\widehat{\\mathbf n}(\\boldsymbol\\ell,\\ell_z)$
using the boxed equation above and let

```{math}
n_{\mathrm{tot}}=\sum_i\widehat n_i,
\qquad
c_a=\sum_i\nu_{ia}\widehat n_i.
```

Choose a reference element $r$ with $b_r>0$. A square, dimensionless reduced
residual is:

### Pressure

```{math}
R_P=\log\left(\frac{k_BT n_{\mathrm{tot}}}{P}\right).
```

### Independent element ratios

For every $a\\ne r$,

```{math}
R_a=\log\left(\frac{c_a b_r}{c_r b_a}\right).
```

Only $e-1$ element equations are independent because the feed vector is known
only up to a common particle scale once the calculation is written in density
variables.

### Charge neutrality

When present, use a bounded dimensionless residual such as

```{math}
R_z=\frac{\sum_i z_i\widehat n_i}{n_{\mathrm{tot}}}.
```

There are $e+\\chi$ unknown potentials and
$1+(e-1)+\\chi=e+\\chi$ residuals.

Pressure is a physical scale closure: without it, feed ratios and charge leave
one scalar degree of freedom corresponding to equilibrium at different total
number densities. This should not be confused with rank-induced multiplier
non-uniqueness. If the columns of $A$ are dependent, the potentials themselves
are non-unique; the implementation must remove the redundant column, impose an
explicit gauge, or reject the problem with a rank diagnostic.

## Explicit ionisation-lowering closures

The current Stewart--Pyatt lowering depends on electron density and the
effective positive-ion charge

```{math}
z^*=\frac{\sum_{i:z_i>0}z_i^2n_i}
{\sum_{i:z_i>0}z_in_i}.
```

Reference energies inherit this dependence along each ionisation chain, and
the monatomic partition sums use the lowered electronic cutoff. Direct
substitution would make the reconstruction implicit in itself. Hiding that
fixed point in a nested solver would obscure convergence and derivative
costs, so the proposed system retains two logarithmic auxiliary variables:

```{math}
\eta=\log n_e,
\qquad
\xi=\log z^*.
```

At a candidate $(\\boldsymbol\\ell,\\ell_z,\\eta,\\xi)$:

1. calculate lowering from $T$, $n_e=\\exp\\eta$, and $z^{\*}=\\exp\\xi$;

1. construct the reference-energy chains and fixed-active-set partitions;

1. reconstruct every $\\widehat n_i$; and

1. add the closures

   ```{math}
   R_e=\log\widehat n_e-\eta,
   ```

   and

   ```{math}
   R_*=\log\left(
   \frac{\sum_{i:z_i>0}z_i^2\widehat n_i}
        {\sum_{i:z_i>0}z_i\widehat n_i}
   \right)-\xi.
   ```

The complete electron-containing state is

```{math}
\mathbf w=(\ell_1,\ldots,\ell_e,\ell_z,\eta,\xi)
\in\mathbb R^{e+3}.
```

Its residual contains pressure, $e-1$ element ratios, charge neutrality, and
the two lowering closures: also $e+3$ equations. The SiCO case has $e=3$, so
the proposed dimension is six. If there are no positive ions or no electron,
the Stewart--Pyatt closure is disabled and these two auxiliaries must not be
created.

## Equivalence result and proof obligations

The useful equivalence claim is about roots of the implemented stationarity
and closure equations on a fixed active set. It is narrower than a proof of a
unique global Gibbs minimum.

### Proposition

Assume the domain and rank conditions above, a positive elemental target
vector, and identical thermodynamic evaluations in the full and reduced
systems. Then:

1. every positive full-system root maps to a reduced root; and
1. every reduced root lifts to a positive full-system root for the chosen
   elemental target scale.

The mapping preserves species densities, element ratios, charge neutrality,
pressure, ionisation-lowering closures, and the fixed electronic active set.

### Full root to reduced root

Let $(\\mathbf N,\\boldsymbol\\ell,\\ell_z)$ satisfy full chemical stationarity,

```{math}
B^T\mathbf N=\mathbf b,
\qquad
\mathbf z^T\mathbf N=0,
\qquad
V=\frac{N_{\mathrm{tot}}k_BT}{P}.
```

Set $\\mathbf n=\\mathbf N/V$. Dividing elemental conservation by $V$ gives

```{math}
B^T\mathbf n=\frac{\mathbf b}{V},
```

so every elemental ratio residual is zero. Charge neutrality survives division
by $V$. The ideal-volume equation gives $k_BT\\sum_i n_i=P$, so the pressure
residual is zero. Chemical stationarity rearranges to the species
reconstruction equation. If lowering is active, choose
$\\eta=\\log n_e$ and $\\xi=\\log z^{\*}$ from this same density vector; both
auxiliary closures are then zero.

### Reduced root to full root

Let a reduced root reconstruct $\\mathbf n$. Zero element-ratio residuals imply
that there is a scalar $\\kappa>0$ such that

```{math}
B^T\mathbf n=\kappa\mathbf b.
```

Choose $V=1/\\kappa$ and $\\mathbf N=V\\mathbf n$. Then
$B^T\\mathbf N=\\mathbf b$. Charge neutrality is preserved by multiplication by
$V$. Since $R_P=0$,

```{math}
\sum_i n_i=\frac{P}{k_BT},
```

and therefore

```{math}
\frac{N_{\mathrm{tot}}k_BT}{P}
=\frac{V(\sum_i n_i)k_BT}{P}=V.
```

The lifted particle numbers thus reproduce the full solver's ideal volume.
The reconstruction equation is precisely full chemical stationarity after
using $Z_i=Vq_i$ and $N_i=Vn_i$. The explicit $n_e$ and $z^\*$ closures ensure
that both formulations evaluate the same lowering when it is active.

### What this proposition does not prove

Root equivalence does not by itself prove:

- existence of a root;
- uniqueness of the reconstructed root;
- global convergence of either Newton method;
- selection of the global Gibbs minimum;
- equivalence across different electronic active sets; or
- thermodynamic consistency of a composition-dependent lowering model.

With fixed $q_i$ and $E_i^0$, the ideal-mixture Gibbs model has a convex
structure, subject to the usual rank and feasible-set qualifications. A
minimum claim can then be supported by convexity or by a positive reduced
second variation on the feasible tangent space. With composition-dependent
reference energies and a hard cutoff, the implemented stationarity residual
is piecewise and need not be the gradient of one globally smooth scalar
objective. The branch's Gibbs-value comparison is therefore a local branch
diagnostic, not a general proof of global optimality.

## Analytical reduced Jacobian

The reduced Jacobian can be assembled without differentiating through a
nonlinear species solve because reconstruction is explicit once
$(\\eta,\\xi)$ is present.

Define

```{math}
L_{ip}=\frac{\partial\log\widehat n_i}{\partial w_p},
\qquad
\frac{\partial\widehat n_i}{\partial w_p}
=\widehat n_i L_{ip}.
```

The direct potential columns are

```{math}
L_{i,\ell_a}=-\nu_{ia},
\qquad
L_{i,\ell_z}=-z_i.
```

Within a fixed active set, the lowering columns are

```{math}
L_{i,\eta}
=\frac{\partial\log q_i}{\partial\eta}
-\frac{1}{k_BT}\frac{\partial E_i^0}{\partial\eta},
```

```{math}
L_{i,\xi}
=\frac{\partial\log q_i}{\partial\xi}
-\frac{1}{k_BT}\frac{\partial E_i^0}{\partial\xi}.
```

These derivatives include the lowering propagated through ionisation chains
and through the active electronic partition sums. The active masks themselves
are held fixed.

Let $x_i=\\widehat n_i/n\_{\\mathrm{tot}}$. The pressure row is

```{math}
\frac{\partial R_P}{\partial w_p}=\sum_i x_iL_{ip}.
```

Define elemental weights

```{math}
\omega_i^{(a)}=\frac{\nu_{ia}\widehat n_i}{c_a}.
```

Then an element-ratio row is

```{math}
\frac{\partial R_a}{\partial w_p}
=\sum_i\left(\omega_i^{(a)}-\omega_i^{(r)}\right)L_{ip}.
```

For $R_z=(\\mathbf z^T\\widehat{\\mathbf n})/n\_{\\mathrm{tot}}$,

```{math}
\frac{\partial R_z}{\partial w_p}
=\sum_i x_i(z_i-R_z)L_{ip}.
```

The electron closure row is

```{math}
\frac{\partial R_e}{\partial w_p}
=L_{e p}-\mathbf 1_{p=\eta}.
```

For positive-ion moments

```{math}
M_m=\sum_{i:z_i>0}z_i^m\widehat n_i,
\qquad
\rho_i^{(m)}=\frac{z_i^m\widehat n_i}{M_m},
```

the effective-charge closure row is

```{math}
\frac{\partial R_*}{\partial w_p}
=\sum_{i:z_i>0}
\left(\rho_i^{(2)}-\rho_i^{(1)}\right)L_{ip}
-\mathbf 1_{p=\xi}.
```

These expressions are also a useful implementation decomposition: one packed
thermodynamic pass returns $\\log\\widehat n$, $L$, active-set diagnostics, and
the moment reductions; residual and Jacobian assembly then uses only small
weighted contractions.

## Temperature tangent

Let the converged reduced residual be
$\\mathbf R(\\mathbf w,T,P)=0$. At fixed pressure and fixed active set,

```{math}
J_R\frac{d\mathbf w}{dT}
=-\left.\frac{\partial\mathbf R}{\partial T}\right|_{\mathbf w,P}.
```

After this reduced solve, lift the tangent through reconstruction:

```{math}
\frac{d\log n_i}{dT}
=\left.\frac{\partial\log n_i}{\partial T}\right|_{\mathbf w}
+\sum_pL_{ip}\frac{dw_p}{dT},
```

```{math}
\frac{dn_i}{dT}
=n_i\frac{d\log n_i}{dT}.
```

The mole-fraction derivative follows by normalization. Equivalence with the
full log tangent is a required numerical and algebraic check; it must not be
assumed merely because the equilibrium compositions agree.

## Active-set and ionisation-lowering policy

The reviewed branch policy retains the hard electronic cutoff as a real,
piecewise thermodynamic feature. The reduced system must therefore:

- compute the same strict active mask $E\_{ij}\\lt I_i-\\Delta E_i$;
- hold that mask fixed while forming a local Jacobian and tangent;
- attach the deterministic active-level fingerprint and nearest-cutoff margin
  to diagnostic results;
- permit local probes on both sides of the closest cutoff;
- compare distinct local roots with the same qualified Gibbs diagnostic used
  by the full prototype; and
- avoid claiming a global search over all active masks.

The known Si$^+$ crossing near 20,862 K is a required research case. The full
prototype observes locally competing roots with 27/28 and 28/29 active levels.
A reduced solve is equivalent only when it reproduces the same active mask and
lowering closure, not merely a nearby mole-fraction vector.

## Numerical design questions

The reduced dimension alone does not determine numerical quality. A prototype
should answer the following questions explicitly.

### Stable reconstruction

Store and manipulate $\\log n_i$. Pressure, elemental moments, and positive-ion
moments should use shifted exponentials or log-sum-exp reductions. Directly
forming every exponential before a finite-domain check risks overflow even
when the final normalized residual is representable.

### Residual and variable scaling

The proposed residuals are dimensionless, but that does not guarantee good
scaling. Report the scales used for potentials, $\\eta$, and $\\xi$. Compare
condition estimates only after documenting row and column scaling, because a
condition number is coordinate-dependent.

### Globalisation

Exponential reconstruction may amplify a potential step across many species.
Compare Armijo backtracking, a trust region, and a nonlinear least-squares
strategy if plain damped Newton has a narrow basin. Acceptance should require
finite reconstructed moments, valid lowering variables, and a decrease in a
documented merit function.

### Initialisation and continuation

Test a neutral or weakly ionised potential estimate, a least-squares fit to a
generic density vector, deliberately perturbed potentials, and temperature
continuation in both directions. Reusing the full solver's answer may be a
validation aid but must not be required for an independent reduced solve.

### Rank and zero-feed handling

Perform a rank-revealing factorization of the elemental/charge columns before
iteration. For the first prototype, a zero elemental target should produce a
precise unsupported-domain error rather than an ill-defined log ratio. A later
boundary formulation could analytically set all species containing an absent
element to zero, but that is a distinct active-boundary design and should not
be smuggled into the interior proof.

## Validation matrix

The full log prototype is the primary root-equivalence reference. Production
results remain useful where the governed solver converges cleanly.

### Algebra and derivatives

- Verify reconstruction substituted into every chemical residual to roundoff.
- Compare the analytical reduced Jacobian with central finite differences at
  representative fixed-active-set states.
- Compare the lifted reduced tangent with the full log-system tangent.
- Check pressure, every retained element ratio, charge, $n_e$, and $z^\*$
  separately rather than relying only on one residual norm.
- Compare Gibbs diagnostics only after matching target scale and active set.

### Mixtures

- Simple oxygen ionisation: `O2`, `O2+`, `O`, `O-`, `O+`, `O++`, plus the
  appended electron.
- Multi-element SiCO: the existing 15 supplied species plus the electron,
  explicitly covering `CO`, `CO+`, `SiO`, and `SiO+`.
- A minimal molecular case such as `CO`, `C`, and `O` to isolate the
  $\\ell_C+\\ell_O$ coupling.
- A small charged molecular case to combine multi-element and charge coupling.
- An electron-free mixture using the reduced system without $\\ell_z$, $\\eta$,
  or $\\xi$.
- Unsupported zero-feed and rank-deficient constructions, with precise
  diagnostics.

### State envelope

- 1,000--25,000 K;
- 1,013.25 Pa--10.1325 MPa;
- ascending and descending temperature continuation;
- cold, warm, and deliberately perturbed starts;
- existing SiO feed fractions 0.1, 0.5, and 0.9; and
- the known Si$^+$ cutoff window near 20,862 K.

### Comparisons

Report:

- maximum absolute mole-fraction error;
- log-density error for trace species;
- physical closure errors;
- residual evaluations, nonlinear iterations, backtracks, and failures;
- convergence basin under controlled perturbations;
- row/column-scaled Jacobian condition estimates;
- active-level fingerprints and cutoff margins;
- equilibrium and temperature-tangent timings; and
- component timings for reconstruction, packed thermodynamics, lowering,
  Jacobian assembly, and the dense solve.

Mixture construction and species-data loading should be timed both separately
and as part of an end-to-end workload.

## Performance interpretation

The packed full log prototype reduced nonlinear iterations by about 10.5 times
and measured a 2.62-times equilibrium speedup on its recorded SiCO sweep. Those
are results for that prototype, not forecasts for the reduced system.

Likewise, the reported approximately 4% is specifically the dense
`numpy.linalg.solve` share **within the packed log-equilibrium prototype**. It
does not mean:

- that the dense solve is 4% of the production GFE solver;
- that equilibrium is 4% of a full property calculation;
- that reducing $20\\times20$ to $6\\times6$ can save at most 4% end to end; or
- that the proposed formulation is already benchmarked.

The reduced formulation changes residual evaluation, reconstruction,
globalisation, and possibly the number of thermodynamic evaluations. Its value
must therefore be measured through both component profiles and complete
equilibrium/property workloads.

## Risks

- Exponential reconstruction can stiffen globalisation or overflow.
- Removing species variables can hide useful local correction directions.
- Trace species can dominate log-space errors while barely affecting bulk
  composition.
- Rank-deficient constraint columns can make potentials non-unique.
- Zero-feed elements imply boundary solutions outside the positive interior
  derivation.
- Ionisation lowering couples all positive ions through $n_e$ and $z^\*$.
- Hard cutoffs retain discontinuities and locally competing roots.
- A smaller linear system may not reduce the dominant packed-thermodynamic
  cost.
- Root agreement does not prove global-minimum selection.
- An apparently improved condition number may be an artifact of different
  variable or residual scaling.

## Suggested staged outputs and decision gates

### Stage 1: derivation note

Output: this note, reviewed for thermodynamic notation, reconstruction signs,
constraint counting, scale handling, and proof scope.

Decision gate: agree that root equivalence on a fixed active set is the right
initial claim and that the zero-feed boundary problem can remain unsupported.

### Stage 2: fixed-lowering prototype

Output: a research-only reduced solver using the packed thermodynamic kernel
with lowering disabled or frozen.

Decision gate: reproduce atomic, molecular, and electron-free full-log roots
and analytical Jacobians without using a full-solver initial state.

### Stage 3: lowering auxiliaries

Output: explicit $\\eta$ and $\\xi$ closures and their analytical derivatives.

Decision gate: reproduce oxygen and SiCO roots, lowering values, and tangents
through the pressure/temperature envelope.

### Stage 4: continuation and active sets

Output: robust globalisation, two-way continuation, active fingerprints, and
local cutoff probes.

Decision gate: characterize rather than conceal alternate roots and failures.

### Stage 5: benchmark and recommendation

Output: reproducible robustness, conditioning, tangent-cost, component, and
end-to-end comparisons with both existing formulations.

Decision gate: recommend one of continued research, a revised formulation, a
separate production proposal, or retention as a documented negative result.

## Acceptance criteria for the research programme

- The fixed participating species set is stated in the derivation and API
  design.
- Multi-element molecular reconstruction is demonstrated explicitly.
- Full-system and reduced roots are mapped in both directions with assumptions
  stated.
- Pressure scale, rank deficiency, any chosen gauge, and zero-feed behaviour
  are distinguished and documented.
- The reduced Jacobian agrees with numerical differentiation to a documented
  tolerance.
- Reconstructed oxygen and SiCO states agree with the full log reference across
  the stated temperature and pressure matrix.
- The lifted temperature tangent agrees with the full-system tangent on a
  fixed active set.
- Ionisation lowering is validated first in isolation and then through the
  explicit $n_e$ and $z^\*$ closures.
- Known cutoff branches are identified by active-level fingerprint and are not
  presented as a global active-set search.
- Robustness and conditioning are reported in addition to speed.
- Component and end-to-end timings keep the approximately 4% dense-solve
  observation in its correct prototype-only context.
- Any production integration is proposed separately after the research gates.

## Current recommendation

Proceed to a small fixed-lowering prototype only after review of this
derivation. The reduction is algebraically plausible and offers a cleaner
nonlinear state, but its strongest prospective benefit is better robustness
and conditioning, not the arithmetic cost of a smaller dense solve. Retain the
full log formulation as the reference implementation throughout the study.
