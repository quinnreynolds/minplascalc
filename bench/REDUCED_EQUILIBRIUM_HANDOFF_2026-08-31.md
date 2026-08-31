# Reduced equilibrium research handoff — 2026-08-31

This is the durable checkpoint for the potential-only equilibrium research
prototype associated with issue #100. The implementation is intentionally
outside the public package API and is banked on
`codex/research-reduced-equilibrium` for later study.

The branch starts from the shared analytical-temperature-property and full
log-equilibrium work on `feature/consolidate-computation-state`. Its reduced
formulation history begins with commit `9dd572b` and the checkpoint code ends
at `68fe6ee`, before this handoff note.

## What is implemented

- Species-density reconstruction from element and charge potentials.
- Fixed and coupled Stewart--Pyatt ionisation lowering.
- Analytical residual Jacobians and fixed-active-set temperature tangents.
- Oxygen, SiCO, electron-free, and multi-element molecular validation.
- Temperature continuation, exact active-level fingerprints, and local
  competing-cutoff branch diagnostics.
- Shared packed equilibrium thermodynamics and value-keyed reconstruction
  reuse across residual/Jacobian callbacks.
- A reproducible benchmark against production and the full log formulation.

The derivation and stage-by-stage evidence are in
`docs/theory/Reduced_Equilibrium_Research_Note.md`. Detailed timings are in
`bench/REDUCED_EQUILIBRIUM_2026-08-28.md`.

## Current evidence

After packed-kernel and reconstruction-cache consolidation, the 60-state SiCO
equilibrium workload measures 0.160236 s end to end. That is 2.03 times faster
than production but 1.49 times slower than the full log prototype. Reduced
SiCO reconstruction performs 727 actual reconstructions for 1,531 optimizer
callbacks.

The non-speed comparison is mixed:

| Metric | Full log | Reduced | Interpretation |
|---|---:|---:|---|
| Generic direct starts, all tested states | 33/42 | 37/42 | Reduced gains four complex-mixture starts; production achieves 38/42. |
| Generic direct starts, SiCO | 17/21 | 21/21 | The clearest reduced robustness benefit. |
| Large random start perturbations | 30/30 in the hardest sampled cases | 22--30/30 | Full log has the larger measured basin and usually needs about half as many residual evaluations. |
| Warm SiCO equilibrated condition number | 7.78 | 3.68 | Reduced is better conditioned in the representative warm state. |
| Cold SiCO equilibrated residual amplification | 3.09e10 | 3.11e14 | Reduced is roughly 10,000 times more sensitive in this trace-ionisation state. |
| Maximum SiCO mole-fraction disagreement | reference | 2.9e-10 | Bulk states are effectively equivalent. |

A separate 723-state analytical heat-capacity comparison used 241
temperatures from 1,000 K to 25,000 K for SiO feed fractions 0.1, 0.5, and
0.9. Full and reduced active-level fingerprints agreed at all 723 states. The
largest relative heat-capacity difference was 4.38e-10, the largest
mole-fraction difference was 4.23e-10, and discrete-curvature statistics were
identical to reported precision. The reduced formulation therefore shows no
additional property-smoothness benefit over full log: that improvement comes
from log variables and the analytical tangent shared by both formulations.

Both formulations find the same locally competing active sets at the known
Si+ cutoff and select the same Gibbs branch. Cold trace-lowering roots can
differ greatly in log density while remaining indistinguishable in bulk mole
fractions. Active fingerprints diagnose this, but reduction does not remove
the multiplicity.

## Current recommendation

Keep this branch as a research implementation and equivalence oracle. The
smaller nonlinear system is scientifically useful, gives better warm
conditioning, and improves direct-start robustness for the complex test
mixture. It does not currently justify production integration because it is
slower than full log, has a narrower perturbed-start basin, is much more
sensitive in cold SiCO trace states, and requires substantially more
implementation and test code.

The likely production path is the full log solver with the shared packed
thermodynamic and caching ideas retained. Revisit the reduced system if a new
globalisation strategy can preserve its SiCO direct-start advantage while
matching full log on perturbation robustness, cold sensitivity, and speed.

## Reproducing the checkpoint

```console
PYTHONPATH=. uv run pytest tests -q
PYTHONPATH=. uv run python bench/bench_reduced_equilibrium.py \
    --warmup 1 --repeats 7
```

When restarting the study, first repeat the benchmark on mains power and
record both independent-start and continuation results. Do not compare raw
condition numbers between formulations without row/column equilibration, and
do not treat trace log-density disagreement as bulk-composition error.
