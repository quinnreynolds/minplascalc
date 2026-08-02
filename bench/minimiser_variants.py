r"""Alternative step controls for the Gibbs minimiser (issue #16).

The Newton step itself is sound: ``gfe_matrix`` is the exact Hessian of the
ideal-gas part of G, and the bordered system is the KKT system for the
linear element-balance and charge constraints.  What is ad hoc is the step
control.

The shipped "governor" caps the change in every species at
``governor_factor * N_i``, *symmetrically*:

.. code:: python

    max_allowed_delta_Ni = governor_factor * self.__Ni
    delta_Ni = delta_Ni.clip(min=max_allowed_delta_Ni)
    relaxation_factor = (max_allowed_delta_Ni / delta_Ni).min()

With ``governor_factor <= 0.9`` a species can therefore grow by at most a
factor of 1.9 per iteration, and the ``min`` runs over *all* species, so one
species needing a large relative move throttles the entire step.

Only the *decrease* direction actually threatens the solve -- ``N_i`` must
stay positive for the log and the ``1/N_i`` Hessian entries.  The standard
safeguard is the fraction-to-boundary rule from interior-point methods:
limit only steps that would drive a species non-positive, and leave growth
free.  ``rule="ftb"`` does that.

Two axes are separated here, because changing both at once makes the
comparison meaningless:

* ``rule``: ``"governor"`` (reproduces the shipped step control) or
  ``"ftb"``.
* ``convergence``: ``"shipped"`` (relative movement of the most abundant
  species only, as shipped) or ``"all"`` (every species).
"""

import numpy as np

import minplascalc.units as u


def build_constraints(mixture):
    """Element-balance and charge-neutrality constraint matrix and vector."""
    species = mixture.species
    unique_elements = sorted({s for sp in species for s in sp.stoichiometry})

    A = np.zeros((len(species), len(unique_elements) + 1))
    b = np.zeros(len(unique_elements) + 1)
    for k, element in enumerate(unique_elements):
        coeff = np.array([sp.stoichiometry.get(element, 0) for sp in species])
        A[:, k] = coeff
        b[k] = sum(1e24 * c * x for c, x in zip(coeff, mixture.x0))
    A[:, -1] = [sp.charge_number for sp in species]
    return A, b


def _newton_step(mixture, Ni, A, b):
    """Unrelaxed new particle numbers from the bordered KKT system at Ni."""
    nb = len(mixture.species)
    kbt = u.k_b * mixture.T

    # E0 and dE depend on the composition, refreshed each iteration exactly
    # as the shipped solver does.
    mixture._LTE__Ni = Ni
    E0, dE = mixture._LTE__get_reference_energies()

    N_tot = Ni.sum()
    V = N_tot * kbt / mixture.P

    dof = nb + A.shape[1]
    M = np.zeros((dof, dof))
    v = np.zeros(dof)
    M[:nb, nb:] = A
    M[nb:, :nb] = A.T
    v[nb:] = b
    M[:nb, :nb] = -kbt / N_tot + np.diag(kbt / Ni)

    total = np.array(
        [
            sp.total_partition_function(V, mixture.T, d)
            for sp, d in zip(mixture.species, dE)
        ]
    )
    v[:nb] = -(-kbt * np.log(total / Ni) + E0)
    return np.linalg.solve(M, v)[:nb]


def solve(
    mixture,
    rule="ftb",
    convergence="shipped",
    rtol=1e-10,
    max_iter=1000,
    tau=0.995,
):
    """Minimise G with the chosen step rule and convergence test.

    Returns
    -------
    tuple[np.ndarray, int, bool]
        Number densities, total Newton iterations, and whether it
        converged.  Raises :class:`FloatingPointError` if the iterate
        underflows -- see the notes in ``check_minimiser_variants.py``.
    """
    nb = len(mixture.species)
    kbt = u.k_b * mixture.T
    A, b = build_constraints(mixture)

    Ni = np.full(nb, mixture.gfe_initial_particles)
    governor_factors = np.linspace(0.9, 0.1, 9)
    iters = 0
    converged = False

    for governor_factor in governor_factors:
        if converged:
            break
        for _ in range(max_iter):
            if not np.all(np.isfinite(Ni)) or np.any(Ni <= 0):
                raise FloatingPointError("iterate left the positive orthant")

            new_Ni = _newton_step(mixture, Ni, A, b)
            delta = new_Ni - Ni
            adelta = np.abs(delta)

            if convergence == "shipped":
                k = int(new_Ni.argmax())
                tol = adelta[k] / new_Ni[k]
            else:
                with np.errstate(divide="ignore", invalid="ignore"):
                    rel = adelta / np.abs(new_Ni)
                rel = rel[np.isfinite(rel)]
                tol = float(rel.max()) if rel.size else 0.0
            if tol <= rtol:
                converged = True
                break

            if rule == "governor":
                cap = governor_factor * Ni
                lam = float((cap / np.maximum(adelta, cap)).min())
            else:
                shrink = delta < 0
                lam = (
                    min(1.0, tau * float(np.min(Ni[shrink] / adelta[shrink])))
                    if shrink.any()
                    else 1.0
                )

            Ni = Ni + lam * delta
            iters += 1

    mixture._LTE__Ni = Ni
    V = Ni.sum() * kbt / mixture.P
    return Ni / V, iters, converged
