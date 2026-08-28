"""Research-only reduced equilibrium solver.

This module is deliberately outside the public package API.  It implements
the fixed-ionisation-lowering member of the reduced system described in
``docs/theory/Reduced_Equilibrium_Research_Note.md``: elemental and (when
present) charge potentials are the only nonlinear unknowns and all species
densities are reconstructed from chemical stationarity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from minplascalc import species as species_module
from minplascalc import units as u


def _logsumexp(values: np.ndarray) -> float:
    """Return a scalar log-sum-exp without forming an unstable sum."""
    values = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(values))
    if not np.isfinite(maximum):
        return maximum
    return maximum + float(np.log(np.exp(values - maximum).sum()))


@dataclass(frozen=True)
class ReducedEquilibriumResult:
    """Converged reduced state and Newton diagnostics."""

    potentials: np.ndarray
    log_number_densities: np.ndarray
    number_densities: np.ndarray
    residual_norm: float
    iterations: int
    residual_evaluations: int
    backtracks: int
    jacobian_condition: float


class ReducedEquilibriumSystem:
    """Fixed-lowering reduced equilibrium system for an LTE mixture."""

    def __init__(
        self,
        mixture,
        *,
        fixed_ionization_lowering: np.ndarray | None = None,
    ):
        self.mixture = mixture
        self.species = tuple(mixture.species)
        self.species_count = len(self.species)
        if self.species_count == 0:
            raise ValueError(
                "The reduced system requires at least one species."
            )
        if not np.isfinite(mixture.T) or mixture.T <= 0:
            raise ValueError("The reduced system requires T > 0.")
        if not np.isfinite(mixture.P) or mixture.P <= 0:
            raise ValueError("The reduced system requires P > 0.")

        self.element_names = sorted(
            {
                element
                for species in self.species
                for element in species.stoichiometry
            }
        )
        if not self.element_names:
            raise ValueError(
                "The reduced system requires at least one element."
            )
        self.element_count = len(self.element_names)

        self.stoichiometry = np.array(
            [
                [
                    species.stoichiometry.get(element, 0)
                    for element in self.element_names
                ]
                for species in self.species
            ],
            dtype=np.float64,
        )
        self.charges = np.asarray(mixture.charge_numbers, dtype=np.float64)
        if self.charges.shape != (self.species_count,):
            raise ValueError(
                "The mixture charge-number vector has the wrong shape."
            )
        self.electron_index = next(
            (
                i
                for i, species in enumerate(self.species)
                if species.name == "e"
            ),
            -1,
        )
        self.has_charge_constraint = self.electron_index >= 0
        if self.has_charge_constraint:
            self.constraint_matrix = np.column_stack(
                (self.stoichiometry, self.charges)
            )
        else:
            self.constraint_matrix = self.stoichiometry.copy()
        self.potential_count = self.constraint_matrix.shape[1]

        x0 = np.asarray(mixture.x0, dtype=np.float64)
        if x0.shape != (self.species_count,):
            raise ValueError("The mixture feed vector has the wrong shape.")
        if np.any(~np.isfinite(x0)) or np.any(x0 < 0):
            raise ValueError("Feed fractions must be finite and non-negative.")
        self.targets = self.stoichiometry.T @ x0 * 1e24
        zero = np.flatnonzero(self.targets <= 0)
        if zero.size:
            names = ", ".join(self.element_names[i] for i in zero)
            raise ValueError(
                f"The reduced system requires positive feed totals; zero-feed "
                f"element(s): {names}."
            )

        if fixed_ionization_lowering is None:
            lowering = np.zeros(self.species_count, dtype=np.float64)
        else:
            lowering = np.asarray(fixed_ionization_lowering, dtype=np.float64)
            if lowering.shape != (self.species_count,):
                raise ValueError(
                    "fixed_ionization_lowering must have one value per "
                    "species."
                )
            if np.any(~np.isfinite(lowering)):
                raise ValueError("fixed_ionization_lowering must be finite.")
            if np.any(lowering < 0):
                raise ValueError(
                    "fixed_ionization_lowering must be nonnegative."
                )
            lowering = lowering.copy()
        lowering.setflags(write=False)
        self.fixed_ionization_lowering = lowering

        rank = np.linalg.matrix_rank(self.constraint_matrix)
        if rank < self.potential_count:
            raise ValueError(
                "The elemental/charge potential matrix is rank deficient "
                f"(rank {rank}, expected {self.potential_count})."
            )

        self.base_reference_energies = self._reference_energies()
        self._log_q = self._log_partition_per_volume()
        if np.any(~np.isfinite(self._log_q)):
            raise ValueError(
                "The fixed lowering must leave every species partition "
                "factor finite and positive."
            )
        self._base_log_densities = (
            self._log_q - self.base_reference_energies / (u.k_b * mixture.T)
        )
        self.residual_evaluations = 0

    def _reference_energies(self) -> np.ndarray:
        """Reproduce the mixture reference-energy chain at fixed lowering."""
        reference = np.zeros(self.species_count, dtype=np.float64)
        for index, species in enumerate(self.species):
            if sum(species.stoichiometry.values()) >= 2:
                reference[index] = -species.dissociation_energy

        lowering = self.fixed_ionization_lowering
        for neutral in (s for s in self.species if s.charge_number == 0):
            positive = sorted(
                (
                    (i, species)
                    for i, species in enumerate(self.species)
                    if species.stoichiometry == neutral.stoichiometry
                    and species.charge_number >= 0
                ),
                key=lambda item: item[1].charge_number,
            )
            negative = sorted(
                (
                    (i, species)
                    for i, species in enumerate(self.species)
                    if species.stoichiometry == neutral.stoichiometry
                    and species.charge_number <= 0
                ),
                key=lambda item: item[1].charge_number,
                reverse=True,
            )
            for (source, source_species), (target, _) in zip(
                positive[:-1], positive[1:]
            ):
                reference[target] = (
                    reference[source]
                    + source_species.ionisation_energy
                    - lowering[source]
                )
            for (source, _), (target, target_species) in zip(
                negative[:-1], negative[1:]
            ):
                reference[target] = (
                    reference[source]
                    - target_species.ionisation_energy
                    + lowering[target]
                )
        return reference

    def _log_partition_per_volume(self) -> np.ndarray:
        """Evaluate ``log(q_i)`` without introducing a volume or state."""
        temperature = float(self.mixture.T)
        result = np.empty(self.species_count, dtype=np.float64)
        for index, species in enumerate(self.species):
            if isinstance(species, species_module.Monatomic):
                terms = np.log(species._degeneracies) - (
                    species._level_energies / (u.k_b * temperature)
                )
                active = species._level_energies < (
                    species.ionisation_energy
                    - self.fixed_ionization_lowering[index]
                )
                if not np.any(active):
                    result[index] = -np.inf
                    continue
                log_internal = _logsumexp(terms[active])
            else:
                internal = species.internal_partition_function(
                    temperature, self.fixed_ionization_lowering[index]
                )
                if not np.isfinite(internal) or internal <= 0:
                    result[index] = np.nan
                    continue
                log_internal = float(np.log(internal))
            translational = species.translational_partition_function(
                temperature
            )
            result[index] = float(np.log(translational) + log_internal)
        return result

    def _reconstruct(
        self, potentials: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return logs, normalized densities, and stationarity slopes."""
        log_densities = (
            self._base_log_densities - self.constraint_matrix @ potentials
        )
        log_total = _logsumexp(log_densities)
        with np.errstate(over="ignore", invalid="ignore"):
            fractions = np.exp(log_densities - log_total)
        slopes = -self.constraint_matrix
        return log_densities, fractions, slopes

    def _evaluate_state(
        self, potentials: np.ndarray, *, jacobian: bool
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        self.residual_evaluations += 1
        log_densities, fractions, slopes = self._reconstruct(potentials)
        log_total = _logsumexp(log_densities)
        if not np.isfinite(log_total) or np.any(~np.isfinite(fractions)):
            residual = np.full(
                1 + self.element_count - 1 + self.has_charge_constraint, np.inf
            )
            return residual, None, log_densities

        residual = [
            np.log(u.k_b * self.mixture.T / self.mixture.P) + log_total
        ]
        log_moments = []
        for column in range(self.element_count):
            coefficients = self.stoichiometry[:, column]
            present = coefficients > 0
            log_moments.append(
                _logsumexp(
                    log_densities[present] + np.log(coefficients[present])
                )
            )
        reference = 0
        for column in range(1, self.element_count):
            residual.append(
                log_moments[column]
                - log_moments[reference]
                + np.log(self.targets[reference])
                - np.log(self.targets[column])
            )
        charge_residual = float(fractions @ self.charges)
        if self.has_charge_constraint:
            residual.append(charge_residual)
        residual_array = np.asarray(residual, dtype=np.float64)

        if not jacobian:
            return residual_array, None, log_densities

        derivative = np.empty((residual_array.size, self.potential_count))
        derivative[0] = fractions @ slopes
        for row, column in enumerate(range(1, self.element_count), start=1):
            weights = np.zeros(self.species_count)
            present = self.stoichiometry[:, column] > 0
            weights[present] = self.stoichiometry[present, column] * np.exp(
                log_densities[present] - log_moments[column]
            )
            reference_weights = np.zeros(self.species_count)
            present_reference = self.stoichiometry[:, reference] > 0
            reference_weights[present_reference] = self.stoichiometry[
                present_reference, reference
            ] * np.exp(
                log_densities[present_reference] - log_moments[reference]
            )
            derivative[row] = (weights - reference_weights) @ slopes
        if self.has_charge_constraint:
            derivative[-1] = (
                fractions * (self.charges - charge_residual)
            ) @ slopes
        return residual_array, derivative, log_densities

    def evaluate(
        self, potentials: np.ndarray, *, jacobian: bool = True
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Evaluate the reduced residual and, optionally, its Jacobian."""
        values = np.asarray(potentials, dtype=np.float64)
        if values.shape != (self.potential_count,):
            raise ValueError(
                f"potentials must have shape ({self.potential_count},)."
            )
        if np.any(~np.isfinite(values)):
            raise ValueError("potentials must be finite.")
        residual, derivative, _ = self._evaluate_state(
            values, jacobian=jacobian
        )
        return residual, derivative

    def initial_state(self) -> np.ndarray:
        """Return a generic potential estimate without a full-system solve."""
        x0 = np.asarray(self.mixture.x0, dtype=np.float64)
        weights = np.maximum(x0, 1e-12)
        weights /= weights.sum()
        desired = np.log(self.mixture.P / (u.k_b * self.mixture.T)) + np.log(
            weights
        )
        return np.linalg.lstsq(
            self.constraint_matrix,
            self._base_log_densities - desired,
            rcond=None,
        )[0]

    def solve(
        self,
        initial: np.ndarray | None = None,
        *,
        tolerance: float = 1e-10,
        max_iterations: int = 80,
        max_backtracks: int = 30,
    ) -> ReducedEquilibriumResult:
        """Solve with analytical Newton steps and Armijo backtracking."""
        potentials = (
            self.initial_state()
            if initial is None
            else np.array(initial, dtype=float, copy=True)
        )
        if potentials.shape != (self.potential_count,):
            raise ValueError(
                f"initial must have shape ({self.potential_count},)."
            )
        if np.any(~np.isfinite(potentials)):
            raise ValueError("initial potentials must be finite.")
        self.residual_evaluations = 0
        backtracks = 0
        for iteration in range(max_iterations + 1):
            residual, jacobian, log_densities = self._evaluate_state(
                potentials, jacobian=True
            )
            residual_norm = float(np.linalg.norm(residual, ord=np.inf))
            if residual_norm < tolerance:
                densities = np.exp(log_densities)
                condition = float(np.linalg.cond(jacobian))
                return ReducedEquilibriumResult(
                    potentials=potentials.copy(),
                    log_number_densities=log_densities.copy(),
                    number_densities=densities,
                    residual_norm=residual_norm,
                    iterations=iteration,
                    residual_evaluations=self.residual_evaluations,
                    backtracks=backtracks,
                    jacobian_condition=condition,
                )
            if iteration == max_iterations:
                break
            if jacobian is None:
                raise RuntimeError(
                    "Reduced-equilibrium state is outside its finite domain."
                )
            try:
                update = np.linalg.solve(jacobian, -residual)
            except np.linalg.LinAlgError as error:
                raise RuntimeError(
                    "Reduced-equilibrium Jacobian is singular."
                ) from error
            merit = 0.5 * float(residual @ residual)
            step = 1.0
            for _ in range(max_backtracks + 1):
                candidate_potentials = potentials + step * update
                if np.all(np.isfinite(candidate_potentials)):
                    candidate, _, candidate_logs = self._evaluate_state(
                        candidate_potentials, jacobian=False
                    )
                    candidate_merit = 0.5 * float(candidate @ candidate)
                    if (
                        np.isfinite(candidate_merit)
                        and candidate_merit < (1 - 1e-4 * step) * merit
                    ):
                        potentials = candidate_potentials
                        break
                step *= 0.5
                backtracks += 1
            else:
                raise RuntimeError(
                    f"Reduced-equilibrium line search stalled at residual "
                    f"{residual_norm:.3e}."
                )
        raise RuntimeError(
            f"Reduced-equilibrium solver did not converge after "
            f"{max_iterations} "
            f"iterations; residual={residual_norm:.3e}."
        )

    def active_level_fingerprint(self, result: ReducedEquilibriumResult):
        """Return active-level diagnostics for a reduced result."""
        del result
        return self.mixture._active_level_fingerprint(
            self.fixed_ionization_lowering
        )
