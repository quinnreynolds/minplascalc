"""Experimental full-equilibrium Newton solver in log particle numbers.

This deliberately lives outside the package API. It reuses the production
thermodynamic functions while testing a different nonlinear formulation:
chemical-potential equilibrium and every conservation law are solved together,
with positivity guaranteed by ``u_i = log(N_i)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from minplascalc import units as u
from minplascalc.mixture import LTE


@dataclass(frozen=True)
class LogEquilibriumResult:
    """Converged prototype state and solver diagnostics."""

    log_particles: np.ndarray
    scaled_multipliers: np.ndarray
    number_densities: np.ndarray
    residual_norm: float
    iterations: int
    residual_evaluations: int


@dataclass(frozen=True)
class LogEquilibriumPathResult:
    """Requested temperature states and aggregate continuation diagnostics."""

    states: tuple[LogEquilibriumResult, ...]
    total_iterations: int
    total_residual_evaluations: int
    continuation_solves: int


class LogEquilibriumSystem:
    """Coupled dimensionless equilibrium residual in log variables."""

    def __init__(self, mixture: LTE):
        self.mixture = mixture
        self.species_count = len(mixture.species)
        self.constraints = mixture._constraint_matrix()
        self.constraint_count = self.constraints.shape[1]
        self.element_names = sorted(
            {
                element
                for species in mixture.species
                for element in species.stoichiometry
            }
        )
        self.element_count = len(self.element_names)
        self.targets = np.array(
            [
                sum(
                    1e24 * species.stoichiometry.get(element, 0) * fraction
                    for species, fraction in zip(mixture.species, mixture.x0)
                )
                for element in self.element_names
            ]
        )
        if np.any(self.targets <= 0):
            raise ValueError(
                "The log prototype requires positive totals for every element."
            )
        self.has_charge_constraint = self.constraint_count > self.element_count
        self.residual_evaluations = 0

    def _set_particle_numbers(self, particle_numbers: np.ndarray) -> None:
        """Set the private iterate on the prototype-owned mixture."""
        self.mixture._LTE__Ni = particle_numbers

    def evaluate(
        self,
        log_particles: np.ndarray,
        scaled_multipliers: np.ndarray,
        *,
        jacobian: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Evaluate the complete dimensionless residual and its Jacobian."""
        self.residual_evaluations += 1
        particle_numbers = np.exp(log_particles)
        self._set_particle_numbers(particle_numbers)
        particle_total = particle_numbers.sum()
        temperature = self.mixture.T
        kbt = u.k_b * temperature
        volume = particle_total * kbt / self.mixture.P

        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            reference_energies, lowering = (
                self.mixture._LTE__get_reference_energies()
            )
            partitions = np.array(
                [
                    species.total_partition_function(volume, temperature, dE)
                    for species, dE in zip(self.mixture.species, lowering)
                ]
            )
            chemical = (
                reference_energies / kbt
                - np.log(partitions)
                + log_particles
                + self.constraints @ scaled_multipliers
            )

        conserved = self.constraints.T @ particle_numbers
        element_residual = np.log(
            conserved[: self.element_count] / self.targets
        )
        if self.has_charge_constraint:
            charge_residual = np.array([conserved[-1] / particle_total])
            residual = np.concatenate(
                (chemical, element_residual, charge_residual)
            )
        else:
            residual = np.concatenate((chemical, element_residual))

        if not jacobian:
            return residual, None

        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            reference_dN, _ = (
                self.mixture._LTE__get_reference_energy_derivatives()
            )
        system_size = self.species_count + self.constraint_count
        derivative = np.zeros((system_size, system_size))

        # Chain d(mu/kT)/dN through N=exp(u). The ideal-mixture block is
        # delta_ij - N_j/N_total; reference_dN supplies ionisation lowering.
        chemical_log_derivative = (
            np.eye(self.species_count)
            - particle_numbers[np.newaxis, :] / particle_total
            + reference_dN * particle_numbers[np.newaxis, :] / kbt
        )
        derivative[: self.species_count, : self.species_count] = (
            chemical_log_derivative
        )
        derivative[: self.species_count, self.species_count :] = (
            self.constraints
        )

        for row in range(self.element_count):
            derivative[self.species_count + row, : self.species_count] = (
                self.constraints[:, row] * particle_numbers / conserved[row]
            )
        if self.has_charge_constraint:
            charge = conserved[-1]
            derivative[-1, : self.species_count] = (
                self.constraints[:, -1] * particle_numbers / particle_total
                - charge * particle_numbers / particle_total**2
            )
        return residual, derivative

    def initial_state(
        self, particle_scale: float = 1e20
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a generic positive state with fitted initial multipliers."""
        log_particles = np.full(self.species_count, np.log(particle_scale))
        zero_multipliers = np.zeros(self.constraint_count)
        residual, _ = self.evaluate(
            log_particles, zero_multipliers, jacobian=False
        )
        chemical = residual[: self.species_count]
        scaled_multipliers = np.linalg.lstsq(
            self.constraints, -chemical, rcond=None
        )[0]
        return log_particles, scaled_multipliers

    def solve(
        self,
        initial: tuple[np.ndarray, np.ndarray] | None = None,
        *,
        tolerance: float = 1e-10,
        max_iterations: int = 80,
        max_backtracks: int = 30,
    ) -> LogEquilibriumResult:
        """Solve the full system with damped analytical Newton steps."""
        if initial is None:
            log_particles, scaled_multipliers = self.initial_state()
        else:
            log_particles = np.array(initial[0], copy=True)
            scaled_multipliers = np.array(initial[1], copy=True)

        self.residual_evaluations = 0
        for iteration in range(max_iterations + 1):
            residual, derivative = self.evaluate(
                log_particles, scaled_multipliers
            )
            residual_norm = float(np.linalg.norm(residual, ord=np.inf))
            if residual_norm < tolerance:
                particle_numbers = np.exp(log_particles)
                volume = (
                    particle_numbers.sum()
                    * u.k_b
                    * self.mixture.T
                    / self.mixture.P
                )
                return LogEquilibriumResult(
                    log_particles=log_particles,
                    scaled_multipliers=scaled_multipliers,
                    number_densities=particle_numbers / volume,
                    residual_norm=residual_norm,
                    iterations=iteration,
                    residual_evaluations=self.residual_evaluations,
                )
            if iteration == max_iterations:
                break

            assert derivative is not None
            update = np.linalg.solve(derivative, -residual)
            log_update = update[: self.species_count]
            multiplier_update = update[self.species_count :]
            merit = 0.5 * residual @ residual
            step = 1.0
            for _ in range(max_backtracks):
                candidate_log = log_particles + step * log_update
                candidate_multipliers = (
                    scaled_multipliers + step * multiplier_update
                )
                if np.max(np.abs(candidate_log)) < 700:
                    candidate, _ = self.evaluate(
                        candidate_log,
                        candidate_multipliers,
                        jacobian=False,
                    )
                    candidate_merit = 0.5 * candidate @ candidate
                    if (
                        np.isfinite(candidate_merit)
                        and candidate_merit < (1 - 1e-4 * step) * merit
                    ):
                        log_particles = candidate_log
                        scaled_multipliers = candidate_multipliers
                        break
                step *= 0.5
            else:
                raise RuntimeError(
                    f"Log-equilibrium line search stalled at residual "
                    f"{residual_norm:.3e}."
                )

        raise RuntimeError(
            f"Log-equilibrium solver did not converge after {max_iterations} "
            f"iterations; residual={residual_norm:.3e}."
        )

    def solve_temperature_path(
        self,
        temperatures: np.ndarray,
        *,
        bootstrap_temperature: float = 12000.0,
        maximum_temperature_step: float = 1000.0,
        tolerance: float = 1e-9,
    ) -> LogEquilibriumPathResult:
        """Solve requested states using an independent midrange bootstrap."""
        requested = np.asarray(temperatures, dtype=np.float64)
        if requested.ndim != 1 or requested.size == 0:
            raise ValueError("temperatures must be a non-empty vector")

        total_iterations = 0
        total_evaluations = 0
        solve_count = 0

        self.mixture.T = bootstrap_temperature
        current = self.solve(tolerance=tolerance)
        total_iterations += current.iterations
        total_evaluations += current.residual_evaluations
        solve_count += 1

        def advance(target: float) -> LogEquilibriumResult:
            nonlocal current, total_iterations, total_evaluations, solve_count
            start = self.mixture.T
            step_count = max(
                1,
                int(np.ceil(abs(target - start) / maximum_temperature_step)),
            )
            initial = (current.log_particles, current.scaled_multipliers)
            for temperature in np.linspace(start, target, step_count + 1)[1:]:
                self.mixture.T = float(temperature)
                current = self.solve(initial, tolerance=tolerance)
                initial = (current.log_particles, current.scaled_multipliers)
                total_iterations += current.iterations
                total_evaluations += current.residual_evaluations
                solve_count += 1
            return current

        states = []
        for temperature in requested:
            states.append(advance(float(temperature)))

        return LogEquilibriumPathResult(
            states=tuple(states),
            total_iterations=total_iterations,
            total_residual_evaluations=total_evaluations,
            continuation_solves=solve_count,
        )
