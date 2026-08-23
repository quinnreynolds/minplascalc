"""Experimental full-equilibrium Newton solver in log particle numbers.

This deliberately lives outside the package API. It reuses the production
thermodynamic functions while testing a different nonlinear formulation:
chemical-potential equilibrium and every conservation law are solved together,
with positivity guaranteed by ``u_i = log(N_i)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from minplascalc import species as species_module
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


@dataclass(frozen=True)
class _PackedThermodynamics:
    """Thermodynamic values shared by the residual and Jacobian."""

    particle_numbers: np.ndarray
    particle_total: float
    reference_energies: np.ndarray
    ionization_lowering: np.ndarray
    log_partitions: np.ndarray
    reference_dN: np.ndarray | None
    active_level_counts: np.ndarray


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
        self._prepare_packed_thermodynamics()

    def _prepare_packed_thermodynamics(self) -> None:
        """Pack immutable species data for vectorized evaluation."""
        count = self.species_count
        self.charges = np.asarray(
            self.mixture.charge_numbers, dtype=np.float64
        )
        self.electron_index = next(
            (
                index
                for index, species in enumerate(self.mixture.species)
                if species.name == "e"
            ),
            -1,
        )
        self.log_translational_prefactors = 1.5 * np.log(
            2
            * np.pi
            * np.array(
                [species.molar_mass for species in self.mixture.species]
            )
            * u.k_b
            / (u.N_a * u.h**2)
        )

        monatomic = [
            (index, species)
            for index, species in enumerate(self.mixture.species)
            if isinstance(species, species_module.Monatomic)
        ]
        self.monatomic_indices = np.array(
            [index for index, _ in monatomic], dtype=np.int64
        )
        self.monatomic_ionization = np.zeros(count)
        level_owners = []
        level_energies = []
        level_degeneracies = []
        for index, species in monatomic:
            self.monatomic_ionization[index] = species.ionisation_energy
            level_owners.extend([index] * len(species._level_energies))
            level_energies.extend(species._level_energies)
            level_degeneracies.extend(species._degeneracies)
        self.level_owners = np.asarray(level_owners, dtype=np.int64)
        self.level_energies = np.asarray(level_energies, dtype=np.float64)
        self.level_degeneracies = np.asarray(
            level_degeneracies, dtype=np.float64
        )

        diatomic = [
            (index, species)
            for index, species in enumerate(self.mixture.species)
            if isinstance(species, species_module.Diatomic)
        ]
        self.diatomic_indices = np.array(
            [index for index, _ in diatomic], dtype=np.int64
        )
        self.diatomic_g0 = np.array([species.g0 for _, species in diatomic])
        self.diatomic_w = np.array([species.w_e for _, species in diatomic])
        self.diatomic_rotation = np.array(
            [species.sigma_s * species.b_e for _, species in diatomic]
        )
        known = set(self.monatomic_indices) | set(self.diatomic_indices)
        if self.electron_index >= 0:
            known.add(self.electron_index)
        self.fallback_indices = np.array(
            [index for index in range(count) if index not in known],
            dtype=np.int64,
        )

        self.base_reference_energies = np.zeros(count)
        for index, species in enumerate(self.mixture.species):
            if sum(species.stoichiometry.values()) >= 2:
                self.base_reference_energies[
                    index
                ] = -species.dissociation_energy

        self.reference_chains = []
        neutral_species = [
            species
            for species in self.mixture.species
            if species.charge_number == 0
        ]
        for neutral in neutral_species:
            negative = sorted(
                (
                    (index, species)
                    for index, species in enumerate(self.mixture.species)
                    if species.stoichiometry == neutral.stoichiometry
                    and species.charge_number <= 0
                ),
                key=lambda item: item[1].charge_number,
                reverse=True,
            )
            positive = sorted(
                (
                    (index, species)
                    for index, species in enumerate(self.mixture.species)
                    if species.stoichiometry == neutral.stoichiometry
                    and species.charge_number >= 0
                ),
                key=lambda item: item[1].charge_number,
            )
            for (source, source_species), (target, _) in zip(
                positive[:-1], positive[1:]
            ):
                self.reference_chains.append(
                    (
                        source,
                        target,
                        source,
                        -1.0,
                        source_species.ionisation_energy,
                    )
                )
            for (source, _), (target, target_species) in zip(
                negative[:-1], negative[1:]
            ):
                self.reference_chains.append(
                    (
                        source,
                        target,
                        target,
                        1.0,
                        -target_species.ionisation_energy,
                    )
                )

    def _set_particle_numbers(self, particle_numbers: np.ndarray) -> None:
        """Set the private iterate on the prototype-owned mixture."""
        self.mixture._LTE__Ni = particle_numbers

    def _packed_lowering(
        self, particle_numbers: np.ndarray, *, derivatives: bool
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Evaluate ionisation lowering and its particle-number Jacobian."""
        count = self.species_count
        lowering = np.zeros(count)
        derivative = np.zeros((count, count)) if derivatives else None
        if self.electron_index < 0:
            return lowering, derivative

        positive = self.charges > 0
        charge_sum = particle_numbers[positive] @ self.charges[positive]
        charge_square_sum = (
            particle_numbers[positive] @ self.charges[positive] ** 2
        )
        z_star = charge_square_sum / charge_sum
        denominator = z_star + 1
        total_particles = particle_numbers.sum()
        electron_particles = particle_numbers[self.electron_index]
        volume = total_particles * u.k_b * self.mixture.T / self.mixture.P
        electron_density = electron_particles / volume
        debye_pow3 = (
            u.epsilon_0
            * u.k_b
            * self.mixture.T
            / (4 * np.pi * denominator * electron_density * u.e**2)
        ) ** 1.5

        if derivatives:
            dzstar_dN = np.zeros(count)
            dzstar_dN[positive] = (
                self.charges[positive] ** 2 * charge_sum
                - charge_square_sum * self.charges[positive]
            ) / charge_sum**2
            dlog_ratio_dN = (
                1.5 * dzstar_dN / denominator - 0.5 / total_particles
            )
            dlog_ratio_dN[self.electron_index] += 0.5 / electron_particles

        for index in np.flatnonzero(positive):
            ion_sphere_pow3 = (
                3 * self.charges[index] / (4 * np.pi * electron_density)
            )
            ratio = ion_sphere_pow3 / debye_pow3
            shape = (ratio + 1) ** (2 / 3) - 1
            lowering[index] = (
                u.k_b * self.mixture.T * shape / (2 * denominator)
            )
            if derivatives:
                shape_derivative = 2 / 3 * (ratio + 1) ** (-1 / 3)
                ratio_dN = ratio * dlog_ratio_dN
                assert derivative is not None
                derivative[index] = (
                    u.k_b
                    * self.mixture.T
                    / 2
                    * (
                        shape_derivative * ratio_dN / denominator
                        - shape * dzstar_dN / denominator**2
                    )
                )
        return lowering, derivative

    def _packed_thermodynamics(
        self, log_particles: np.ndarray, *, derivatives: bool
    ) -> _PackedThermodynamics:
        """Evaluate all species thermodynamics in packed numeric arrays."""
        particle_numbers = np.exp(log_particles)
        particle_total = float(particle_numbers.sum())
        temperature = self.mixture.T
        kbt = u.k_b * temperature
        volume = particle_total * kbt / self.mixture.P
        lowering, lowering_dN = self._packed_lowering(
            particle_numbers, derivatives=derivatives
        )

        reference = self.base_reference_energies.copy()
        reference_dN = (
            np.zeros((self.species_count, self.species_count))
            if derivatives
            else None
        )
        for (
            source,
            target,
            lowering_index,
            sign,
            offset,
        ) in self.reference_chains:
            reference[target] = (
                reference[source] + offset + sign * lowering[lowering_index]
            )
            if derivatives:
                assert reference_dN is not None and lowering_dN is not None
                reference_dN[target] = (
                    reference_dN[source] + sign * lowering_dN[lowering_index]
                )

        log_internal = np.empty(self.species_count)
        active_counts = np.zeros(self.species_count, dtype=np.int64)
        if self.monatomic_indices.size:
            owners = self.level_owners
            active = self.level_energies < (
                self.monatomic_ionization[owners] - lowering[owners]
            )
            level_terms = (
                self.level_degeneracies
                * active
                * np.exp(-self.level_energies / kbt)
            )
            sums = np.bincount(
                owners, weights=level_terms, minlength=self.species_count
            )
            active_counts = np.bincount(
                owners, weights=active, minlength=self.species_count
            ).astype(np.int64)
            log_internal[self.monatomic_indices] = np.log(
                sums[self.monatomic_indices]
            )
        if self.diatomic_indices.size:
            vibration_ratio = self.diatomic_w / kbt
            log_internal[self.diatomic_indices] = (
                np.log(self.diatomic_g0)
                - vibration_ratio / 2
                - np.log1p(-np.exp(-vibration_ratio))
                + np.log(kbt / self.diatomic_rotation)
            )
        if self.electron_index >= 0:
            log_internal[self.electron_index] = np.log(2.0)
        for index in self.fallback_indices:
            log_internal[index] = np.log(
                self.mixture.species[index].internal_partition_function(
                    temperature, lowering[index]
                )
            )

        log_partitions = (
            np.log(volume)
            + self.log_translational_prefactors
            + 1.5 * np.log(temperature)
            + log_internal
        )
        return _PackedThermodynamics(
            particle_numbers=particle_numbers,
            particle_total=particle_total,
            reference_energies=reference,
            ionization_lowering=lowering,
            log_partitions=log_partitions,
            reference_dN=reference_dN,
            active_level_counts=active_counts,
        )

    def evaluate(
        self,
        log_particles: np.ndarray,
        scaled_multipliers: np.ndarray,
        *,
        jacobian: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Evaluate the complete dimensionless residual and its Jacobian."""
        self.residual_evaluations += 1
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            thermodynamics = self._packed_thermodynamics(
                log_particles, derivatives=jacobian
            )
            chemical = (
                thermodynamics.reference_energies / (u.k_b * self.mixture.T)
                - thermodynamics.log_partitions
                + log_particles
                + self.constraints @ scaled_multipliers
            )

        particle_numbers = thermodynamics.particle_numbers
        particle_total = thermodynamics.particle_total
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

        reference_dN = thermodynamics.reference_dN
        assert reference_dN is not None
        system_size = self.species_count + self.constraint_count
        derivative = np.zeros((system_size, system_size))

        # Chain d(mu/kT)/dN through N=exp(u). The ideal-mixture block is
        # delta_ij - N_j/N_total; reference_dN supplies ionisation lowering.
        chemical_log_derivative = (
            np.eye(self.species_count)
            - particle_numbers[np.newaxis, :] / particle_total
            + reference_dN
            * particle_numbers[np.newaxis, :]
            / (u.k_b * self.mixture.T)
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
