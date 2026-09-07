"""Full-equilibrium Newton solver in log particle numbers.

Chemical-potential equilibrium and every conservation law are solved together,
with positivity guaranteed by ``u_i = log(N_i)``. The particle-number solver
remains available inside :mod:`minplascalc.mixture` as a regression oracle and
fallback for mixtures with a zero conserved element total.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from minplascalc import species as species_module
from minplascalc import units as u

if TYPE_CHECKING:
    from minplascalc.mixture import LTE


@dataclass(frozen=True)
class LogEquilibriumResult:
    """Converged log-equilibrium state and solver diagnostics."""

    log_particles: np.ndarray
    scaled_multipliers: np.ndarray
    number_densities: np.ndarray
    residual_norm: float
    iterations: int
    residual_evaluations: int
    cutoff_branches: int = 0


@dataclass(frozen=True)
class LogEquilibriumPathResult:
    """Requested temperature states and aggregate continuation diagnostics."""

    states: tuple[LogEquilibriumResult, ...]
    total_iterations: int
    total_residual_evaluations: int
    continuation_solves: int


@dataclass(frozen=True)
class LogEquilibriumBranchResult:
    """Locally competing cutoff branches and the lowest-G selection."""

    selected: LogEquilibriumResult
    candidates: tuple[LogEquilibriumResult, ...]
    dimensionless_gibbs: tuple[float, ...]
    nearest_cutoff_distance: float


@dataclass(frozen=True)
class LogEquilibriumTemperatureTangent:
    """Temperature derivative recovered from the log-system Jacobian."""

    log_particle_derivative: np.ndarray
    particle_derivative: np.ndarray
    mole_fraction_derivative: np.ndarray
    reference_energy_derivative: np.ndarray


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


class ZeroElementTotalError(ValueError):
    """A conserved element total cannot be represented in log variables."""


class CutoffConvergenceError(RuntimeError):
    """No consistent root was found in the explored hard-cutoff branches.

    This is a local search failure, not proof that no distant root exists.
    ``residual_norm`` is the full residual at the stalled Newton iterate.
    """

    def __init__(
        self,
        temperature: float,
        pressure: float,
        species_name: str,
        residual_norm: float,
        attempted_branches: int,
        iterations: int,
        residual_evaluations: int,
    ):
        self.temperature = temperature
        self.pressure = pressure
        self.species_name = species_name
        self.residual_norm = residual_norm
        self.attempted_branches = attempted_branches
        self.iterations = iterations
        self.residual_evaluations = residual_evaluations
        super().__init__(
            f"No self-consistent equilibrium found in {attempted_branches} "
            f"local hard-cutoff branches near {species_name} at "
            f"T={temperature:g} K, P={pressure:g} Pa; "
            f"stalled residual={residual_norm:.3e}. "
            "The discrete cutoff may leave a gap between adjacent roots."
        )


class _NewtonFailure(RuntimeError):
    """Retain the failed iterate for bounded cutoff recovery."""

    def __init__(self, message, initial, residual_norm, iterations):
        super().__init__(message)
        self.initial = initial
        self.residual_norm = residual_norm
        self.iterations = iterations


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
            raise ZeroElementTotalError(
                "Log equilibrium requires positive totals for every element."
            )
        self.has_charge_constraint = self.constraint_count > self.element_count
        self.residual_evaluations = 0
        self._prepare_packed_thermodynamics()
        self._cached_temperature = np.nan

    def _prepare_packed_thermodynamics(self) -> None:
        """Pack immutable species data for vectorized evaluation."""
        count = self.species_count
        self.charges = np.asarray(
            self.mixture.charge_numbers, dtype=np.float64
        )
        self.positive_indices = np.flatnonzero(self.charges > 0)
        self.positive_charges = self.charges[self.positive_indices]
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
        level_energies: list[float] = []
        level_degeneracies: list[float] = []
        for index, atom in monatomic:
            self.monatomic_ionization[index] = atom.ionisation_energy
            level_owners.extend([index] * len(atom._level_energies))
            level_energies.extend(atom._level_energies)
            level_degeneracies.extend(atom._degeneracies)
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

    def _prepare_temperature_thermodynamics(self) -> None:
        """Cache factors invariant during a fixed-temperature solve."""
        temperature = self.mixture.T
        if temperature == self._cached_temperature:
            return

        kbt = u.k_b * temperature
        self.level_boltzmann_terms = self.level_degeneracies * np.exp(
            -self.level_energies / kbt
        )
        self.temperature_log_internal = np.full(self.species_count, np.nan)
        if self.diatomic_indices.size:
            vibration_ratio = self.diatomic_w / kbt
            self.temperature_log_internal[self.diatomic_indices] = (
                np.log(self.diatomic_g0)
                - vibration_ratio / 2
                - np.log1p(-np.exp(-vibration_ratio))
                + np.log(kbt / self.diatomic_rotation)
            )
        if self.electron_index >= 0:
            self.temperature_log_internal[self.electron_index] = np.log(2.0)
        self.temperature_log_partition = (
            self.log_translational_prefactors + 1.5 * np.log(temperature)
        )
        self._cached_temperature = temperature

    def _set_particle_numbers(self, particle_numbers: np.ndarray) -> None:
        """Set the private particle-number iterate on the owned mixture."""
        self.mixture._set_particle_numbers(particle_numbers)

    def _packed_lowering(
        self, particle_numbers: np.ndarray, *, derivatives: bool
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Evaluate ionisation lowering and its particle-number Jacobian."""
        count = self.species_count
        lowering = np.zeros(count)
        derivative = np.zeros((count, count)) if derivatives else None
        if self.electron_index < 0:
            return lowering, derivative

        positive_numbers = particle_numbers[self.positive_indices]
        charge_sum = positive_numbers @ self.positive_charges
        charge_square_sum = positive_numbers @ self.positive_charges**2
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
            dzstar_dN[self.positive_indices] = (
                self.positive_charges**2 * charge_sum
                - charge_square_sum * self.positive_charges
            ) / charge_sum**2
            dlog_ratio_dN = (
                1.5 * dzstar_dN / denominator - 0.5 / total_particles
            )
            dlog_ratio_dN[self.electron_index] += 0.5 / electron_particles

        ion_sphere_pow3 = (
            3 * self.positive_charges / (4 * np.pi * electron_density)
        )
        ratio = ion_sphere_pow3 / debye_pow3
        shape = (ratio + 1) ** (2 / 3) - 1
        lowering[self.positive_indices] = (
            u.k_b * self.mixture.T * shape / (2 * denominator)
        )
        if derivatives:
            shape_derivative = 2 / 3 * (ratio + 1) ** (-1 / 3)
            ratio_dN = ratio[:, np.newaxis] * dlog_ratio_dN
            assert derivative is not None
            derivative[self.positive_indices] = (
                u.k_b
                * self.mixture.T
                / 2
                * (
                    shape_derivative[:, np.newaxis] * ratio_dN / denominator
                    - shape[:, np.newaxis] * dzstar_dN / denominator**2
                )
            )
        return lowering, derivative

    def _packed_thermodynamics(
        self,
        log_particles: np.ndarray,
        *,
        derivatives: bool,
        active_levels: np.ndarray | None = None,
    ) -> _PackedThermodynamics:
        """Evaluate all species thermodynamics in packed numeric arrays."""
        particle_numbers = np.exp(log_particles)
        particle_total = float(particle_numbers.sum())
        temperature = self.mixture.T
        kbt = u.k_b * temperature
        self._prepare_temperature_thermodynamics()
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

        log_internal = self.temperature_log_internal.copy()
        active_counts = np.zeros(self.species_count, dtype=np.int64)
        if self.monatomic_indices.size:
            owners = self.level_owners
            active = (
                self.level_energies
                < (self.monatomic_ionization[owners] - lowering[owners])
                if active_levels is None
                else active_levels
            )
            level_terms = self.level_boltzmann_terms * active
            sums = np.bincount(
                owners, weights=level_terms, minlength=self.species_count
            )
            active_counts = np.bincount(
                owners, weights=active, minlength=self.species_count
            ).astype(np.int64)
            log_internal[self.monatomic_indices] = np.log(
                sums[self.monatomic_indices]
            )
        for index in self.fallback_indices:
            log_internal[index] = np.log(
                self.mixture.species[index].internal_partition_function(
                    temperature, lowering[index]
                )
            )

        log_partitions = (
            np.log(volume) + self.temperature_log_partition + log_internal
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

    def _residual_from_thermodynamics(
        self,
        thermodynamics: _PackedThermodynamics,
        log_particles: np.ndarray,
        scaled_multipliers: np.ndarray,
    ) -> np.ndarray:
        """Assemble a residual from an already evaluated candidate state."""
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
        return residual

    def _reference_derivative_from_particles(
        self, particle_numbers: np.ndarray
    ) -> np.ndarray:
        """Build only the reference-energy Jacobian for a cached state."""
        _, lowering_dN = self._packed_lowering(
            particle_numbers, derivatives=True
        )
        assert lowering_dN is not None
        reference_dN = np.zeros((self.species_count, self.species_count))
        for source, target, lowering_index, sign, _ in self.reference_chains:
            reference_dN[target] = (
                reference_dN[source] + sign * lowering_dN[lowering_index]
            )
        return reference_dN

    def _jacobian_from_thermodynamics(
        self, thermodynamics: _PackedThermodynamics
    ) -> np.ndarray:
        """Assemble the Jacobian, deriving values absent from the cache."""
        reference_dN = thermodynamics.reference_dN
        if reference_dN is None:
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                reference_dN = self._reference_derivative_from_particles(
                    thermodynamics.particle_numbers
                )
        particle_numbers = thermodynamics.particle_numbers
        particle_total = thermodynamics.particle_total
        conserved = self.constraints.T @ particle_numbers
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
        return derivative

    def _evaluate_with_state(
        self,
        log_particles: np.ndarray,
        scaled_multipliers: np.ndarray,
        *,
        jacobian: bool,
        cache_derivatives: bool = False,
        active_levels: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None, _PackedThermodynamics]:
        """Evaluate a new thermodynamic state and retain it for reuse."""
        self.residual_evaluations += 1
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            thermodynamics = self._packed_thermodynamics(
                log_particles,
                derivatives=jacobian or cache_derivatives,
                active_levels=active_levels,
            )
            residual = self._residual_from_thermodynamics(
                thermodynamics, log_particles, scaled_multipliers
            )
            derivative = (
                self._jacobian_from_thermodynamics(thermodynamics)
                if jacobian
                else None
            )
        return residual, derivative, thermodynamics

    def evaluate(
        self,
        log_particles: np.ndarray,
        scaled_multipliers: np.ndarray,
        *,
        jacobian: bool = True,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Evaluate the complete dimensionless residual and its Jacobian."""
        residual, derivative, _ = self._evaluate_with_state(
            log_particles, scaled_multipliers, jacobian=jacobian
        )
        return residual, derivative

    def initial_state(
        self, particle_scale: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a generic positive state with fitted initial multipliers."""
        if particle_scale is None:
            particle_scale = self.mixture.gfe_initial_particles
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
        max_cutoff_branches: int = 8,
    ) -> LogEquilibriumResult:
        """Solve with damped Newton and bounded hard-cutoff branch recovery.

        After a stall near a cutoff, solve locally fixed electronic active
        sets and accept only roots consistent with the original discrete
        model. ``max_cutoff_branches`` bounds these extra Newton solves; zero
        disables recovery. An unresolved local gap raises
        :class:`CutoffConvergenceError`, never an approximate result.
        """
        self.residual_evaluations = 0
        try:
            return self._solve_newton(
                initial,
                tolerance=tolerance,
                max_iterations=max_iterations,
                max_backtracks=max_backtracks,
            )
        except _NewtonFailure as failure:
            if max_cutoff_branches <= 0 or max_iterations <= 0:
                raise
            return self._recover_cutoff_branches(
                failure,
                tolerance=tolerance,
                max_iterations=max_iterations,
                max_backtracks=max_backtracks,
                max_branches=max_cutoff_branches,
            )

    def _solve_newton(
        self,
        initial: tuple[np.ndarray, np.ndarray] | None,
        *,
        tolerance: float,
        max_iterations: int,
        max_backtracks: int,
        active_levels: np.ndarray | None = None,
    ) -> LogEquilibriumResult:
        """Take Newton steps on the original model or a fixed active set."""
        if initial is None:
            log_particles, scaled_multipliers = self.initial_state()
        else:
            log_particles = np.array(initial[0], copy=True)
            scaled_multipliers = np.array(initial[1], copy=True)

        cached_residual = None
        cached_thermodynamics = None
        for iteration in range(max_iterations + 1):
            if cached_residual is None:
                residual, derivative, thermodynamics = (
                    self._evaluate_with_state(
                        log_particles,
                        scaled_multipliers,
                        jacobian=True,
                        active_levels=active_levels,
                    )
                )
            else:
                residual = cached_residual
                assert cached_thermodynamics is not None
                thermodynamics = cached_thermodynamics
                derivative = self._jacobian_from_thermodynamics(thermodynamics)
            residual_norm = float(np.linalg.norm(residual, ord=np.inf))
            if residual_norm < tolerance:
                particle_numbers = thermodynamics.particle_numbers
                volume = (
                    thermodynamics.particle_total
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
                    candidate, _, candidate_thermodynamics = (
                        self._evaluate_with_state(
                            candidate_log,
                            candidate_multipliers,
                            jacobian=False,
                            cache_derivatives=True,
                            active_levels=active_levels,
                        )
                    )
                    candidate_merit = 0.5 * candidate @ candidate
                    if (
                        np.isfinite(candidate_merit)
                        and candidate_merit < (1 - 1e-4 * step) * merit
                    ):
                        log_particles = candidate_log
                        scaled_multipliers = candidate_multipliers
                        cached_residual = candidate
                        cached_thermodynamics = candidate_thermodynamics
                        break
                step *= 0.5
            else:
                raise _NewtonFailure(
                    f"Log-equilibrium line search stalled at residual "
                    f"{residual_norm:.3e}.",
                    (log_particles, scaled_multipliers),
                    residual_norm,
                    iteration,
                )

        raise _NewtonFailure(
            f"Log-equilibrium solver did not converge after {max_iterations} "
            f"iterations; residual={residual_norm:.3e}.",
            (log_particles, scaled_multipliers),
            residual_norm,
            max_iterations,
        )

    def _recover_cutoff_branches(
        self,
        failure: _NewtonFailure,
        *,
        tolerance: float,
        max_iterations: int,
        max_backtracks: int,
        max_branches: int,
    ) -> LogEquilibriumResult:
        """Explore nearby active sets without smoothing the physical model."""
        thermodynamics = self._packed_thermodynamics(
            failure.initial[0], derivatives=False
        )
        owners = self.level_owners
        margins = (
            self.monatomic_ionization[owners]
            - thermodynamics.ionization_lowering[owners]
            - self.level_energies
        )
        # Neutral cutoffs cannot move with composition in this model.
        nearby = np.flatnonzero(
            (np.abs(margins) < 2e-5 * u.k_b * self.mixture.T)
            & (self.charges[owners] > 0)
        )
        if self.electron_index < 0 or not nearby.size:
            raise failure
        nearby = nearby[np.argsort(np.abs(margins[nearby]))]
        active = margins > 0
        queue = [(active, failure.initial)]
        queued = {active.tobytes()}

        def enqueue(mask, initial, *, priority=False):
            key = mask.tobytes()
            if key not in queued:
                queued.add(key)
                if priority:
                    queue.insert(0, (mask, initial))
                else:
                    queue.append((mask, initial))

        # Flip equal-energy levels together: splitting them would create an
        # active set that the strict cutoff can never represent.
        for level in nearby:
            alternate = active.copy()
            group = (owners == owners[level]) & (
                self.level_energies == self.level_energies[level]
            )
            alternate[group] = ~alternate[group]
            enqueue(alternate, failure.initial)
            if len(queue) >= max_branches:
                break

        candidates = []
        total_iterations = failure.iterations
        attempted = 0
        while queue and attempted < max_branches:
            mask, initial = queue.pop(0)
            attempted += 1
            try:
                candidate = self._solve_newton(
                    initial,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                    max_backtracks=max_backtracks,
                    active_levels=mask,
                )
            except _NewtonFailure as branch_failure:
                total_iterations += branch_failure.iterations
                continue
            except np.linalg.LinAlgError:
                # A speculative branch can have a singular Jacobian.
                continue
            total_iterations += candidate.iterations
            residual, _, actual = self._evaluate_with_state(
                candidate.log_particles,
                candidate.scaled_multipliers,
                jacobian=False,
            )
            actual_active = self.level_energies < (
                self.monatomic_ionization[owners]
                - actual.ionization_lowering[owners]
            )
            norm = float(np.linalg.norm(residual, ord=np.inf))
            if np.array_equal(mask, actual_active) and norm < tolerance:
                candidates.append(replace(candidate, residual_norm=norm))
            else:
                # Follow the active set implied by this branch's root.
                # Deduplication stops the two-branch cycle of a cutoff gap.
                enqueue(
                    actual_active,
                    (candidate.log_particles, candidate.scaled_multipliers),
                    priority=True,
                )

        if not candidates:
            raise CutoffConvergenceError(
                self.mixture.T,
                self.mixture.P,
                self.mixture.species[owners[nearby[0]]].name,
                failure.residual_norm,
                attempted,
                total_iterations,
                self.residual_evaluations,
            ) from failure
        selected = min(candidates, key=self.dimensionless_gibbs)
        return replace(
            selected,
            iterations=total_iterations,
            residual_evaluations=self.residual_evaluations,
            cutoff_branches=attempted,
        )

    def dimensionless_gibbs(self, result: LogEquilibriumResult) -> float:
        """Return ``G/(k_B T)`` on the solver's arbitrary particle scale."""
        thermodynamics = self._packed_thermodynamics(
            result.log_particles, derivatives=False
        )
        chemical_potential = (
            thermodynamics.reference_energies / (u.k_b * self.mixture.T)
            - thermodynamics.log_partitions
            + result.log_particles
        )
        return float(thermodynamics.particle_numbers @ chemical_potential)

    def temperature_tangent(
        self, result: LogEquilibriumResult
    ) -> LogEquilibriumTemperatureTangent:
        """Differentiate the coupled log equilibrium at a converged state."""
        thermodynamics = self._packed_thermodynamics(
            result.log_particles, derivatives=True
        )
        derivative = self._jacobian_from_thermodynamics(thermodynamics)
        self._set_particle_numbers(thermodynamics.particle_numbers)
        reference_dN, reference_dT = (
            self.mixture._reference_energy_derivatives()
        )
        temperature = self.mixture.T
        kbt = u.k_b * temperature
        dlog_partition_dT = np.array(
            [
                1 / temperature
                + species.dlog_total_partition_dT(temperature, lowering)
                for species, lowering in zip(
                    self.mixture.species,
                    thermodynamics.ionization_lowering,
                )
            ]
        )
        explicit_derivative = np.zeros(
            self.species_count + self.constraint_count
        )
        explicit_derivative[: self.species_count] = (
            reference_dT / kbt
            - thermodynamics.reference_energies / (kbt * temperature)
            - dlog_partition_dT
        )
        state_derivative = np.linalg.solve(derivative, -explicit_derivative)
        log_particle_derivative = state_derivative[: self.species_count]
        particle_derivative = (
            thermodynamics.particle_numbers * log_particle_derivative
        )
        mole_fractions = (
            thermodynamics.particle_numbers / thermodynamics.particle_total
        )
        mole_fraction_derivative = mole_fractions * (
            log_particle_derivative - mole_fractions @ log_particle_derivative
        )
        return LogEquilibriumTemperatureTangent(
            log_particle_derivative=log_particle_derivative,
            particle_derivative=particle_derivative,
            mole_fraction_derivative=mole_fraction_derivative,
            reference_energy_derivative=(
                reference_dT + reference_dN @ particle_derivative
            ),
        )

    def heat_capacity(self, result: LogEquilibriumResult) -> float:
        """Return piecewise analytical Cp at a log-equilibrium result."""
        thermodynamics = self._packed_thermodynamics(
            result.log_particles, derivatives=False
        )
        tangent = self.temperature_tangent(result)
        temperature = self.mixture.T
        species = self.mixture.species
        internal_energies = np.array(
            [
                item.internal_energy(temperature, lowering)
                for item, lowering in zip(
                    species, thermodynamics.ionization_lowering
                )
            ]
        )
        internal_energy_derivatives = np.array(
            [
                item.dinternal_energy_dT(temperature, lowering)
                for item, lowering in zip(
                    species, thermodynamics.ionization_lowering
                )
            ]
        )
        enthalpies = (
            internal_energies
            + thermodynamics.reference_energies
            + u.k_b * temperature
        )
        enthalpy_derivatives = (
            internal_energy_derivatives
            + tangent.reference_energy_derivative
            + u.k_b
        )
        minimum = int(np.argmin(thermodynamics.reference_energies))
        masses = self.mixture.masses
        mass_ratios = masses / masses[minimum]
        relative_enthalpies = (
            enthalpies
            - thermodynamics.reference_energies[minimum] * mass_ratios
        )
        relative_enthalpy_derivatives = (
            enthalpy_derivatives
            - tangent.reference_energy_derivative[minimum] * mass_ratios
        )
        mole_fractions = (
            thermodynamics.particle_numbers / thermodynamics.particle_total
        )
        mean_mass = mole_fractions @ masses
        mean_mass_derivative = tangent.mole_fraction_derivative @ masses
        enthalpy_per_particle = mole_fractions @ relative_enthalpies
        enthalpy_derivative = (
            tangent.mole_fraction_derivative @ relative_enthalpies
            + mole_fractions @ relative_enthalpy_derivatives
        )
        return float(
            (
                enthalpy_derivative * mean_mass
                - enthalpy_per_particle * mean_mass_derivative
            )
            / mean_mass**2
        )

    def solve_lowest_gibbs_branch(
        self,
        initial: tuple[np.ndarray, np.ndarray] | None = None,
        *,
        tolerance: float = 1e-10,
        max_iterations: int = 80,
        cutoff_window: float = 2e-5,
        probe_distance: float = 1e-5,
    ) -> LogEquilibriumBranchResult:
        """Probe both sides of the nearest electronic cutoff and select G.

        ``cutoff_window`` and ``probe_distance`` are fractions of ``k_B T``.
        This local policy detects the pair of piecewise roots created as a
        level crosses the ionisation-lowered continuum, without pretending to
        perform a global combinatorial search over every electronic active
        set.
        """
        primary = self.solve(
            initial,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        thermodynamics = self._packed_thermodynamics(
            primary.log_particles, derivatives=False
        )
        if not self.level_energies.size:
            return LogEquilibriumBranchResult(
                selected=primary,
                candidates=(primary,),
                dimensionless_gibbs=(self.dimensionless_gibbs(primary),),
                nearest_cutoff_distance=float("inf"),
            )
        margins = (
            self.monatomic_ionization[self.level_owners]
            - thermodynamics.ionization_lowering[self.level_owners]
            - self.level_energies
        )
        nearest = int(np.argmin(np.abs(margins)))
        kbt = u.k_b * self.mixture.T
        cutoff_distance = float(abs(margins[nearest]) / kbt)
        candidates = [primary]

        if cutoff_distance < cutoff_window:
            owner = self.level_owners[nearest]
            _, lowering_dN = self._packed_lowering(
                thermodynamics.particle_numbers, derivatives=True
            )
            assert lowering_dN is not None
            margin_gradient = (
                -lowering_dN[owner] * thermodynamics.particle_numbers
            )
            gradient_square = float(margin_gradient @ margin_gradient)
            if gradient_square > 0:
                displacement = (
                    probe_distance * kbt * margin_gradient / gradient_square
                )
                for sign in (-1.0, 1.0):
                    try:
                        candidate = self.solve(
                            (
                                primary.log_particles + sign * displacement,
                                primary.scaled_multipliers,
                            ),
                            tolerance=tolerance,
                            max_iterations=max_iterations,
                        )
                    except RuntimeError:
                        # A probe may fall in the discontinuity gap where no
                        # root exists. The already converged primary remains
                        # valid, so a failed optional probe is not fatal.
                        continue
                    candidate_thermodynamics = self._packed_thermodynamics(
                        candidate.log_particles, derivatives=False
                    )
                    active_counts = (
                        candidate_thermodynamics.active_level_counts
                    )
                    if not any(
                        np.array_equal(
                            active_counts,
                            self._packed_thermodynamics(
                                existing.log_particles, derivatives=False
                            ).active_level_counts,
                        )
                        for existing in candidates
                    ):
                        candidates.append(candidate)

        gibbs = tuple(self.dimensionless_gibbs(item) for item in candidates)
        return LogEquilibriumBranchResult(
            selected=candidates[int(np.argmin(gibbs))],
            candidates=tuple(candidates),
            dimensionless_gibbs=gibbs,
            nearest_cutoff_distance=cutoff_distance,
        )

    def solve_temperature_path(
        self,
        temperatures: np.ndarray,
        *,
        bootstrap_temperature: float = 12000.0,
        maximum_temperature_step: float = 1000.0,
        tolerance: float = 1e-9,
        max_iterations: int = 80,
    ) -> LogEquilibriumPathResult:
        """Solve a temperature path, restoring the input T on any failure.

        On success the mixture remains at the last requested temperature.
        A cutoff gap at the bootstrap or an unrequested intermediate point
        may be bypassed. Requested points must always satisfy the full model.
        Skipping an intermediate gap can exceed the nominal temperature step.
        """
        requested = np.asarray(temperatures, dtype=np.float64)
        if requested.ndim != 1 or requested.size == 0:
            raise ValueError("temperatures must be a non-empty vector")

        original_temperature = self.mixture.T
        try:
            total_iterations = 0
            total_evaluations = 0
            solve_count = 0

            self.mixture.T = bootstrap_temperature
            try:
                current = self.solve(
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                )
            except CutoffConvergenceError as failure:
                if requested[0] == bootstrap_temperature:
                    raise
                total_iterations += failure.iterations
                total_evaluations += failure.residual_evaluations
                solve_count += 1
                # A bootstrap gap says nothing about the requested state.
                self.mixture.T = float(requested[0])
                current = self.solve(
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                )
            total_iterations += current.iterations
            total_evaluations += current.residual_evaluations
            solve_count += 1

            def advance(target: float) -> LogEquilibriumResult:
                nonlocal current, total_iterations
                nonlocal total_evaluations, solve_count
                start = self.mixture.T
                step_count = max(
                    1,
                    int(
                        np.ceil(abs(target - start) / maximum_temperature_step)
                    ),
                )
                initial = (current.log_particles, current.scaled_multipliers)
                for temperature in np.linspace(start, target, step_count + 1)[
                    1:
                ]:
                    self.mixture.T = float(temperature)
                    try:
                        current = self.solve(
                            initial,
                            tolerance=tolerance,
                            max_iterations=max_iterations,
                        )
                    except CutoffConvergenceError as failure:
                        if temperature == target:
                            raise
                        total_iterations += failure.iterations
                        total_evaluations += failure.residual_evaluations
                        solve_count += 1
                        # Keep the last converged initial state. No result
                        # from the gap is installed or returned to the caller.
                        continue
                    initial = (
                        current.log_particles,
                        current.scaled_multipliers,
                    )
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
        except Exception:
            self.mixture.T = original_temperature
            raise
