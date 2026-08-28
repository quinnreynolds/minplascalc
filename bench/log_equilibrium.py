"""Experimental full-equilibrium Newton solver in log particle numbers.

This deliberately lives outside the package API. It reuses the production
thermodynamic functions while testing a different nonlinear formulation:
chemical-potential equilibrium and every conservation law are solved together,
with positivity guaranteed by ``u_i = log(N_i)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bench.equilibrium_thermodynamics import PackedEquilibriumThermodynamics
from minplascalc import units as u
from minplascalc.mixture import LTE, ActiveLevelFingerprint


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


class LogEquilibriumSystem(PackedEquilibriumThermodynamics):
    """Coupled dimensionless equilibrium residual in log variables."""

    def __init__(
        self,
        mixture: LTE,
        *,
        fixed_ionization_lowering: np.ndarray | None = None,
    ):
        self.mixture = mixture
        self.species_count = len(mixture.species)
        if fixed_ionization_lowering is None:
            self.fixed_ionization_lowering = None
        else:
            fixed_ionization_lowering = np.asarray(
                fixed_ionization_lowering, dtype=np.float64
            )
            if fixed_ionization_lowering.shape != (self.species_count,):
                raise ValueError(
                    "fixed_ionization_lowering must have one value per "
                    "species."
                )
            if not np.all(np.isfinite(fixed_ionization_lowering)):
                raise ValueError(
                    "fixed_ionization_lowering must contain only finite "
                    "values."
                )
            if np.any(fixed_ionization_lowering < 0):
                raise ValueError(
                    "fixed_ionization_lowering must be nonnegative."
                )
            self.fixed_ionization_lowering = fixed_ionization_lowering.copy()
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

    def _set_particle_numbers(self, particle_numbers: np.ndarray) -> None:
        """Set the private iterate on the prototype-owned mixture."""
        self.mixture._LTE__Ni = particle_numbers

    def _packed_lowering(
        self, particle_numbers: np.ndarray, *, derivatives: bool
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Evaluate ionisation lowering and its particle-number Jacobian."""
        count = self.species_count
        if self.fixed_ionization_lowering is not None:
            derivative = np.zeros((count, count)) if derivatives else None
            return self.fixed_ionization_lowering.copy(), derivative

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
        reference, reference_dN = self._packed_reference_from_lowering(
            lowering, lowering_dN
        )
        log_partition_density, active_counts = (
            self._packed_log_partition_per_volume(lowering)
        )
        log_partitions = np.log(volume) + log_partition_density
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
        _, reference_dN = self._packed_reference_from_lowering(
            np.zeros(self.species_count), lowering_dN
        )
        assert reference_dN is not None
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
    ) -> tuple[np.ndarray, np.ndarray | None, _PackedThermodynamics]:
        """Evaluate a new thermodynamic state and retain it for reuse."""
        self.residual_evaluations += 1
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            thermodynamics = self._packed_thermodynamics(
                log_particles, derivatives=jacobian or cache_derivatives
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
        cached_residual = None
        cached_thermodynamics = None
        for iteration in range(max_iterations + 1):
            if cached_residual is None:
                residual, derivative, thermodynamics = (
                    self._evaluate_with_state(
                        log_particles,
                        scaled_multipliers,
                        jacobian=True,
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
                raise RuntimeError(
                    f"Log-equilibrium line search stalled at residual "
                    f"{residual_norm:.3e}."
                )

        raise RuntimeError(
            f"Log-equilibrium solver did not converge after {max_iterations} "
            f"iterations; residual={residual_norm:.3e}."
        )

    def dimensionless_gibbs(self, result: LogEquilibriumResult) -> float:
        """Return ``G/(k_B T)`` on the prototype's arbitrary particle scale."""
        thermodynamics = self._packed_thermodynamics(
            result.log_particles, derivatives=False
        )
        chemical_potential = (
            thermodynamics.reference_energies / (u.k_b * self.mixture.T)
            - thermodynamics.log_partitions
            + result.log_particles
        )
        return float(thermodynamics.particle_numbers @ chemical_potential)

    def active_level_fingerprint(
        self, result: LogEquilibriumResult
    ) -> ActiveLevelFingerprint:
        """Return active-level diagnostics for a converged prototype state."""
        thermodynamics = self._packed_thermodynamics(
            result.log_particles, derivatives=False
        )
        return self.mixture._active_level_fingerprint(
            thermodynamics.ionization_lowering
        )

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
            self.mixture._LTE__get_reference_energy_derivatives()
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
        cutoff_window: float = 2e-5,
        probe_distance: float = 1e-5,
    ) -> LogEquilibriumBranchResult:
        """Probe both sides of the nearest electronic cutoff and select G.

        ``cutoff_window`` and ``probe_distance`` are fractions of ``k_B T``.
        This is deliberately a local policy for the exploratory solver: it
        detects the pair of piecewise roots created as a level crosses the
        ionisation-lowered continuum, without pretending to perform a global
        combinatorial search over every electronic active set.
        """
        primary = self.solve(initial, tolerance=tolerance)
        thermodynamics = self._packed_thermodynamics(
            primary.log_particles, derivatives=False
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
