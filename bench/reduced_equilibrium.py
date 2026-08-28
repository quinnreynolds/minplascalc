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

from bench.equilibrium_thermodynamics import PackedEquilibriumThermodynamics
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
    ionization_lowering: np.ndarray
    temperature: float = 0.0
    method: str = "newton"


@dataclass(frozen=True)
class ReducedEquilibriumPathResult:
    """Requested temperature states and continuation diagnostics."""

    states: tuple[ReducedEquilibriumResult, ...]
    total_iterations: int
    total_residual_evaluations: int
    total_backtracks: int
    continuation_solves: int


@dataclass(frozen=True)
class ReducedEquilibriumBranchResult:
    """Locally competing active-set branches and their Gibbs diagnostics."""

    selected: ReducedEquilibriumResult
    candidates: tuple[ReducedEquilibriumResult, ...]
    dimensionless_gibbs: tuple[float, ...]
    nearest_cutoff_distance: float


@dataclass(frozen=True)
class ReducedEquilibriumTemperatureTangent:
    """Piecewise fixed-active-set temperature tangent of a reduced state."""

    log_number_density_derivative: np.ndarray
    number_density_derivative: np.ndarray
    mole_fraction_derivative: np.ndarray
    potential_derivative: np.ndarray
    reference_energy_derivative: np.ndarray

    @property
    def log_density_derivative(self) -> np.ndarray:
        """Alias using the shorter density terminology."""
        return self.log_number_density_derivative

    @property
    def density_derivative(self) -> np.ndarray:
        """Alias using the shorter density terminology."""
        return self.number_density_derivative

    @property
    def log_particle_derivative(self) -> np.ndarray:
        """Compatibility alias for the full log-space prototype."""
        return self.log_number_density_derivative

    @property
    def particle_derivative(self) -> np.ndarray:
        """Compatibility alias for the full log-space prototype."""
        return self.number_density_derivative


class ReducedEquilibriumSystem(PackedEquilibriumThermodynamics):
    """Fixed-lowering reduced equilibrium system for an LTE mixture."""

    def __init__(
        self,
        mixture,
        *,
        fixed_ionization_lowering: np.ndarray | None = None,
        coupled_ionization_lowering: bool = False,
    ):
        self.mixture = mixture
        self.coupled_ionization_lowering = bool(coupled_ionization_lowering)
        if (
            self.coupled_ionization_lowering
            and fixed_ionization_lowering is not None
        ):
            raise ValueError(
                "coupled ionisation lowering cannot be combined with a "
                "fixed lowering vector."
            )
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
        self.positive_indices = np.flatnonzero(self.charges > 0)
        if self.coupled_ionization_lowering:
            if not self.has_charge_constraint:
                raise ValueError(
                    "Coupled ionisation lowering requires an electron."
                )
            if self.positive_indices.size == 0:
                raise ValueError(
                    "Coupled ionisation lowering requires positive ions."
                )
        if self.has_charge_constraint:
            self.constraint_matrix = np.column_stack(
                (self.stoichiometry, self.charges)
            )
        else:
            self.constraint_matrix = self.stoichiometry.copy()
        self.base_potential_count = self.constraint_matrix.shape[1]
        self.potential_count = self.base_potential_count + (
            2 if self.coupled_ionization_lowering else 0
        )

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
        if rank < self.base_potential_count:
            raise ValueError(
                "The elemental/charge potential matrix is rank deficient "
                f"(rank {rank}, expected {self.base_potential_count})."
            )

        self._prepare_packed_thermodynamics()
        self.base_reference_energies = self._reference_energies()
        self._cached_temperature = np.nan
        self._log_q = np.empty(self.species_count)
        self._base_log_densities = np.empty(self.species_count)
        self._refresh_temperature_cache()
        if not self.coupled_ionization_lowering and np.any(
            ~np.isfinite(self._log_q)
        ):
            raise ValueError(
                "The fixed lowering must leave every species partition "
                "factor finite and positive."
            )
        self.residual_evaluations = 0

    def _refresh_temperature_cache(self) -> None:
        """Refresh fixed-lowering quantities after ``mixture.T`` changes."""
        temperature = float(self.mixture.T)
        if temperature == self._cached_temperature:
            return
        self._log_q = self._log_partition_per_volume()
        self._base_log_densities = (
            self._log_q - self.base_reference_energies / (u.k_b * temperature)
        )
        self._cached_temperature = temperature

    def _reference_energies(self) -> np.ndarray:
        """Reproduce the mixture reference-energy chain at fixed lowering."""
        reference, _ = self._packed_reference_from_lowering(
            self.fixed_ionization_lowering
        )
        return reference

    def _log_partition_per_volume(
        self, lowering: np.ndarray | None = None
    ) -> np.ndarray:
        """Evaluate ``log(q_i)`` without introducing a volume or state."""
        if lowering is None:
            lowering = self.fixed_ionization_lowering
        result, _ = self._packed_log_partition_per_volume(lowering)
        return result

    def _coupled_lowering(
        self, eta: float, xi: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return lowering and its derivatives with respect to eta and xi."""
        temperature = float(self.mixture.T)
        kbt = u.k_b * temperature
        with np.errstate(over="ignore", invalid="ignore"):
            electron_density = np.exp(eta)
            z_star = np.exp(xi)
        denominator = z_star + 1.0
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            debye_pow3 = (
                u.epsilon_0
                * kbt
                / (4 * np.pi * denominator * electron_density * u.e**2)
            ) ** 1.5
            ion_sphere_pow3 = (
                3
                * self.charges[self.positive_indices]
                / (4 * np.pi * electron_density)
            )
            ratio = ion_sphere_pow3 / debye_pow3
            shape = (ratio + 1.0) ** (2.0 / 3.0) - 1.0
            shape_derivative = (2.0 / 3.0) * (ratio + 1.0) ** (-1.0 / 3.0)

        lowering = np.zeros(self.species_count)
        d_eta = np.zeros(self.species_count)
        d_xi = np.zeros(self.species_count)
        with np.errstate(all="ignore"):
            lowering[self.positive_indices] = kbt * shape / (2 * denominator)
            dlog_ratio_dxi = 1.5 * z_star / denominator
            ratio_d_eta = 0.5 * ratio
            ratio_d_xi = ratio * dlog_ratio_dxi
            d_eta[self.positive_indices] = (
                kbt / (2 * denominator) * shape_derivative * ratio_d_eta
            )
            d_xi[self.positive_indices] = (
                kbt
                / 2
                * (
                    shape_derivative * ratio_d_xi / denominator
                    - shape * z_star / denominator**2
                )
            )
        return lowering, d_eta, d_xi

    def _coupled_temperature_lowering_derivative(
        self, eta: float, xi: float
    ) -> np.ndarray:
        """Return explicit d(lowering)/dT at fixed eta and xi."""
        temperature = float(self.mixture.T)
        kbt = u.k_b * temperature
        with np.errstate(over="ignore", invalid="ignore"):
            electron_density = np.exp(eta)
            z_star = np.exp(xi)
        denominator = z_star + 1.0
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            debye_pow3 = (
                u.epsilon_0
                * kbt
                / (4 * np.pi * denominator * electron_density * u.e**2)
            ) ** 1.5
            ion_sphere_pow3 = (
                3
                * self.charges[self.positive_indices]
                / (4 * np.pi * electron_density)
            )
            ratio = ion_sphere_pow3 / debye_pow3
            shape = (ratio + 1.0) ** (2.0 / 3.0) - 1.0
            shape_derivative = (2.0 / 3.0) * (ratio + 1.0) ** (-1.0 / 3.0)
        derivative = np.zeros(self.species_count)
        with np.errstate(all="ignore"):
            derivative[self.positive_indices] = (
                u.k_b
                / (2 * denominator)
                * (shape - 1.5 * ratio * shape_derivative)
            )
        return derivative

    def _reference_from_lowering(
        self,
        lowering: np.ndarray,
        lowering_derivatives: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Build reference energies and optional eta/xi derivatives."""
        derivative_matrix = (
            None
            if lowering_derivatives is None
            else np.column_stack(lowering_derivatives)
        )
        return self._packed_reference_from_lowering(
            lowering, derivative_matrix
        )

    def _reconstruct(
        self, potentials: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return logs, normalized densities, and stationarity slopes."""
        self._refresh_temperature_cache()
        base = potentials[: self.base_potential_count]
        if self.coupled_ionization_lowering:
            lowering, d_eta, d_xi = self._coupled_lowering(
                potentials[-2], potentials[-1]
            )
            reference, reference_derivative = self._reference_from_lowering(
                lowering, (d_eta, d_xi)
            )
            log_q = self._log_partition_per_volume(lowering)
            base_log_densities = log_q - reference / (u.k_b * self.mixture.T)
            slopes = np.column_stack(
                (
                    -self.constraint_matrix,
                    -reference_derivative / (u.k_b * self.mixture.T),
                )
            )
        else:
            lowering = self.fixed_ionization_lowering
            base_log_densities = self._base_log_densities
            slopes = -self.constraint_matrix
        log_densities = base_log_densities - self.constraint_matrix @ base
        log_total = _logsumexp(log_densities)
        with np.errstate(over="ignore", invalid="ignore"):
            fractions = np.exp(log_densities - log_total)
        return log_densities, fractions, slopes, lowering

    def _evaluate_state(
        self, potentials: np.ndarray, *, jacobian: bool
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
        self.residual_evaluations += 1
        with np.errstate(all="ignore"):
            log_densities, fractions, slopes, lowering = self._reconstruct(
                potentials
            )
        log_total = _logsumexp(log_densities)
        if not np.isfinite(log_total) or np.any(~np.isfinite(fractions)):
            size = 1 + self.element_count - 1 + self.has_charge_constraint
            if self.coupled_ionization_lowering:
                size += 2
            residual = np.full(size, np.inf)
            return residual, None, log_densities, lowering

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
        if self.coupled_ionization_lowering:
            positive = self.positive_indices
            log_moment_one = _logsumexp(
                log_densities[positive] + np.log(self.charges[positive])
            )
            log_moment_two = _logsumexp(
                log_densities[positive] + 2 * np.log(self.charges[positive])
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
        if self.coupled_ionization_lowering:
            eta, xi = potentials[-2:]
            residual.extend(
                (
                    log_densities[self.electron_index] - eta,
                    log_moment_two - log_moment_one - xi,
                )
            )
        residual_array = np.asarray(residual, dtype=np.float64)

        if not jacobian:
            return residual_array, None, log_densities, lowering

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
            charge_row = self.element_count
            derivative[charge_row] = (
                fractions * (self.charges - charge_residual)
            ) @ slopes
        if self.coupled_ionization_lowering:
            electron_row = self.element_count + self.has_charge_constraint
            zstar_row = electron_row + 1
            derivative[electron_row] = slopes[self.electron_index]
            derivative[electron_row, -2] -= 1.0
            positive = self.positive_indices
            rho_one = np.zeros(self.species_count)
            rho_two = np.zeros(self.species_count)
            rho_one[positive] = self.charges[positive] * np.exp(
                log_densities[positive] - log_moment_one
            )
            rho_two[positive] = self.charges[positive] ** 2 * np.exp(
                log_densities[positive] - log_moment_two
            )
            derivative[zstar_row] = (rho_two - rho_one) @ slopes
            derivative[zstar_row, -1] -= 1.0
        return residual_array, derivative, log_densities, lowering

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
        residual, derivative, _, _ = self._evaluate_state(
            values, jacobian=jacobian
        )
        return residual, derivative

    def initial_state(self) -> np.ndarray:
        """Return a generic potential estimate without a full-system solve."""
        self._refresh_temperature_cache()
        x0 = np.asarray(self.mixture.x0, dtype=np.float64)
        weights = np.maximum(x0, 1e-12)
        weights /= weights.sum()
        desired = np.log(self.mixture.P / (u.k_b * self.mixture.T)) + np.log(
            weights
        )
        base = np.linalg.lstsq(
            self.constraint_matrix,
            self._base_log_densities - desired,
            rcond=None,
        )[0]
        if not self.coupled_ionization_lowering:
            return base
        electron_density = self.mixture.P / (u.k_b * self.mixture.T)
        electron_density *= weights[self.electron_index]
        positive = self.positive_indices
        z_star = (weights[positive] @ self.charges[positive] ** 2) / (
            weights[positive] @ self.charges[positive]
        )
        return np.concatenate(
            (
                base,
                [np.log(electron_density), np.log(z_star)],
            )
        )

    def solve(
        self,
        initial: np.ndarray | None = None,
        *,
        tolerance: float = 1e-10,
        max_iterations: int = 80,
        max_backtracks: int = 30,
        method: str = "newton",
    ) -> ReducedEquilibriumResult:
        """Solve with Newton or analytical trust-region least squares."""
        if method in {"least_squares", "trust_region", "trf"}:
            return self._solve_least_squares(
                initial,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
        if method != "newton":
            raise ValueError("method must be 'newton' or 'least_squares'.")
        return self._solve_newton(
            initial,
            tolerance=tolerance,
            max_iterations=max_iterations,
            max_backtracks=max_backtracks,
        )

    def _initial_potentials(self, initial: np.ndarray | None) -> np.ndarray:
        """Validate and copy a starting reduced state."""
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
        return potentials

    def _solve_newton(
        self,
        initial: np.ndarray | None = None,
        *,
        tolerance: float = 1e-10,
        max_iterations: int = 80,
        max_backtracks: int = 30,
    ) -> ReducedEquilibriumResult:
        """Solve with analytical Newton steps and Armijo backtracking."""
        potentials = self._initial_potentials(initial)
        self.residual_evaluations = 0
        backtracks = 0
        for iteration in range(max_iterations + 1):
            residual, jacobian, log_densities, lowering = self._evaluate_state(
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
                    ionization_lowering=lowering.copy(),
                    temperature=float(self.mixture.T),
                    method="newton",
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
                    candidate, _, _, _ = self._evaluate_state(
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

    def _solve_least_squares(
        self,
        initial: np.ndarray | None = None,
        *,
        tolerance: float = 1e-10,
        max_iterations: int = 80,
    ) -> ReducedEquilibriumResult:
        """Solve with scipy's trust-region reflective least-squares method."""
        from scipy.optimize import least_squares

        potentials = self._initial_potentials(initial)
        self.residual_evaluations = 0

        def residual_function(values: np.ndarray) -> np.ndarray:
            residual, _, _, _ = self._evaluate_state(values, jacobian=False)
            return residual

        def jacobian_function(values: np.ndarray) -> np.ndarray:
            _, jacobian, _, _ = self._evaluate_state(values, jacobian=True)
            if jacobian is None:
                return np.zeros((self.potential_count, self.potential_count))
            return jacobian

        optimizer = least_squares(
            residual_function,
            potentials,
            jac=jacobian_function,
            method="trf",
            x_scale="jac",
            ftol=max(tolerance * 0.1, 1e-14),
            xtol=max(tolerance * 0.1, 1e-14),
            gtol=max(tolerance * 0.1, 1e-14),
            max_nfev=max_iterations,
        )
        residual, jacobian, logs, lowering = self._evaluate_state(
            optimizer.x, jacobian=True
        )
        residual_norm = float(np.linalg.norm(residual, ord=np.inf))
        if (
            not optimizer.success
            or jacobian is None
            or not np.isfinite(residual_norm)
            or residual_norm >= tolerance
        ):
            raise RuntimeError(
                "Reduced-equilibrium trust-region solve did not meet the "
                f"residual tolerance (status={optimizer.status}, "
                f"residual={residual_norm:.3e})."
            )
        return ReducedEquilibriumResult(
            potentials=optimizer.x.copy(),
            log_number_densities=logs.copy(),
            number_densities=np.exp(logs),
            residual_norm=residual_norm,
            iterations=int(optimizer.nfev),
            residual_evaluations=self.residual_evaluations,
            backtracks=0,
            jacobian_condition=float(np.linalg.cond(jacobian)),
            ionization_lowering=lowering.copy(),
            temperature=float(self.mixture.T),
            method="least_squares",
        )

    def solve_temperature_path(
        self,
        temperatures: np.ndarray,
        *,
        bootstrap_temperature: float = 12000.0,
        max_temperature_step: float = 1000.0,
        method: str = "least_squares",
        tolerance: float = 1e-10,
        max_iterations: int = 80,
        max_backtracks: int = 30,
    ) -> ReducedEquilibriumPathResult:
        """Solve requested temperatures by continuation from 12,000 K."""
        requested = np.asarray(temperatures, dtype=np.float64)
        if requested.ndim != 1 or requested.size == 0:
            raise ValueError(
                "temperatures must be a non-empty one-dimensional array."
            )
        if np.any(~np.isfinite(requested)) or np.any(requested <= 0):
            raise ValueError("temperatures must be finite and positive.")
        if (
            not np.isfinite(bootstrap_temperature)
            or bootstrap_temperature <= 0
            or not np.isfinite(max_temperature_step)
            or max_temperature_step <= 0
        ):
            raise ValueError(
                "bootstrap temperature and step must be positive."
            )

        original_temperature = float(self.mixture.T)
        total_iterations = 0
        total_evaluations = 0
        total_backtracks = 0
        continuation_solves = 0

        def record(state: ReducedEquilibriumResult) -> None:
            nonlocal total_iterations, total_evaluations
            nonlocal total_backtracks, continuation_solves
            total_iterations += state.iterations
            total_evaluations += state.residual_evaluations
            total_backtracks += state.backtracks
            continuation_solves += 1

        try:
            self.mixture.T = float(bootstrap_temperature)
            bootstrap = self.solve(
                method=method,
                tolerance=tolerance,
                max_iterations=max_iterations,
                max_backtracks=max_backtracks,
            )
            record(bootstrap)
            states: dict[float, ReducedEquilibriumResult] = {
                float(bootstrap_temperature): bootstrap
            }

            lower = sorted(
                {
                    float(value)
                    for value in requested
                    if value < bootstrap_temperature
                },
                reverse=True,
            )
            upper = sorted(
                {
                    float(value)
                    for value in requested
                    if value > bootstrap_temperature
                }
            )
            for direction in (lower, upper):
                previous_temperature = float(bootstrap_temperature)
                previous = bootstrap
                for target in direction:
                    distance = abs(target - previous_temperature)
                    steps = max(
                        1, int(np.ceil(distance / max_temperature_step))
                    )
                    for step in range(1, steps + 1):
                        temperature = (
                            previous_temperature
                            + (target - previous_temperature) * step / steps
                        )
                        self.mixture.T = temperature
                        previous = self.solve(
                            initial=previous.potentials,
                            method=method,
                            tolerance=tolerance,
                            max_iterations=max_iterations,
                            max_backtracks=max_backtracks,
                        )
                        record(previous)
                    previous_temperature = target
                    states[target] = previous
        finally:
            self.mixture.T = original_temperature
            self._refresh_temperature_cache()

        return ReducedEquilibriumPathResult(
            states=tuple(states[float(value)] for value in requested),
            total_iterations=total_iterations,
            total_residual_evaluations=total_evaluations,
            total_backtracks=total_backtracks,
            continuation_solves=continuation_solves,
        )

    def dimensionless_gibbs(self, result: ReducedEquilibriumResult) -> float:
        """Return the reduced branch's qualified ``G/(k_B T)`` diagnostic."""
        original_temperature = float(self.mixture.T)
        try:
            self.mixture.T = float(result.temperature)
            logs = np.asarray(result.log_number_densities, dtype=np.float64)
            lowering = np.asarray(result.ionization_lowering, dtype=np.float64)
            reference, _ = self._reference_from_lowering(lowering)
            log_q = self._log_partition_per_volume(lowering)
            with np.errstate(over="ignore", invalid="ignore"):
                densities = np.exp(logs)
                chemical = (
                    reference / (u.k_b * result.temperature) - log_q + logs
                )
                value = densities @ chemical
            return float(value)
        finally:
            self.mixture.T = original_temperature
            self._refresh_temperature_cache()

    def solve_lowest_gibbs_branch(
        self,
        initial: np.ndarray | None = None,
        *,
        method: str = "least_squares",
        tolerance: float = 1e-10,
        max_iterations: int = 80,
        max_backtracks: int = 30,
        cutoff_window: float = 2e-5,
        probe_distance: float = 1e-5,
    ) -> ReducedEquilibriumBranchResult:
        """Probe both sides of the nearest cutoff and select lowest ``G``."""
        if not self.coupled_ionization_lowering:
            raise ValueError(
                "Cutoff branch probing requires coupled lowering."
            )
        if not np.isfinite(cutoff_window) or cutoff_window < 0:
            raise ValueError("cutoff_window must be finite and nonnegative.")
        if not np.isfinite(probe_distance) or probe_distance <= 0:
            raise ValueError("probe_distance must be finite and positive.")
        base = self.solve(
            initial=initial,
            method=method,
            tolerance=tolerance,
            max_iterations=max_iterations,
            max_backtracks=max_backtracks,
        )
        fingerprint = self.active_level_fingerprint(base)
        nearest_index = fingerprint.nearest_cutoff_species_index
        nearest_level = fingerprint.nearest_cutoff_level_index
        if nearest_index is None or nearest_level is None:
            return ReducedEquilibriumBranchResult(
                selected=base,
                candidates=(base,),
                dimensionless_gibbs=(self.dimensionless_gibbs(base),),
                nearest_cutoff_distance=np.inf,
            )
        _, d_eta, d_xi = self._coupled_lowering(*base.potentials[-2:])
        gradient = np.zeros(self.potential_count)
        gradient[-2:] = (-d_eta[nearest_index], -d_xi[nearest_index])
        gradient_norm_squared = float(gradient @ gradient)
        distance = fingerprint.nearest_cutoff_margin
        dimensionless_distance = (
            np.inf
            if distance is None
            else abs(distance / (u.k_b * self.mixture.T))
        )
        if (
            distance is None
            or not np.isfinite(distance)
            or not np.isfinite(gradient_norm_squared)
            or gradient_norm_squared == 0
            or dimensionless_distance >= cutoff_window
        ):
            return ReducedEquilibriumBranchResult(
                selected=base,
                candidates=(base,),
                dimensionless_gibbs=(self.dimensionless_gibbs(base),),
                nearest_cutoff_distance=dimensionless_distance,
            )
        probe_margin = probe_distance * u.k_b * self.mixture.T
        candidates = [base]
        seen = {fingerprint.fingerprint}
        for sign in (-1.0, 1.0):
            target_margin = sign * probe_margin
            trial = base.potentials + (
                (target_margin - distance) / gradient_norm_squared * gradient
            )
            try:
                candidate = self.solve(
                    initial=trial,
                    method=method,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                    max_backtracks=max_backtracks,
                )
            except RuntimeError:
                continue
            candidate_fingerprint = self.active_level_fingerprint(candidate)
            if candidate_fingerprint.fingerprint not in seen:
                seen.add(candidate_fingerprint.fingerprint)
                candidates.append(candidate)
        gibbs = tuple(
            self.dimensionless_gibbs(candidate) for candidate in candidates
        )
        selected = candidates[int(np.argmin(gibbs))]
        return ReducedEquilibriumBranchResult(
            selected=selected,
            candidates=tuple(candidates),
            dimensionless_gibbs=gibbs,
            nearest_cutoff_distance=dimensionless_distance,
        )

    def temperature_tangent(
        self, result: ReducedEquilibriumResult
    ) -> ReducedEquilibriumTemperatureTangent:
        """Differentiate a converged reduced state at fixed pressure."""
        potentials = np.asarray(result.potentials, dtype=np.float64)
        residual, jacobian, log_densities, lowering = self._evaluate_state(
            potentials, jacobian=True
        )
        del residual
        if jacobian is None:
            raise RuntimeError(
                "Cannot differentiate a non-finite reduced state."
            )

        temperature = float(self.mixture.T)
        kbt = u.k_b * temperature
        if self.coupled_ionization_lowering:
            _, d_lowering_deta, d_lowering_dxi = self._coupled_lowering(
                potentials[-2], potentials[-1]
            )
            reference, reference_auxiliary_derivative = (
                self._reference_from_lowering(
                    lowering, (d_lowering_deta, d_lowering_dxi)
                )
            )
            assert reference_auxiliary_derivative is not None
            d_lowering_dT = self._coupled_temperature_lowering_derivative(
                potentials[-2], potentials[-1]
            )
            _, reference_derivative = self._reference_from_lowering(
                lowering, (d_lowering_dT, np.zeros(self.species_count))
            )
            assert reference_derivative is not None
            reference_dT = reference_derivative[:, 0]
        else:
            reference_dT = np.zeros(self.species_count)
            reference = self.base_reference_energies
            reference_auxiliary_derivative = None
        dlog_q_dT = np.array(
            [
                species.dlog_total_partition_dT(temperature, dE)
                for species, dE in zip(self.species, lowering)
            ]
        )
        explicit_species = (
            dlog_q_dT - reference_dT / kbt + reference / (kbt * temperature)
        )
        _, fractions, slopes, _ = self._reconstruct(potentials)
        explicit = [1.0 / temperature + float(fractions @ explicit_species)]
        log_moments = []
        for column in range(self.element_count):
            present = self.stoichiometry[:, column] > 0
            log_moments.append(
                _logsumexp(
                    log_densities[present]
                    + np.log(self.stoichiometry[present, column])
                )
            )
        for column in range(1, self.element_count):
            weights = np.zeros(self.species_count)
            present = self.stoichiometry[:, column] > 0
            weights[present] = self.stoichiometry[present, column] * np.exp(
                log_densities[present] - log_moments[column]
            )
            reference_weights = np.zeros(self.species_count)
            present_reference = self.stoichiometry[:, 0] > 0
            reference_weights[present_reference] = self.stoichiometry[
                present_reference, 0
            ] * np.exp(log_densities[present_reference] - log_moments[0])
            explicit.append((weights - reference_weights) @ explicit_species)
        charge_residual = float(fractions @ self.charges)
        if self.has_charge_constraint:
            explicit.append(
                (fractions * (self.charges - charge_residual))
                @ explicit_species
            )
        if self.coupled_ionization_lowering:
            positive = self.positive_indices
            log_moment_one = _logsumexp(
                log_densities[positive] + np.log(self.charges[positive])
            )
            log_moment_two = _logsumexp(
                log_densities[positive] + 2 * np.log(self.charges[positive])
            )
            rho_one = np.zeros(self.species_count)
            rho_two = np.zeros(self.species_count)
            rho_one[positive] = self.charges[positive] * np.exp(
                log_densities[positive] - log_moment_one
            )
            rho_two[positive] = self.charges[positive] ** 2 * np.exp(
                log_densities[positive] - log_moment_two
            )
            explicit.append(explicit_species[self.electron_index])
            explicit.append((rho_two - rho_one) @ explicit_species)
        explicit_array = np.asarray(explicit)
        potential_derivative = np.linalg.solve(jacobian, -explicit_array)
        log_density_derivative = (
            explicit_species + slopes @ potential_derivative
        )
        total_reference_derivative = reference_dT
        if reference_auxiliary_derivative is not None:
            total_reference_derivative = (
                reference_dT
                + reference_auxiliary_derivative @ potential_derivative[-2:]
            )
        densities = np.exp(log_densities)
        number_density_derivative = densities * log_density_derivative
        mole_fractions = fractions
        mole_fraction_derivative = mole_fractions * (
            log_density_derivative - mole_fractions @ log_density_derivative
        )
        return ReducedEquilibriumTemperatureTangent(
            log_number_density_derivative=log_density_derivative,
            number_density_derivative=number_density_derivative,
            mole_fraction_derivative=mole_fraction_derivative,
            potential_derivative=potential_derivative,
            reference_energy_derivative=total_reference_derivative,
        )

    def active_level_fingerprint(self, result: ReducedEquilibriumResult):
        """Return active-level diagnostics for a reduced result."""
        lowering = np.asarray(result.ionization_lowering, dtype=np.float64)
        if lowering.shape != (self.species_count,):
            raise ValueError(
                "result has an invalid ionisation-lowering shape."
            )
        original_temperature = float(self.mixture.T)
        try:
            self.mixture.T = float(result.temperature)
            return self.mixture._active_level_fingerprint(lowering)
        finally:
            self.mixture.T = original_temperature
            self._refresh_temperature_cache()
