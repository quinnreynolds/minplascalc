"""Stage-4 continuation checks for the coupled reduced prototype."""

from __future__ import annotations

import numpy as np
import pytest

from bench.log_equilibrium import LogEquilibriumSystem
from bench.reduced_equilibrium import ReducedEquilibriumSystem
from bench.workloads import make_sico, make_simple
from minplascalc import units as u

TEMPERATURES = np.linspace(1000.0, 25000.0, 9)


def _path(factory, pressure, descending):
    temperatures = TEMPERATURES[::-1] if descending else TEMPERATURES
    mixture = factory(T=float(temperatures[0]), P=pressure)
    system = ReducedEquilibriumSystem(
        mixture, coupled_ionization_lowering=True
    )
    path = system.solve_temperature_path(
        temperatures,
        bootstrap_temperature=12000.0,
        max_temperature_step=1000.0,
        tolerance=1e-9,
    )
    return temperatures, mixture, system, path


def _reduced_seed_from_full(system, result, temperature):
    """Map an interior full-log root into the reduced coordinates."""
    original_temperature = float(system.mixture.T)
    try:
        system.mixture.T = float(temperature)
        logs = np.log(result.number_densities)
        positive = system.positive_indices
        z_star = (
            result.number_densities[positive] @ system.charges[positive] ** 2
        ) / (result.number_densities[positive] @ system.charges[positive])
        eta = logs[system.electron_index]
        xi = np.log(z_star)
        lowering, _, _ = system._coupled_lowering(eta, xi)
        reference, _ = system._reference_from_lowering(lowering)
        log_q = system._log_partition_per_volume(lowering)
        base = np.linalg.lstsq(
            system.constraint_matrix,
            log_q - reference / (u.k_b * temperature) - logs,
            rcond=None,
        )[0]
        seed = np.concatenate((base, [eta, xi]))
        residual, _ = system.evaluate(seed)
        reconstructed, _, _, _ = system._reconstruct(seed)
        return seed, residual, reconstructed
    finally:
        system.mixture.T = original_temperature
        system._refresh_temperature_cache()


@pytest.mark.parametrize("factory", [make_simple, make_sico])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("pressure", [101325.0, 10132500.0])
def test_coupled_temperature_path_is_finite_and_diagnostic(
    factory, descending, pressure
):
    temperatures, _, system, path = _path(factory, pressure, descending)

    assert len(path.states) == len(temperatures)
    assert path.continuation_solves >= len(temperatures)
    assert path.total_iterations > 0
    assert path.total_residual_evaluations >= path.continuation_solves
    for temperature, state in zip(temperatures, path.states):
        assert state.temperature == pytest.approx(temperature)
        assert state.residual_norm < 1e-8
        assert state.residual_evaluations > 0
        assert np.all(np.isfinite(state.number_densities))
        assert np.all(state.number_densities > 0)
        assert np.all(np.isfinite(state.ionization_lowering))
        assert np.all(state.ionization_lowering >= 0)
        assert state.potentials.size == system.potential_count


@pytest.mark.parametrize("factory", [make_simple, make_sico])
@pytest.mark.parametrize("pressure", [101325.0, 10132500.0])
def test_cold_endpoint_requires_continuation(factory, pressure):
    cold_mixture = factory(T=1000.0, P=pressure)
    cold_system = ReducedEquilibriumSystem(
        cold_mixture, coupled_ionization_lowering=True
    )
    with pytest.raises(RuntimeError, match="line search|converge|finite"):
        cold_system.solve(tolerance=1e-9)

    _, _, _, path = _path(factory, pressure, descending=False)
    cold = path.states[0]
    assert cold.temperature == pytest.approx(1000.0)
    assert cold.residual_norm < 1e-8
    assert np.all(np.isfinite(cold.number_densities))
    assert np.all(cold.number_densities > 0)


@pytest.mark.parametrize("factory", [make_simple, make_sico])
@pytest.mark.parametrize("pressure", [101325.0, 10132500.0])
def test_continuation_endpoints_match_full_log(factory, pressure):
    temperatures, _, reduced_system, reduced_path = _path(
        factory, pressure, descending=False
    )
    full = LogEquilibriumSystem(factory(T=float(temperatures[0]), P=pressure))
    full_path = full.solve_temperature_path(
        temperatures[[0, -1]],
        bootstrap_temperature=12000.0,
        maximum_temperature_step=1000.0,
        tolerance=1e-9,
    )

    for temperature, reduced, reference in zip(
        (temperatures[0], temperatures[-1]),
        (reduced_path.states[0], reduced_path.states[-1]),
        (full_path.states[0], full_path.states[-1]),
    ):
        reduced_fractions = reduced.number_densities / np.sum(
            reduced.number_densities
        )
        reference_fractions = reference.number_densities / np.sum(
            reference.number_densities
        )
        assert np.sum(reduced.number_densities) == pytest.approx(
            np.sum(reference.number_densities), rel=5e-9
        )
        # At cold endpoints the lowering closure has multiple roots which
        # differ only in trace species.  Compare the observable composition,
        # then prove separately that the full root lies on the reduced system.
        assert np.sum(np.abs(reduced_fractions - reference_fractions)) < 1e-8
        bulk = reference_fractions > 1e-8
        assert reduced.number_densities[bulk] == pytest.approx(
            reference.number_densities[bulk], rel=5e-6
        )
        _, residual, reconstructed = _reduced_seed_from_full(
            reduced_system, reference, temperature
        )
        assert np.linalg.norm(residual, ord=np.inf) < 1e-8
        assert (
            np.max(np.abs(reconstructed - np.log(reference.number_densities)))
            < 2e-9
        )


def test_trust_region_rejects_false_optimizer_convergence_if_exposed():
    mixture = make_sico(T=12000.0)
    system = ReducedEquilibriumSystem(
        mixture, coupled_ionization_lowering=True
    )
    with pytest.raises(RuntimeError, match="residual|converge|stationarity"):
        system.solve(
            initial=np.zeros(system.potential_count),
            method="least_squares",
            max_iterations=1,
        )
