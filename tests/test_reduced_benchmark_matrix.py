"""Stage-5 robustness and conditioning checks for the reduced prototype."""

from __future__ import annotations

import numpy as np
import pytest

from bench.log_equilibrium import LogEquilibriumSystem
from bench.reduced_equilibrium import ReducedEquilibriumSystem
from bench.workloads import make_sico, make_simple
from minplascalc import units as u


def _equilibrated_condition(jacobian):
    """Condition after one infinity-norm row and column equilibration."""
    row_norms = np.max(np.abs(jacobian), axis=1)
    scaled = jacobian / np.where(row_norms > 0, row_norms, 1)[:, None]
    column_norms = np.max(np.abs(scaled), axis=0)
    scaled /= np.where(column_norms > 0, column_norms, 1)[None, :]
    return np.linalg.cond(scaled)


@pytest.mark.parametrize("factory", [make_simple, make_sico])
@pytest.mark.parametrize("pressure", [1013.25, 101325.0, 10132500.0])
@pytest.mark.parametrize("temperature", [12000.0, 20000.0])
def test_reduced_benchmark_matrix_matches_full_observables(
    factory, pressure, temperature
):
    reduced_mixture = factory(T=temperature, P=pressure)
    reduced = ReducedEquilibriumSystem(
        reduced_mixture, coupled_ionization_lowering=True
    )
    reduced_result = reduced.solve_temperature_path(
        np.array([temperature]),
        bootstrap_temperature=12000.0,
        max_temperature_step=1000.0,
        tolerance=1e-9,
    ).states[0]

    full_mixture = factory(T=temperature, P=pressure)
    full = LogEquilibriumSystem(full_mixture)
    full_result = full.solve(tolerance=1e-9)

    reduced_fractions = reduced_result.number_densities / (
        reduced_result.number_densities.sum()
    )
    full_fractions = full_result.number_densities / (
        full_result.number_densities.sum()
    )
    assert reduced_result.residual_norm < 1e-8
    assert full_result.residual_norm < 1e-8
    assert reduced_fractions == pytest.approx(
        full_fractions, rel=5e-6, abs=1e-12
    )
    assert np.all(np.isfinite(reduced_result.number_densities))
    assert np.all(reduced_result.number_densities > 0)
    assert np.all(np.isfinite(reduced_result.ionization_lowering))
    assert np.all(reduced_result.ionization_lowering >= 0)

    total = reduced_result.number_densities.sum()
    assert total * u.k_b * temperature / pressure == pytest.approx(
        1.0, rel=1e-8
    )
    concentrations = reduced.stoichiometry.T @ reduced_result.number_densities
    target_ratios = concentrations / reduced.targets
    assert target_ratios == pytest.approx(
        target_ratios[0], rel=2e-8, abs=2e-12
    )
    charge_residual = (
        np.dot(reduced.charges, reduced_result.number_densities) / total
    )
    assert charge_residual == pytest.approx(0.0, abs=2e-8)


@pytest.mark.parametrize("factory", [make_simple, make_sico])
def test_reduced_unknown_count_and_conditioning_are_finite(factory):
    temperature = 12000.0
    pressure = 101325.0
    reduced = ReducedEquilibriumSystem(
        factory(T=temperature, P=pressure),
        coupled_ionization_lowering=True,
    )
    result = reduced.solve(tolerance=1e-9)
    _, reduced_jacobian = reduced.evaluate(result.potentials)
    assert reduced_jacobian is not None
    full = LogEquilibriumSystem(factory(T=temperature, P=pressure))
    full_result = full.solve(tolerance=1e-9)
    _, full_jacobian = full.evaluate(
        full_result.log_particles,
        full_result.scaled_multipliers,
    )
    assert full_jacobian is not None

    assert reduced.base_potential_count == (
        reduced.element_count + int(reduced.has_charge_constraint)
    )
    assert reduced.potential_count == reduced.base_potential_count + 2
    assert reduced.potential_count < (
        full.species_count + full.constraint_count
    )
    assert np.isfinite(result.jacobian_condition)
    assert result.jacobian_condition > 0
    assert np.isfinite(np.linalg.cond(full_jacobian))
    assert _equilibrated_condition(reduced_jacobian) < (
        _equilibrated_condition(full_jacobian)
    )


@pytest.mark.parametrize("factory", [make_simple, make_sico])
@pytest.mark.parametrize("pressure", [1013.25, 101325.0, 10132500.0])
def test_cold_trace_root_is_reproducible_in_both_directions(factory, pressure):
    temperatures = np.array([1000.0, 12000.0, 25000.0])
    paths = []
    for descending in (False, True):
        requested = temperatures[::-1] if descending else temperatures
        system = ReducedEquilibriumSystem(
            factory(T=12000.0, P=pressure),
            coupled_ionization_lowering=True,
        )
        path = system.solve_temperature_path(
            requested,
            bootstrap_temperature=12000.0,
            max_temperature_step=1000.0,
            tolerance=1e-9,
        )
        states = {
            float(temperature): state
            for temperature, state in zip(requested, path.states)
        }
        cold = states[1000.0]
        assert cold.residual_norm < 1e-8
        assert cold.residual_evaluations > 0
        assert np.all(np.isfinite(cold.number_densities))
        assert np.all(cold.number_densities > 0)
        paths.append((system, cold))

    first_system, first = paths[0]
    second_system, second = paths[1]
    assert first.number_densities == pytest.approx(
        second.number_densities, rel=2e-8, abs=1e-12
    )
    assert first.ionization_lowering == pytest.approx(
        second.ionization_lowering, rel=2e-8, abs=1e-30
    )
    assert first_system.active_level_fingerprint(first) == (
        second_system.active_level_fingerprint(second)
    )
