"""Tests for the isolated log-space equilibrium formulation."""

import numpy as np
import pytest

from bench.log_equilibrium import LogEquilibriumSystem
from bench.workloads import make_sico, make_simple


def _mole_fractions(number_densities):
    return number_densities / number_densities.sum()


def test_log_equilibrium_analytical_jacobian():
    system = LogEquilibriumSystem(make_sico(T=12000.0))
    result = system.solve(tolerance=1e-11)
    values = np.concatenate((result.log_particles, result.scaled_multipliers))
    _, analytical = system.evaluate(
        values[: system.species_count],
        values[system.species_count :],
    )
    assert analytical is not None

    step = 1e-4
    numerical = np.empty_like(analytical)
    for column in range(values.size):
        high = values.copy()
        low = values.copy()
        high[column] += step
        low[column] -= step
        residual_high, _ = system.evaluate(
            high[: system.species_count],
            high[system.species_count :],
            jacobian=False,
        )
        residual_low, _ = system.evaluate(
            low[: system.species_count],
            low[system.species_count :],
            jacobian=False,
        )
        numerical[:, column] = (residual_high - residual_low) / (2 * step)

    assert analytical == pytest.approx(numerical, rel=2e-8, abs=2e-9)


@pytest.mark.parametrize("factory", [make_simple, make_sico])
@pytest.mark.parametrize("temperature", [8000.0, 12000.0, 25000.0])
def test_log_equilibrium_cold_midrange_matches_production(
    factory, temperature
):
    reference = factory(T=temperature)
    expected = _mole_fractions(reference.calculate_composition())

    system = LogEquilibriumSystem(factory(T=temperature))
    result = system.solve(tolerance=1e-9)
    actual = _mole_fractions(result.number_densities)

    assert result.iterations <= 10
    assert result.residual_norm < 1e-9
    assert actual == pytest.approx(expected, rel=2e-7, abs=2e-10)


@pytest.mark.parametrize("factory", [make_simple, make_sico])
@pytest.mark.parametrize("descending", [False, True])
def test_log_equilibrium_temperature_continuation(factory, descending):
    temperatures = np.linspace(1000.0, 25000.0, 15)
    if descending:
        temperatures = temperatures[::-1]
    system = LogEquilibriumSystem(factory(T=float(temperatures[0])))
    path = system.solve_temperature_path(temperatures)

    assert len(path.states) == len(temperatures)
    assert all(state.residual_norm < 1e-9 for state in path.states)
    assert path.total_iterations / path.continuation_solves < 6

    for index in (0, -1):
        reference = factory(T=float(temperatures[index]))
        expected = _mole_fractions(reference.calculate_composition())
        actual = _mole_fractions(path.states[index].number_densities)
        assert actual == pytest.approx(expected, rel=2e-7, abs=3e-10)
