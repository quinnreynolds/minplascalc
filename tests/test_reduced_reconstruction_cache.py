"""Value-keyed reconstruction-cache checks for the reduced prototype."""

from __future__ import annotations

import numpy as np
import pytest

from bench.reduced_equilibrium import ReducedEquilibriumSystem
from bench.workloads import make_sico, make_simple


@pytest.mark.parametrize("factory", [make_simple, make_sico])
def test_identical_state_reuses_reconstruction_for_residual_and_jacobian(
    factory,
):
    system = ReducedEquilibriumSystem(
        factory(T=12000.0), coupled_ionization_lowering=True
    )
    potentials = system.initial_state()

    residual_only, _ = system.evaluate(potentials, jacobian=False)
    residual, jacobian = system.evaluate(potentials.copy(), jacobian=True)
    repeated, repeated_jacobian = system.evaluate(
        potentials.copy(), jacobian=True
    )

    assert system.reconstruction_evaluations == 1
    assert system.reconstruction_cache_hits == 2
    assert residual == pytest.approx(residual_only, rel=0, abs=0)
    assert repeated == pytest.approx(residual, rel=0, abs=0)
    assert repeated_jacobian == pytest.approx(jacobian, rel=0, abs=0)


@pytest.mark.parametrize("factory", [make_simple, make_sico])
def test_cache_invalidates_on_temperature_but_not_pressure(factory):
    mixture = factory(T=12000.0, P=101325.0)
    system = ReducedEquilibriumSystem(
        mixture, coupled_ionization_lowering=True
    )
    potentials = system.initial_state()
    original, original_jacobian = system.evaluate(potentials, jacobian=True)

    mixture.P *= 2
    pressure_changed, pressure_jacobian = system.evaluate(
        potentials, jacobian=True
    )
    assert system.reconstruction_evaluations == 1
    assert system.reconstruction_cache_hits == 1
    assert pressure_changed[0] - original[0] == pytest.approx(-np.log(2))
    assert pressure_jacobian == pytest.approx(original_jacobian, rel=0, abs=0)

    mixture.T = 13000.0
    hotter, hotter_jacobian = system.evaluate(potentials, jacobian=True)
    assert system.reconstruction_evaluations == 2
    assert not np.allclose(hotter, pressure_changed, rtol=0, atol=1e-8)
    assert not np.allclose(
        hotter_jacobian, pressure_jacobian, rtol=0, atol=1e-8
    )

    mixture.T = 12000.0
    restored, restored_jacobian = system.evaluate(potentials, jacobian=True)
    assert system.reconstruction_evaluations == 3
    assert restored == pytest.approx(pressure_changed, rel=0, abs=1e-13)
    assert restored_jacobian == pytest.approx(
        pressure_jacobian, rel=0, abs=1e-13
    )


@pytest.mark.parametrize("factory", [make_simple, make_sico])
def test_least_squares_reports_reconstruction_cache_savings(factory):
    system = ReducedEquilibriumSystem(
        factory(T=12000.0), coupled_ionization_lowering=True
    )
    result = system.solve(method="least_squares", tolerance=1e-9)

    assert result.reconstruction_evaluations < result.residual_evaluations
    assert result.reconstruction_cache_hits > 0
    assert (
        result.reconstruction_evaluations + result.reconstruction_cache_hits
        == result.residual_evaluations
    )
    before_reconstructions = system.reconstruction_evaluations
    before_hits = system.reconstruction_cache_hits
    system.temperature_tangent(result)
    assert system.reconstruction_evaluations == before_reconstructions
    assert system.reconstruction_cache_hits == before_hits + 1


def test_path_aggregates_reconstruction_diagnostics():
    system = ReducedEquilibriumSystem(
        make_sico(T=12000.0), coupled_ionization_lowering=True
    )
    path = system.solve_temperature_path(
        np.array([10000.0, 12000.0, 14000.0]), tolerance=1e-9
    )

    assert path.total_reconstruction_evaluations >= sum(
        state.reconstruction_evaluations for state in path.states
    )
    assert path.total_reconstruction_cache_hits >= sum(
        state.reconstruction_cache_hits for state in path.states
    )
    assert (
        path.total_reconstruction_evaluations
        + path.total_reconstruction_cache_hits
        == path.total_residual_evaluations
    )
    assert path.total_reconstruction_evaluations < (
        path.total_residual_evaluations
    )
