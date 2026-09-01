"""Tests for consolidated equilibrium-state caching."""

import pytest

import minplascalc as mpc


@pytest.fixture
def oxygen_mixture():
    return mpc.mixture.lte_from_names(
        ["O2", "O", "O+"],
        x0=[1.0, 0.0, 0.0],
        T=10_000.0,
        P=101_325.0,
    )


@pytest.mark.parametrize(
    "attribute, value",
    [
        ("T", 10_100.0),
        ("P", 202_650.0),
        ("x0", [0.8, 0.2, 0.0]),
        ("gfe_initial_particles", 2e20),
        ("gfe_rtol", 2e-10),
        ("gfe_max_iter", 999),
    ],
)
def test_mutable_inputs_invalidate_derived_state(
    oxygen_mixture, attribute, value
):
    original_state = oxygen_mixture._equilibrium_state()
    original_workspace = oxygen_mixture._transport_workspace()

    setattr(oxygen_mixture, attribute, value)

    assert oxygen_mixture._equilibrium_state() is not original_state
    assert oxygen_mixture._transport_workspace() is not original_workspace


def test_temporary_temperature_restores_cached_state(oxygen_mixture):
    original_state = oxygen_mixture._equilibrium_state()
    original_workspace = oxygen_mixture._transport_workspace()

    with oxygen_mixture._at_temperature(10_100.0):
        assert oxygen_mixture.T == 10_100.0
        assert oxygen_mixture._equilibrium_state() is not original_state
        assert oxygen_mixture._transport_workspace() is not original_workspace

    assert oxygen_mixture.T == 10_000.0
    assert oxygen_mixture._equilibrium_state() is original_state
    assert oxygen_mixture._transport_workspace() is original_workspace
