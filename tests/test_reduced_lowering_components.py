"""Component checks for the coupled Stewart--Pyatt reduced closures."""

from __future__ import annotations

import numpy as np
import pytest

from bench.log_equilibrium import LogEquilibriumSystem
from bench.reduced_equilibrium import ReducedEquilibriumSystem
from bench.workloads import make_sico, make_simple
from minplascalc import mixture as mixture_module
from minplascalc import species as species_module
from minplascalc import units as u


def _system(factory=make_sico):
    mixture = factory(T=12000.0)
    return mixture, ReducedEquilibriumSystem(
        mixture, coupled_ionization_lowering=True
    )


def _matched_particle_state(mixture, system):
    """Make positive particle numbers with known electron and z-star values."""
    numbers = np.full(system.species_count, 1.0e-6)
    numbers[system.electron_index] = 0.1
    positive = system.positive_indices
    charge_one = positive[system.charges[positive] == 1]
    charge_two = positive[system.charges[positive] == 2]
    numbers[charge_one] = 0.2 / charge_one.size
    numbers[charge_two] = 0.3 / charge_two.size
    numbers /= numbers.sum()
    volume = numbers.sum() * u.k_b * mixture.T / mixture.P
    electron_density = numbers[system.electron_index] / volume
    z_star = (numbers[positive] @ system.charges[positive] ** 2) / (
        numbers[positive] @ system.charges[positive]
    )
    return numbers, np.log(electron_density), np.log(z_star)


@pytest.mark.parametrize("factory", [make_simple, make_sico])
def test_coupled_lowering_matches_full_log_at_matched_electron_moments(
    factory,
):
    mixture, reduced = _system(factory)
    numbers, eta, xi = _matched_particle_state(mixture, reduced)
    full = LogEquilibriumSystem(mixture)

    expected, _ = full._packed_lowering(numbers, derivatives=False)
    actual, _, _ = reduced._coupled_lowering(eta, xi)

    assert actual == pytest.approx(expected, rel=2e-14, abs=1e-30)


def test_coupled_lowering_auxiliary_derivatives_match_central_difference():
    _, system = _system()
    eta = np.log(8.0e22)
    xi = np.log(1.4)
    lowering, d_eta, d_xi = system._coupled_lowering(eta, xi)
    step = 1.0e-5

    eta_high = system._coupled_lowering(eta + step, xi)[0]
    eta_low = system._coupled_lowering(eta - step, xi)[0]
    xi_high = system._coupled_lowering(eta, xi + step)[0]
    xi_low = system._coupled_lowering(eta, xi - step)[0]

    assert np.all(np.isfinite(lowering))
    assert d_eta == pytest.approx(
        (eta_high - eta_low) / (2 * step), rel=3e-9, abs=1e-30
    )
    assert d_xi == pytest.approx(
        (xi_high - xi_low) / (2 * step), rel=3e-9, abs=1e-30
    )


def test_coupled_lowering_temperature_derivative_matches_central_difference():
    mixture, system = _system()
    eta = np.log(8.0e22)
    xi = np.log(1.4)
    analytical = system._coupled_temperature_lowering_derivative(eta, xi)
    temperature = mixture.T
    step = 1.0e-4 * temperature

    mixture.T = temperature + step
    high = system._coupled_lowering(eta, xi)[0]
    mixture.T = temperature - step
    low = system._coupled_lowering(eta, xi)[0]
    mixture.T = temperature

    assert analytical == pytest.approx(
        (high - low) / (2 * step), rel=3e-8, abs=1e-30
    )


def test_reference_chain_auxiliary_derivatives_match_central_difference():
    _, system = _system()
    eta = np.log(8.0e22)
    xi = np.log(1.4)
    lowering, d_eta, d_xi = system._coupled_lowering(eta, xi)
    reference, derivative = system._reference_from_lowering(
        lowering, (d_eta, d_xi)
    )
    assert derivative is not None
    step = 1.0e-5
    reference_eta_high = system._reference_from_lowering(
        system._coupled_lowering(eta + step, xi)[0]
    )[0]
    reference_eta_low = system._reference_from_lowering(
        system._coupled_lowering(eta - step, xi)[0]
    )[0]
    reference_xi_high = system._reference_from_lowering(
        system._coupled_lowering(eta, xi + step)[0]
    )[0]
    reference_xi_low = system._reference_from_lowering(
        system._coupled_lowering(eta, xi - step)[0]
    )[0]
    numerical = np.column_stack(
        (
            (reference_eta_high - reference_eta_low) / (2 * step),
            (reference_xi_high - reference_xi_low) / (2 * step),
        )
    )

    assert np.all(np.isfinite(reference))
    assert derivative == pytest.approx(numerical, rel=3e-9, abs=1e-30)


def test_coupled_residual_appends_electron_and_zstar_closures():
    _, system = _system()
    state = system.initial_state()
    residual, _, logs, _ = system._evaluate_state(state, jacobian=False)
    positive = system.positive_indices
    shifted = logs[positive] - np.max(logs[positive])
    weights = np.exp(shifted)
    expected_electron = logs[system.electron_index] - state[-2]
    expected_z_star = (
        np.log(np.sum(weights * system.charges[positive] ** 2))
        - np.log(np.sum(weights * system.charges[positive]))
        - state[-1]
    )
    closure_start = system.element_count + int(system.has_charge_constraint)

    assert residual.size == system.potential_count
    assert residual[-2:] == pytest.approx(
        [expected_electron, expected_z_star], rel=2e-12, abs=2e-12
    )
    assert closure_start == system.potential_count - 2


def test_coupled_partition_uses_strict_active_level_cutoff():
    _, system = _system()
    index, species = next(
        (i, item)
        for i, item in enumerate(system.species)
        if isinstance(item, species_module.Monatomic)
        and len(item._level_energies) > 1
    )
    level = species._level_energies[-1]
    lowering = np.zeros(system.species_count)
    lowering[index] = species.ionisation_energy - level
    packed = system._log_partition_per_volume(lowering)
    active = species._level_energies < (
        species.ionisation_energy - lowering[index]
    )
    terms = np.log(species._degeneracies[active]) - (
        species._level_energies[active] / (u.k_b * system.mixture.T)
    )
    expected = np.log(
        species.translational_partition_function(system.mixture.T)
    )
    expected += max(terms) + np.log(np.exp(terms - max(terms)).sum())

    assert not active[-1]
    assert packed[index] == pytest.approx(expected, rel=2e-14)


def test_coupled_mode_rejects_invalid_modes_and_domains():
    mixture = make_sico(T=12000.0)
    with pytest.raises(ValueError, match="combined|fixed"):
        ReducedEquilibriumSystem(
            mixture,
            fixed_ionization_lowering=np.zeros(len(mixture.species)),
            coupled_ionization_lowering=True,
        )

    electron_free = mixture_module.lte_from_names(
        ["O2", "O"],
        [1.0, 0.0],
        12000.0,
        101325.0,
        electrons_yn=False,
    )
    with pytest.raises(ValueError, match="electron"):
        ReducedEquilibriumSystem(
            electron_free, coupled_ionization_lowering=True
        )

    no_positive_ions = mixture_module.lte_from_names(
        ["O2"],
        [1.0],
        12000.0,
        101325.0,
        electrons_yn=True,
    )
    with pytest.raises(ValueError, match="positive ions"):
        ReducedEquilibriumSystem(
            no_positive_ions, coupled_ionization_lowering=True
        )

    _, system = _system()
    with pytest.raises(ValueError, match="finite"):
        system.evaluate(np.full(system.potential_count, np.nan))
