"""Tests for the isolated log-space equilibrium formulation."""

import numpy as np
import pytest

from bench.log_equilibrium import LogEquilibriumSystem
from bench.workloads import make_sico, make_simple


def _mole_fractions(number_densities):
    return number_densities / number_densities.sum()


@pytest.mark.parametrize("factory", [make_simple, make_sico])
@pytest.mark.parametrize("temperature", [2000.0, 12000.0, 25000.0])
def test_packed_thermodynamics_matches_species_evaluation(
    factory, temperature
):
    reference = factory(T=temperature)
    reference.calculate_composition()
    state = reference._equilibrium_state()

    system = LogEquilibriumSystem(factory(T=temperature))
    packed = system._packed_thermodynamics(
        np.log(state.particle_numbers), derivatives=True
    )
    system._set_particle_numbers(state.particle_numbers)
    expected_reference, expected_lowering = (
        system.mixture._LTE__get_reference_energies()
    )
    expected_reference_dN, _ = (
        system.mixture._LTE__get_reference_energy_derivatives()
    )
    expected_log_partitions = np.log(
        [
            species.total_partition_function(
                state.volume, temperature, lowering
            )
            for species, lowering in zip(
                system.mixture.species, expected_lowering
            )
        ]
    )

    assert packed.ionization_lowering == pytest.approx(
        expected_lowering, rel=3e-15
    )
    assert packed.reference_energies == pytest.approx(
        expected_reference, rel=3e-15
    )
    assert packed.reference_dN == pytest.approx(
        expected_reference_dN, rel=3e-15
    )
    assert packed.log_partitions == pytest.approx(
        expected_log_partitions, rel=3e-15
    )


@pytest.mark.parametrize("factory", [make_simple, make_sico])
def test_zero_frozen_lowering_converges_to_finite_state(factory):
    mixture = factory(T=12000.0)
    fixed_lowering = np.zeros(len(mixture.species))
    system = LogEquilibriumSystem(
        mixture, fixed_ionization_lowering=fixed_lowering
    )

    result = system.solve(tolerance=1e-10)

    assert result.residual_norm < 1e-10
    assert np.all(np.isfinite(result.number_densities))
    assert np.all(result.number_densities > 0)


def test_frozen_lowering_and_reference_derivatives_are_constant():
    mixture = make_sico(T=12000.0)
    fixed_lowering = np.linspace(0.0, 1.0e-22, len(mixture.species))
    system = LogEquilibriumSystem(
        mixture, fixed_ionization_lowering=fixed_lowering
    )
    log_particles, _ = system.initial_state()
    particle_numbers = np.exp(log_particles)

    lowering, lowering_dN = system._packed_lowering(
        particle_numbers, derivatives=True
    )
    packed = system._packed_thermodynamics(log_particles, derivatives=True)

    assert lowering == pytest.approx(fixed_lowering)
    assert lowering_dN is not None
    assert lowering_dN == pytest.approx(
        np.zeros((system.species_count, system.species_count))
    )
    assert packed.reference_dN is not None
    assert packed.reference_dN == pytest.approx(
        np.zeros((system.species_count, system.species_count))
    )


def test_frozen_lowering_is_validated():
    mixture = make_simple(T=12000.0)

    invalid_lowerings = (
        np.zeros(len(mixture.species) - 1),
        np.full(len(mixture.species), np.nan),
        np.full(len(mixture.species), -1.0),
    )
    for fixed_lowering in invalid_lowerings:
        with pytest.raises(ValueError):
            LogEquilibriumSystem(
                mixture, fixed_ionization_lowering=fixed_lowering
            )


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


def test_log_equilibrium_temperature_tangent_matches_production():
    production = make_sico(T=12000.0)
    expected = production.calculate_composition_temperature_derivative()
    expected_cp = production.calculate_heat_capacity()

    system = LogEquilibriumSystem(make_sico(T=12000.0))
    result = system.solve(tolerance=1e-11)
    tangent = system.temperature_tangent(result)

    assert tangent.mole_fraction_derivative == pytest.approx(
        expected, rel=2e-8, abs=2e-12
    )
    assert system.heat_capacity(result) == pytest.approx(expected_cp, rel=2e-8)


def test_log_heat_capacity_survives_production_log_zero_case():
    system = LogEquilibriumSystem(make_sico(T=1000.0, P=10132500.0))
    result = system.solve_temperature_path(np.array([1000.0])).states[0]

    heat_capacity = system.heat_capacity(result)
    assert np.isfinite(heat_capacity)
    assert heat_capacity > 0


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


@pytest.mark.parametrize("factory", [make_simple, make_sico])
@pytest.mark.parametrize("pressure", [1013.25, 10132500.0])
@pytest.mark.parametrize("descending", [False, True])
def test_log_equilibrium_pressure_range(factory, pressure, descending):
    temperatures = np.linspace(1000.0, 25000.0, 13)
    if descending:
        temperatures = temperatures[::-1]
    system = LogEquilibriumSystem(
        factory(T=float(temperatures[0]), P=pressure)
    )
    path = system.solve_temperature_path(temperatures)

    assert all(state.residual_norm < 1e-9 for state in path.states)
    assert all(
        np.all(np.isfinite(state.number_densities)) for state in path.states
    )
    assert all(np.all(state.number_densities > 0) for state in path.states)
    assert path.total_iterations / path.continuation_solves < 7


def test_log_equilibrium_selects_lower_gibbs_cutoff_branch():
    temperature = 20862.0
    system = LogEquilibriumSystem(make_sico(T=temperature))
    continued = system.solve_temperature_path(np.array([temperature])).states[
        0
    ]
    branches = system.solve_lowest_gibbs_branch(
        (continued.log_particles, continued.scaled_multipliers)
    )

    assert branches.nearest_cutoff_distance < 2e-5
    assert len(branches.candidates) == 2
    assert branches.dimensionless_gibbs[0] != pytest.approx(
        branches.dimensionless_gibbs[1], rel=1e-9
    )
    assert system.dimensionless_gibbs(branches.selected) == min(
        branches.dimensionless_gibbs
    )

    active_counts = [
        system._packed_thermodynamics(
            candidate.log_particles, derivatives=False
        ).active_level_counts
        for candidate in branches.candidates
    ]
    differing_species = np.flatnonzero(active_counts[0] != active_counts[1])
    assert [
        system.mixture.species[index].name for index in differing_species
    ] == ["Si+"]
    assert (
        abs(
            active_counts[0][differing_species[0]]
            - active_counts[1][differing_species[0]]
        )
        == 1
    )

    fingerprints = [
        system.active_level_fingerprint(candidate)
        for candidate in branches.candidates
    ]
    assert fingerprints[0].fingerprint != fingerprints[1].fingerprint
    assert {
        fingerprint.nearest_cutoff_species_name for fingerprint in fingerprints
    } == {"Si+"}
    per_species = [
        {state.species_name: state for state in fingerprint.species}
        for fingerprint in fingerprints
    ]
    assert (
        per_species[0]["Si+"].fingerprint != per_species[1]["Si+"].fingerprint
    )
