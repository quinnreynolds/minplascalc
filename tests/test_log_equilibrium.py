"""Tests for the isolated log-space equilibrium formulation."""

import numpy as np
import pytest

from bench.workloads import make_sico, make_simple
from minplascalc import mixture as mixture_module
from minplascalc import units as u
from minplascalc.log_equilibrium import (
    CutoffConvergenceError,
    LogEquilibriumSystem,
)


def _mole_fractions(number_densities):
    return number_densities / number_densities.sum()


@pytest.mark.parametrize("factory", [make_simple, make_sico])
@pytest.mark.parametrize("temperature", [2000.0, 12000.0, 25000.0])
def test_packed_thermodynamics_matches_species_evaluation(
    factory, temperature
):
    reference = factory(T=temperature)
    reference._calculate_composition_particle_numbers()
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
    production._calculate_composition_particle_numbers()
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


def test_public_composition_uses_positive_log_state():
    mixture = make_sico(T=1000.0, P=10132500.0)

    composition = mixture.calculate_composition()

    assert np.all(np.isfinite(composition))
    assert np.all(composition > 0)
    assert mixture._LTE__log_equilibrium_result.residual_norm < 1e-9


def test_public_log_composition_matches_particle_number_oracle():
    production = make_sico(T=12000.0)
    actual = _mole_fractions(production.calculate_composition())
    oracle = make_sico(T=12000.0)
    expected = _mole_fractions(
        oracle._calculate_composition_particle_numbers()
    )

    assert actual == pytest.approx(expected, rel=2e-7, abs=2e-10)


def test_zero_element_total_uses_particle_number_fallback():
    mixture = make_sico(T=12000.0)
    mixture.x0 = [1.0, *np.zeros(len(mixture.species) - 2)]

    composition = mixture.calculate_composition()

    assert np.all(np.isfinite(composition))
    assert np.all(composition >= 0)
    assert mixture._LTE__log_equilibrium_result is None


@pytest.mark.parametrize("factory", [make_simple, make_sico])
@pytest.mark.parametrize("temperature", [8000.0, 12000.0, 25000.0])
def test_log_equilibrium_cold_midrange_matches_production(
    factory, temperature
):
    reference = factory(T=temperature)
    expected = _mole_fractions(
        reference._calculate_composition_particle_numbers()
    )

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
        expected = _mole_fractions(
            reference._calculate_composition_particle_numbers()
        )
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


def _assert_original_equilibrium(system, result, tolerance=1e-10):
    """Check recovered roots using unfrozen species partition functions."""
    particles = np.exp(result.log_particles)
    volume = particles.sum() * u.k_b * system.mixture.T / system.mixture.P
    system._set_particle_numbers(particles)
    reference, lowering = system.mixture._LTE__get_reference_energies()
    partitions = np.array(
        [
            sp.total_partition_function(volume, system.mixture.T, de)
            for sp, de in zip(system.mixture.species, lowering)
        ]
    )
    chemical = (
        reference / (u.k_b * system.mixture.T)
        - np.log(partitions)
        + result.log_particles
        + system.constraints @ result.scaled_multipliers
    )
    assert np.linalg.norm(chemical, np.inf) < tolerance
    conserved = system.constraints.T @ particles
    assert conserved[: system.element_count] == pytest.approx(
        system.targets, rel=tolerance
    )
    if system.has_charge_constraint:
        assert abs(conserved[-1] / particles.sum()) < tolerance
    assert np.all(result.number_densities > 0)
    assert result.number_densities.sum() * u.k_b * system.mixture.T == (
        pytest.approx(system.mixture.P, rel=tolerance)
    )


def test_recover_cutoff_stall_without_changing_partition_model():
    # Independent public-data case; no production inputs are required.
    mixture = make_sico(sio=0.9, T=24600.0)
    mixture.calculate_composition()
    initial = mixture._LTE__log_equilibrium_initial
    mixture.T = 24650.0
    system = LogEquilibriumSystem(mixture)
    with pytest.raises(RuntimeError, match="line search stalled"):
        system.solve(initial, max_cutoff_branches=0)
    with pytest.raises(CutoffConvergenceError) as limited:
        system.solve(initial, max_cutoff_branches=1)
    assert limited.value.attempted_branches == 1

    result = system.solve(initial)

    assert 1 < result.cutoff_branches <= 8
    assert result.residual_evaluations == system.residual_evaluations
    assert result.residual_norm < 1e-10
    _assert_original_equilibrium(system, result)


@pytest.mark.parametrize("warm", [False, True])
def test_public_composition_recovers_cutoff_stall(warm):
    mixture = make_sico(sio=0.9, T=24600.0 if warm else 24650.0)
    if warm:
        mixture.calculate_composition()
        mixture.T = 24650.0

    composition = mixture.calculate_composition()

    assert mixture.T == 24650.0
    _assert_original_equilibrium(
        mixture._LTE__log_equilibrium_system,
        mixture._LTE__log_equilibrium_result,
    )
    assert mixture.calculate_composition() == pytest.approx(composition)
    assert np.isfinite(mixture.calculate_heat_capacity())


def test_public_cold_continuation_reaches_requested_temperature():
    # The penultimate continuation step is the recoverable 24650 K cutoff.
    target = 12000.0 + (24650.0 - 12000.0) * 14 / 13
    mixture = make_sico(sio=0.9, T=target)

    mixture.calculate_composition()

    assert mixture.T == target
    _assert_original_equilibrium(
        mixture._LTE__log_equilibrium_system,
        mixture._LTE__log_equilibrium_result,
    )


@pytest.mark.parametrize("warm", [False, True])
def test_local_cutoff_gap_never_returns_inconsistent_root(warm):
    mixture = make_sico(T=11850.0 if warm else 11900.0, P=10132500.0)
    if warm:
        mixture.calculate_composition()
        mixture.T = 11900.0

    with pytest.raises(CutoffConvergenceError) as caught:
        mixture.calculate_composition()

    error = caught.value
    assert error.species_name == "Si+"
    assert error.temperature == 11900.0
    assert error.pressure == 10132500.0
    assert error.residual_norm > mixture.gfe_rtol
    assert error.attempted_branches == 2
    assert mixture.T == 11900.0
    assert mixture._LTE__log_equilibrium_result is None
    assert not mixture._LTE__isLTE
    # Failure must not poison the next point in a caller-managed sweep.
    mixture.T = 12000.0
    mixture.calculate_composition()
    _assert_original_equilibrium(
        mixture._LTE__log_equilibrium_system,
        mixture._LTE__log_equilibrium_result,
    )


def test_temperature_path_restores_input_on_requested_cutoff_gap():
    mixture = make_sico(T=10000.0, P=10132500.0)
    system = LogEquilibriumSystem(mixture)

    with pytest.raises(CutoffConvergenceError):
        system.solve_temperature_path(np.array([11900.0, 10000.0]))

    assert mixture.T == 10000.0


@pytest.mark.parametrize("bootstrap", [11900.0, 12000.0])
def test_temperature_path_bypasses_unrequested_cutoff_gap(bootstrap):
    mixture = make_sico(T=11800.0, P=10132500.0)
    system = LogEquilibriumSystem(mixture)

    path = system.solve_temperature_path(
        np.array([11800.0]),
        bootstrap_temperature=bootstrap,
        maximum_temperature_step=100.0,
        tolerance=1e-10,
    )

    assert mixture.T == 11800.0
    assert len(path.states) == 1
    _assert_original_equilibrium(system, path.states[0])


@pytest.mark.parametrize("failed_temperature", [12000.0, 11000.0])
def test_public_composition_restores_temperature_on_other_failure(
    monkeypatch, failed_temperature
):
    mixture = make_sico(T=2000.0)
    original = LogEquilibriumSystem.solve

    def fail_at_temperature(system, *args, **kwargs):
        if system.mixture.T == failed_temperature:
            raise RuntimeError("injected failure")
        return original(system, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(LogEquilibriumSystem, "solve", fail_at_temperature)
        with pytest.raises(RuntimeError, match="injected failure"):
            mixture.calculate_composition()

    assert mixture.T == 2000.0
    assert not mixture._LTE__isLTE
    mixture.calculate_composition()
    _assert_original_equilibrium(
        mixture._LTE__log_equilibrium_system,
        mixture._LTE__log_equilibrium_result,
    )


def test_branch_selection_without_electronic_levels():
    mixture = mixture_module.lte_from_names(
        ["O2", "CO"], [0.5, 0.5], 3000.0, 101325.0, electrons_yn=False
    )

    mixture.calculate_composition()

    _assert_original_equilibrium(
        mixture._LTE__log_equilibrium_system,
        mixture._LTE__log_equilibrium_result,
    )
