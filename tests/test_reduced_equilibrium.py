"""Stage-2 checks for the fixed-ionisation-lowering reduced prototype.

The reduced formulation is deliberately tested against the full log prototype
on the *same frozen thermodynamics*.  This keeps these tests about the
potential-only reduction (and its closures), rather than conflating Stage 2
with the composition-dependent Stewart--Pyatt fixed point or a cutoff probe.
"""

from __future__ import annotations

import numpy as np
import pytest

from bench.log_equilibrium import LogEquilibriumSystem
from bench.reduced_equilibrium import ReducedEquilibriumSystem
from bench.workloads import make_sico, make_simple
from minplascalc import mixture as mixture_module
from minplascalc import units as u


def _frozen(mixture):
    """Use the same zero lowering array in both prototype formulations."""
    return np.zeros(len(mixture.species), dtype=float)


def _densities(result):
    values = getattr(result, "number_densities", None)
    if values is None:
        values = getattr(result, "densities", None)
    if values is None:
        raise AssertionError(
            "ReducedEquilibriumResult must expose number_densities."
        )
    values = np.asarray(values, dtype=float)
    assert values.ndim == 1
    assert np.all(np.isfinite(values))
    assert np.all(values > 0)
    return values


def _state(result):
    """Extract the reduced unknown vector from a result.

    ``state`` is the preferred spelling.  The two descriptive aliases make
    this validation suite useful while the research-only result dataclass is
    being stabilised, without accepting a particle-number state by accident.
    """
    for name in ("state", "potentials", "log_potentials", "variables"):
        value = getattr(result, name, None)
        if value is not None:
            return np.asarray(value, dtype=float).reshape(-1)
    raise AssertionError(
        "ReducedEquilibriumResult must expose its potential state as state."
    )


def _initial_state(system):
    value = system.initial_state()
    # A reduced state is a single vector.  Accept a one-item tuple while the
    # implementation is in motion, but reject full-log-style two-part states.
    if isinstance(value, tuple):
        if len(value) != 1:
            raise AssertionError(
                "ReducedEquilibriumSystem.initial_state must return one "
                "potential vector."
            )
        value = value[0]
    return np.asarray(value, dtype=float).reshape(-1)


def _reconstruct(system, state):
    for name in ("reconstruct", "reconstruct_number_densities", "densities"):
        method = getattr(system, name, None)
        if method is not None:
            values = method(state)
            if isinstance(values, tuple):
                values = values[0]
            return np.asarray(values, dtype=float).reshape(-1)
    # The initial research implementation keeps this operation private and
    # returns (log densities, mole fractions, slopes).  Use the logs here so
    # the coupling assertion remains independent of the arbitrary pressure
    # scale.
    method = getattr(system, "_reconstruct", None)
    if method is not None:
        values = method(state)
        if isinstance(values, tuple) and len(values) >= 1:
            return np.exp(np.asarray(values[0], dtype=float).reshape(-1))
    raise AssertionError(
        "ReducedEquilibriumSystem must expose explicit species reconstruction."
    )


def _solve_pair(factory, temperature, *, electrons=True):
    reduced_mixture = factory(T=temperature)
    if not electrons:
        if factory is make_sico:
            reduced_mixture = mixture_module.lte_from_names(
                ["CO", "C", "O"],
                [1.0, 0.0, 0.0],
                temperature,
                101325.0,
                electrons_yn=False,
            )
        else:
            reduced_mixture = mixture_module.lte_from_names(
                ["O2", "O"],
                [1.0, 0.0],
                temperature,
                101325.0,
                electrons_yn=False,
            )
    frozen = _frozen(reduced_mixture)
    reduced = ReducedEquilibriumSystem(
        reduced_mixture, fixed_ionization_lowering=frozen
    )
    full = LogEquilibriumSystem(
        factory(T=temperature) if electrons else reduced_mixture,
        fixed_ionization_lowering=(
            _frozen(factory(T=temperature)) if electrons else frozen
        ),
    )
    reduced_result = reduced.solve(tolerance=1e-10)
    full_result = full.solve(tolerance=1e-10)
    return reduced, reduced_result, full, full_result


@pytest.mark.parametrize("factory", [make_simple, make_sico])
@pytest.mark.parametrize("temperature", [8000.0, 12000.0, 18000.0])
def test_reduced_root_matches_frozen_full_log_root(factory, temperature):
    reduced, result, full, full_result = _solve_pair(factory, temperature)

    actual = _densities(result)
    reference = _densities(full_result)
    assert result.residual_norm < 1e-9
    assert actual == pytest.approx(reference, rel=3e-7, abs=2e-10)
    assert actual.sum() * u.k_b * temperature / factory(T=temperature).P > 0
    assert reduced.species_count == full.species_count


def test_reduced_analytical_jacobian_matches_central_difference():
    mixture = make_sico(T=12000.0)
    frozen = _frozen(mixture)
    system = ReducedEquilibriumSystem(
        mixture, fixed_ionization_lowering=frozen
    )
    result = system.solve(tolerance=1e-11)
    state = _state(result)
    residual, analytical = system.evaluate(state, jacobian=True)
    assert analytical is not None
    assert residual.size == state.size
    assert analytical.shape == (state.size, state.size)

    step = 1e-5
    numerical = np.empty_like(analytical)
    for column in range(state.size):
        high = state.copy()
        low = state.copy()
        high[column] += step
        low[column] -= step
        residual_high, _ = system.evaluate(high, jacobian=False)
        residual_low, _ = system.evaluate(low, jacobian=False)
        numerical[:, column] = (residual_high - residual_low) / (2 * step)

    assert analytical == pytest.approx(numerical, rel=3e-6, abs=3e-8)


def test_reduced_explicitly_couples_carbon_monoxide_to_both_elements():
    mixture = mixture_module.lte_from_names(
        ["CO", "C", "O"],
        [1.0, 0.0, 0.0],
        12000.0,
        101325.0,
        electrons_yn=False,
    )
    system = ReducedEquilibriumSystem(
        mixture, fixed_ionization_lowering=_frozen(mixture)
    )
    state = _initial_state(system)
    names = list(getattr(system, "element_names", ["C", "O"]))
    co_index = [sp.name for sp in mixture.species].index("CO")
    base = _reconstruct(system, state)
    assert base[co_index] > 0

    for element in ("C", "O"):
        assert element in names
        shifted = state.copy()
        shifted[names.index(element)] += 0.25
        shifted_density = _reconstruct(system, shifted)
        # The boxed reconstruction gives d log(n_CO)/d ell_a = -nu_CO,a.
        observed = np.log(shifted_density[co_index] / base[co_index])
        assert observed == pytest.approx(-0.25, rel=2e-8, abs=2e-10)


def test_reduced_closures_are_small_individually():
    mixture = make_sico(T=12000.0)
    system = ReducedEquilibriumSystem(
        mixture, fixed_ionization_lowering=_frozen(mixture)
    )
    result = system.solve(tolerance=1e-11)
    densities = _densities(result)
    n_total = densities.sum()
    assert abs(np.log(u.k_b * mixture.T * n_total / mixture.P)) < 2e-10

    elements = sorted(
        {element for sp in mixture.species for element in sp.stoichiometry}
    )
    concentrations = np.array(
        [
            sum(
                density * sp.stoichiometry.get(element, 0)
                for sp, density in zip(mixture.species, densities)
            )
            for element in elements
        ]
    )
    reference = concentrations[0]
    feed = np.array(
        [
            sum(
                fraction * sp.stoichiometry.get(element, 0)
                for sp, fraction in zip(mixture.species, mixture.x0)
            )
            for element in elements
        ]
    )
    element_residual = np.log(
        concentrations[1:] * feed[0] / (reference * feed[1:])
    )
    assert element_residual == pytest.approx(0.0, abs=2e-10)
    charge_residual = np.dot(mixture.charge_numbers, densities) / n_total
    assert charge_residual == pytest.approx(0.0, abs=2e-10)


def test_reduced_fingerprint_is_deterministic_and_matches_full_log():
    mixture = make_sico(T=12000.0)
    frozen = _frozen(mixture)
    reduced = ReducedEquilibriumSystem(
        mixture, fixed_ionization_lowering=frozen
    )
    result = reduced.solve(tolerance=1e-11)
    repeated = reduced.solve(tolerance=1e-11)
    first = reduced.active_level_fingerprint(result)
    second = reduced.active_level_fingerprint(repeated)
    assert first == second
    assert len(first.fingerprint) == 64

    full = LogEquilibriumSystem(mixture, fixed_ionization_lowering=frozen)
    full_result = full.solve(tolerance=1e-11)
    assert first == full.active_level_fingerprint(full_result)


def test_reduced_electron_free_case_has_no_charge_potential_or_lowering():
    mixture = mixture_module.lte_from_names(
        ["O2", "O"],
        [1.0, 0.0],
        12000.0,
        101325.0,
        electrons_yn=False,
    )
    system = ReducedEquilibriumSystem(
        mixture, fixed_ionization_lowering=_frozen(mixture)
    )
    result = system.solve(tolerance=1e-11)
    full = LogEquilibriumSystem(
        mixture, fixed_ionization_lowering=_frozen(mixture)
    ).solve(tolerance=1e-11)
    assert result.residual_norm < 1e-9
    assert _densities(result) == pytest.approx(_densities(full), rel=3e-7)
    assert system.element_count == 1
    assert not getattr(system, "has_charge_constraint", False)
    assert _state(result).size == 1


def test_reduced_solver_does_not_require_full_state_and_accepts_perturbation():
    mixture = make_simple(T=12000.0)
    system = ReducedEquilibriumSystem(
        mixture, fixed_ionization_lowering=_frozen(mixture)
    )
    independent = system.solve(tolerance=1e-10)
    initial = _initial_state(system)
    perturbed = initial + np.linspace(-0.2, 0.2, initial.size)
    restarted = system.solve(initial=perturbed, tolerance=1e-10)
    assert restarted.residual_norm < 1e-9
    assert _densities(restarted) == pytest.approx(
        _densities(independent), rel=3e-7, abs=2e-10
    )


def test_reduced_rejects_zero_element_feed():
    mixture = make_sico(sio=0.0, T=12000.0)
    with pytest.raises(ValueError, match="(?i)positive|zero|feed|element"):
        ReducedEquilibriumSystem(
            mixture, fixed_ionization_lowering=_frozen(mixture)
        )


def test_reduced_rejects_rank_deficient_element_matrix():
    mixture = mixture_module.lte_from_names(
        ["CO", "SiO"],
        [0.5, 0.5],
        12000.0,
        101325.0,
        electrons_yn=False,
    )
    with pytest.raises(ValueError, match="(?i)rank|dependent|independent"):
        ReducedEquilibriumSystem(
            mixture, fixed_ionization_lowering=_frozen(mixture)
        )


def test_fixed_lowering_shape_and_values_are_validated_by_log_reference():
    mixture = make_simple(T=12000.0)
    with pytest.raises(ValueError, match="one value per species"):
        ReducedEquilibriumSystem(
            mixture, fixed_ionization_lowering=np.zeros(2)
        )
    with pytest.raises(ValueError, match="nonnegative"):
        ReducedEquilibriumSystem(
            mixture,
            fixed_ionization_lowering=np.full(len(mixture.species), -1.0),
        )
    reduced = ReducedEquilibriumSystem(
        mixture, fixed_ionization_lowering=_frozen(mixture)
    )
    with pytest.raises(ValueError, match="potentials.*shape"):
        reduced.evaluate(np.zeros(1))
    with pytest.raises(ValueError, match="one value per species"):
        LogEquilibriumSystem(mixture, fixed_ionization_lowering=np.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        LogEquilibriumSystem(
            mixture,
            fixed_ionization_lowering=np.full(len(mixture.species), np.nan),
        )
    with pytest.raises(ValueError, match="nonnegative"):
        LogEquilibriumSystem(
            mixture,
            fixed_ionization_lowering=np.full(len(mixture.species), -1.0),
        )


def _coupled_lowering(result):
    values = getattr(result, "ionization_lowering", None)
    if values is None:
        values = getattr(result, "lowering", None)
    if values is None:
        raise AssertionError(
            "Coupled reduced results must expose ionization_lowering."
        )
    values = np.asarray(values, dtype=float)
    assert values.ndim == 1
    assert np.all(np.isfinite(values))
    assert np.all(values >= 0)
    return values


def _assert_lowering_matches(actual, expected, *, relative):
    """Compare the exact-zero neutral entries separately from ion lowering."""
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape
    zero = expected == 0
    assert actual[zero] == pytest.approx(0.0, abs=1e-30)
    assert actual[~zero] == pytest.approx(expected[~zero], rel=relative)


def _assert_reference_matches(actual, expected, *, relative):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape
    zero = expected == 0
    assert actual[zero] == pytest.approx(0.0, abs=1e-30)
    assert actual[~zero] == pytest.approx(expected[~zero], rel=relative)


def _coupled_reference_energies(system, result):
    values = getattr(result, "reference_energies", None)
    if values is None:
        method = getattr(system, "_reference_from_lowering", None)
        if method is None:
            raise AssertionError(
                "Coupled reduced results must expose reference_energies "
                "or a reference-energy reconstruction method."
            )
        values = method(_coupled_lowering(result), None)[0]
    values = np.asarray(values, dtype=float)
    assert values.ndim == 1
    assert np.all(np.isfinite(values))
    return values


def _coupled_system(factory, temperature, pressure=101325.0):
    mixture = factory(T=temperature, P=pressure)
    return ReducedEquilibriumSystem(mixture, coupled_ionization_lowering=True)


@pytest.mark.parametrize("factory", [make_simple, make_sico])
@pytest.mark.parametrize("temperature", [8000.0, 12000.0])
def test_coupled_reduced_root_matches_ordinary_full_log_root(
    factory, temperature
):
    reduced = _coupled_system(factory, temperature)
    full = LogEquilibriumSystem(reduced.mixture)
    reduced_result = reduced.solve(tolerance=1e-10)
    full_result = full.solve(tolerance=1e-10)

    assert reduced_result.residual_norm < 1e-9
    assert _densities(reduced_result) == pytest.approx(
        _densities(full_result), rel=2e-8, abs=3e-10
    )
    _assert_lowering_matches(
        _coupled_lowering(reduced_result),
        full._packed_thermodynamics(
            full_result.log_particles, derivatives=False
        ).ionization_lowering,
        relative=3e-7,
    )
    _assert_reference_matches(
        _coupled_reference_energies(reduced, reduced_result),
        full._packed_thermodynamics(
            full_result.log_particles, derivatives=False
        ).reference_energies,
        relative=3e-7,
    )
    reduced_fingerprint = reduced.active_level_fingerprint(reduced_result)
    full_fingerprint = full.active_level_fingerprint(full_result)
    assert reduced_fingerprint.fingerprint == full_fingerprint.fingerprint
    assert [
        (state.species_name, state.active_level_count)
        for state in reduced_fingerprint.species
    ] == [
        (state.species_name, state.active_level_count)
        for state in full_fingerprint.species
    ]
    assert reduced_fingerprint.nearest_cutoff_species_name == (
        full_fingerprint.nearest_cutoff_species_name
    )
    assert reduced_fingerprint.nearest_cutoff_level_index == (
        full_fingerprint.nearest_cutoff_level_index
    )


def test_coupled_reduced_analytical_jacobian_matches_central_difference():
    system = _coupled_system(make_sico, 12000.0)
    result = system.solve(tolerance=1e-10)
    state = _state(result)
    residual, analytical = system.evaluate(state, jacobian=True)
    assert analytical is not None
    assert analytical.shape == (state.size, state.size)
    assert residual.size == state.size

    step = 2e-5
    numerical = np.empty_like(analytical)
    for column in range(state.size):
        high = state.copy()
        low = state.copy()
        high[column] += step
        low[column] -= step
        residual_high, _ = system.evaluate(high, jacobian=False)
        residual_low, _ = system.evaluate(low, jacobian=False)
        numerical[:, column] = (residual_high - residual_low) / (2 * step)

    assert analytical == pytest.approx(numerical, rel=8e-5, abs=5e-7)


def _density_tangent(tangent):
    for name in ("number_density_derivative", "density_derivative"):
        value = getattr(tangent, name, None)
        if value is not None:
            return np.asarray(value, dtype=float)
    return None


def _log_density_tangent(tangent):
    for name in ("log_number_density_derivative", "log_density_derivative"):
        value = getattr(tangent, name, None)
        if value is not None:
            return np.asarray(value, dtype=float)
    return None


def test_coupled_reduced_temperature_tangent_matches_full_log():
    # At 12,000 K the active set is well away from the known Si+ crossing.
    mixture = make_sico(T=12000.0, P=101325.0)
    reduced = ReducedEquilibriumSystem(
        mixture, coupled_ionization_lowering=True
    )
    full = LogEquilibriumSystem(mixture)
    reduced_result = reduced.solve(tolerance=1e-11)
    full_result = full.solve(tolerance=1e-11)
    reduced_tangent = reduced.temperature_tangent(reduced_result)
    full_tangent = full.temperature_tangent(full_result)

    assert reduced_tangent.mole_fraction_derivative == pytest.approx(
        full_tangent.mole_fraction_derivative, rel=3e-7, abs=2e-12
    )
    assert reduced_tangent.reference_energy_derivative == pytest.approx(
        full_tangent.reference_energy_derivative, rel=5e-7, abs=2e-27
    )
    reduced_log = _log_density_tangent(reduced_tangent)
    if reduced_log is not None:
        full_densities = _densities(full_result)
        full_fractions = full_densities / full_densities.sum()
        # The full log variables are particle-number logs.  At fixed pressure
        # d(log V)/dT = 1/T + sum_i x_i d(log N_i)/dT.
        full_log = full_tangent.log_particle_derivative - (
            1 / mixture.T
            + full_fractions @ full_tangent.log_particle_derivative
        )
        assert reduced_log == pytest.approx(full_log, rel=5e-7, abs=2e-11)
        reduced_density = _density_tangent(reduced_tangent)
        assert reduced_density is not None
        expected_density = _densities(full_result) * full_log
        assert reduced_density == pytest.approx(
            expected_density, rel=5e-7, abs=2e-6
        )


@pytest.mark.parametrize("pressure", [1013.25, 10132500.0])
def test_coupled_reduced_pressure_probes_match_full_log(pressure):
    reduced = _coupled_system(make_simple, 12000.0, pressure)
    full = LogEquilibriumSystem(reduced.mixture)
    reduced_result = reduced.solve(tolerance=1e-10)
    full_result = full.solve(tolerance=1e-10)
    assert reduced_result.residual_norm < 1e-9
    assert _densities(reduced_result) == pytest.approx(
        _densities(full_result), rel=2e-8, abs=4e-10
    )
    _assert_lowering_matches(
        _coupled_lowering(reduced_result),
        full._packed_thermodynamics(
            full_result.log_particles, derivatives=False
        ).ionization_lowering,
        relative=6e-7,
    )
    _assert_reference_matches(
        _coupled_reference_energies(reduced, reduced_result),
        full._packed_thermodynamics(
            full_result.log_particles, derivatives=False
        ).reference_energies,
        relative=6e-7,
    )


def test_coupled_reduced_independent_and_perturbed_initial_state():
    system = _coupled_system(make_simple, 12000.0)
    independent = system.solve(tolerance=1e-10)
    initial = _initial_state(system)
    perturbed = initial + np.linspace(-0.1, 0.1, initial.size)
    restarted = system.solve(initial=perturbed, tolerance=1e-10)
    assert restarted.residual_norm < 1e-9
    assert _densities(restarted) == pytest.approx(
        _densities(independent), rel=6e-6, abs=4e-10
    )
    assert _coupled_lowering(restarted) == pytest.approx(
        _coupled_lowering(independent), rel=5e-7, abs=2e-16
    )
