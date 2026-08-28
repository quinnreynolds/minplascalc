"""Checks for the packed thermodynamics shared by equilibrium prototypes."""

from __future__ import annotations

import numpy as np
import pytest

from bench.log_equilibrium import LogEquilibriumSystem
from bench.reduced_equilibrium import ReducedEquilibriumSystem
from bench.workloads import make_sico, make_simple
from minplascalc import species as species_module
from minplascalc import units as u


def _partition(system, lowering):
    """Call the shared partition kernel."""
    log_partition, active_counts = system._packed_log_partition_per_volume(
        np.asarray(lowering, dtype=np.float64)
    )
    return np.asarray(log_partition), np.asarray(active_counts)


def _reference(system, lowering, derivatives=None):
    """Call the shared reference chain with optional derivatives."""
    reference, derivative = system._packed_reference_from_lowering(
        np.asarray(lowering), derivatives
    )
    return np.asarray(reference), (
        None if derivative is None else np.asarray(derivative)
    )


def _log_particles(system):
    """Build a finite, deterministic state without invoking a solver."""
    return np.full(system.species_count, np.log(1.0e20), dtype=np.float64)


def _lowering_for(system, scale=0.0):
    """Return a nonnegative lowering vector suitable for both prototypes."""
    lowering = np.zeros(system.species_count, dtype=np.float64)
    if scale:
        positive = system.positive_indices
        lowering[positive] = scale * u.k_b * system.mixture.T
    return lowering


def _auxiliary_state(system, packed):
    """Recover reduced eta/xi coordinates from a packed particle state."""
    volume = (
        packed.particle_total * u.k_b * system.mixture.T / system.mixture.P
    )
    eta = np.log(packed.particle_numbers[system.electron_index] / volume)
    positive = system.positive_indices
    charges = system.charges[positive]
    xi = np.log(packed.particle_numbers[positive] @ charges**2)
    xi -= np.log(packed.particle_numbers[positive] @ charges)
    return eta, xi


def _legacy_log_partition(system, lowering):
    """Compute per-volume partitions using the pre-packed species routines."""
    values = np.empty(system.species_count)
    active_counts = np.zeros(system.species_count, dtype=np.int64)
    for index, species in enumerate(system.mixture.species):
        if isinstance(species, species_module.Monatomic):
            terms = np.log(species._degeneracies) - (
                species._level_energies / (u.k_b * system.mixture.T)
            )
            active = species._level_energies < (
                species.ionisation_energy - lowering[index]
            )
            active_counts[index] = np.count_nonzero(active)
            values[index] = np.log(
                species.translational_partition_function(system.mixture.T)
            )
            if np.any(active):
                maximum = np.max(terms[active])
                values[index] += maximum + np.log(
                    np.exp(terms[active] - maximum).sum()
                )
            else:
                values[index] = -np.inf
        else:
            internal = species.internal_partition_function(
                system.mixture.T, lowering[index]
            )
            values[index] = np.log(
                species.translational_partition_function(system.mixture.T)
            ) + np.log(internal)
    return values, active_counts


@pytest.mark.parametrize("factory", [make_simple, make_sico])
@pytest.mark.parametrize("temperature", [8000.0, 12000.0, 25000.0])
@pytest.mark.parametrize("scale", [0.0, 1.0e-3])
def test_packed_partition_and_reference_match_full_and_reduced(
    factory, temperature, scale
):
    """Packed values agree with legacy species and reduced calculations."""
    mixture = factory(T=temperature)
    reduced = ReducedEquilibriumSystem(mixture)
    lowering = _lowering_for(reduced, scale=scale)
    full = LogEquilibriumSystem(mixture, fixed_ionization_lowering=lowering)

    log_density, active_counts = _partition(full, lowering)
    legacy_density, legacy_counts = _legacy_log_partition(full, lowering)
    np.testing.assert_allclose(log_density, legacy_density)
    np.testing.assert_array_equal(active_counts, legacy_counts)
    np.testing.assert_allclose(
        log_density, reduced._log_partition_per_volume(lowering)
    )

    reference, derivative = _reference(full, lowering)
    expected_reference, expected_derivative = reduced._reference_from_lowering(
        lowering
    )
    np.testing.assert_allclose(reference, expected_reference)
    assert derivative is None
    assert expected_derivative is None

    packet = full._packed_thermodynamics(
        _log_particles(full), derivatives=False
    )
    volume = packet.particle_total * u.k_b * temperature / mixture.P
    np.testing.assert_allclose(
        packet.log_partitions - np.log(volume), log_density
    )


@pytest.mark.parametrize("factory", [make_simple, make_sico])
@pytest.mark.parametrize("temperature", [8000.0, 12000.0, 25000.0])
def test_packed_auxiliary_derivatives_match_full_log_and_reduced(
    factory, temperature
):
    """Lowering/reference derivatives agree with both prototype paths."""
    mixture = factory(T=temperature)
    full = LogEquilibriumSystem(mixture)
    reduced = ReducedEquilibriumSystem(
        mixture, coupled_ionization_lowering=True
    )
    packet = full._packed_thermodynamics(
        _log_particles(full), derivatives=True
    )
    lowering_dN = full._packed_lowering(
        packet.particle_numbers, derivatives=True
    )[1]
    assert lowering_dN is not None
    reference, reference_dN = _reference(
        full, packet.ionization_lowering, lowering_dN
    )
    np.testing.assert_allclose(reference, packet.reference_energies)
    np.testing.assert_allclose(reference_dN, packet.reference_dN)

    eta, xi = _auxiliary_state(reduced, packet)
    lowering, d_eta, d_xi = reduced._coupled_lowering(eta, xi)
    expected_reference, reference_auxiliary = reduced._reference_from_lowering(
        lowering, (d_eta, d_xi)
    )
    np.testing.assert_allclose(lowering, packet.ionization_lowering)
    np.testing.assert_allclose(expected_reference, packet.reference_energies)
    assert reference_auxiliary is not None
    assert reference_auxiliary.shape == (full.species_count, 2)
    _, shared_auxiliary = _reference(
        full,
        lowering,
        np.column_stack((d_eta, d_xi)),
    )
    np.testing.assert_allclose(shared_auxiliary, reference_auxiliary)


@pytest.mark.parametrize("temperature", [20861.5, 20862.0, 20862.5])
def test_packed_active_levels_match_cutoff_adjacent_sico_state(temperature):
    """Strict level activity stays aligned on either side of Si+ cutoff."""
    mixture = make_sico(T=temperature)
    full = LogEquilibriumSystem(mixture)
    reduced = ReducedEquilibriumSystem(
        mixture, coupled_ionization_lowering=True
    )
    result = full.solve(tolerance=1.0e-10)
    packet = full._packed_thermodynamics(
        result.log_particles, derivatives=False
    )
    _, actual_counts = _partition(full, packet.ionization_lowering)
    np.testing.assert_array_equal(actual_counts, packet.active_level_counts)

    eta, xi = _auxiliary_state(reduced, packet)
    lowering, _, _ = reduced._coupled_lowering(eta, xi)
    _, reduced_counts = _partition(reduced, lowering)
    np.testing.assert_array_equal(reduced_counts, actual_counts)
    fingerprint = mixture._active_level_fingerprint(lowering)
    assert fingerprint.nearest_cutoff_species_name == "Si+"
