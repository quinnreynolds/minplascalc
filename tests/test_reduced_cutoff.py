"""Stage-4 local active-set checks around the reviewed Si+ crossing."""

from __future__ import annotations

import numpy as np
import pytest

from bench.log_equilibrium import LogEquilibriumSystem
from bench.reduced_equilibrium import ReducedEquilibriumSystem
from bench.workloads import make_sico

CROSSING_TEMPERATURE = 20862.0


def _mole_fractions(result):
    densities = np.asarray(result.number_densities, dtype=float)
    return densities / densities.sum()


def _fingerprint(system, result):
    return system.active_level_fingerprint(result)


def _species_counts(fingerprint):
    return {
        state.species_name: state.active_level_count
        for state in fingerprint.species
    }


def _matching_candidates(reduced, reduced_branch, full, full_branch):
    reduced_by_fingerprint = {
        _fingerprint(reduced, candidate).fingerprint: candidate
        for candidate in reduced_branch.candidates
    }
    full_by_fingerprint = {
        _fingerprint(full, candidate).fingerprint: candidate
        for candidate in full_branch.candidates
    }
    assert set(reduced_by_fingerprint) == set(full_by_fingerprint)
    return reduced_by_fingerprint, full_by_fingerprint


def test_local_cutoff_branches_match_full_log_by_exact_fingerprint():
    mixture = make_sico(T=CROSSING_TEMPERATURE, P=101325.0)
    reduced = ReducedEquilibriumSystem(
        mixture, coupled_ionization_lowering=True
    )
    full = LogEquilibriumSystem(mixture)
    reduced_branch = reduced.solve_lowest_gibbs_branch(tolerance=1e-10)
    full_branch = full.solve_lowest_gibbs_branch(tolerance=1e-10)

    assert len(full_branch.candidates) == 2
    assert len(reduced_branch.candidates) == 2
    reduced_by_fingerprint, full_by_fingerprint = _matching_candidates(
        reduced, reduced_branch, full, full_branch
    )
    assert len(reduced_by_fingerprint) == len(reduced_branch.candidates)

    for fingerprint, reduced_candidate in reduced_by_fingerprint.items():
        full_candidate = full_by_fingerprint[fingerprint]
        reduced_state = _fingerprint(reduced, reduced_candidate)
        full_state = _fingerprint(full, full_candidate)
        assert reduced_state.fingerprint == full_state.fingerprint
        assert _species_counts(reduced_state) == _species_counts(full_state)
        assert reduced_state.nearest_cutoff_species_name == "Si+"
        assert reduced_state.nearest_cutoff_species_name == (
            full_state.nearest_cutoff_species_name
        )
        assert reduced_state.nearest_cutoff_level_index == (
            full_state.nearest_cutoff_level_index
        )
        assert reduced_state.nearest_cutoff_margin_over_kbt == pytest.approx(
            full_state.nearest_cutoff_margin_over_kbt,
            rel=2e-5,
            abs=2e-10,
        )
        assert _mole_fractions(reduced_candidate) == pytest.approx(
            _mole_fractions(full_candidate), rel=3e-6, abs=2e-11
        )

    reduced_selected = _fingerprint(
        reduced, reduced_branch.selected
    ).fingerprint
    full_selected = _fingerprint(full, full_branch.selected).fingerprint
    assert reduced_selected == full_selected
    assert reduced_branch.dimensionless_gibbs[
        list(reduced_by_fingerprint).index(reduced_selected)
    ] == min(reduced_branch.dimensionless_gibbs)


def test_cutoff_probe_deduplicates_by_exact_fingerprint_and_is_local():
    mixture = make_sico(T=CROSSING_TEMPERATURE, P=101325.0)
    reduced = ReducedEquilibriumSystem(
        mixture, coupled_ionization_lowering=True
    )
    branch = reduced.solve_lowest_gibbs_branch(tolerance=1e-10)
    fingerprints = [
        _fingerprint(reduced, candidate).fingerprint
        for candidate in branch.candidates
    ]
    assert len(fingerprints) == len(set(fingerprints))
    assert branch.nearest_cutoff_distance < 2e-5
    assert len(branch.candidates) <= 3

    # A window narrower than the measured distance must suppress the optional
    # local probe; it must not trigger a global active-set search.
    narrow = reduced.solve_lowest_gibbs_branch(
        tolerance=1e-10, cutoff_window=1e-7
    )
    assert len(narrow.candidates) == 1


@pytest.mark.parametrize(
    "temperature, expected_sign, expected_count",
    [(20861.5, -1, 27), (20862.5, 1, 29)],
)
def test_neighbouring_temperatures_track_cutoff_sign_and_active_count(
    temperature, expected_sign, expected_count
):
    mixture = make_sico(T=temperature, P=101325.0)
    reduced = ReducedEquilibriumSystem(
        mixture, coupled_ionization_lowering=True
    )
    full = LogEquilibriumSystem(mixture)
    reduced_result = reduced.solve_temperature_path(
        np.array([temperature]),
        bootstrap_temperature=12000.0,
        max_temperature_step=1000.0,
        tolerance=1e-10,
    ).states[0]
    full_result = full.solve(tolerance=1e-10)
    reduced_state = _fingerprint(reduced, reduced_result)
    full_state = _fingerprint(full, full_result)

    assert reduced_state.fingerprint == full_state.fingerprint
    assert reduced_state.nearest_cutoff_species_name == "Si+"
    assert reduced_state.nearest_cutoff_margin_over_kbt * expected_sign > 0
    assert reduced_state.nearest_cutoff_margin_over_kbt == pytest.approx(
        full_state.nearest_cutoff_margin_over_kbt, rel=2e-5, abs=2e-10
    )
    assert _species_counts(reduced_state)["Si+"] == expected_count
    assert _species_counts(reduced_state) == _species_counts(full_state)
    assert _mole_fractions(reduced_result) == pytest.approx(
        _mole_fractions(full_result), rel=3e-6, abs=2e-11
    )
