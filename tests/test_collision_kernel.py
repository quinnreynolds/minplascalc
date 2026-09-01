"""Numerical equivalence checks for the compiled collision-pair kernel."""

import numpy as np
import pytest

import minplascalc as mpc
from minplascalc import functions_transport as ft

SPECIES = [
    "O2",
    "O2+",
    "O",
    "O+",
    "O++",
    "CO",
    "CO+",
    "C",
    "C+",
    "C++",
    "SiO",
    "SiO+",
    "Si",
    "Si+",
    "Si++",
]


@pytest.mark.parametrize("T", [2000.0, 12000.0, 25000.0])
def test_collision_kernel_matches_scalar_pair_evaluation(T):
    mixture = mpc.mixture.lte_from_names(
        SPECIES,
        x0=[0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0],
        T=T,
        P=101325.0,
    )
    state = mixture._equilibrium_state()
    actual = mixture._collision_model().evaluate(
        state.number_densities, T, ft.LS_PAIRS
    )

    n = len(mixture.species)
    expected = {
        moment: np.empty((n, n), dtype=np.float64) for moment in ft.LS_PAIRS
    }
    for i, (n_i, species_i) in enumerate(
        zip(state.number_densities, mixture.species)
    ):
        for j, (n_j, species_j) in enumerate(
            zip(state.number_densities, mixture.species)
        ):
            pair = ft._pair_integrals(
                species_i,
                n_i,
                species_j,
                n_j,
                T,
                ft.LS_PAIRS,
            )
            for moment, value in pair.items():
                expected[moment][i, j] = value

    for moment in ft.LS_PAIRS:
        assert actual[moment] == pytest.approx(expected[moment], rel=3e-15)
