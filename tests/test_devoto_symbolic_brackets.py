"""Cross-check Devoto kernels against symbolic collision brackets."""

from types import SimpleNamespace

import numpy as np
import pytest

from bench.devoto_brackets_symbolic import assemble_bracket_block
from minplascalc import functions_transport as ft


@pytest.fixture(scope="module")
def artificial_transport_system():
    rng = np.random.default_rng(1966)
    species_count = 3
    masses = np.array([1.7, 2.9, 5.3])
    number_densities = np.array([2.3, 3.1, 4.7])
    collision_integrals = {}
    for moment in ft.LS_PAIRS:
        values = rng.uniform(0.5, 2.0, (species_count, species_count))
        collision_integrals[moment] = (values + values.T) / 2

    state = SimpleNamespace(
        masses=masses,
        number_densities=number_densities,
    )

    class ArtificialMixture:
        species = tuple(range(species_count))

        def __init__(self):
            self.masses = masses

        @staticmethod
        def calculate_composition():
            return number_densities

        @staticmethod
        def _equilibrium_state():
            return state

    mixture = ArtificialMixture()
    return masses, number_densities, collision_integrals, mixture


@pytest.mark.parametrize(
    "left_order, right_order",
    [(0, 0), (0, 1), (1, 1)],
)
def test_symbolic_viscosity_brackets_match_devoto_kernels(
    artificial_transport_system, left_order, right_order
):
    masses, number_densities, collision_integrals, mixture = (
        artificial_transport_system
    )
    actual_matrix = ft.qhat(mixture, collision_integrals)
    species_count = len(masses)
    actual = actual_matrix[
        left_order * species_count : (left_order + 1) * species_count,
        right_order * species_count : (right_order + 1) * species_count,
    ]
    derived = assemble_bracket_block(
        masses,
        number_densities,
        collision_integrals,
        rank=2,
        left_order=left_order,
        right_order=right_order,
    )
    assert actual == pytest.approx(derived, rel=2e-13)


@pytest.mark.parametrize(
    "left_order, right_order",
    [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (2, 2),
        (2, 3),
    ],
    ids=["A3", "A4", "A7", "A12", "A11", "A16"],
)
def test_symbolic_diffusion_brackets_match_devoto_kernels(
    artificial_transport_system, left_order, right_order
):
    masses, number_densities, collision_integrals, mixture = (
        artificial_transport_system
    )
    actual_matrix = ft.q(mixture, collision_integrals)
    species_count = len(masses)
    actual = actual_matrix[
        left_order * species_count : (left_order + 1) * species_count,
        right_order * species_count : (right_order + 1) * species_count,
    ]
    derived = assemble_bracket_block(
        masses,
        number_densities,
        collision_integrals,
        rank=1,
        left_order=left_order,
        right_order=right_order,
    )
    assert actual == pytest.approx(derived, rel=2e-13)
