"""Regression tests for the Devoto appendix q-matrix expressions."""

import numpy as np
import pytest

import minplascalc as mpc
from minplascalc import functions_transport as ft

SICO_SPECIES = [
    "Si",
    "C",
    "O",
    "SiO",
    "CO",
    "O2",
    "Si+",
    "C+",
    "O+",
    "Si++",
    "C++",
    "O++",
]


def _expected_q22_q22_term(masses, number_densities, q22):
    count = len(masses)
    expected = np.zeros((count, count))
    for i in range(count):
        for j in range(count):
            total = 0.0
            for l in range(count):
                total += (
                    number_densities[l]
                    * masses[l] ** 0.5
                    / (masses[i] + masses[l]) ** 4.5
                    * (ft.delta(i, j) + ft.delta(j, l))
                    * 7
                    * masses[j]
                    * masses[l]
                    * (4 * masses[j] ** 2 + 7 * masses[l] ** 2)
                    * q22[i, l]
                )
            expected[i, j] = (
                8
                * number_densities[i]
                * (masses[i] / masses[j]) ** 2.5
                * total
            )
    return expected


def _expected_q23_q22_term(masses, number_densities, q22):
    count = len(masses)
    expected = np.zeros((count, count))
    for i in range(count):
        for j in range(count):
            total = 0.0
            for l in range(count):
                total += (
                    number_densities[l]
                    * masses[l] ** 1.5
                    / (masses[i] + masses[l]) ** 5.5
                    * (ft.delta(i, j) + ft.delta(j, l))
                    * 63
                    / 4
                    * masses[j]
                    * masses[l]
                    * (8 * masses[j] ** 2 + 7 * masses[l] ** 2)
                    * q22[i, l]
                )
            expected[i, j] = (
                8
                * number_densities[i]
                * (masses[i] / masses[j]) ** 3.5
                * total
            )
    return expected


@pytest.fixture
def q22_only_inputs():
    masses = np.array([2.0, 3.0])
    number_densities = np.array([5.0, 7.0])
    q22 = np.array([[11.0, 13.0], [17.0, 19.0]])
    zeros = np.zeros_like(q22)
    return masses, number_densities, q22, zeros


def test_q22_uses_devoto_a11_q22_coefficient(q22_only_inputs):
    masses, number_densities, q22, zeros = q22_only_inputs
    actual = ft._q22_jit(
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        q22,
        zeros,
        zeros,
        zeros,
        masses,
        len(masses),
        number_densities,
    )
    expected = _expected_q22_q22_term(masses, number_densities, q22)
    assert actual == pytest.approx(expected)


def test_q23_uses_devoto_a16_q22_coefficient(q22_only_inputs):
    masses, number_densities, q22, zeros = q22_only_inputs
    actual = ft._q23_jit(
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        q22,
        zeros,
        zeros,
        zeros,
        zeros,
        zeros,
        masses,
        len(masses),
        number_densities,
    )
    expected = _expected_q23_q22_term(masses, number_densities, q22)
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize(
    "x0, center_temperature",
    [
        (
            [
                0.19754580488363843,
                0.30290296560926583,
                0.4995512295070957,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            17800.0,
        ),
        (
            [
                0.1321622922145921,
                0.37018534501125755,
                0.4976523627741504,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            15100.0,
        ),
        (
            [
                0.4525320530234709,
                0.45677295691031883,
                0.09069499006621033,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            16300.0,
        ),
    ],
    ids=["Tbath-2342K", "Tbath-2529K", "Tbath-3211K"],
)
def test_reported_thermal_conductivity_outliers_are_smooth(
    x0, center_temperature
):
    mixture = mpc.mixture.lte_from_names(
        SICO_SPECIES,
        x0=x0,
        T=center_temperature - 1,
        P=101325.0,
    )
    conductivities = []
    for temperature in (
        center_temperature - 1,
        center_temperature,
        center_temperature + 1,
    ):
        mixture.T = temperature
        conductivities.append(mixture.calculate_thermal_conductivity())

    assert np.all(np.isfinite(conductivities))
    assert np.all(np.asarray(conductivities) > 0)
    midpoint = (conductivities[0] + conductivities[2]) / 2
    assert conductivities[1] == pytest.approx(midpoint, abs=1e-4)
