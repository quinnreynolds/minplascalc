import numpy as np
import pytest

import minplascalc as mpc


@pytest.fixture
def diatomic_sample_species():
    return mpc.species.from_name("O2")


@pytest.fixture
def monatomic_sample_species():
    return mpc.species.from_name("O+")


def test_translational_partition_function(diatomic_sample_species):
    partition_function = (
        diatomic_sample_species.translational_partition_function(0)
    )
    assert partition_function == 0


def test_internal_partition_function(diatomic_sample_species):
    partition_function = diatomic_sample_species.internal_partition_function(
        300, 0
    )
    assert partition_function == pytest.approx(4.921932)


@pytest.mark.parametrize(
    "T, energy, tol",
    [
        (1000, 2.070973e-20, 1e-26),
        (25000, 7.619842e-19, 1e-25),
    ],
)
def test_monatomic_internal_energy(monatomic_sample_species, T, energy, tol):
    internal_energy = monatomic_sample_species.internal_energy(T, 0)
    assert internal_energy == pytest.approx(energy, abs=tol)


@pytest.mark.parametrize(
    "T, energy, tol",
    [
        (1000, 5.381335e-20, 1e-26),
        (25000, 1.208306e-18, 1e-24),
    ],
)
def test_diatomic_internal_energy(diatomic_sample_species, T, energy, tol):
    internal_energy = diatomic_sample_species.internal_energy(T, 0)
    assert internal_energy == pytest.approx(energy, abs=tol)


@pytest.mark.parametrize("species_name", ["O2", "O+"])
@pytest.mark.parametrize("T", [2000.0, 12000.0, 25000.0])
def test_partition_log_temperature_derivative(species_name, T):
    species = mpc.species.from_name(species_name)
    volume = 0.37
    lowering = 0.0
    delta_T = T * 1e-5

    log_high = np.log(
        species.total_partition_function(volume, T + delta_T, lowering)
    )
    log_low = np.log(
        species.total_partition_function(volume, T - delta_T, lowering)
    )
    finite_difference = (log_high - log_low) / (2 * delta_T)

    assert species.dlog_total_partition_dT(T, lowering) == pytest.approx(
        finite_difference, rel=2e-8
    )


def test_electron_partition_log_temperature_derivative():
    electron = mpc.species.Electron()
    T = 12000.0
    delta_T = T * 1e-5
    volume = 0.37
    log_high = np.log(
        electron.total_partition_function(volume, T + delta_T, 0.0)
    )
    log_low = np.log(
        electron.total_partition_function(volume, T - delta_T, 0.0)
    )

    assert electron.dlog_total_partition_dT(T, 0.0) == pytest.approx(
        (log_high - log_low) / (2 * delta_T), rel=2e-8
    )
