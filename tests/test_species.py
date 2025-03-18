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
