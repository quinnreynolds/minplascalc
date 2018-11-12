import minplascalc as mpc
import pytest


def test_molarmass():
    mm = mpc.molar_mass_calculator(8, 8, 8)
    assert mm == pytest.approx(0.016, abs=1e-3)


@pytest.fixture
def diatomic_sample_species():
    return mpc.species_from_name("O2")


@pytest.fixture
def monatomic_sample_species():
    return mpc.species_from_name("O+")


def test_partitionfunction_translational(diatomic_sample_species):
    partition_function = diatomic_sample_species.partitionfunction_translational(0.)
    assert partition_function == 0.


def test_partitionfunction_internal(diatomic_sample_species):
    partition_function = diatomic_sample_species.partitionfunction_internal(300)
    assert partition_function == pytest.approx(217.6606)

@pytest.mark.parametrize("T, energy, tol", [
    (1000, 2.070973e-20, 1e-26),
    (25000, 7.619840e-19, 1e-25),
])
def test_monatomic_internal_energy(monatomic_sample_species,
                                   T, energy, tol):
    internal_energy = monatomic_sample_species.internal_energy(T)
    assert internal_energy == pytest.approx(energy, abs=tol)

@pytest.mark.parametrize("T, energy, tol", [
    (1000, 3.811852e-20, 1e-26),
    (25000, 1.192611e-18, 1e-24),
])
def test_diatomic_internal_energy(diatomic_sample_species,
                                  T, energy, tol):
    internal_energy = diatomic_sample_species.internal_energy(T)
    assert internal_energy == pytest.approx(energy, abs=tol)

