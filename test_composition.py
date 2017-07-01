import pytest
import MinPlasCalc as mpc

@pytest.fixture
def composition():
    return mpc.compositionGFE(
        compositionFile="Compositions/OxygenPlasma5sp.json",
        T=1000.,
        P=101325.)

def test_solver(composition):
    composition.initialiseNi([1e20 for i in range(len(composition.species))])
    composition.T = 1000.
    composition.solveGfe()

    assert composition.calculateDensity() == pytest.approx(0.3899566)

def test_calculateHeatCapacity(composition):
    with pytest.raises(NotImplementedError):
        composition.calculateHeatCapacity()

def test_calculateViscosity(composition):
    with pytest.raises(NotImplementedError):
        composition.calculateViscosity()

def test_calculateThermalConductivity(composition):
    with pytest.raises(NotImplementedError):
        composition.calculateThermalConductivity()

def test_calculateElectricalConductivity(composition):
    with pytest.raises(NotImplementedError):
        composition.calculateElectricalConductivity()

def test_calculateTotalEmissionCoefficient(composition):
    with pytest.raises(NotImplementedError):
        composition.calculateTotalEmissionCoefficient()

