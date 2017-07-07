import pytest
import MinPlasCalc as mpc


@pytest.fixture
def composition():
    c = mpc.compositionGFE(compositionFile="Compositions/OxygenPlasma5sp.json",
                           T=1000., P=101325.)
    c.initialiseNi([1e20]*len(c.species))
    return c


def test_solver(composition):
    composition.T = 1000.
    composition.solveGfe()

    assert composition.calculateDensity() == pytest.approx(0.3899566)


LOW_T = 1000.
HIGH_T = 25000.
LOW_P = 10132.5
HIGH_P = 1013250.

@pytest.mark.parametrize("temperature, pressure, result, tol", [
    (LOW_T, LOW_P, 1081.252, 1e-3),
    (HIGH_T, LOW_P, 23194.70, 1e-2),
    (LOW_T, HIGH_P, 1081.252, 1e-3),
    (HIGH_T, HIGH_P, 5829.618, 1e-3),
])
def test_heat_capacity(composition, temperature, pressure, result, tol):
    composition.T = temperature
    composition.P = pressure

    thisresult = composition.calculate_heat_capacity()

    assert thisresult == pytest.approx(result, abs=tol)

@pytest.mark.parametrize("temperature, pressure, result, tol", [
    (LOW_T, LOW_P, -1.455147e7, 1e1),
    (HIGH_T, LOW_P, 1.893154e8, 1e2),
    (LOW_T, HIGH_P, -1.455147e7, 1e1),
    (HIGH_T, HIGH_P, 1.483881e8, 1e2),
])
def test_enthalpy(composition, temperature, pressure, result, tol):
    composition.T = temperature
    composition.P = pressure
    composition.solveGfe()

    assert composition.calculate_enthalpy() == pytest.approx(result, abs=tol)


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

