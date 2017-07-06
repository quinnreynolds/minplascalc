import pytest
import MinPlasCalc as mpc


@pytest.fixture
def composition():
    return mpc.compositionGFE(
        compositionFile="Compositions/OxygenPlasma5sp.json",
        T=1000.,
        P=101325.)


def test_solver(composition):
    composition.initialiseNi([1e20]*len(composition.species))
    composition.T = 1000.
    composition.solveGfe()

    assert composition.calculateDensity() == pytest.approx(0.3899566)


def test_heat_capacity_lowt_lowp(composition):
    composition.T = 1000.
    composition.P = 10132.5

    assert composition.calculate_heat_capacity() == pytest.approx(1081.252, abs=1e-3)


def test_heat_capacity_hight_lowp(composition):
    composition.T = 25000.
    composition.P = 10132.5

    assert composition.calculate_heat_capacity() == pytest.approx(23194.70, abs=1e-2)


def test_heat_capacity_lowt_highp(composition):
    composition.T = 1000.
    composition.P = 1013250.

    assert composition.calculate_heat_capacity() == pytest.approx(1081.252, abs=1e-3)


def test_heat_capacity_hight_highp(composition):
    composition.T = 25000.
    composition.P = 1013250.

    assert composition.calculate_heat_capacity() == pytest.approx(5829.618, abs=1e-3)


def test_enthalpy_lowt_lowp(composition):
    composition.initialiseNi([1e20]*len(composition.species))
    composition.T = 1000.
    composition.P = 10132.5
    composition.solveGfe()

    assert composition.calculate_enthalpy() == pytest.approx(-1.455147e7, abs=1e1)


def test_enthalpy_hight_lowp(composition):
    composition.initialiseNi([1e20]*len(composition.species))
    composition.T = 25000.
    composition.P = 10132.5
    composition.solveGfe()

    assert composition.calculate_enthalpy() == pytest.approx(1.893154e8, abs=1e2)


def test_enthalpy_lowt_highp(composition):
    composition.initialiseNi([1e20]*len(composition.species))
    composition.T = 1000.
    composition.P = 1013250.
    composition.solveGfe()

    assert composition.calculate_enthalpy() == pytest.approx(-1.455147e7, abs=1e1)


def test_enthalpy_hight_highp(composition):
    composition.initialiseNi([1e20]*len(composition.species))
    composition.T = 25000.
    composition.P = 1013250.
    composition.solveGfe()

    assert composition.calculate_enthalpy() == pytest.approx(1.483881e8, abs=1e2)


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

