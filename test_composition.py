import pytest
import minplascalc as mpc


@pytest.fixture
def mixture():
    c = mpc.Mixture(str(mpc.MIXTUREPATH / "OxygenPlasma5sp.json"),
                    T=1000., P=101325.)
    c.initialiseNi([1e20]*len(c.species))
    return c


def test_solver(mixture):
    mixture.T = 1000.
    mixture.solveGfe()

    assert mixture.calculateDensity() == pytest.approx(0.3899566)


LOW_T = 1000.
HIGH_T = 25000.
LOW_P = 10132.5
HIGH_P = 1013250.

@pytest.mark.parametrize("temperature, pressure, result, tol", [
    (LOW_T, LOW_P, 1081.252, 1e-3),
    (HIGH_T, LOW_P, 23193.93, 1e-2),
    (LOW_T, HIGH_P, 1081.252, 1e-3),
    (HIGH_T, HIGH_P, 5829.864, 1e-3),
])
def test_heat_capacity(mixture, temperature, pressure, result, tol):
    mixture.T = temperature
    mixture.P = pressure

    thisresult = mixture.calculate_heat_capacity()

    assert thisresult == pytest.approx(result, abs=tol)

@pytest.mark.parametrize("temperature, pressure, result, tol", [
    (LOW_T, LOW_P, -1.455147e7, 1e1),
    (HIGH_T, LOW_P, 1.893144e8, 1e2),
    (LOW_T, HIGH_P, -1.455147e7, 1e1),
    (HIGH_T, HIGH_P, 1.483885e8, 1e2),
])
def test_enthalpy(mixture, temperature, pressure, result, tol):
    mixture.T = temperature
    mixture.P = pressure
    mixture.solveGfe()

    assert mixture.calculate_enthalpy() == pytest.approx(result, abs=tol)


def test_calculateViscosity(mixture):
    with pytest.raises(NotImplementedError):
        mixture.calculateViscosity()


def test_calculateThermalConductivity(mixture):
    with pytest.raises(NotImplementedError):
        mixture.calculateThermalConductivity()


def test_calculateElectricalConductivity(mixture):
    with pytest.raises(NotImplementedError):
        mixture.calculateElectricalConductivity()


def test_calculateTotalEmissionCoefficient(mixture):
    with pytest.raises(NotImplementedError):
        mixture.calculateTotalEmissionCoefficient()

