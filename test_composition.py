import pytest
import minplascalc as mpc


@pytest.fixture
def mixture():
    c = mpc.Mixture(str(mpc.MIXTUREPATH / "OxygenPlasma5sp.json"),
                    temperature=1000., pressure=101325.)
    c.initialise_ni([1e20]*len(c.species))
    return c


def test_solver(mixture):
    mixture.temperature = 1000.
    mixture.solve_gfe()

    assert mixture.calculate_density() == pytest.approx(0.3899566)


LOW_T = 1000.
HIGH_T = 25000.
LOW_P = 10132.5
HIGH_P = 1013250.

@pytest.mark.parametrize("temperature, pressure, result, tol", [
    (LOW_T, LOW_P, 1081.252, 1e-3),
    (HIGH_T, LOW_P, 23193.92, 1e-2),
    (LOW_T, HIGH_P, 1081.252, 1e-3),
    (HIGH_T, HIGH_P, 5829.868, 1e-3),
])
def test_heat_capacity(mixture, temperature, pressure, result, tol):
    mixture.temperature = temperature
    mixture.pressure = pressure

    thisresult = mixture.calculate_heat_capacity()

    assert thisresult == pytest.approx(result, abs=tol)

@pytest.mark.parametrize("temperature, pressure, result, tol", [
    (LOW_T, LOW_P, -1.455147e7, 1e1),
    (HIGH_T, LOW_P, 1.893144e8, 1e2),
    (LOW_T, HIGH_P, -1.455147e7, 1e1),
    (HIGH_T, HIGH_P, 1.483885e8, 1e2),
])
def test_enthalpy(mixture, temperature, pressure, result, tol):
    mixture.temperature = temperature
    mixture.pressure = pressure
    mixture.solve_gfe()

    assert mixture.calculate_enthalpy() == pytest.approx(result, abs=tol)


def test_calculate_viscosity(mixture):
    with pytest.raises(NotImplementedError):
        mixture.calculate_viscosity()


def test_calculate_thermal_conductivity(mixture):
    with pytest.raises(NotImplementedError):
        mixture.calculate_thermal_conductivity()


def test_calculate_electrical_conductivity(mixture):
    with pytest.raises(NotImplementedError):
        mixture.calculate_electrical_conductivity()


def test_calculate_total_emission_coefficient(mixture):
    with pytest.raises(NotImplementedError):
        mixture.calculate_total_emission_coefficient()

