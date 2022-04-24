import pytest
import minplascalc as mpc


@pytest.fixture
def mixture_simple():
    c = mpc.mixture.LTE(species=[mpc.species.from_name(sp) for sp in 
                                 ['O2','O2+','O','O-','O+','O++']],
                        x0=[1,0,0,0,0,0],
                        T=1000, P=101325,
                        gfe_ni0=1e20, gfe_reltol=1e-10, gfe_maxiter=1000)
    return c

@pytest.fixture
def mixture_complex():
    c = mpc.mixture.LTE(species=[mpc.species.from_name(sp) for sp in 
                                 ['O2','O2+','O','O+','O++','CO','CO+','C',
                                  'C+','C++','SiO','SiO+','Si','Si+', 'Si++']],
                        x0=[0,0,0,0,0,0.5,0,0,0,0,0.5,0,0,0,0],
                        T=10000, P=101325,
                        gfe_ni0=1e20, gfe_reltol=1e-10, gfe_maxiter=1000)
    return c

LOW_T, MID_T, HIGH_T = 1000, 10000, 25000
LOW_P, MID_P, HIGH_P = 10132.5, 101325, 1013250
LOW_X0 = [0,0,0,0,0,0.1,0,0,0,0,0.9,0,0,0,0]
MID_X0 = [0,0,0,0,0,0.5,0,0,0,0,0.5,0,0,0,0]
HIGH_X0 = [0,0,0,0,0,0.9,0,0,0,0,0.1,0,0,0,0]

@pytest.mark.parametrize("T, P, result, tol", [
    (MID_T, MID_P, 0.01911387, 1e-7),
    (LOW_T, LOW_P, 0.03899566, 1e-7),
    (HIGH_T, LOW_P, 0.00035649, 1e-7),
    (LOW_T, HIGH_P, 3.89956460, 1e-7),
    (HIGH_T, HIGH_P, 0.04029165, 1e-7),
])
def test_density_simple(mixture_simple, T, P, result, tol):
    mixture_simple.T = T
    mixture_simple.P = P

    thisresult = mixture_simple.calculate_density()

    assert thisresult == pytest.approx(result, abs=tol)


@pytest.mark.parametrize("x0, result, tol", [
    (LOW_X0, 0.02059268, 1e-7),
    (MID_X0, 0.01871773, 1e-7),
    (HIGH_X0, 0.01668042, 1e-7),
])
def test_density_complex(mixture_complex, x0, result, tol):
    mixture_complex.x0 = x0

    thisresult = mixture_complex.calculate_density()

    assert thisresult == pytest.approx(result, abs=tol)

@pytest.mark.parametrize("T, P, result, tol", [
    (MID_T, MID_P, 15633838, 1e1),
    (LOW_T, LOW_P, -14256097, 1e1),
    (HIGH_T, LOW_P, 200223180, 1e2),
    (LOW_T, HIGH_P, -14256097, 1e1),
    (HIGH_T, HIGH_P, 150794540, 1e2),
])
def test_enthalpy_simple(mixture_simple, T, P, result, tol):
    mixture_simple.T = T
    mixture_simple.P = P
    
    thisresult = mixture_simple.calculate_enthalpy()

    assert thisresult == pytest.approx(result, abs=tol)

@pytest.mark.parametrize("x0, result, tol", [
    (LOW_X0, 22596860, 1e1),
    (MID_X0, 22146645, 1e1),
    (HIGH_X0, 21406432, 1e1),
])
def test_enthalpy_complex(mixture_complex, x0, result, tol):
    mixture_complex.x0 = x0
    
    thisresult = mixture_complex.calculate_enthalpy()

    assert thisresult == pytest.approx(result, abs=tol)


@pytest.mark.parametrize("T, P, result, tol", [
    (MID_T, MID_P, 3249.103, 1e-2),
    (LOW_T, LOW_P, 1081.252, 1e-2),
    (HIGH_T, LOW_P, 27605.70, 1e-1),
    (LOW_T, HIGH_P, 1081.252, 1e-2),
    (HIGH_T, HIGH_P, 7297.106, 1e-2),
])
def test_heat_capacity_simple(mixture_simple, T, P, result, tol):
    mixture_simple.T = T
    mixture_simple.P = P

    thisresult = mixture_simple.calculate_heat_capacity()

    assert thisresult == pytest.approx(result, abs=tol)


@pytest.mark.parametrize("x0, result, tol", [
    (LOW_X0, 6028.144, 1e-2),
    (MID_X0, 5508.598, 1e-2),
    (HIGH_X0, 6072.258, 1e-2),
])
def test_heat_capacity_complex(mixture_complex, x0, result, tol):
    mixture_complex.x0 = x0

    thisresult = mixture_complex.calculate_heat_capacity()

    assert thisresult == pytest.approx(result, abs=tol)

@pytest.mark.parametrize("T, P, result, tol", [
    (MID_T, MID_P, 15633838, 1e1),
    (LOW_T, LOW_P, -14256097, 1e1),
    (HIGH_T, LOW_P, 200223180, 1e2),
    (LOW_T, HIGH_P, -14256097, 1e1),
    (HIGH_T, HIGH_P, 150794540, 1e2),
])
def test_LTE_fallthrough(mixture_simple, T, P, result, tol):
    mixture_simple.T = T
    mixture_simple.P = P
    
    mixture_simple.calculate_composition()
    thisresult = mixture_simple.calculate_enthalpy()

    assert thisresult == pytest.approx(result, abs=tol)


def test_calculate_viscosity(mixture_simple):
    with pytest.raises(NotImplementedError):
        mixture_simple.calculate_viscosity()


def test_calculate_thermal_conductivity(mixture_simple):
    with pytest.raises(NotImplementedError):
        mixture_simple.calculate_thermal_conductivity()


def test_calculate_electrical_conductivity(mixture_simple):
    with pytest.raises(NotImplementedError):
        mixture_simple.calculate_electrical_conductivity()


def test_calculate_total_emission_coefficient(mixture_simple):
    with pytest.raises(NotImplementedError):
        mixture_simple.calculate_total_emission_coefficient()


def test_species_setter_exception(mixture_simple):
    with pytest.raises(TypeError):
        mixture_simple.species = [mpc.species.from_name(sp) 
                                  for sp in ['O','O+']]


def test_species_item_exception(mixture_simple):
    with pytest.raises(TypeError):
        mixture_simple.species[0] = mpc.species.from_name('O')


def test_x0_item_exception(mixture_simple):
    with pytest.raises(TypeError):
        mixture_simple.x0[0] = 0.5
