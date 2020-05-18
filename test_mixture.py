import pytest
import minplascalc as mpc


@pytest.fixture
def mixture_simple():
    c = mpc.Mixture(species=[mpc.species_from_name(sp) for sp in 
                             ['O2','O2+','O','O+','O++']],
                    x0=[1,0,0,0,0],
                    T=1000, P=101325,
                    gfe_ni0=1e20, gfe_reltol=1e-10, gfe_maxiter=1000)
    return c


LOW_T, MID_T, HIGH_T = 1000, 10000, 25000
LOW_P, MID_P, HIGH_P = 10132.5, 101325, 1013250

@pytest.mark.parametrize("T, P, result, tol", [
    (MID_T, MID_P,   0.01911388, 1e-7),
    (LOW_T, LOW_P,   0.03899566, 1e-7),
    (HIGH_T, LOW_P,  0.00036478, 1e-7),
    (LOW_T, HIGH_P,  3.89956587, 1e-7),
    (HIGH_T, HIGH_P, 0.04053962, 1e-7),
])
def test_density(mixture_simple, T, P, result, tol):
    mixture_simple.T = T
    mixture_simple.P = P

    thisresult = mixture_simple.calculate_density()

    assert thisresult == pytest.approx(result, abs=tol)


@pytest.mark.parametrize("T, P, result, tol", [
    (MID_T, MID_P, 3249.165, 1e-2),
    (LOW_T, LOW_P, 1081.252, 1e-2),
    (HIGH_T, LOW_P, 23193.92, 1e-1),
    (LOW_T, HIGH_P, 1081.252, 1e-2),
    (HIGH_T, HIGH_P, 5829.868, 1e-2),
])
def test_heat_capacity(mixture_simple, T, P, result, tol):
    mixture_simple.T = T
    mixture_simple.P = P

    thisresult = mixture_simple.calculate_heat_capacity()

    assert thisresult == pytest.approx(result, abs=tol)


@pytest.mark.parametrize("T, P, result, tol", [
    (MID_T, MID_P, 1.5633632e7, 1e1),
    (LOW_T, LOW_P, -1.455147e7, 1e1),
    (HIGH_T, LOW_P, 1.893144e8, 1e2),
    (LOW_T, HIGH_P, -1.455147e7, 1e1),
    (HIGH_T, HIGH_P, 1.483885e8, 1e2),
])
def test_enthalpy(mixture_simple, T, P, result, tol):
    mixture_simple.T = T
    mixture_simple.P = P
    
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
        mixture_simple.species = [mpc.species_from_name(sp) 
                                  for sp in ['O','O+']]


def test_species_item_exception(mixture_simple):
    with pytest.raises(TypeError):
        mixture_simple.species[0] = mpc.species_from_name('O')


def test_x0_item_exception(mixture_simple):
    with pytest.raises(TypeError):
        mixture_simple.x0[0] = 0.5
