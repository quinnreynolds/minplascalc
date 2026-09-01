import pytest

import minplascalc as mpc


@pytest.fixture
def mixture_simple():
    c = mpc.mixture.lte_from_names(
        ["O2", "O", "C", "CO"],
        x0=[0.5, 0, 0.5, 0],
        T=3000,
        P=101325,
        electrons_yn=False,
    )
    return c


@pytest.fixture
def mixture_complex():
    c = mpc.mixture.lte_from_names(
        ["SiO", "Si", "CO", "C"],
        x0=[0.5, 0, 0.5, 0],
        T=5000,
        P=101325,
        electrons_yn=False,
    )
    return c


LOW_T, MID_T, HIGH_T = 1000, 3000, 5000
LOW_P, MID_P, HIGH_P = 10132.5, 101325, 1013250
LOW_X0 = [0.1, 0, 0.9, 0]
MID_X0 = [0.5, 0, 0.5, 0]
HIGH_X0 = [0.9, 0, 0.1, 0]


def test_type(mixture_simple):
    assert isinstance(mixture_simple, mpc.mixture.LTEWithoutElectrons)
    assert isinstance(mixture_simple, mpc.mixture.LTE)


def test_no_electrons_in_species(mixture_simple):
    species_names = [sp.name for sp in mixture_simple.species]
    assert "e" not in species_names


def test_electrical_conductivity_is_zero(mixture_simple):
    assert mixture_simple.calculate_electrical_conductivity() == 0.0


@pytest.mark.parametrize(
    "T, P, result, tol",
    [
        (MID_T, MID_P, 0.11386602, 1e-7),
        (LOW_T, LOW_P, 0.03575511, 1e-7),
        (HIGH_T, LOW_P, 0.00535685, 1e-7),
        (LOW_T, HIGH_P, 3.57551082, 1e-7),
        (HIGH_T, HIGH_P, 0.54916274, 1e-7),
    ],
)
def test_density_simple(mixture_simple, T, P, result, tol):
    mixture_simple.T = T
    mixture_simple.P = P

    thisresult = mixture_simple.calculate_density()

    assert thisresult == pytest.approx(result, abs=tol)


@pytest.mark.parametrize(
    "x0, result, tol",
    [
        (LOW_X0, 0.07218760, 1e-7),
        (MID_X0, 0.08785933, 1e-7),
        (HIGH_X0, 0.10353106, 1e-7),
    ],
)
def test_density_complex(mixture_complex, x0, result, tol):
    mixture_complex.x0 = x0

    thisresult = mixture_complex.calculate_density()

    assert thisresult == pytest.approx(result, abs=tol)


@pytest.mark.parametrize(
    "T, P, result, tol",
    [
        (MID_T, MID_P, 3741.395, 1e-2),
        (LOW_T, LOW_P, 1143.126, 1e-2),
        (HIGH_T, LOW_P, 1742.402, 1e-2),
        (LOW_T, HIGH_P, 1143.126, 1e-2),
        (HIGH_T, HIGH_P, 2451.842, 1e-2),
    ],
)
def test_heat_capacity_simple(mixture_simple, T, P, result, tol):
    mixture_simple.T = T
    mixture_simple.P = P

    thisresult = mixture_simple.calculate_heat_capacity()

    assert thisresult == pytest.approx(result, abs=tol)


@pytest.mark.parametrize(
    "x0, result, tol",
    [
        (LOW_X0, 1254.924, 1e-2),
        (MID_X0, 1033.044, 1e-2),
        (HIGH_X0, 878.337, 1e-2),
    ],
)
def test_heat_capacity_complex(mixture_complex, x0, result, tol):
    mixture_complex.x0 = x0

    thisresult = mixture_complex.calculate_heat_capacity()

    assert thisresult == pytest.approx(result, abs=tol)


@pytest.mark.parametrize(
    "T, P, result, tol",
    [
        (MID_T, MID_P, 9.7707373124692715e-05, 1e-9),
        (LOW_T, LOW_P, 4.7692984929475872e-05, 1e-9),
        (HIGH_T, LOW_P, 1.4123464934312000e-04, 1e-9),
        (LOW_T, HIGH_P, 4.7692984929348614e-05, 1e-9),
        (HIGH_T, HIGH_P, 1.4101162926984635e-04, 1e-9),
    ],
)
def test_viscosity_simple(mixture_simple, T, P, result, tol):
    mixture_simple.T = T
    mixture_simple.P = P

    thisresult = mixture_simple.calculate_viscosity()

    assert thisresult == pytest.approx(result, abs=tol)


@pytest.mark.parametrize(
    "x0, result, tol",
    [
        (LOW_X0, 1.2699613997787544e-04, 1e-9),
        (MID_X0, 1.2435092204444808e-04, 1e-9),
        (HIGH_X0, 1.2192724970350315e-04, 1e-9),
    ],
)
def test_viscosity_complex(mixture_complex, x0, result, tol):
    mixture_complex.x0 = x0

    thisresult = mixture_complex.calculate_viscosity()

    assert thisresult == pytest.approx(result, abs=tol)


@pytest.mark.parametrize(
    "T, P, result, tol",
    [
        (MID_T, MID_P, 6.800857920705657e-01, 1e-5),
        (LOW_T, LOW_P, 5.903704964428261e-02, 1e-7),
        (HIGH_T, LOW_P, 2.4280416800390991e-01, 1e-5),
        (LOW_T, HIGH_P, 5.9037026025524894e-02, 1e-7),
        (HIGH_T, HIGH_P, 4.855691838684442e-01, 1e-5),
    ],
)
def test_thermal_conductivity_simple(mixture_simple, T, P, result, tol):
    mixture_simple.T = T
    mixture_simple.P = P

    thisresult = mixture_simple.calculate_thermal_conductivity()

    assert thisresult == pytest.approx(result, abs=tol)


@pytest.mark.parametrize(
    "x0, result, tol",
    [
        (LOW_X0, 1.4006276613762003e-01, 1e-5),
        (MID_X0, 1.1822894617782748e-01, 1e-5),
        (HIGH_X0, 9.237671740729556e-02, 1e-5),
    ],
)
def test_thermal_conductivity_complex(mixture_complex, x0, result, tol):
    mixture_complex.x0 = x0

    thisresult = mixture_complex.calculate_thermal_conductivity()

    assert thisresult == pytest.approx(result, abs=tol)
