"""Exercise the warm-started temperature-sweep usage pattern (issue #94).

The example notebooks repeatedly reassign ``mixture.T`` and recalculate on
the same object as part of a sweep, but until now that usage pattern was
untested: existing tests only ever solve a mixture once per fixture
instance. These tests drive a persistent mixture object through a full
temperature sweep, both ascending and descending, to make sure the warm
start from the previous solution's composition converges correctly at
every step. A few points in each sweep are also cross-checked against a
freshly constructed (cold-started) mixture at the same conditions, to catch
cases where a warm start converges to a plausible-looking but wrong answer.
"""

import numpy as np
import pytest

import minplascalc as mpc

SIMPLE_SPECIES = ["O2", "O2+", "O", "O-", "O+", "O++"]
SIMPLE_X0 = [1, 0, 0, 0, 0, 0]
SIMPLE_P = 101325

COMPLEX_SPECIES = [
    "O2",
    "O2+",
    "O",
    "O+",
    "O++",
    "CO",
    "CO+",
    "C",
    "C+",
    "C++",
    "SiO",
    "SiO+",
    "Si",
    "Si+",
    "Si++",
]
COMPLEX_X0 = [0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0]
COMPLEX_P = 101325

CROSS_CHECK_RTOL = 1e-6


@pytest.fixture
def mixture_simple():
    return mpc.mixture.lte_from_names(
        SIMPLE_SPECIES, x0=SIMPLE_X0, T=1000, P=SIMPLE_P
    )


@pytest.fixture
def mixture_complex():
    return mpc.mixture.lte_from_names(
        COMPLEX_SPECIES, x0=COMPLEX_X0, T=10000, P=COMPLEX_P
    )


def _assert_physically_sane(mixture):
    n_i = mixture.calculate_composition()
    assert np.all(np.isfinite(n_i))
    assert np.all(n_i >= 0)

    assert np.isfinite(mixture.calculate_density())
    assert mixture.calculate_density() > 0

    assert np.isfinite(mixture.calculate_enthalpy())


def _assert_matches_cold_start(mixture, species, x0, P, T):
    """Warm-started results should match a freshly solved mixture."""
    cold = mpc.mixture.lte_from_names(species, x0=x0, T=T, P=P)

    assert mixture.calculate_composition() == pytest.approx(
        cold.calculate_composition(), rel=CROSS_CHECK_RTOL
    )
    assert mixture.calculate_density() == pytest.approx(
        cold.calculate_density(), rel=CROSS_CHECK_RTOL
    )
    assert mixture.calculate_enthalpy() == pytest.approx(
        cold.calculate_enthalpy(), rel=CROSS_CHECK_RTOL
    )


def _check_indices(temperatures):
    """First, middle and last indices of a sweep, for cold-start checks."""
    return {0, len(temperatures) // 2, len(temperatures) - 1}


@pytest.mark.parametrize(
    "temperatures",
    [
        np.linspace(1000, 25000, 30),
        np.linspace(25000, 1000, 30),
    ],
    ids=["ascending", "descending"],
)
def test_temperature_sweep_simple(mixture_simple, temperatures):
    check_indices = _check_indices(temperatures)
    for i, T in enumerate(temperatures):
        mixture_simple.T = T
        _assert_physically_sane(mixture_simple)
        if i in check_indices:
            _assert_matches_cold_start(
                mixture_simple, SIMPLE_SPECIES, SIMPLE_X0, SIMPLE_P, T
            )


@pytest.mark.parametrize(
    "temperatures",
    [
        np.linspace(1000, 25000, 30),
        np.linspace(25000, 1000, 30),
    ],
    ids=["ascending", "descending"],
)
def test_temperature_sweep_complex(mixture_complex, temperatures):
    check_indices = _check_indices(temperatures)
    for i, T in enumerate(temperatures):
        mixture_complex.T = T
        _assert_physically_sane(mixture_complex)
        if i in check_indices:
            _assert_matches_cold_start(
                mixture_complex, COMPLEX_SPECIES, COMPLEX_X0, COMPLEX_P, T
            )
