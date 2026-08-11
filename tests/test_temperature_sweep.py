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

from typing import NamedTuple

import numpy as np
import pytest

import minplascalc as mpc


class MixtureCase(NamedTuple):
    """A mixture specification to sweep, and where to start the sweep from."""

    name: str
    species: list[str]
    x0: list[float]
    P: float
    T0: float

    def new_mixture(self):
        """Build a freshly constructed (cold-started) mixture at ``T0``."""
        return mpc.mixture.lte_from_names(
            self.species, x0=self.x0, T=self.T0, P=self.P
        )


SIMPLE = MixtureCase(
    name="simple",
    species=["O2", "O2+", "O", "O-", "O+", "O++"],
    x0=[1, 0, 0, 0, 0, 0],
    P=101325,
    T0=1000,
)

COMPLEX = MixtureCase(
    name="complex",
    species=[
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
    ],
    x0=[0, 0, 0, 0, 0, 0.5, 0, 0, 0, 0, 0.5, 0, 0, 0, 0],
    P=101325,
    T0=10000,
)

CROSS_CHECK_RTOL = 1e-6


def _assert_physically_sane(mixture):
    n_i = mixture.calculate_composition()
    assert np.all(np.isfinite(n_i))
    assert np.all(n_i >= 0)

    assert np.isfinite(mixture.calculate_density())
    assert mixture.calculate_density() > 0

    assert np.isfinite(mixture.calculate_enthalpy())


def _assert_matches_cold_start(mixture, case, T):
    """Warm-started results should match a freshly solved mixture."""
    cold = case.new_mixture()
    cold.T = T

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


@pytest.mark.parametrize("case", [SIMPLE, COMPLEX], ids=lambda case: case.name)
@pytest.mark.parametrize(
    "temperatures",
    [
        np.linspace(1000, 25000, 30),
        np.linspace(25000, 1000, 30),
    ],
    ids=["ascending", "descending"],
)
def test_temperature_sweep(case, temperatures):
    mixture = case.new_mixture()
    check_indices = _check_indices(temperatures)
    for i, T in enumerate(temperatures):
        mixture.T = T
        _assert_physically_sane(mixture)
        if i in check_indices:
            _assert_matches_cold_start(mixture, case, T)
