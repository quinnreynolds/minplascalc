from typing import TYPE_CHECKING

import numpy as np

from minplascalc.transport.collision_cross_section import Qij_mix
from minplascalc.transport.q_hat_matrix_jit import (
    qhat00_jit,
    qhat01_jit,
    qhat10_jit,
    qhat11_jit,
)
from minplascalc.units import Units

if TYPE_CHECKING:
    from minplascalc.mixture import LTE

u = Units()


def qhat(mixture: "LTE") -> np.ndarray:
    """Calculate the qhat-matrix for a mixture of species.

    Parameters
    ----------
    mixture : LTE
        Mixture of species.

    Returns
    -------
    np.ndarray
        qhat-matrix.

    Notes
    -----
    The various elements of the qhat-matrix are calculated from the appendix of [Devoto1966]_, from
    equation A19 to A22.
    """
    nb_species = len(mixture.species)
    number_densities = mixture.calculate_composition()
    masses = np.array([sp.molarmass / u.N_a for sp in mixture.species])

    Q11 = Qij_mix(mixture, 1, 1)
    Q12 = Qij_mix(mixture, 1, 2)
    Q13 = Qij_mix(mixture, 1, 3)
    Q22 = Qij_mix(mixture, 2, 2)
    Q23 = Qij_mix(mixture, 2, 3)
    Q24 = Qij_mix(mixture, 2, 4)
    Q33 = Qij_mix(mixture, 3, 3)

    # Equation A19 of [Devoto1966]_.
    qhat00 = qhat00_jit(nb_species, number_densities, masses, Q11, Q22)

    # Equation A20 of [Devoto1966]_.
    qhat01 = qhat01_jit(nb_species, number_densities, masses, Q11, Q12, Q22, Q23)

    # Equation A22 of [Devoto1966]_.
    qhat11 = qhat11_jit(
        nb_species, number_densities, masses, Q11, Q12, Q13, Q22, Q23, Q24, Q33
    )
    # Equation A21 of [Devoto1966]_.
    qhat10 = qhat10_jit(nb_species, masses, qhat01)

    qq = np.zeros((2 * nb_species, 2 * nb_species))

    qq[0 * nb_species : 1 * nb_species, 0 * nb_species : 1 * nb_species] = qhat00
    qq[0 * nb_species : 1 * nb_species, 1 * nb_species : 2 * nb_species] = qhat01

    qq[1 * nb_species : 2 * nb_species, 0 * nb_species : 1 * nb_species] = qhat10
    qq[1 * nb_species : 2 * nb_species, 1 * nb_species : 2 * nb_species] = qhat11

    return qq
