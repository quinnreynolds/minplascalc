from typing import TYPE_CHECKING

import numpy as np

from minplascalc.transport.collision_cross_section import Qij_mix
from minplascalc.transport.q_matrix_jit import (
    q00_jit,
    q01_jit,
    q02_jit,
    q03_jit,
    q10_jit,
    q11_jit,
    q12_jit,
    q13_jit,
    q20_jit,
    q21_jit,
    q22_jit,
    q23_jit,
    q30_jit,
    q31_jit,
    q32_jit,
    q33_jit,
)
from minplascalc.units import Units

if TYPE_CHECKING:
    from minplascalc.mixture import LTE

u = Units()


### q-matrix calculations ######################################################


def q(mixture: "LTE") -> np.ndarray:
    """Calculate the q-matrix for a mixture of species.

    Parameters
    ----------
    mixture : LTE
        Mixture of species.

    Returns
    -------
    np.ndarray
        q-matrix.

    Notes
    -----
    The various elements of the q-matrix are calculated from the appendix of [Devoto1966]_.
    """
    nb_species = len(mixture.species)
    number_densities = mixture.calculate_composition()  # m^-3
    masses = np.array([species.molarmass / u.N_a for species in mixture.species])  # kg

    # Calculate the collision integrals for the mixture.
    Q11 = Qij_mix(mixture, 1, 1)
    Q12 = Qij_mix(mixture, 1, 2)
    Q13 = Qij_mix(mixture, 1, 3)
    Q14 = Qij_mix(mixture, 1, 4)
    Q15 = Qij_mix(mixture, 1, 5)
    Q16 = Qij_mix(mixture, 1, 6)
    Q17 = Qij_mix(mixture, 1, 7)
    Q22 = Qij_mix(mixture, 2, 2)
    Q23 = Qij_mix(mixture, 2, 3)
    Q24 = Qij_mix(mixture, 2, 4)
    Q25 = Qij_mix(mixture, 2, 5)
    Q26 = Qij_mix(mixture, 2, 6)
    Q33 = Qij_mix(mixture, 3, 3)
    Q34 = Qij_mix(mixture, 3, 4)
    Q35 = Qij_mix(mixture, 3, 5)
    Q44 = Qij_mix(mixture, 4, 4)

    # Equation A3 of [Devoto1966]_.
    q00 = q00_jit(nb_species, number_densities, masses, Q11)

    # Equation A4 of [Devoto1966]_.
    q01 = q01_jit(nb_species, number_densities, masses, Q11, Q12)

    # Equation A6 of [Devoto1966]_.
    q11 = q11_jit(nb_species, number_densities, masses, Q11, Q12, Q13, Q22)

    # Equation A7 of [Devoto1966]_.
    q02 = q02_jit(nb_species, number_densities, masses, Q11, Q12, Q13)

    # Equation A9 of [Devoto1966]_.
    q12 = q12_jit(nb_species, number_densities, masses, Q11, Q12, Q13, Q14, Q22, Q23)

    # Equation A11 of [Devoto1966]_.
    q22 = q22_jit(
        nb_species,
        number_densities,
        masses,
        Q11,
        Q12,
        Q13,
        Q14,
        Q15,
        Q22,
        Q23,
        Q24,
        Q33,
    )

    # Equation A12 of [Devoto1966]_.
    q03 = q03_jit(nb_species, number_densities, masses, Q11, Q12, Q13, Q14)

    # Equation A14 of [Devoto1966]_.
    q13 = q13_jit(
        nb_species, number_densities, masses, Q11, Q12, Q13, Q14, Q15, Q22, Q23, Q24
    )

    # Equation A16 of [Devoto1966]_.
    q23 = q23_jit(
        nb_species,
        number_densities,
        masses,
        Q11,
        Q12,
        Q13,
        Q14,
        Q15,
        Q16,
        Q22,
        Q23,
        Q24,
        Q25,
        Q33,
        Q34,
    )

    # Equation A18 of [Devoto1966]_.
    q33 = q33_jit(
        nb_species,
        number_densities,
        masses,
        Q11,
        Q12,
        Q13,
        Q14,
        Q15,
        Q16,
        Q17,
        Q22,
        Q23,
        Q24,
        Q25,
        Q26,
        Q33,
        Q34,
        Q35,
        Q44,
    )

    # Equation A5 of [Devoto1966]_.
    q10 = q10_jit(nb_species, masses, q01)

    # Equation A8 of [Devoto1966]_.
    q20 = q20_jit(nb_species, masses, q02)

    # Equation A10 of [Devoto1966]_.
    q21 = q21_jit(nb_species, masses, q12)

    # Equation A13 of [Devoto1966]_.
    q30 = q30_jit(nb_species, masses, q03)

    # Equation A15 of [Devoto1966]_.
    q31 = q31_jit(nb_species, masses, q13)

    # Equation A17 of [Devoto1966]_.
    q32 = q32_jit(nb_species, masses, q23)

    # Combine the q-matrix elements into a single matrix.
    qq = np.zeros((4 * nb_species, 4 * nb_species))

    qq[0 * nb_species : 1 * nb_species, 0 * nb_species : 1 * nb_species] = q00
    qq[0 * nb_species : 1 * nb_species, 1 * nb_species : 2 * nb_species] = q01
    qq[0 * nb_species : 1 * nb_species, 2 * nb_species : 3 * nb_species] = q02
    qq[0 * nb_species : 1 * nb_species, 3 * nb_species : 4 * nb_species] = q03

    qq[1 * nb_species : 2 * nb_species, 0 * nb_species : 1 * nb_species] = q10
    qq[1 * nb_species : 2 * nb_species, 1 * nb_species : 2 * nb_species] = q11
    qq[1 * nb_species : 2 * nb_species, 2 * nb_species : 3 * nb_species] = q12
    qq[1 * nb_species : 2 * nb_species, 3 * nb_species : 4 * nb_species] = q13

    qq[2 * nb_species : 3 * nb_species, 0 * nb_species : 1 * nb_species] = q20
    qq[2 * nb_species : 3 * nb_species, 1 * nb_species : 2 * nb_species] = q21
    qq[2 * nb_species : 3 * nb_species, 2 * nb_species : 3 * nb_species] = q22
    qq[2 * nb_species : 3 * nb_species, 3 * nb_species : 4 * nb_species] = q23

    qq[3 * nb_species : 4 * nb_species, 0 * nb_species : 1 * nb_species] = q30
    qq[3 * nb_species : 4 * nb_species, 1 * nb_species : 2 * nb_species] = q31
    qq[3 * nb_species : 4 * nb_species, 2 * nb_species : 3 * nb_species] = q32
    qq[3 * nb_species : 4 * nb_species, 3 * nb_species : 4 * nb_species] = q33

    return qq
