from typing import TYPE_CHECKING

import numpy as np

from minplascalc.transport.collision_cross_section import Qij_mix
from minplascalc.transport.potential_functions import delta
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
    q00 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            sum_val = 0.0
            for l in range(nb_species):
                term1 = (
                    number_densities[l]
                    * masses[i] ** (1 / 2)
                    / (masses[i] + masses[l]) ** (1 / 2)
                )
                term2 = number_densities[i] * (masses[l] / masses[j]) ** (1 / 2) * (
                    delta(i, j) - delta(j, l)
                ) - number_densities[j] * (masses[l] * masses[j]) ** (1 / 2) / masses[
                    i
                ] * (1 - delta(i, l))
                sum_val += term1 * Q11[i, l] * term2
            q00[i, j] = 8 * sum_val

    # Equation A4 of [Devoto1966]_.
    q01 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            sum_val = 0.0
            for l in range(nb_species):
                term1 = (
                    number_densities[l]
                    * masses[l] ** (3 / 2)
                    / (masses[i] + masses[l]) ** (3 / 2)
                )
                term2 = (delta(i, j) - delta(j, l)) * (
                    5 / 2 * Q11[i, l] - 3 * Q12[i, l]
                )
                sum_val += term1 * term2
            q01[i, j] = (
                8 * number_densities[i] * (masses[i] / masses[j]) ** (3 / 2) * sum_val
            )

    # Equation A6 of [Devoto1966]_.
    q11 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            sum_val = 0.0
            for l in range(nb_species):
                term1 = (
                    number_densities[l]
                    * masses[l] ** (1 / 2)
                    / (masses[i] + masses[l]) ** (5 / 2)
                )
                term2 = (delta(i, j) - delta(j, l)) * (
                    5 / 4 * (6 * masses[j] ** 2 + 5 * masses[l] ** 2) * Q11[i, l]
                    - 15 * masses[l] ** 2 * Q12[i, l]
                    + 12 * masses[l] ** 2 * Q13[i, l]
                ) + (delta(i, j) + delta(j, l)) * 4 * masses[j] * masses[l] * Q22[i, l]
                sum_val += term1 * term2
            q11[i, j] = (
                8 * number_densities[i] * (masses[i] / masses[j]) ** (3 / 2) * sum_val
            )

    # Equation A7 of [Devoto1966]_.
    q02 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            sum_val = 0.0
            for l in range(nb_species):
                term1 = (
                    number_densities[l]
                    * masses[l] ** (5 / 2)
                    / (masses[i] + masses[l]) ** (5 / 2)
                )
                term2 = (delta(i, j) - delta(j, l)) * (
                    35 / 8 * Q11[i, l] - 21 / 2 * Q12[i, l] + 6 * Q13[i, l]
                )
                sum_val += term1 * term2
            q02[i, j] = (
                8 * number_densities[i] * (masses[i] / masses[j]) ** (5 / 2) * sum_val
            )

    # Equation A9 of [Devoto1966]_.
    q12 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            sum_val = 0.0
            for l in range(nb_species):
                term1 = (
                    number_densities[l]
                    * masses[l] ** (3 / 2)
                    / (masses[i] + masses[l]) ** (7 / 2)
                )
                term2 = (delta(i, j) - delta(j, l)) * (
                    35 / 16 * (12 * masses[j] ** 2 + 5 * masses[l] ** 2) * Q11[i, l]
                    - 63 / 2 * (masses[j] ** 2 + 5 / 4 * masses[l] ** 2) * Q12[i, l]
                    + 57 * masses[l] ** 2 * Q13[i, l]
                    - 30 * masses[l] ** 2 * Q14[i, l]
                ) + (delta(i, j) + delta(j, l)) * (
                    14 * masses[j] * masses[l] * Q22[i, l]
                    - 16 * masses[j] * masses[l] * Q23[i, l]
                )
                sum_val += term1 * term2
            q12[i, j] = (
                8 * number_densities[i] * (masses[i] / masses[j]) ** (5 / 2) * sum_val
            )

    # Equation A11 of [Devoto1966]_.
    q22 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            sum_val = 0.0
            for l in range(nb_species):
                term1 = (
                    number_densities[l]
                    * masses[l] ** (1 / 2)
                    / (masses[i] + masses[l]) ** (9 / 2)
                )
                term2 = (delta(i, j) - delta(j, l)) * (
                    35
                    / 64
                    * (
                        40 * masses[j] ** 4
                        + 168 * (masses[j] * masses[l]) ** 2
                        + 35 * masses[l] ** 4
                    )
                    * Q11[i, l]
                    - 21
                    / 8
                    * masses[l] ** 2
                    * (84 * masses[j] ** 2 + 35 * masses[l] ** 2)
                    * Q12[i, l]
                    + 3
                    / 2
                    * masses[l] ** 2
                    * (108 * masses[j] ** 2 + 133 * masses[l] ** 2)
                    * Q13[i, l]
                    - 210 * masses[l] ** 4 * Q14[i, l]
                    + 90 * masses[l] ** 4 * Q15[i, l]
                    + 24
                    * (masses[j] * masses[l]) ** 2
                    * Q33[i, j]  # TODO: Error here? Should be Q33[i, l]?
                ) + (delta(i, j) + delta(j, l)) * (
                    7
                    * masses[j]
                    * masses[l]
                    * (4 * (masses[j] ** 2 + 7 * masses[l] ** 2))
                    * Q22[i, l]
                    - 112 * masses[j] * masses[l] ** 3 * Q23[i, l]
                    + 80 * masses[j] * masses[l] ** 3 * Q24[i, l]
                )
                sum_val += term1 * term2
            q22[i, j] = (
                8 * number_densities[i] * (masses[i] / masses[j]) ** (5 / 2) * sum_val
            )

    # Equation A12 of [Devoto1966]_.
    q03 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            sum_val = 0.0
            for l in range(nb_species):
                term1 = (
                    number_densities[l]
                    * masses[l] ** (7 / 2)
                    / (masses[i] + masses[l]) ** (7 / 2)
                )
                term2 = (delta(i, j) - delta(j, l)) * (
                    105 / 16 * Q11[i, l]
                    - 189 / 8 * Q12[i, l]
                    + 27 * Q13[i, l]
                    - 10 * Q14[i, l]
                )
                sum_val += term1 * term2
            q03[i, j] = (
                8 * number_densities[i] * (masses[i] / masses[j]) ** (7 / 2) * sum_val
            )

    # Equation A14 of [Devoto1966]_.
    q13 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            sum_val = 0.0
            for l in range(nb_species):
                term1 = (
                    number_densities[l]
                    * masses[l] ** (5 / 2)
                    / (masses[i] + masses[l]) ** (9 / 2)
                )
                term2 = (delta(i, j) - delta(j, l)) * (
                    105 / 32 * (18 * masses[j] ** 2 + 5 * masses[l] ** 2) * Q11[i, l]
                    - 63 / 4 * (9 * masses[j] ** 2 + 5 * masses[l] ** 2) * Q12[i, l]
                    + 81 * (masses[j] ** 2 + 2 * masses[l] ** 2) * Q13[i, l]
                    - 160 * masses[l] ** 2 * Q14[i, l]
                    + 60 * masses[l] ** 2 * Q15[i, l]
                ) + (delta(i, j) + delta(j, l)) * masses[j] * masses[l] * (
                    63 / 2 * Q22[i, l] - 72 * Q23[i, l] + 40 * Q24[i, l]
                )
                sum_val += term1 * term2
            q13[i, j] = (
                8 * number_densities[i] * (masses[i] / masses[j]) ** (7 / 2) * sum_val
            )

    # Equation A16 of [Devoto1966]_.
    q23 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            sum_val = 0.0
            for l in range(nb_species):
                term1 = (
                    number_densities[l]
                    * masses[l] ** (3 / 2)
                    / (masses[i] + masses[l]) ** (11 / 2)
                )
                term2 = (delta(i, j) - delta(j, l)) * (
                    105
                    / 128
                    * (
                        120 * masses[j] ** 4
                        + 252 * (masses[j] * masses[l]) ** 2
                        + 35 * masses[l] ** 4
                    )
                    * Q11[i, l]
                    - 63
                    / 64
                    * (
                        120 * masses[j] ** 4
                        + 756 * (masses[j] * masses[l]) ** 2
                        + 175 * masses[l] ** 4
                    )
                    * Q12[i, l]
                    + 9
                    / 4
                    * masses[l] ** 2
                    * (450 * masses[j] ** 2 + 217 * masses[l] ** 2)
                    * Q13[i, l]
                    + 5  # TODO: Error here? Should be a minus?
                    / 2
                    * masses[l] ** 2
                    * (198 * masses[j] ** 2 + 301 * masses[l] ** 2)
                    * Q14[i, l]
                    + 615 * masses[l] ** 4 * Q15[i, l]
                    - 210 * masses[l] ** 4 * Q16[i, l]
                    + 108
                    * (masses[j] * masses[l]) ** 2
                    * Q33[i, j]  # TODO: Error here? Should be Q33[i, l]?
                    - 120
                    * (masses[j] * masses[l]) ** 2
                    * Q34[i, j]  # TODO: Error here? Should be Q34[i, l]?
                ) + (delta(i, j) + delta(j, l)) * (
                    63
                    / 4
                    * masses[j]
                    * masses[l]
                    * (8 * (masses[j] ** 2 + 7 * masses[l] ** 2))
                    * Q22[i, l]
                    - 18
                    * masses[j]
                    * masses[l]
                    * (8 * masses[j] ** 2 + 21 * masses[l] ** 2)
                    * Q23[i, l]
                    + 500 * masses[j] * masses[l] ** 3 * Q24[i, l]
                    - 240 * masses[j] * masses[l] ** 3 * Q25[i, l]
                )
                sum_val += term1 * term2
            q23[i, j] = (
                8 * number_densities[i] * (masses[i] / masses[j]) ** (7 / 2) * sum_val
            )

    # TODO: Equation A18 of [Devoto1966]_.
    q33 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            sum_val = 0.0
            for l in range(nb_species):
                term1 = (
                    number_densities[l]
                    * masses[l] ** (1 / 2)
                    / (masses[i] + masses[l]) ** (13 / 2)
                )
                term2 = (delta(i, j) - delta(j, l)) * (
                    105
                    / 256
                    * (
                        112 * masses[j] ** 6
                        + 1080 * masses[j] ** 4 * masses[l] ** 2
                        + 1134 * masses[j] ** 2 * masses[l] ** 4
                        + 105 * masses[l] ** 6
                    )
                    * Q11[i, l]
                    - 567
                    / 64
                    * masses[l] ** 2
                    * (
                        120 * masses[j] ** 4
                        + 252 * (masses[j] * masses[l]) ** 2
                        + 35 * masses[l] ** 4
                    )
                    * Q12[i, l]
                    + 27
                    / 16
                    * masses[l] ** 2
                    * (
                        440 * masses[j] ** 4
                        + 2700 * (masses[j] * masses[l]) ** 2
                        + 651 * masses[l] ** 4
                    )
                    * Q13[i, l]
                    + 15  # TODO: Error here? Should be a minus?
                    / 2
                    * masses[l] ** 4
                    * (594 * masses[j] ** 2 + 301 * masses[l] ** 2)
                    * Q14[i, l]
                    + 135
                    / 2
                    * masses[l] ** 4
                    * (26 * masses[j] ** 2 + 41 * masses[l] ** 2)
                    * Q15[i, l]
                    - 1890 * masses[l] ** 6 * Q16[i, l]
                    - 560  # TODO: Error here? Should be a plus?
                    * masses[l] ** 6
                    * Q17[i, l]
                    + 18
                    * (masses[j] * masses[l]) ** 2
                    * (10 * masses[j] ** 2 + 27 * masses[l] ** 2)
                    * Q33[i, j]  # TODO: Error here? Should be Q33[i, l]?
                    - 1080
                    * masses[j] ** 2
                    * masses[l] ** 4
                    * Q34[i, j]  # TODO: Error here? Should be Q34[i, l]?
                    + 720
                    * masses[j] ** 2
                    * masses[l] ** 4
                    * Q35[i, j]  # TODO: Error here? Should be Q35[i, l]?
                ) + (delta(i, j) + delta(j, l)) * (
                    189
                    / 16
                    * masses[j]
                    * masses[l]
                    * (
                        8 * masses[j] ** 4
                        + 48 * (masses[j] * masses[l]) ** 2
                        + 21 * masses[l] ** 4
                    )
                    * Q22[i, l]
                    - 162
                    * masses[j]
                    * masses[l] ** 3
                    * (8 * masses[j] ** 2 + 7 * masses[l] ** 2)
                    * Q23[i, l]
                    + 10
                    * masses[j]
                    * masses[l] ** 3
                    * (88 * masses[j] ** 2 + 225 * masses[l] ** 2)
                    * Q24[i, l]
                    - 2160 * masses[j] * masses[l] ** 5 * Q25[i, l]
                    + 840 * masses[j] * masses[l] ** 5 * Q26[i, l]
                    + 64 * (masses[j] * masses[l]) ** 3 * Q44[i, l]
                )
                sum_val += term1 * term2
            q33[i, j] = (
                8 * number_densities[i] * (masses[i] / masses[j]) ** (7 / 2) * sum_val
            )

    # Equation A5 of [Devoto1966]_.
    q10 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            q10[i, j] = masses[j] / masses[i] * q01[i, j]

    # Equation A8 of [Devoto1966]_.
    q20 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            q20[i, j] = (masses[j] / masses[i]) ** 2 * q02[i, j]

    # Equation A10 of [Devoto1966]_.
    q21 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            q21[i, j] = masses[j] / masses[i] * q12[i, j]

    # Equation A13 of [Devoto1966]_.
    q30 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            q30[i, j] = (masses[j] / masses[i]) ** 3 * q03[i, j]

    # Equation A15 of [Devoto1966]_.
    q31 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            q31[i, j] = (masses[j] / masses[i]) ** 2 * q13[i, j]

    # Equation A17 of [Devoto1966]_.
    q32 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            q32[i, j] = masses[j] / masses[i] * q23[i, j]

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
    qhat00 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            sum_val = 0.0
            for l in range(nb_species):
                term1 = (
                    number_densities[l]
                    * masses[l] ** (1 / 2)
                    / (masses[i] + masses[l]) ** (3 / 2)
                )
                term2 = (delta(i, j) - delta(j, l)) * 10 / 3 * masses[j] * Q11[i, l] + (
                    delta(i, j) + delta(j, l)
                ) * 2 * masses[l] * Q22[i, l]
                sum_val += term1 * term2
            qhat00[i, j] = 8 * number_densities[i] * (masses[i] / masses[j]) * sum_val

    # Equation A20 of [Devoto1966]_.
    qhat01 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            sum_val = 0.0
            for l in range(nb_species):
                term1 = (
                    number_densities[l]
                    * masses[l] ** (3 / 2)
                    / (masses[i] + masses[l]) ** (5 / 2)
                )
                term2 = (delta(i, j) - delta(j, l)) * masses[j] * (
                    35 / 3 * Q11[i, l] - 14 * Q12[i, l]
                ) + (delta(i, j) + delta(j, l)) * masses[l] * (
                    7 * Q22[i, l] - 8 * Q23[i, l]
                )
                sum_val += term1 * term2
            qhat01[i, j] = (
                8 * number_densities[i] * (masses[i] / masses[j]) ** 2 * sum_val
            )

    # Equation A22 of [Devoto1966]_.
    qhat11 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            sum_val = 0.0
            for l in range(nb_species):
                term1 = (
                    number_densities[l]
                    * masses[l] ** (1 / 2)
                    / (masses[i] + masses[l]) ** (7 / 2)
                )
                term2 = (delta(i, j) - delta(j, l)) * masses[j] * (
                    1 / 6 * (140 * masses[j] ** 2 + 245 * masses[l] ** 2) * Q11[i, l]
                    - masses[l] ** 2
                    * (98 * Q12[i, l] - 64 * Q13[i, l] - 24 * Q33[i, l])
                ) + (delta(i, j) + delta(j, l)) * masses[l] * (
                    1 / 6 * (154 * masses[j] ** 2 + 147 * masses[l] ** 2) * Q22[i, l]
                    - masses[l] ** 2 * (56 * Q23[i, l] - 40 * Q24[i, l])
                )
                sum_val += term1 * term2
            qhat11[i, j] = (
                8 * number_densities[i] * (masses[i] / masses[j]) ** 2 * sum_val
            )

    # Equation A21 of [Devoto1966]_.
    qhat10 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            qhat10[i, j] = masses[j] / masses[i] * qhat01[i, j]

    qq = np.zeros((2 * nb_species, 2 * nb_species))

    qq[0 * nb_species : 1 * nb_species, 0 * nb_species : 1 * nb_species] = qhat00
    qq[0 * nb_species : 1 * nb_species, 1 * nb_species : 2 * nb_species] = qhat01

    qq[1 * nb_species : 2 * nb_species, 0 * nb_species : 1 * nb_species] = qhat10
    qq[1 * nb_species : 2 * nb_species, 1 * nb_species : 2 * nb_species] = qhat11

    return qq


### Transport property calculations ############################################


def Dij(mixture: "LTE") -> np.ndarray:
    r"""Diffusion coefficients.

    Diffusion coefficents, calculation per [Devoto1966]_ (eq. 3 and eq. 6).
    Fourth-order approximation.

    Parameters
    ----------
    mixture : LTE
        Mixture of species.

    Returns
    -------
    np.ndarray
        Diffusion coefficients.

    Notes
    -----
    The diffusion coefficients are given by equation 3 of [Devoto1966]_:

    .. math::

        D_{ij} = \frac{\rho n_i}{2 n_{\text{tot}} m_j} \sqrt{\frac{2 k_B T}{m_i}} c_{i 0}^{j i}

    where:

    - :math:`D_{ij}` is the diffusion coefficient between species :math:`i` and :math:`j`,
    - :math:`\rho` is the density of the mixture,
    - :math:`n_i` is the number density of species :math:`i`,
    - :math:`n_{\text{tot}}` is the total number density of the mixture,
    - :math:`m_j` is the molar mass of species :math:`j`,
    - :math:`k_B` is the Boltzmann constant,
    - :math:`T` is the temperature of the mixture,


    The elements of :math:`c_{i 0}^{j i}` are given by equation 6 of [Devoto1966]_:

    .. math::

        \begin{aligned}
            & \sum_{j=1}^\nu \sum_{p=0}^M q_{i j}^{m p} c_{j p}^{h k}
                = 3 \pi^{\frac{1}{2}}\left(\delta_{i k}-\delta_{i h}\right) \delta_{m 0} \\
            & \quad(i=1,2, \cdots \nu ; m=0, \cdots M)
        \end{aligned}

    where:

    - :math:`\nu` is the number of species in the mixture,
    - :math:`M` is the order of the approximation (:math:`M=3` in this case, and goes from 0 to 3 so
      that a fourth-order approximation is used),
    - :math:`q_{i j}^{m p}` are the elements of the :math:`q`-matrix.


    TODO: Add how the code works.
    TODO: Why not use equation 8?
    """
    nb_species = len(mixture.species)
    number_densities = mixture.calculate_composition()  # m^-3
    n_tot = np.sum(number_densities)  # m^-3
    masses = np.array([sp.molarmass / u.N_a for sp in mixture.species])  # kg
    rho = mixture.calculate_density()  # kg/m^3

    diffusion_matrix = np.zeros((nb_species, nb_species))

    qq = q(mixture)  # Size (4*nb_species, 4*nb_species)
    # qq = q(mixture)[:nb_species, :nb_species]
    invq = np.linalg.inv(qq)
    bvec = np.zeros(4 * nb_species)  # 4 for 4th order approximation
    # bvec = np.zeros(nb_species)
    for i in range(nb_species):
        for j in range(nb_species):
            # TODO: Check if this is correct
            # Equation 6 of [Devoto1966]_.
            dij = np.array([delta(h, i) - delta(h, j) for h in range(0, nb_species)])
            bvec[:nb_species] = 3 * np.sqrt(u.pi) * dij
            cflat = invq.dot(bvec)
            cip = cflat.reshape(4, nb_species)
            # cip = cflat.reshape(1, nb_species)

            # Diffusion coefficient, equation 3 of [Devoto1966]_.
            diffusion_matrix[i, j] = (
                rho
                * number_densities[i]
                / (2 * n_tot * masses[j])
                * np.sqrt(2 * u.k_b * mixture.T / masses[i])
                * cip[0, i]
            )

    return diffusion_matrix


def DTi(mixture: "LTE") -> float:
    r"""Thermal diffusion coefficients.

    Thermal diffusion coefficents, calculation per [Devoto1966]_ (eq. 4 and eq. 5).
    Fourth-order approximation.

    Parameters
    ----------
    mixture : LTE
        Mixture of species.

    Returns
    -------
    float
        Thermal diffusion coefficients.

    Notes
    -----
    The thermal diffusion coefficients are given by equation 4 of [Devoto1966]_:

    .. math::

        D_i^T = \frac{1}{2} n_i m_i \sqrt{\frac{2 k_B T}{m_i}} a_{i 0}

    where:

    - :math:`D_i^T` is the thermal diffusion coefficient of species :math:`i`,
    - :math:`n_i` is the number density of species :math:`i`,
    - :math:`m_i` is the molar mass of species :math:`i`,
    - :math:`k_B` is the Boltzmann constant,
    - :math:`T` is the temperature of the mixture.


    The elements of :math:`a_{i 0}}` are given by equation 5 of [Devoto1966]_:

    .. math::

        \begin{aligned}
            & \sum_{j=1}^\nu \sum_{p=0}^M q_{i j}^{m p} a_{j p}
                =-\frac{15 \pi^{\frac{1}{2}} n_i}{2} \delta_{m 1} \\
            & \quad(i=1,2, \cdots \nu ; m=0, \cdots M)
        \end{aligned}

    where:

    - :math:`\nu` is the number of species in the mixture,
    - :math:`M` is the order of the approximation (:math:`M=3` in this case, and goes from 0 to 3 so
      that a fourth-order approximation is used),
    - :math:`q_{i j}^{m p}` are the elements of the :math:`q`-matrix.


    TODO: Add how the code works.
    TODO: Why not use equation 9?
    """
    nb_species = len(mixture.species)
    number_densities = mixture.calculate_composition()
    masses = np.array([sp.molarmass / u.N_a for sp in mixture.species])

    qq = q(mixture)
    invq = np.linalg.inv(qq)
    bvec = np.zeros(4 * nb_species)  # 4 for 4th order approximation
    # Only the first element is non-zero
    bvec[nb_species : 2 * nb_species] = -15 / 2 * np.sqrt(u.pi) * number_densities
    aflat = invq.dot(bvec)
    aip = aflat.reshape(4, nb_species)

    return (
        0.5
        * number_densities
        * masses
        * np.sqrt(2 * u.k_b * mixture.T / masses)
        * aip[0]
    )


def viscosity(mixture: "LTE") -> float:
    r"""Viscosity.

    Viscosity, calculation per [Devoto1966]_ (eq. 19 and eq. 20).
    Second-order approximation.

    Parameters
    ----------
    mixture : LTE
        Mixture of species.

    Returns
    -------
    float
        Viscosity.

    Notes
    -----
    The viscosity is given by equation 19 of [Devoto1966]_:

    .. math::

        \eta=\frac{1}{2} k_b T \sum_{j=1}^\nu n_j b_{j 0}

    where:

    - :math:`\eta` is the viscosity,
    - :math:`k_b` is the Boltzmann constant,
    - :math:`T` is the temperature of the mixture,
    - :math:`n_j` is the number density of species :math:`j`,

    The elements of :math:`b_{j 0}` are given by equation 20 of [Devoto1966]_:

    .. math::

        \begin{aligned}
            & \sum_{j=1}^\nu \sum_{p=0}^1 \hat{q}_{i j}^{m p} b_{j p}
                =5 n_i\left(\frac{2 \pi m_i}{k_b T}\right)^{\frac{1}{2}} \delta_{m 0} \\
            & \quad(i=1,2, \cdots \nu ; m=0, 1)
        \end{aligned}

    where:

    - :math:`\nu` is the number of species in the mixture,
    - :math:`\hat{q}_{i j}^{m p}` are the elements of the :math:`\hat{q}`-matrix.


    TODO: Add how the code works.
    TODO: Why not use equation 21?
    """
    nb_species = len(mixture.species)
    number_densities = mixture.calculate_composition()
    masses = np.array([sp.molarmass / u.N_a for sp in mixture.species])

    qq = qhat(mixture)
    invq = np.linalg.inv(qq)
    bvec = np.zeros(2 * nb_species)  # 2 for 2nd order approximation
    bvec[:nb_species] = (
        5 * number_densities * np.sqrt(2 * u.pi * masses / (u.k_b * mixture.T))
    )
    bflat = invq.dot(bvec)
    bip = bflat.reshape(2, nb_species)

    return 0.5 * u.k_b * mixture.T * np.sum(number_densities * bip[0])


def electricalconductivity(mixture: "LTE") -> float:
    r"""Electrical conductivity.

    Electrical conductivity, calculation per [Devoto1966]_ (eq. 29).
    Fourth-order approximation.
    This simplification neglects heavy ion contributions to the current.

    Parameters
    ----------
    mixture : LTE
        Mixture of species.

    Returns
    -------
    float
        Electrical conductivity.

    Notes
    -----
    The electrical conductivity is given by equation 29 of [Devoto1966]_:

    .. math::

        \sigma=\frac{e^{2} n_{\text{tot}}}{\rho k_{B} T} \sum_{j=2}^{\zeta} n_{j} m_{j} z_{j} D_{1 j}

    where:

    - :math:`\sigma` is the electrical conductivity,
    - :math:`e` is the elementary charge,
    - :math:`n_{\text{tot}}` is the total number density of the mixture,
    - :math:`\rho` is the density of the mixture,
    - :math:`k_{B}` is the Boltzmann constant,
    - :math:`T` is the temperature of the mixture,
    - :math:`n_{j}` is the number density of species :math:`j`,
    - :math:`m_{j}` is the molar mass of species :math:`j`,
    - :math:`z_{j}` is the charge number of species :math:`j`,
    - :math:`D_{1 j}` is the element :math:`D_{1 j}` of the diffusion matrix.

    The sum is over all ionic species in the mixture.
    """
    number_densities = mixture.calculate_composition()
    masses = np.array([sp.molarmass / u.N_a for sp in mixture.species])
    n_tot = np.sum(number_densities)
    rho = mixture.calculate_density()

    D1 = Dij(mixture)[-1, :]

    sum_val = 0.0
    for species_j, D1j, mj, nj in zip(mixture.species, D1, masses, number_densities):
        # TODO: Check if this is correct. Electrons should be discarded.
        # TODO: Check if this is correct. Neutrak species should be discarded (but ok,
        # since they have 0 charge).
        sum_val += nj * mj * species_j.chargenumber * D1j

    premult = u.e**2 * n_tot / (rho * u.k_b * mixture.T)

    return premult * sum_val


def thermalconductivity(
    mixture: "LTE",
    rel_delta_T: float,
    DTterms_yn: bool,
    ni_limit: float,
) -> float:
    r"""Thermal conductivity.

    Thermal conductivity, calculation per [Devoto1966]_ (eq. 2, 13 and 18).
    Fourth-order approximation.
    Numerical derivative performed to obtain :math:`\frac{dx_i}{dT}` for :math:`\vec{\nabla} x`
    in the :math:`\vec{d_i}` expression.

    It assumes that there is no pressure gradient and no external forces.

    Parameters
    ----------
    mixture : LTE
        Mixture of species.
    rel_delta_T : float
        Relative delta T for numerical derivative.
    DTterms_yn : bool
        Flag to include thermal diffusion terms.
    ni_limit : float
        TODO: Number density limit.

    Returns
    -------
    float
        Thermal conductivity.

    Notes
    -----
    The total heat flux :math:`\vec{q}` is given by equation 18 of [Devoto1966]_:

    .. math::

        \begin{array}{r}
            \vec{q} =\sum_{j=1}^\nu\left(\frac{n^2 m_j}{\rho} \sum_{i=1}^\nu m_i h_i D_{i j}
                    -\frac{n k_b T D_j^T}{n_j m_j}\right) \vec{d_i} \\
            -\left(\lambda^{\prime}+\sum_{i=1}^\nu \frac{n^2 h_i D_i^T}{\rho T}\right) \vec{\nabla} T
        \end{array}

    where:

    - :math:`\vec{q}` is the total heat flux,
    - :math:`\nu` is the number of species in the mixture,
    - :math:`n` is the total number density of the mixture,
    - :math:`m_j` is the mass of species :math:`j`,
    - :math:`\rho` is the density of the mixture,
    - :math:`m_i` is the mass of species :math:`i`,
    - :math:`h_i` is the enthalpy of species :math:`i`,
    - :math:`D_{i j}` is the diffusion coefficient between species :math:`i` and :math:`j`,
    - :math:`D_j^T` is the thermal diffusion coefficient of species :math:`j`,
    - :math:`D_i^T` is the thermal diffusion coefficient of species :math:`i`,


    :math:`\vec{d_i}` contains diffusion forces due to concentration :math:`x_i` and pressure gradients,
    and from external forces :math:`\vec{X_i}`, and is given by equation 2 of [Devoto1966]_:

    .. math::

        \vec{d_i} = \vec{\nabla} x_i
                  + \left(x_i\right. & \left.-\frac{\rho_i}{\rho}\right) \vec{\nabla} \ln p
                  - \frac{\rho_i}{p \rho}\left(\frac{\rho}{m_i} X _i-\sum_{i=1}^{\prime} n_l X _l\right)

    Assuming there is no pressure gradient and no external forces, the equation simplifies to:

    .. math::

        \vec{d_i} = \vec{\nabla} x_i

    It can be rewritten as: (TODO: check if correct)

    .. math::

        \vec{d_i} = \frac{d x_i}{d T} \vec{\nabla} T

    Injecting back into the total heat flux equation, we get:

    .. math::

        \vec{q} = \left[\sum_{j=1}^\nu\left(\frac{n^2 m_j}{\rho} \sum_{i=1}^\nu m_i h_i D_{i j}
                -\frac{n k_b T D_j^T}{n_j m_j}\right) \frac{d x_i}{d T}
        -\left(\lambda^{\prime}+\sum_{i=1}^\nu \frac{n^2 h_i D_i^T}{\rho T}\right) \right]\vec{\nabla} T

    The thermal conductivity (:math:`\vec{q} = - \lambda \vec{\nabla} T`) is then given by:

    .. math::

        \lambda = \sum_{j=1}^\nu\left(\frac{n k_b T D_j^T}{n_j m_j}
                    -\frac{n^2 m_j}{\rho} \sum_{i=1}^\nu m_i h_i D_{i j}\right) \frac{d x_i}{d T}
            +\left(\lambda^{\prime}+\sum_{i=1}^\nu \frac{n^2 h_i D_i^T}{\rho T}\right)


    In this equation, :math:`\lambda^{\prime}` is given by equation 13 of [Devoto1966]_:

    .. math::

        \lambda^{\prime} = -\frac{5 k_b}{4}
            \sum_{j=1}^\nu n_j\left(\frac{2 k_b T}{m_j}\right)^{\frac{1}{2}} a_{j 1}
    """
    nb_species = len(mixture.species)
    number_densities = mixture.calculate_composition()
    masses = np.array([sp.molarmass / u.N_a for sp in mixture.species])
    n_tot = np.sum(number_densities)
    rho = mixture.calculate_density()
    hv = mixture.calculate_species_enthalpies()

    # Rescale species enthalpies relative to average molar mass.
    # TODO: why?
    average_molar_mass = rho / n_tot
    hv = hv * masses / average_molar_mass

    ### Translational tk components. ###
    # Solve equation 5 of [Devoto1966]_ to get the `a` matrix.
    qq = q(mixture)
    invq = np.linalg.inv(qq)
    bvec = np.zeros(4 * nb_species)
    bvec[nb_species : 2 * nb_species] = -15 / 2 * np.sqrt(u.pi) * number_densities
    aflat = invq.dot(bvec)
    aip = aflat.reshape(4, nb_species)
    # Equation 13 of [Devoto1966]_.
    kdash = (
        -5
        / 4
        * u.k_b
        * np.sum(number_densities * np.sqrt(2 * u.k_b * mixture.T / masses) * aip[1])
    )
    # TODO: Why not use equation 14?

    ### Thermal diffusion tk components. ###
    if DTterms_yn:
        # TODO: This looks like the second term in the second parenthesis of 18 of [Devoto1966]_.
        # TODO: Check if it is correct. Where are the term n² and rho?
        locDTi = DTi(mixture)
        kdt = np.sum(hv * locDTi / mixture.T)
    else:
        kdt = 0.0

    ### reactional tk components - normal diffusion term ###

    # Compute the derivative of the number densities with respect to temperature.
    # x is the concentration of species i, x = ni / ntot
    Tval = mixture.T
    mixture.T = Tval * (1 + rel_delta_T)
    n_positive = mixture.calculate_composition()
    mixture.T = Tval * (1 - rel_delta_T)
    n_negative = mixture.calculate_composition()
    mixture.T = Tval
    x_positive = n_positive / np.sum(n_positive)
    x_negative = n_negative / np.sum(n_negative)
    dxdT = (x_positive - x_negative) / (2 * rel_delta_T * mixture.T)

    locDij = Dij(mixture)

    krxn_enth = 0.0
    for j in range(nb_species):
        for i in range(nb_species):
            # TODO: This looks like the first term in the first parenthesis of 18 of [Devoto1966]_.
            # TODO: Check if it is correct.
            krxn_enth += masses[j] * masses[i] * hv[i] * locDij[i, j] * dxdT[j]
    # TODO: Why a minus sign below? (Seems to related to going from d_j to d_j/dT dT )
    krxn_enth *= -(n_tot**2) / rho

    ### reactional tk components - thermal diffusion term ###
    if DTterms_yn:
        # TODO: This looks like the second term in the first parenthesis of 18 of [Devoto1966]_.
        # TODO: Check if it is correct.
        dxdTfilt = np.where(number_densities < ni_limit, np.zeros(nb_species), dxdT)
        krxn_therm = (
            n_tot
            * u.k_b
            * Tval
            * np.sum(locDTi * dxdTfilt / (number_densities * masses))
        )
    else:
        krxn_therm = 0.0

    return kdash + kdt + krxn_enth + krxn_therm
