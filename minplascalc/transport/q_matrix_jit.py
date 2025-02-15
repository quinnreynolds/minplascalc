import numpy as np
from numba import njit

from minplascalc.transport.potential_functions import delta


@njit
def q00_jit(
    nb_species: int,
    number_densities: np.ndarray,
    masses: np.ndarray,
    Q11: np.ndarray,
) -> np.ndarray:
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

    return q00


@njit
def q01_jit(
    nb_species: int,
    number_densities: np.ndarray,
    masses: np.ndarray,
    Q11: np.ndarray,
    Q12: np.ndarray,
) -> np.ndarray:
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
    return q01


@njit
def q11_jit(
    nb_species: int,
    number_densities: np.ndarray,
    masses: np.ndarray,
    Q11: np.ndarray,
    Q12: np.ndarray,
    Q13: np.ndarray,
    Q22: np.ndarray,
) -> np.ndarray:
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
    return q11


@njit
def q02_jit(
    nb_species: int,
    number_densities: np.ndarray,
    masses: np.ndarray,
    Q11: np.ndarray,
    Q12: np.ndarray,
    Q13: np.ndarray,
) -> np.ndarray:
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
    return q02


@njit
def q12_jit(
    nb_species: int,
    number_densities: np.ndarray,
    masses: np.ndarray,
    Q11: np.ndarray,
    Q12: np.ndarray,
    Q13: np.ndarray,
    Q14: np.ndarray,
    Q22: np.ndarray,
    Q23: np.ndarray,
) -> np.ndarray:
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
    return q12


@njit
def q22_jit(
    nb_species: int,
    number_densities: np.ndarray,
    masses: np.ndarray,
    Q11: np.ndarray,
    Q12: np.ndarray,
    Q13: np.ndarray,
    Q14: np.ndarray,
    Q15: np.ndarray,
    Q22: np.ndarray,
    Q23: np.ndarray,
    Q24: np.ndarray,
    Q33: np.ndarray,
) -> np.ndarray:
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
    return q22


@njit
def q03_jit(
    nb_species: int,
    number_densities: np.ndarray,
    masses: np.ndarray,
    Q11: np.ndarray,
    Q12: np.ndarray,
    Q13: np.ndarray,
    Q14: np.ndarray,
) -> np.ndarray:
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
    return q03


@njit
def q13_jit(
    nb_species: int,
    number_densities: np.ndarray,
    masses: np.ndarray,
    Q11: np.ndarray,
    Q12: np.ndarray,
    Q13: np.ndarray,
    Q14: np.ndarray,
    Q15: np.ndarray,
    Q22: np.ndarray,
    Q23: np.ndarray,
    Q24: np.ndarray,
) -> np.ndarray:
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
    return q13


@njit
def q23_jit(
    nb_species: int,
    number_densities: np.ndarray,
    masses: np.ndarray,
    Q11: np.ndarray,
    Q12: np.ndarray,
    Q13: np.ndarray,
    Q14: np.ndarray,
    Q15: np.ndarray,
    Q16: np.ndarray,
    Q22: np.ndarray,
    Q23: np.ndarray,
    Q24: np.ndarray,
    Q25: np.ndarray,
    Q33: np.ndarray,
    Q34: np.ndarray,
) -> np.ndarray:
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
    return q23


@njit
def q33_jit(
    nb_species: int,
    number_densities: np.ndarray,
    masses: np.ndarray,
    Q11: np.ndarray,
    Q12: np.ndarray,
    Q13: np.ndarray,
    Q14: np.ndarray,
    Q15: np.ndarray,
    Q16: np.ndarray,
    Q17: np.ndarray,
    Q22: np.ndarray,
    Q23: np.ndarray,
    Q24: np.ndarray,
    Q25: np.ndarray,
    Q26: np.ndarray,
    Q33: np.ndarray,
    Q34: np.ndarray,
    Q35: np.ndarray,
    Q44: np.ndarray,
) -> np.ndarray:
    # Equation A18 of [Devoto1966]_.
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
    return q33


@njit
def q10_jit(
    nb_species: int,
    masses: np.ndarray,
    q01: np.ndarray,
) -> np.ndarray:
    # Equation A5 of [Devoto1966]_.
    q10 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            q10[i, j] = masses[j] / masses[i] * q01[i, j]
    return q10


@njit
def q20_jit(
    nb_species: int,
    masses: np.ndarray,
    q02: np.ndarray,
) -> np.ndarray:
    # Equation A8 of [Devoto1966]_.
    q20 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            q20[i, j] = (masses[j] / masses[i]) ** 2 * q02[i, j]
    return q20


@njit
def q21_jit(
    nb_species: int,
    masses: np.ndarray,
    q12: np.ndarray,
) -> np.ndarray:
    # Equation A10 of [Devoto1966]_.
    q21 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            q21[i, j] = masses[j] / masses[i] * q12[i, j]
    return q21


@njit
def q30_jit(
    nb_species: int,
    masses: np.ndarray,
    q03: np.ndarray,
) -> np.ndarray:
    # Equation A13 of [Devoto1966]_.
    q30 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            q30[i, j] = (masses[j] / masses[i]) ** 3 * q03[i, j]
    return q30


@njit
def q31_jit(
    nb_species: int,
    masses: np.ndarray,
    q13: np.ndarray,
) -> np.ndarray:
    # Equation A15 of [Devoto1966]_.
    q31 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            q31[i, j] = (masses[j] / masses[i]) ** 2 * q13[i, j]
    return q31


@njit
def q32_jit(
    nb_species: int,
    masses: np.ndarray,
    q23: np.ndarray,
) -> np.ndarray:
    # Equation A17 of [Devoto1966]_.
    q32 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            q32[i, j] = masses[j] / masses[i] * q23[i, j]
    return q32
