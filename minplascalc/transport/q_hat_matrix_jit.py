import numpy as np
from numba import njit

from minplascalc.transport.potential_functions import delta


@njit
def qhat00_jit(
    nb_species: int,
    number_densities: np.ndarray,
    masses: np.ndarray,
    Q11: np.ndarray,
    Q22: np.ndarray,
) -> np.ndarray:
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
    return qhat00


@njit
def qhat01_jit(
    nb_species: int,
    number_densities: np.ndarray,
    masses: np.ndarray,
    Q11: np.ndarray,
    Q12: np.ndarray,
    Q22: np.ndarray,
    Q23: np.ndarray,
) -> np.ndarray:
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
    return qhat01


@njit
def qhat11_jit(
    nb_species: int,
    number_densities: np.ndarray,
    masses: np.ndarray,
    Q11: np.ndarray,
    Q12: np.ndarray,
    Q13: np.ndarray,
    Q22: np.ndarray,
    Q23: np.ndarray,
    Q24: np.ndarray,
    Q33: np.ndarray,
) -> np.ndarray:
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
    return qhat11


@njit
def qhat10_jit(
    nb_species: int,
    masses: np.ndarray,
    qhat01: np.ndarray,
) -> np.ndarray:
    # Equation A21 of [Devoto1966]_.
    qhat10 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            qhat10[i, j] = masses[j] / masses[i] * qhat01[i, j]
    return qhat10
