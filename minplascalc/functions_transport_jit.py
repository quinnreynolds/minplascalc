import numpy as np
from numba import njit
from scipy import constants
from scipy.special import gamma

from minplascalc.data_transport import c_in, c_nn
from minplascalc.units import Units

u = Units()

ke = 1 / (4 * u.pi * u.epsilon_0)
a0 = constants.physical_constants["Bohr radius"][0]
egamma = np.euler_gamma


@njit
def pot_parameters_neut_neut_jit(
    alpha_i: float, alpha_j: float, n_eff_i: float, n_eff_j: float
) -> tuple[float, float]:
    r"""Calculate the equilibrium distance and binding energy for a neutral-neutral pair.

    Parameters
    ----------
    alpha_i : float
        Polarisability of species i in m^3.
    alpha_j : float
        Polarisability of species j in m^3.
    n_eff_i : float
        Effective number of electrons of species i.
    n_eff_j : float
        Effective number of electrons of species j.

    Returns
    -------
    tuple[float, float]
        Equilibrium distance and binding energy.

    Notes
    -----
    The equilibrium distance is given by eq. 6 of [Laricchiuta2007]_ by the formula:

    .. math::

        r_e = 1.767 \frac{\alpha_1^{1 / 3}+\alpha_2^{1 / 3}}{\left(\alpha_1 \alpha_2\right)^{0.095}}

    where :math:`\alpha_i` is the polarisability of species :math:`i` in m^3.


    The binding energy is given by eq. 7 of [Laricchiuta2007]_ by the formula:

    .. math::

        \epsilon_0 = 0.72 \frac{C_d}{r_e^6}

    where :math:`C_d` is the effective long-range London coefficient, defined in eq. 8 of [Laricchiuta2007]_
    by the formula:

    .. math::

        C_d = 15.7 \frac{\alpha_1 \alpha_2}{\sqrt{\frac{\alpha_1}{n_1}} + \sqrt{\frac{\alpha_2}{n_2}}}

    with :math:`n_i` the effective number of electrons of species :math:`i`.
    """
    # Effective long-range London coefficient, as defined in eq. 8 of [Laricchiuta2007]_.
    C_d = (
        15.7
        * alpha_i
        * alpha_j
        / (
            np.sqrt(alpha_i / n_eff_i)
            + np.sqrt(alpha_j / n_eff_j)
        )
    )
    # Equilibrium distance r_e, as defined in eq. 6 of [Laricchiuta2007]_.
    r_e = (
        1.767 * (alpha_i ** (1 / 3) + alpha_j ** (1 / 3)) / (alpha_i * alpha_j) ** 0.095
    )
    # Binding energy epsilon_0, as defined in eq. 7 of [Laricchiuta2007]_.
    epsilon_0 = 0.72 * C_d / r_e**6
    # Return the equilibrium distance and the binding energy.
    return r_e, epsilon_0