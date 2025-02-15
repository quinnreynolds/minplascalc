import numpy as np
from numba import njit

from minplascalc.transport.potential_functions import delta
from minplascalc.units import Units

u = Units()
K_B = u.k_b
E = u.e


### Transport property calculations ############################################


@njit
def Dij_jit(
    nb_species: int,
    number_densities: np.ndarray,
    masses: np.ndarray,
    rho: float,
    T: float,
    qq: np.ndarray,
) -> np.ndarray:
    r"""Diffusion coefficients.

    Diffusion coefficents, calculation per [Devoto1966]_ (eq. 3 and eq. 6).
    Fourth-order approximation.

    Parameters
    ----------
    nb_species : int
        Number of species in the mixture.
    number_densities : np.ndarray
        Number densities of the species in the mixture, in m^-3.
    masses : np.ndarray
        Masses of the species in the mixture, in kg.
    rho : float
        Density of the mixture, in kg/m^3.
    T : float
        Temperature of the mixture, in K.
    qq : np.ndarray
        q-matrix, size (4*nb_species, 4*nb_species) for fourth-order approximation.

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
    diffusion_matrix = np.zeros((nb_species, nb_species))

    n_tot = np.sum(number_densities)  # m^-3

    inverse_q = np.linalg.inv(qq)
    b_vec = np.zeros(4 * nb_species)  # 4 for 4th order approximation

    for i in range(nb_species):
        for j in range(nb_species):
            # TODO: Check if this is correct
            # Equation 6 of [Devoto1966]_.
            dij = np.array([delta(h, i) - delta(h, j) for h in range(0, nb_species)])
            b_vec[:nb_species] = 3 * np.sqrt(np.pi) * dij
            cflat = inverse_q.dot(b_vec)
            cip = cflat.reshape(4, nb_species)

            # Diffusion coefficient, equation 3 of [Devoto1966]_.
            diffusion_matrix[i, j] = (
                rho
                * number_densities[i]
                / (2 * n_tot * masses[j])
                * np.sqrt(2 * K_B * T / masses[i])
                * cip[0, i]
            )

    return diffusion_matrix


@njit
def DTi_jit(
    nb_species: int,
    number_densities: np.ndarray,
    masses: np.ndarray,
    T: float,
    qq: np.ndarray,
) -> float:
    r"""Thermal diffusion coefficients.

    Thermal diffusion coefficents, calculation per [Devoto1966]_ (eq. 4 and eq. 5).
    Fourth-order approximation.

    Parameters
    ----------
    nb_species : int
        Number of species in the mixture.
    number_densities : np.ndarray
        Number densities of the species in the mixture, in m^-3.
    masses : np.ndarray
        Masses of the species in the mixture, in kg.
    T : float
        Temperature of the mixture, in K.
    qq : np.ndarray
        q-matrix, size (4*nb_species, 4*nb_species) for fourth-order approximation.

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
    inverse_q = np.linalg.inv(qq)
    b_vec = np.zeros(4 * nb_species)  # 4 for 4th order approximation
    # Only the first element is non-zero
    b_vec[nb_species : 2 * nb_species] = -15 / 2 * np.sqrt(np.pi) * number_densities
    aflat = inverse_q.dot(b_vec)
    aip = aflat.reshape(4, nb_species)

    return 0.5 * number_densities * masses * np.sqrt(2 * K_B * T / masses) * aip[0]


@njit
def viscosity_jit(
    nb_species: int,
    number_densities: np.ndarray,
    masses: np.ndarray,
    T: float,
    qqhat: np.ndarray,
) -> float:
    r"""Viscosity.

    Viscosity, calculation per [Devoto1966]_ (eq. 19 and eq. 20).
    Second-order approximation.

    Parameters
    ----------
    mixture : LTE
        Mixture of species.

    Returns
    -------
    nb_species : int
        Number of species in the mixture.
    number_densities : np.ndarray
        Number densities of the species in the mixture, in m^-3.
    masses : np.ndarray
        Masses of the species in the mixture, in kg.
    T : float
        Temperature of the mixture, in K.
    qqhat : np.ndarray
        qhat-matrix, size (2*nb_species, 2*nb_species) for second-order approximation.

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
    inverse_qhat = np.linalg.inv(qqhat)
    b_vec = np.zeros(2 * nb_species)  # 2 for 2nd order approximation
    b_vec[:nb_species] = 5 * number_densities * np.sqrt(2 * np.pi * masses / (K_B * T))
    bflat = inverse_qhat.dot(b_vec)
    bip = bflat.reshape(2, nb_species)

    return 0.5 * K_B * T * np.sum(number_densities * bip[0])


@njit
def electricalconductivity_jit(
    charge_numbers: np.ndarray,
    number_densities: np.ndarray,
    masses: np.ndarray,
    rho: float,
    T: float,
    D1: np.ndarray,
) -> float:
    r"""Electrical conductivity.

    Electrical conductivity, calculation per [Devoto1966]_ (eq. 29).
    Fourth-order approximation.
    This simplification neglects heavy ion contributions to the current.

    Parameters
    ----------
    charge_numbers : np.ndarray
        Charge numbers of the species in the mixture.
    number_densities : np.ndarray
        Number densities of the species in the mixture, in m^-3.
    masses : np.ndarray
        Masses of the species in the mixture, in kg.
    rho : float
        Density of the mixture, in kg/m^3.
    T : float
        Temperature of the mixture, in K.
    D1 : np.ndarray
        D1-matrix, size (4*nb_species, 4*nb_species) for fourth-order approximation.

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
    n_tot = np.sum(number_densities)

    sum_val = 0.0
    for charge_number, D1j, mj, nj in zip(charge_numbers, D1, masses, number_densities):
        # TODO: Check if this is correct. Electrons should be discarded.
        # TODO: Check if this is correct. Neutral species should be discarded (but ok,
        # since they have 0 charge).
        sum_val += nj * mj * charge_number * D1j

    premult = E**2 * n_tot / (rho * K_B * T)

    return premult * sum_val


@njit
def thermalconductivity_dash_jit(
    nb_species: int,
    number_densities: np.ndarray,
    masses: np.ndarray,
    T: float,
    qq: np.ndarray,
) -> float:
    r"""Thermal conductivity dash.

    Thermal conductivity, calculation per [Devoto1966]_ (eq. 13).
    Fourth-order approximation.

    Parameters
    ----------
    nb_species : int
        Number of species in the mixture.
    number_densities : np.ndarray
        Number densities of the species in the mixture, in m^-3.
    masses : np.ndarray
        Masses of the species in the mixture, in kg.
    T : float
        Temperature of the mixture, in K.
    qq : np.ndarray
        q-matrix, size (4*nb_species, 4*nb_species) for fourth-order approximation

    Returns
    -------
    float
        Thermal conductivity prime.

    Notes
    -----
    :math:`\lambda^{\prime}` is given by equation 13 of [Devoto1966]_:

    .. math::

        \lambda^{\prime} = -\frac{5 k_b}{4}
            \sum_{j=1}^\nu n_j\left(\frac{2 k_b T}{m_j}\right)^{\frac{1}{2}} a_{j 1}
    """
    ### Translational tk components. ###
    # Solve equation 5 of [Devoto1966]_ to get the `a` matrix.
    inverse_q = np.linalg.inv(qq)
    b_vec = np.zeros(4 * nb_species)
    b_vec[nb_species : 2 * nb_species] = -15 / 2 * np.sqrt(np.pi) * number_densities
    aflat = inverse_q.dot(b_vec)
    aip = aflat.reshape(4, nb_species)
    # Equation 13 of [Devoto1966]_.
    k_dash = (
        -5 / 4 * K_B * np.sum(number_densities * np.sqrt(2 * K_B * T / masses) * aip[1])
    )
    # TODO: Why not use equation 14?
    return k_dash
