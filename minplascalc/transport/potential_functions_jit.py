import numpy as np
from numba import njit


@njit
def pot_parameters_neut_neut_jit(
    alpha_i: float, alpha_j: float, n_eff_i: float, n_eff_j: float
) -> tuple[float, float]:
    r"""Calculate the equilibrium distance and binding energy for a neutral-neutral pair.

    Parameters
    ----------
    alpha_i : float
        Polarisability of species i in Å^3.
    alpha_j : float
        Polarisability of species j in Å^3.
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

    where :math:`\alpha_i` is the polarisability of species :math:`i` in Å^3.


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
        / (np.sqrt(alpha_i / n_eff_i) + np.sqrt(alpha_j / n_eff_j))
    )
    # Equilibrium distance r_e, as defined in eq. 6 of [Laricchiuta2007]_.
    r_e = (
        1.767 * (alpha_i ** (1 / 3) + alpha_j ** (1 / 3)) / (alpha_i * alpha_j) ** 0.095
    )
    # Binding energy epsilon_0, as defined in eq. 7 of [Laricchiuta2007]_.
    epsilon_0 = 0.72 * C_d / r_e**6
    # Return the equilibrium distance and the binding energy.
    return r_e, epsilon_0


@njit
def pot_parameters_ion_neut_jit(
    alpha_i: float, alpha_n: float, Z_ion: float
) -> tuple[float, float]:
    r"""Calculate the equilibrium distance and binding energy for an ion-neutral pair.

    Parameters
    ----------
    alpha_i : float
        Polarisability of the ion species in Å^3.
    alpha_n : float
        Polarisability of the neutral species in Å^3.
    Z_ion : float
        Charge number of the ion species

    Returns
    -------
    tuple[float, float]
        Equilibrium distance and binding energy.

    Notes
    -----
    The equilibrium distance is given by eq. 9 of [Laricchiuta2007]_ by the formula:

    .. math::

        r_e = 1.767 \frac{\alpha_i^{1 / 3}+\alpha_n^{1 / 3}}
                         {\left(\alpha_i \alpha_n \[ 1 + \frac{1}{\rho}\] \right)^{0.095}}

    where :math:`\alpha_i` is the polarisability of the ion species and :math:`\alpha_n` is the polarisability
    of the neutral species, both in Å^3.

    The binding energy is given by eq. 10 of [Laricchiuta2007]_ by the formula:

    .. math::

        \epsilon_0 = 5.2 \frac{z^2 \alpha_n}{r_e^4} \left(1 + \rho\right)

    where :math:`z` is the charge number of the ion species, :math:`\alpha_n` is the polarisability of the
    neutral species in Å^3

    :math:`\rho` is representative of the relative role of dispersion and induction attraction components in
    proximity to the equilibrium distance, defined in eq. 11 of [Laricchiuta2007]_ by the formula:

    .. math::

        \rho = \frac{\alpha_i}
                    {z^2 \sqrt{\alpha_n} \left(1 + \left(2 \alpha_i / \alpha_n\right)^{2 / 3}\right)}

    """
    # rho, as defined in eq. 11 of [Laricchiuta2007]_.
    rho = alpha_i / (
        Z_ion**2 * np.sqrt(alpha_n) * (1 + (2 * alpha_i / alpha_n) ** (2 / 3))
    )
    # Equilibrium distance r_e, as defined in eq. 9 of [Laricchiuta2007]_.
    r_e = (
        1.767
        * (alpha_i ** (1 / 3) + alpha_n ** (1 / 3))
        / (alpha_i * alpha_n * (1 + 1 / rho)) ** 0.095
    )
    # Binding energy epsilon_0, as defined in eq. 10 of [Laricchiuta2007]_.
    epsilon_0 = 5.2 * Z_ion**2 * alpha_n * (1 + rho) / r_e**4
    # Return the equilibrium distance and the binding energy.
    return r_e, epsilon_0


@njit
def beta_jit(
    alpha_i: float,
    alpha_j: float,
    spin_multiplicity_i: float,
    spin_multiplicity_j: float,
) -> float:
    r"""Calculate the beta parameter for a pair of species.

    Parameters
    ----------
    alpha_i : float
        Polarisability of species i in Å^3.
    alpha_j : float
        Polarisability of species j in Å^3.
    spin_multiplicity_i : float
        Spin multiplicity of species i.
    spin_multiplicity_j : float
        Spin multiplicity of species j.

    Returns
    -------
    float
        Beta parameter.

    Notes
    -----
    :math:`\beta` is a parameter to estimate the hardness of interacting electronic
    distribution densities, and it is estimated in eq. 5 of [Laricchiuta2007]_:

    .. math::

        \beta = 6 + \frac{5}{s_1 + s_2}

    where :math:`s_i` is the softness.

    The softness s defined as the cubic root of the polarizability. For open-shell atoms and ions
    a multiplicative factor, which is the ground state spin multiplicity, should be also considered:

    .. math::

        s = \frac{\alpha^{1 / 3}}{m}
    """
    # Compute the softness of the species.
    s_i = alpha_i ** (1 / 3) * spin_multiplicity_i
    s_j = alpha_j ** (1 / 3) * spin_multiplicity_j
    # Return the beta parameter.
    return 6 + 5 / (s_i + s_j)


@njit
def coulomb_logarithm_ee_jit(
    ne_cgs: float,
    T_eV: float,
):
    r"""Calculate the Coulomb logarithm for electron-electron collision.

    Parameters
    ----------
    ne_cgs : float
        Number density of the electrons, in cm^-3.
    T_eV : float
        Temperature of the electrons, in eV.

    Returns
    -------
    float
        Coulomb logarithm for electron-electron collision.

    Notes
    -----
    The Coulomb logarithm is defined at page 34 of [NRL2019]_.
    The units of this book is cgs, except for temperature which is in eV.

    For thermal electron-electron collisions:

    .. math::

        \lambda_{e e} = 23.5 - \ln \left(n_e^{1 / 2} T^{-5 / 4}\right)
            - \left[10^{-5}+\left(\ln T - 2 \right)^2 / 16\right]^{1 / 2}

    where:

    - :math:`n_e` is the electron number density in cm^-3
    - :math:`T` is the electron temperature in eV.
    """
    return (
        23.5
        - np.log(ne_cgs ** (1 / 2) * T_eV ** (-5 / 4))
        - (1e-5 + (np.log(T_eV) - 2) ** 2 / 16) ** (1 / 2)
    )


@njit
def coulomb_logarithm_ei_jit(ne_cgs: float, T_eV: float, Z_ion: float):
    r"""Calculate the Coulomb logarithm for electron-ion collision.

    Parameters
    ----------
    ne_cgs : float
        Number density of the electrons, in cm^-3.
    T_eV : float
        Temperature of the electrons, in eV.
    Z_ion : float
        Charge number of the ion species

    Returns
    -------
    float
        Coulomb logarithm for electron-ion collision.

    Notes
    -----
    The Coulomb logarithm is defined at page 34 of [NRL2019]_.
    The units of this book is cgs, except for temperature which is in eV.

    For electron-ion collisions, assuming that :math:`T \lt 10 eV`:

    .. math::

        \lambda_{e i} = 23 - \ln \left(n_e^{1 / 2} Z T^{-3 / 2}\right)

    where:

    - :math:`n_e` is the electron number density in cm^-3
    - :math:`T` is the temperature in eV,
    - :math:`Z` is the charge number of the ion species.
    """
    return 23 - np.log(ne_cgs ** (1 / 2) * abs(Z_ion) * T_eV ** (-3 / 2))


@njit
def coulomb_logarithm_ii_jit(
    ni_cgs: float, nj_cgs: float, T_eV: float, Z_ion_i: float, Z_ion_j: float
):
    r"""Calculate the Coulomb logarithm for ion-ion collision.

    Parameters
    ----------
    ni_cgs : float
        Number density of the ion species i, in cm^-3.
    nj_cgs : float
        Number density of the ion species j, in cm^-3.
    T_eV : float
        Temperature of the electrons, in eV.
    Z_ion_i : float
        Charge number of the ion species i.
    Z_ion_j : float
        Charge number of the ion species j.

    Returns
    -------
    float
        Coulomb logarithm for ion-ion collision.

    Notes
    -----
    The Coulomb logarithm is defined at page 34 of [NRL2019]_.
    The units of this book is cgs, except for temperature which is in eV.

    For (mixed) ion-ion collisions:

    .. math::

        \lambda_{i i'} = 23 - \ln \left( \frac{Z_i Z_i'}{T}
            \left(\frac{n_i Z_i^2}{T} + \frac{n_i' Z_i'^2}{T} \right)^{1 / 2}\right)}

    where:

    - :math:`T` is the temperature in eV,
    - :math:`Z_i` is the charge number of the ion species i,
    - :math:`Z_i'` is the charge number of the ion species i',
    - :math:`n_i` is the number density of the ion species i in cm^-3,
    - :math:`n_i'` is the number density of the ion species i' in cm^-3.
    """
    return 23 - np.log(
        abs(Z_ion_i * Z_ion_j)  # TODO: why abs?
        / T_eV
        * (ni_cgs * abs(Z_ion_i) ** 2 / T_eV + nj_cgs * abs(Z_ion_j) ** 2 / T_eV)
        ** (1 / 2)
    )
