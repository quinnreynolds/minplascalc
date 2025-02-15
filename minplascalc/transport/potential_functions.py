from typing import TYPE_CHECKING

import numpy as np
from numba import njit
from scipy import constants

from minplascalc.transport.potential_functions_jit import (
    beta_jit,
    coulomb_logarithm_ee_jit,
    coulomb_logarithm_ei_jit,
    coulomb_logarithm_ii_jit,
    pot_parameters_ion_neut_jit,
    pot_parameters_neut_neut_jit,
)
from minplascalc.units import Units

if TYPE_CHECKING:
    from minplascalc.species import Species

u = Units()

ke = 1 / (4 * u.pi * u.epsilon_0)
a0 = constants.physical_constants["Bohr radius"][0]
egamma = np.euler_gamma


def n_effective_electrons(nint, nout):
    """Do not seem to be used anywhere."""
    return nout * (1 + (1 - nout / nint) * (nint / (nout + nint)) ** 2)


def pot_parameters_neut_neut(
    species_i: "Species", species_j: "Species"
) -> tuple[float, float]:
    r"""Calculate the equilibrium distance and binding energy for a neutral-neutral pair.

    Parameters
    ----------
    species_i : Species
        First species.
    species_j : Species
        Second species.

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
    # Polarisabilities of the species, in m^3.
    alpha_i = species_i.polarisability * 1e30
    alpha_j = species_j.polarisability * 1e30
    # Effective long-range London coefficient, as defined in eq. 8 of [Laricchiuta2007]_.
    if species_i.effectiveelectrons is None or species_j.effectiveelectrons is None:
        raise ValueError(
            "Effective number of electrons must be provided for neutral species."
        )
    n_eff_i = species_i.effectiveelectrons
    n_eff_j = species_j.effectiveelectrons

    r_e, epsilon_0 = pot_parameters_neut_neut_jit(alpha_i, alpha_j, n_eff_i, n_eff_j)

    # Return the equilibrium distance and the binding energy.
    return r_e, epsilon_0


def pot_parameters_ion_neut(
    species_ion: "Species",
    species_neutral: "Species",
) -> tuple[float, float]:
    r"""Calculate the equilibrium distance and binding energy for an ion-neutral pair.

    Parameters
    ----------
    species_ion : Species
        Ion species.
    species_neutral : Species
        Neutral species.

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
    of the neutral species, both in m^3.

    The binding energy is given by eq. 10 of [Laricchiuta2007]_ by the formula:

    .. math::

        \epsilon_0 = 5.2 \frac{z^2 \alpha_n}{r_e^4} \left(1 + \rho\right)

    where :math:`z` is the charge number of the ion species, :math:`\alpha_n` is the polarisability of the
    neutral species in m^3

    :math:`\rho` is representative of the relative role of dispersion and induction attraction components in
    proximity to the equilibrium distance, defined in eq. 11 of [Laricchiuta2007]_ by the formula:

    .. math::

        \rho = \frac{\alpha_i}
                    {z^2 \sqrt{\alpha_n} \left(1 + \left(2 \alpha_i / \alpha_n\right)^{2 / 3}\right)}

    """
    # Polarisabilities of the species, in m^3.
    alpha_i = species_ion.polarisability * 1e30
    alpha_n = species_neutral.polarisability * 1e30
    # Charge number of the ion species.
    Z_ion = species_ion.chargenumber

    # Return the equilibrium distance and the binding energy.
    r_e, epsilon_0 = pot_parameters_ion_neut_jit(alpha_i, alpha_n, Z_ion)
    return r_e, epsilon_0


def beta(
    species_i: "Species",
    species_j: "Species",
) -> float:
    r"""Calculate the beta parameter for a pair of species.

    Parameters
    ----------
    species_i : Species
        First species.
    species_j : Species
        Second species.

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
    # Polarisabilities of the species, in m^3.
    alpha_i = species_i.polarisability * 1e30
    alpha_j = species_j.polarisability * 1e30
    # Return the beta parameter.
    beta = beta_jit(alpha_i, alpha_j, species_i.multiplicity, species_j.multiplicity)
    return beta


@njit
def x0_neut_neut(beta_value: float) -> float:
    r"""Calculate the x0 parameter for a neutral-neutral pair.

    Parameters
    ----------
    beta_value : float
        Beta parameter.

    Returns
    -------
    float
        x0 parameter.

    Notes
    -----
    :math:`x_0` is defined in eq. 13 of [Laricchiuta2007]_ as a solution to a transcendal equation.
    It can be approximated by eq. 17, with the following formula:

    .. math::

        x_0 = \xi_1 \beta^{\xi_2}

    where :math:`\xi_1 = 0.8002` and :math:`\xi_2 = 0.049256`, as given in Table 3.
    """
    return 0.8002 * beta_value**0.049256


@njit
def x0_ion_neut(beta_value: float) -> float:
    r"""Calculate the x0 parameter for a ion-neutral pair.

    Parameters
    ----------
    beta_value : float
        Beta parameter.

    Returns
    -------
    float
        x0 parameter.

    Notes
    -----
    :math:`x_0` is defined in eq. 13 of [Laricchiuta2007]_ as a solution to a transcendal equation.
    It can be approximated by eq. 17, with the following formula:

    .. math::

        x_0 = \xi_1 \beta^{\xi_2}

    where :math:`\xi_1 = 0.7564` and :math:`\xi_2 = 0.064605`, as given in Table 3.
    """
    return 0.7564 * beta_value**0.064605


def coulomb_logarithm_charged(
    species_i: "Species",
    species_j: "Species",
    n_i: float,
    n_j: float,
    T: float,
):
    r"""Calculate the Coulomb logarithm for a pair of charged species.

    Parameters
    ----------
    species_i : Species
        First species.
    species_j : Species
        Second species.
    n_i : float
        Number density of the first species, in m^-3.
    n_j : float
        Number density of the second species, in m^-3.
    T : float
        Temperature, in K.

    Returns
    -------
    float
        Coulomb logarithm.

    Notes
    -----
    The Coulomb logarithm is defined at page 34 of [NRL2019]_.
    The units of this book is cgs, except for temperature which is in eV.

    For thermal electron-electron collisions:

    .. math::

        \lambda_{e e} = 23.5 - \ln \left(n_e^{1 / 2} T^{-5 / 4}\right)
            - \left[10^{-5}+\left(\ln T - 2 \right)^2 / 16\right]^{1 / 2}

    For electron-ion collisions, assuming that :math:`T \lt 10 eV`:

    .. math::

        \lambda_{e i} = 23 - \ln \left(n_e^{1 / 2} Z T^{-3 / 2}\right)

    For (mixed) ion-ion collisions:

    .. math::

        \lambda_{i i'} = 23 - \ln \left( \frac{Z_i Z_i'}{T}
            \left(\frac{n_i Z_i^2}{T} + \frac{n_i' Z_i'^2}{T} \right)^{1 / 2}\right)}
    """
    T_eV = T * u.K_to_eV  # Convert temperature to eV.
    if species_i.name == "e" and species_j.name == "e":
        # Electron-electron collisions.
        ne_cgs = n_i * 1e-6  # m^-3 to cm^-3
        return coulomb_logarithm_ee_jit(ne_cgs, T_eV)
    elif species_i.name == "e":
        # Electron-ion collisions.
        ne_cgs = n_i * 1e-6  # m^-3 to cm^-3
        z_ion = species_j.chargenumber
        return coulomb_logarithm_ei_jit(ne_cgs, T_eV, z_ion)
    elif species_j.name == "e":
        # Ion-electron collisions, same as electron-ion collisions.
        ne_cgs = n_j * 1e-6  # m^-3 to cm^-3
        z_ion = species_i.chargenumber
        return coulomb_logarithm_ei_jit(ne_cgs, T_eV, z_ion)
    else:
        # Ion-ion collisions.
        ni_cgs, nj_cgs = n_i * 1e-6, n_j * 1e-6  # m^-3 to cm^-3
        z_ion_i = species_i.chargenumber
        z_ion_j = species_j.chargenumber
        return coulomb_logarithm_ii_jit(ni_cgs, nj_cgs, T_eV, z_ion_i, z_ion_j)


@njit
def psiconst(s: int) -> float:
    if s == 1:
        return 0
    else:
        return np.sum(1 / np.arange(1, s))


def A(ionisation_energy: float) -> float:
    """...TODO...

    Parameters
    ----------
    ionisation_energy : float
        First ionisation energy of the species, in J.

    Returns
    -------
    float
        ...

    Notes
    -----
    ...

    References
    ----------
    ...
    """
    ie_eV = ionisation_energy * u.J_to_eV  # Convert ionisation energy to eV.
    return np.sqrt(u.pi) * 9.81867945e-09 / ie_eV**0.729218856


def B(ionisation_energy: float) -> float:
    """...TODO...

    Parameters
    ----------
    ionisation_energy : float
        First ionisation energy of the species, in J.

    Returns
    -------
    float
        ...

    Notes
    -----
    ...

    References
    ----------
    ...
    """
    ie_eV = ionisation_energy * u.J_to_eV  # Convert ionisation energy to eV.
    return np.sqrt(u.pi) * 4.78257679e-10 / ie_eV**0.657012657


@njit
def sum1(s: int) -> float:
    r"""Sum of the first s+1 terms of the harmonic series, minus Euler's constant.

    Parameters
    ----------
    s : int
        Number of terms to sum.

    Returns
    -------
    float
        Sum of the first s+1 terms of the harmonic series, minus Euler's constant.

    Notes
    -----
    :math:`\zeta_1` is defined as:

    .. math::

        \zeta_1(s) = \sum_{n=1}^{s+1} \frac{1}{n} - \gamma
    """
    return np.sum(1 / np.arange(1, s + 2)) - egamma


@njit
def sum2(s: int) -> float:
    r"""Sum of the first s+1 terms squared of the harmonic series.

    Parameters
    ----------
    s : int
        Number of terms to sum.

    Returns
    -------
    float
        Sum of the first s+1 terms squared of the harmonic series.

    Notes
    -----
    :math:`\zeta_2` is defined as:

    .. math::

        \zeta_1(s) = \sum_{n=1}^{s+1} \frac{1}{n^2}
    """
    return np.sum(1 / np.arange(1, s + 2) ** 2)


@njit
def delta(i: int, j: int) -> int:
    """Kronecker delta.

    Parameters
    ----------
    i : int
        First integer.
    j : int
        Second integer.

    Returns
    -------
    int
        Returns 1 if i == j, 0 otherwise.
    """
    if i == j:
        return 1
    else:
        return 0
