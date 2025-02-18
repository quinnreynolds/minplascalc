from typing import TYPE_CHECKING

import numpy as np
from scipy import constants
from scipy.special import gamma

from minplascalc.transport.collision_cross_section_jit import Qin_jit, Qnn_jit, Qtr_jit
from minplascalc.transport.potential_functions import (
    A,
    B,
    coulomb_logarithm_charged,
    psiconst,
)
from minplascalc.units import Units

if TYPE_CHECKING:
    from minplascalc.mixture import LTE
    from minplascalc.species import Species

u = Units()

ke = 1 / (4 * u.pi * u.epsilon_0)
a0 = constants.physical_constants["Bohr radius"][0]
egamma = np.euler_gamma


### Collision cross section calculations #######################################


def Qe(species_i: "Species", l: int, s: int, T: float) -> float:
    r"""Electron-neutral collision integrals.

    Parameters
    ----------
    species_i : Species
        Neutral species.
    l : int
        TODO: Angular momentum quantum number? Or integer moment?
    s : int
        TODO: Principal quantum number? Or integer moment?
    T : float
        Temperature, in :math:`\text{K}`.

    Returns
    -------
    float
        Electron-neutral collision integral.

    Notes
    -----
    Calculation of the electron-neutral collision integral :math:`\theta_e` from
    first principles is an extremely complex process and requires detailed knowledge
    of quantum mechanical properties of the target species.
    The complexity also increases rapidly as the atomic mass of the target increases
    and multiple excited states become relevant.
    In light of this, minplascalc opts for a simple empirical formulation which can
    be fitted to experimental or theoretical data to obtain an estimate of the
    collision integral for the neutral species of interest:

    .. math::

        \Omega_{ej}^{(l)} \approx D_1 + D_2 \left( \frac{m_r g}{\hbar} \right) ^{D_3}
            \exp \left( -D_4 \left( \frac{m_r g}{\hbar} \right)^2 \right)

    In cases where insufficient data is available, a very crude hard sphere cross section
    approximation can be implemented by specifying only :math:`D_1` and setting the remaining
    :math:`D_i` to zero. In all other cases, the :math:`D_i` are fitted to momentum cross
    section curves obtained from literature. Performing the second collision integral
    integration step then yields:

    .. math::

        \theta_e = D_1 + \frac{\Gamma(s+2+D_3/2) D_2 \tau^{D_3}}
            {\Gamma(s+2) \left( D_4 \tau^2 + 1\right) ^ {s+2+D_3/2}}

    where :math:`\tau = \frac{\sqrt{2 m_r k_B T}}{\hbar}`.

    References
    ----------
    TODO: Add references.

    See Also
    --------
    - LXCat Database: http://www.lxcat.net/
    """
    if isinstance(species_i.electroncrosssection, (tuple, list)):
        D1, D2, D3, D4 = species_i.electroncrosssection
    elif isinstance(species_i.electroncrosssection, float):
        D1, D2, D3, D4 = species_i.electroncrosssection, 0, 0, 0
    else:
        raise ValueError("Invalid electron cross section data.")
    barg = D3 / 2 + s + 2
    tau = np.sqrt(2 * u.m_e * u.k_b * T) / u.hbar
    return D1 + D2 * tau**D3 * gamma(barg) / (gamma(s + 2) * (D4 * tau**2 + 1) ** barg)


def Qnn(
    species_i: "Species",
    species_j: "Species",
    l: int,
    s: int,
    T: float,
) -> float:
    r"""Neutral-neutral elastic collision integrals.

    Parameters
    ----------
    species_i : Species
        First neutral species.
    species_j : Species
        Second neutral species.
    l : int
        TODO: Angular momentum quantum number? Or integer moment?
    s : int
        TODO: Principal quantum number? Or integer moment?
    T : float
        Temperature, in :math:`\text{K}`.

    Returns
    -------
    float
        Neutral-neutral elastic collision integral.

    Notes
    -----
    The reduced collision integral :math:`\Omega^{(\ell, s) \star}` is computed, using
    eq. 15 of [Laricchiuta2007]_, as:

    .. math::

        \begin{aligned}
            \ln \Omega^{(\ell, s) \star} =
            & {\left[a_1(\beta)+a_2(\beta) x\right]
                \frac{e^{\left(x-a_3(\beta)\right) / a_4(\beta)}}
                {e^{\left(x-a_3(\beta)\right) / a_4(\beta)}+e^{\left(a_3(\beta)-x\right) / a_4(\beta)}} } \\
            & +a_5(\beta)
                \frac{e^{\left(x-a_6(\beta)\right) / a_7(\beta)}}
                {e^{\left(x-a_6(\beta)\right) / a_7(\beta)}+e^{\left(a_6(\beta)-x\right) / a_7(\beta)}}
        \end{aligned}

    where :math:`x=\ln T^{\star}`.

    The fitting parameters are :math:`c_j`.
    They are used in in eq. (16) of [Laricchiuta2007]_ to compute the polynomial
    functions :math:`a_i(\beta)`.

    .. math::

        a_i(\beta)=\sum_{j=0}^2 c_j \beta^j

    where :math:`\beta` is a parameter to estimate the hardness of interacting electronic
    distribution densities, and it is estimated in eq. 5 of [Laricchiuta2007]_.

    The reduced temperature is defined as :math:`T^{\star}=\frac{k_b T}{\epsilon}` in eq. 12
    of [Laricchiuta2007]_, where :math:`\epsilon` is the binding energy, defined in eq. 7 of
    [Laricchiuta2007]_.
    """
    if (
        (l == 1 and s >= 6)
        or (l == 2 and s >= 5)
        or (l == 3 and s >= 4)
        or (l == 4 and s >= 5)
    ):
        # Eq. 18 of [Laricchiuta2007]_.
        # Recursion relation for the collision integral.
        negT, posT = T - 0.5, T + 0.5
        return Qnn(species_i, species_j, l, s - 1, T) + T / (s + 1) * (
            Qnn(species_i, species_j, l, s - 1, posT)
            - Qnn(species_i, species_j, l, s - 1, negT)
        )

    # Get the equilibrium distance r_e and binding energy epsilon_0.
    alpha_i, alpha_j = species_i.polarisability * 1e30, species_j.polarisability * 1e30
    n_eff_i, n_eff_j = species_i.effectiveelectrons, species_j.effectiveelectrons
    spin_multiplicity_i, spin_multiplicity_j = (
        species_i.multiplicity,
        species_j.multiplicity,
    )
    T_eV = T * u.K_to_eV

    omega = Qnn_jit(
        alpha_i,
        alpha_j,
        n_eff_i,
        n_eff_j,
        spin_multiplicity_i,
        spin_multiplicity_j,
        l,
        s,
        T_eV,
    )
    return omega


def Qin(
    species_i: "Species",
    species_j: "Species",
    l: int,
    s: int,
    T: float,
) -> float:
    r"""Ion-neutral elastic collision integrals.

    Parameters
    ----------
    species_i : Species
        First ion species.
    species_j : Species
        Second neutral species.
    l : int
        TODO: Angular momentum quantum number? Or integer moment?
    s : int
        TODO: Principal quantum number? Or integer moment?
    T : float
        Temperature, in :math:`\text{K}`.

    Returns
    -------
    float
        Ion-neutral elastic collision integral.

    Notes
    -----
    The reduced collision integral :math:`\Omega^{(\ell, s) \star}` is computed, using
    eq. 15 of [Laricchiuta2007]_, as:

    .. math::

        \begin{aligned}
            \ln \Omega^{(\ell, s) \star} =
            & {\left[a_1(\beta)+a_2(\beta) x\right]
                \frac{e^{\left(x-a_3(\beta)\right) / a_4(\beta)}}
                {e^{\left(x-a_3(\beta)\right) / a_4(\beta)}+e^{\left(a_3(\beta)-x\right) / a_4(\beta)}} } \\
            & +a_5(\beta)
                \frac{e^{\left(x-a_6(\beta)\right) / a_7(\beta)}}
                {e^{\left(x-a_6(\beta)\right) / a_7(\beta)}+e^{\left(a_6(\beta)-x\right) / a_7(\beta)}}
        \end{aligned}

    where :math:`x=\ln T^{\star}`.

    The fitting parameters are :math:`c_j`.
    They are used in in eq. (16) of [Laricchiuta2007]_ to compute the polynomial
    functions :math:`a_i(\beta)`.

    .. math::

        a_i(\beta)=\sum_{j=0}^2 c_j \beta^j

    where :math:`\beta` is a parameter to estimate the hardness of interacting electronic
    distribution densities, and it is estimated in eq. 5 of [Laricchiuta2007]_.

    The reduced temperature is defined as :math:`T^{\star}=\frac{k_b T}{\epsilon}` in eq. 12
    of [Laricchiuta2007]_, where :math:`\epsilon` is the binding energy, defined in eq. 10 of
    [Laricchiuta2007]_.
    """
    if (
        (l == 1 and s >= 6)
        or (l == 2 and s >= 5)
        or (l == 3 and s >= 4)
        or (l == 4 and s >= 5)
    ):
        # Eq. 18 of [Laricchiuta2007]_.
        # Recursion relation for the collision integral.
        negT, posT = T - 0.5, T + 0.5
        return Qin(species_i, species_j, l, s - 1, T) + T / (s + 1) * (
            Qin(species_i, species_j, l, s - 1, posT)
            - Qin(species_i, species_j, l, s - 1, negT)
        )
    # Get the equilibrium distance r_e and binding energy epsilon_0.
    alpha_i, alpha_j = species_i.polarisability * 1e30, species_j.polarisability * 1e30
    z_ion = species_i.chargenumber
    spin_multiplicity_i, spin_multiplicity_j = (
        species_i.multiplicity,
        species_j.multiplicity,
    )
    T_eV = T * u.K_to_eV
    omega = Qin_jit(
        alpha_i,
        alpha_j,
        z_ion,
        spin_multiplicity_i,
        spin_multiplicity_j,
        l,
        s,
        T_eV,
    )
    return omega


def Qtr(
    species_i: "Species",
    species_j: "Species",
    s: int,
    T: float,
) -> float:
    r"""Ion-neutral resonant charge transfer collision integral.

    Parameters
    ----------
    species_i : Species
        First species.
    species_j : Species
        Second species.
    s : int
        TODO: Principal quantum number? Or integer moment?
    T : float
        Temperature, in :math:`\text{K}`.

    Returns
    -------
    float
        Ion-neutral resonant charge transfer collision integral.

    Notes
    -----
    The resonant charge transfer collision integral is given by eq. 12 of [Devoto1967]_:

    .. math::

        \begin{aligned}
            \bar{Q}^{(1, s)}=A^2- & A B x+\left(\frac{B x}{2}\right)^2+\frac{B \zeta}{2}(B x-2 A) \\
            & +\frac{B^2}{4}\left(\frac{\pi^2}{6}-\sum_{n=1}^{s+1} \frac{1}{n^2}+\zeta^2\right) \\
            & +\frac{B}{2}[B(x+\zeta)-2 A] \ln \frac{T}{M} \\
            & +\left(\frac{B}{2} \ln \frac{T}{M}\right)^2
        \end{aligned}

    where:

    - :math:`A` and :math:`B` are given by ...TODO...,
    - :math:`x=\ln (4 R)`, with :math:`R` the gas constant,
    - :math:`\zeta=\sum_{n=1}^{s+1} \frac{1}{n} - \gamma`,
    - :math:`\gamma` is Euler's constant,
    - :math:`M` is the molar mass of the species, in :math:`\text{kg.mol}^{-1}`.

    See Also
    --------
    - https://www.wellesu.com/10.1007/978-1-4419-8172-1_4
    """
    if species_i.chargenumber < species_j.chargenumber:
        a, b = A(species_i.ionisationenergy), B(species_i.ionisationenergy)
        M = species_i.molarmass
    else:
        a, b = A(species_j.ionisationenergy), B(species_j.ionisationenergy)
        M = species_j.molarmass
    ln_term = np.log(4 * u.R * T / M)

    # Same as eq. 12 of [Devoto1967], with rearranged terms.
    return Qtr_jit(s, a, b, ln_term)


def Qc(
    species_i: "Species",
    n_i: float,
    species_j: "Species",
    n_j: float,
    l: int,
    s: int,
    T: float,
) -> float:
    r"""Coulomb collision integral.

    Parameters
    ----------
    species_i : Species
        First species.
    n_i : float
        Number density of the first species, in :math:`\text{m}^{-3}`.
    species_j : Species
        Second species.
    n_j : float
        Number density of the second species, in :math:`\text{m}^{-3}`.
    l : int
        TODO: Angular momentum quantum number? Or integer moment?
    s : int
        TODO: Principal quantum number? Or integer moment?
    T : float
        Temperature, in :math:`\text{K}`.

    Returns
    -------
    float
        Coulomb collision integral.

    Notes
    -----
    The Coulomb collision integral is given by (TODO: add reference):

    .. math::

        \theta_c=\frac{C_1 \pi}{s(s+1)}\left(\frac{z_i z_j e^2}{2 k_B T}\right)^2
            \left[\ln \Lambda-C_2-2 \bar{\gamma}+\sum_{n=1}^{s-1} \frac{1}{n}\right]

    References
    ----------
    TODO: Add references.
    """
    C1 = [4, 12, 12, 16]
    C2 = [1 / 2, 1, 7 / 6, 4 / 3]
    term1 = C1[l - 1] * u.pi / (s * (s + 1))
    term2 = (
        (
            ke  # TODO: with is there a facotr ke=1/(4*pi*eps0)? Error in documentation or code?
            * species_i.chargenumber
            * species_j.chargenumber
            * u.e**2
            / (2 * u.k_b * T)
        )
        ** 2
    )
    term3 = (
        coulomb_logarithm_charged(species_i, species_j, n_i, n_j, T)
        + np.log(2)  # TODO: why log(2)? Error in documentation or code?
        - C2[l - 1]
        - 2 * egamma
        + psiconst(s)
    )
    return term1 * term2 * term3


### Unified cross section calculations #########################################


def Qij(
    species_i: "Species",
    ni: float,
    species_j: "Species",
    nj: float,
    l: int,
    s: int,
    T: float,
) -> float:
    r"""Calculate the collision integral for a pair of species.

    Parameters
    ----------
    species_i : Species
        First species.
    ni : float
        Number density of the first species, in :math:`\text{m}^{-3}`.
    species_j : Species
        Second species.
    nj : float
        Number density of the second species, in :math:`\text{m}^{-3}`.
    l : int
        TODO: Angular momentum quantum number? Or integer moment?
    s : int
        TODO: Principal quantum number? Or integer moment?
    T : float
        Temperature, in :math:`\text{K}`.

    Returns
    -------
    float
        Collision integral.

    Raises
    ------
    ValueError
        If the collision type is unknown.
    """
    if species_i.chargenumber != 0 and species_j.chargenumber != 0:
        # For charged species, like ion-ion collisions, use the Coulomb collision integral.
        return Qc(species_i, ni, species_j, nj, l, s, T)
    elif species_j.name == "e":
        # For neutral-electron collisions, use the electron-neutral collision integral.
        return Qe(species_i, l, s, T)
    elif species_i.name == "e":
        # For electron-neutral collisions, use the electron-neutral collision integral.
        return Qe(species_j, l, s, T)
    elif species_i.chargenumber == 0 and species_j.chargenumber == 0:
        # For neutral-neutral collisions, use the neutral-neutral collision integral.
        return Qnn(species_i, species_j, l, s, T)
    elif (
        species_i.stoichiometry == species_j.stoichiometry
        and abs(species_i.chargenumber - species_j.chargenumber) == 1
        and l % 2 == 1
    ):
        # For neutral-ion (with ion charge difference of 1) --> resonant charge transfer collisions.
        return Qtr(species_i, species_j, s, T)
    elif species_i.chargenumber == 0:
        # For neutral-ion collisions, use the ion-neutral collision integral.
        return Qin(species_j, species_i, l, s, T)
    elif species_j.chargenumber == 0:
        # For ion-neutral collisions, use the ion-neutral collision integral.
        return Qin(species_i, species_j, l, s, T)
    else:
        raise ValueError("Unknown collision type")


def Qij_mix(mixture: "LTE", l: int, s: int) -> np.ndarray:
    """Calculate the collision integral matrix for a mixture of species.

    Parameters
    ----------
    mixture : LTE
        Mixture of species.
    l : int
        TODO: Angular momentum quantum number? Or integer moment?
    s : int
        TODO: Principal quantum number? Or integer moment?

    Returns
    -------
    np.ndarray
        Collision integral matrix.
    """
    # Square matrix to store the collision integrals.
    Q_values = np.zeros((len(mixture.species), len(mixture.species)))
    # Get the number densities of the species in the mixture.
    number_densities = mixture.calculate_composition()  # in m^-3

    # For all pairs of species in the mixture, calculate the corresponding collision integral.
    for i, (ndi, species_i) in enumerate(zip(number_densities, mixture.species)):
        for j, (ndj, species_j) in enumerate(zip(number_densities, mixture.species)):
            Q_values[i, j] = Qij(species_i, ndi, species_j, ndj, l, s, mixture.T)

    # Return the collision integral matrix.
    return Q_values
