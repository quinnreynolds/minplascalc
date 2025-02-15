from typing import TYPE_CHECKING

import numpy as np
from scipy import constants
from scipy.special import gamma

from minplascalc.data_transport import c_in, c_nn
from minplascalc.transport.potential_functions import (
    A,
    B,
    beta,
    coulomb_logarithm_charged,
    delta,
    pot_parameters_ion_neut,
    pot_parameters_neut_neut,
    psiconst,
    sum1,
    sum2,
    x0_ion_neut,
    x0_neut_neut,
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
        Temperature, in K.

    Returns
    -------
    float
        Electron-neutral collision integral.

    Note
    ----
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
        Temperature, in K.

    Returns
    -------
    float
        Neutral-neutral elastic collision integral.

    Notes
    -----
    The reduced collision integral \Omega^{(\ell, s) \star} is computed, using
    eq. 16 of [Laricchiuta2007]_, as:

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
    # (eq. 6 and 7 of [Laricchiuta2007]).
    r_e, epsilon_0 = pot_parameters_neut_neut(species_i, species_j)
    # Calculate the beta parameter (eq. 5 of [Laricchiuta2007]).
    beta_value = beta(species_i, species_j)
    # Calculate the x0 parameter (eq. 17 of [Laricchiuta2007]).
    x0 = x0_neut_neut(beta_value)
    # Evaluate the polynomial coefficients a (eq. 16 of [Laricchiuta2007]_).
    a = c_nn[l - 1, s - 1].dot([1, beta_value, beta_value**2])
    # Get the parameter sigma (Paragraph above eq. 13 of [Laricchiuta2007]).
    sigma = r_e * x0
    # Compute T* (eq. 12 of [Laricchiuta2007]).
    T_star = (
        u.K_to_eV * T / epsilon_0
    )  # TODO: Check this: K_to_eV or k_b? (units seem good.)
    # Calculate the parameter x (Paragraph above eq. 16 of [Laricchiuta2007]).
    x = np.log(T_star)
    # Calculate the reduced collision integral (eq. 15 of [Laricchiuta2007]).
    lnS1 = (
        (a[0] + a[1] * x)
        * np.exp((x - a[2]) / a[3])
        / (np.exp((x - a[2]) / a[3]) + np.exp((a[2] - x) / a[3]))
    )
    lnS2 = (
        a[4]
        * np.exp((x - a[5]) / a[6])
        / (np.exp((x - a[5]) / a[6]) + np.exp((a[5] - x) / a[6]))
    )
    omega_reduced = np.exp(lnS1 + lnS2)
    # Dimensional collision integral (Paragraph above eq. 17).
    omega = omega_reduced * u.pi * sigma**2 * 1e-20  # TODO: why pi?
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
        First neutral species.
    species_j : Species
        Second neutral species.
    l : int
        TODO: Angular momentum quantum number? Or integer moment?
    s : int
        TODO: Principal quantum number? Or integer moment?
    T : float
        Temperature, in K.

    Returns
    -------
    float
        Neutral-neutral elastic collision integral.

    Notes
    -----
    The reduced collision integral \Omega^{(\ell, s) \star} is computed, using
    eq. 16 of [Laricchiuta2007]_, as:

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
        return Qin(species_i, species_j, l, s - 1, T) + T / (s + 1) * (
            Qin(species_i, species_j, l, s - 1, posT)
            - Qin(species_i, species_j, l, s - 1, negT)
        )
    # Get the equilibrium distance r_e and binding energy epsilon_0.
    # (eq. 9 and 10 of [Laricchiuta2007]).
    r_e, epsilon_0 = pot_parameters_ion_neut(species_i, species_j)
    # Calculate the beta parameter (eq. 5 of [Laricchiuta2007]).
    beta_value = beta(species_i, species_j)
    # Calculate the x0 parameter (eq. 17 of [Laricchiuta2007]).
    x0 = x0_ion_neut(beta_value)
    # Evaluate the polynomial coefficients a (eq. 16 of [Laricchiuta2007]_).
    a = c_in[l - 1, s - 1].dot([1, beta_value, beta_value**2])
    # Get the parameter sigma (Paragraph above eq. 13 of [Laricchiuta2007]).
    sigma = r_e * x0
    # Compute T* (eq. 12 of [Laricchiuta2007]).
    T_star = (
        u.K_to_eV * T / epsilon_0
    )  # TODO: Check this: K_to_eV or k_b? (units seem good.)
    # Calculate the parameter x (Paragraph above eq. 16 of [Laricchiuta2007]).
    x = np.log(T_star)
    # Calculate the reduced collision integral (eq. 15 of [Laricchiuta2007]).
    lnS1 = (
        (a[0] + a[1] * x)
        * np.exp((x - a[2]) / a[3])
        / (np.exp((x - a[2]) / a[3]) + np.exp((a[2] - x) / a[3]))
    )
    lnS2 = (
        a[4]
        * np.exp((x - a[5]) / a[6])
        / (np.exp((x - a[5]) / a[6]) + np.exp((a[5] - x) / a[6]))
    )
    omega_reduced = np.exp(lnS1 + lnS2)
    # Dimensional collision integral (Paragraph above eq. 17).
    omega = omega_reduced * u.pi * sigma**2 * 1e-20  # TODO: why pi?
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
        Temperature, in K.

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
    - :math:`M` is the molar mass of the species.

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

    zeta_1 = sum1(s)
    zeta_2 = sum2(s)

    # Same as eq. 12 of [Devoto1967], with rearranged terms.
    return (
        a**2
        - zeta_1 * a * b
        + (b / 2) ** 2 * (u.pi**2 / 6 - zeta_2 + zeta_1**2)
        + (b / 2) ** 2 * ln_term**2
        + (zeta_1 * b**2 / 2 - a * b) * ln_term
    )


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
        Number density of the first species, in m^-3.
    species_j : Species
        Second species.
    n_j : float
        Number density of the second species, in m^-3.
    l : int
        TODO: Angular momentum quantum number? Or integer moment?
    s : int
        TODO: Principal quantum number? Or integer moment?
    T : float
        Temperature, in K.

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
    """Calculate the collision integral for a pair of species.

    Parameters
    ----------
    species_i : Species
        First species.
    ni : float
        Number density of the first species, in m^-3.
    species_j : Species
        Second species.
    nj : float
        Number density of the second species, in m^-3.
    l : int
        TODO: Angular momentum quantum number? Or integer moment?
    s : int
        TODO: Principal quantum number? Or integer moment?
    T : float
        Temperature, in K.

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
