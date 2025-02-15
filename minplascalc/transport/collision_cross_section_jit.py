import numpy as np
from numba import njit
from scipy import constants
from scipy.special import gamma

from minplascalc.data_transport import c_in, c_nn
from minplascalc.transport.potential_functions import (
    x0_ion_neut,
    x0_neut_neut,
    sum1,
    sum2,
)
from minplascalc.transport.potential_functions_jit import (
    beta_jit,
    coulomb_logarithm_ee_jit,
    coulomb_logarithm_ei_jit,
    coulomb_logarithm_ii_jit,
    pot_parameters_ion_neut_jit,
    pot_parameters_neut_neut_jit,
)
from minplascalc.units import Units


@njit
def Qnn_jit(
    alpha_i: float,
    alpha_j: float,
    n_eff_i: float,
    n_eff_j: float,
    spin_multiplicity_i: float,
    spin_multiplicity_j: float,
    l: int,
    s: int,
    T_eV: float,
) -> float:
    r"""Neutral-neutral elastic collision integrals.

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
    spin_multiplicity_i : float
        Spin multiplicity of species i.
    spin_multiplicity_j : float
        Spin multiplicity of species j.
    l : int
        TODO: Angular momentum quantum number? Or integer moment?
    s : int
        TODO: Principal quantum number? Or integer moment?
    T_eV : float
        Temperature, in eV.

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
    # Get the equilibrium distance r_e and binding energy epsilon_0.
    # (eq. 6 and 7 of [Laricchiuta2007]).
    r_e, epsilon_0 = pot_parameters_neut_neut_jit(alpha_i, alpha_j, n_eff_i, n_eff_j)
    # Calculate the beta parameter (eq. 5 of [Laricchiuta2007]).
    beta_value = beta_jit(alpha_i, alpha_j, spin_multiplicity_i, spin_multiplicity_j)
    # Calculate the x0 parameter (eq. 17 of [Laricchiuta2007]).
    x0 = x0_neut_neut(beta_value)
    # Evaluate the polynomial coefficients a (eq. 16 of [Laricchiuta2007]_).
    beta_array = np.array([1, beta_value, beta_value**2], dtype=np.float64)
    a = np.dot(
        c_nn[l - 1, s - 1],
        beta_array,
        out=np.zeros((7,), dtype=np.float64),
    )
    # Get the parameter sigma (Paragraph above eq. 13 of [Laricchiuta2007]).
    sigma = r_e * x0
    # Compute T* (eq. 12 of [Laricchiuta2007]).
    T_star = T_eV / epsilon_0
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
    omega = omega_reduced * np.pi * sigma**2 * 1e-20  # TODO: why pi?
    return omega


@njit
def Qin_jit(
    alpha_i: float,
    alpha_j: float,
    Z_ion: float,
    spin_multiplicity_i: float,
    spin_multiplicity_j: float,
    l: int,
    s: int,
    T_eV: float,
) -> float:
    r"""Ion-neutral elastic collision integrals.

    Parameters
    ----------
    alpha_i : float
        Polarisability of species i in Å^3.
    alpha_j : float
        Polarisability of species j in Å^3.
    Z_ion : float
        Charge of the ion in elementary charges.
    spin_multiplicity_i : float
        Spin multiplicity of species i.
    spin_multiplicity_j : float
        Spin multiplicity of species j.
    l : int
        TODO: Angular momentum quantum number? Or integer moment?
    s : int
        TODO: Principal quantum number? Or integer moment?
    T_eV : float
        Temperature, in eV.

    Returns
    -------
    float
        Ion-neutral elastic collision integral.

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
    # Get the equilibrium distance r_e and binding energy epsilon_0.
    # (eq. 9 and 10 of [Laricchiuta2007]).
    r_e, epsilon_0 = pot_parameters_ion_neut_jit(alpha_i, alpha_j, Z_ion)
    # Calculate the beta parameter (eq. 5 of [Laricchiuta2007]).
    beta_value = beta_jit(alpha_i, alpha_j, spin_multiplicity_i, spin_multiplicity_j)
    # Calculate the x0 parameter (eq. 17 of [Laricchiuta2007]).
    x0 = x0_ion_neut(beta_value)
    # Evaluate the polynomial coefficients a (eq. 16 of [Laricchiuta2007]_).
    beta_array = np.array([1, beta_value, beta_value**2], dtype=np.float64)
    a = np.dot(
        c_in[l - 1, s - 1],
        beta_array,
        out=np.zeros((7,), dtype=np.float64),
    )
    # Get the parameter sigma (Paragraph above eq. 13 of [Laricchiuta2007]).
    sigma = r_e * x0
    # Compute T* (eq. 12 of [Laricchiuta2007]).
    T_star = T_eV / epsilon_0
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
    omega = omega_reduced * np.pi * sigma**2 * 1e-20  # TODO: why pi?
    return omega


@njit
def Qtr_jit(
    s: int,
    a: float,
    b: float,
    ln_term: float,
) -> float:
    r"""Ion-neutral resonant charge transfer collision integral.

    Parameters
    ----------
    s : int
        TODO: Principal quantum number? Or integer moment?
    a : float
        ...
    b : float
        ...
    ln_term : float
        ...

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
    zeta_1 = sum1(s)
    zeta_2 = sum2(s)

    # Same as eq. 12 of [Devoto1967], with rearranged terms.
    return (
        a**2
        - zeta_1 * a * b
        + (b / 2) ** 2 * (np.pi**2 / 6 - zeta_2 + zeta_1**2)
        + (b / 2) ** 2 * ln_term**2
        + (zeta_1 * b**2 / 2 - a * b) * ln_term
    )
