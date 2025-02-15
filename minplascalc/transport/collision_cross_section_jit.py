import numpy as np
from numba import njit
from scipy import constants
from scipy.special import gamma

from minplascalc.data_transport import c_in, c_nn
from minplascalc.transport.potential_functions import x0_neut_neut
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
        Polarisability of species i in m^3.
    alpha_j : float
        Polarisability of species j in m^3.
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
