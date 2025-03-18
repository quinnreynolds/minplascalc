"""Functions for transport properties calculations."""

from typing import TYPE_CHECKING

import numpy as np
from scipy import constants  # type: ignore
from scipy.special import gamma  # type: ignore

from minplascalc import units as u

if TYPE_CHECKING:
    from minplascalc.mixture import LTE
    from minplascalc.species import Species

ke = 1 / (4 * u.pi * u.epsilon_0)
a0 = constants.physical_constants["Bohr radius"][0]
egamma = np.euler_gamma


#############################################################################
# Table 2 of [Laricchiuta2007]_
# "Fitting parameters, entering in Eq. (16), of classical transport collision
# integrals \Omega^{(l,s), \star} for neutral–neutral interactions (m = 6)."
c0_nn_11 = [
    7.884756e-1,
    -2.952759e-1,
    5.020892e-1,
    -9.042460e-1,
    -3.373058,
    4.161981,
    2.462523,
]
c1_nn_11 = [
    -2.438494e-2,
    -1.744149e-3,
    4.316985e-2,
    -4.017103e-2,
    2.458538e-1,
    2.202737e-1,
    3.231308e-1,
]
c2_nn_11 = [0, 0, 0, 0, -4.850047e-3, -1.718010e-2, -2.281072e-2]
c0_nn_12 = [
    7.123565e-1,
    -2.910530e-1,
    4.187065e-2,
    -9.287685e-1,
    -3.598542,
    3.934824,
    2.578084,
]
c1_nn_12 = [
    -2.688875e-2,
    -2.065175e-3,
    4.060236e-2,
    -2.342270e-2,
    2.545120e-1,
    2.699944e-1,
    3.449024e-1,
]
c2_nn_12 = [0, 0, 0, 0, -4.685966e-3, -2.009886e-2, -2.292710e-2]
c0_nn_13 = [
    6.606022e-1,
    -2.870900e-1,
    -2.519690e-1,
    -9.173046e-1,
    -3.776812,
    3.768103,
    2.695440,
]
c1_nn_13 = [
    -2.831448e-2,
    -2.232827e-3,
    3.778211e-2,
    -1.864476e-2,
    2.552528e-1,
    3.155025e-1,
    3.597998e-1,
]
c2_nn_13 = [0, 0, 0, 0, -4.237220e-3, -2.218849e-2, -2.267102e-2]
c0_nn_14 = [
    6.268016e-1,
    -2.830834e-1,
    -4.559927e-1,
    -9.334638e-1,
    -3.947019,
    3.629926,
    2.824905,
]
c1_nn_14 = [
    -2.945078e-2,
    -2.361273e-3,
    3.705640e-2,
    -1.797329e-2,
    2.446843e-1,
    3.761272e-1,
    3.781709e-1,
]
c2_nn_14 = [0, 0, 0, 0, -3.176374e-3, -2.451016e-2, -2.251978e-2]
c0_nn_15 = [
    5.956859e-1,
    -2.804989e-1,
    -5.965551e-1,
    -8.946001e-1,
    -4.076798,
    3.458362,
    2.982260,
]
c1_nn_15 = [
    -2.915893e-2,
    -2.298968e-3,
    3.724395e-2,
    -2.550731e-2,
    1.983892e-1,
    4.770695e-1,
    4.014572e-1,
]
c2_nn_15 = [0, 0, 0, 0, -5.014065e-4, -2.678054e-2, -2.142580e-2]
c0_nn_22 = [
    7.898524e-1,
    -2.998325e-1,
    7.077103e-1,
    -8.946857e-1,
    -2.958969,
    4.348412,
    2.205440,
]
c1_nn_22 = [
    -2.114115e-2,
    -1.243977e-3,
    3.583907e-2,
    -2.473947e-2,
    2.303358e-1,
    1.920321e-1,
    2.567027e-1,
]
c2_nn_22 = [0, 0, 0, 0, -5.226562e-3, -1.496557e-2, -1.861359e-2]
c0_nn_23 = [
    7.269006e-1,
    -2.972304e-1,
    3.904230e-1,
    -9.442201e-1,
    -3.137828,
    4.190370,
    2.319751,
]
c1_nn_23 = [
    -2.233866e-2,
    -1.392888e-3,
    3.231655e-2,
    -1.494805e-2,
    2.347767e-1,
    2.346004e-1,
    2.700236e-1,
]
c2_nn_23 = [0, 0, 0, 0, -4.963979e-3, -1.718963e-2, -1.854217e-2]
c0_nn_24 = [
    6.829159e-1,
    -2.943232e-1,
    1.414623e-1,
    -9.720228e-1,
    -3.284219,
    4.011692,
    2.401249,
]
c1_nn_24 = [
    -2.332763e-2,
    -1.514322e-3,
    3.075351e-2,
    -1.038869e-2,
    2.243767e-1,
    3.005083e-1,
    2.943600e-1,
]
c2_nn_24 = [0, 0, 0, 0, -3.913041e-3, -2.012373e-2, -1.884503e-2]
c0_nn_33 = [
    7.468781e-1,
    -2.947438e-1,
    2.234096e-1,
    -9.974591e-1,
    -3.381787,
    4.094540,
    2.476087,
]
c1_nn_33 = [
    -2.518134e-2,
    -1.811571e-3,
    3.681114e-2,
    -2.670805e-2,
    2.372932e-1,
    2.756466e-1,
    3.300898e-1,
]
c2_nn_33 = [0, 0, 0, 0, -4.239629e-3, -2.009227e-2, -2.223317e-2]
c0_nn_44 = [
    7.365470e-1,
    -2.968650e-1,
    3.747555e-1,
    -9.944036e-1,
    -3.136655,
    4.145871,
    2.315532,
]
c1_nn_44 = [
    -2.242357e-2,
    -1.396696e-3,
    2.847063e-2,
    -1.378926e-2,
    2.176409e-1,
    2.855836e-1,
    2.842981e-1,
]
c2_nn_44 = [0, 0, 0, 0, -3.899247e-3, -1.939452e-2, -1.874462e-2]
c_nn_11 = np.array(
    [c0_nn_11, c1_nn_11, c2_nn_11], dtype=np.float64
).transpose()
c_nn_12 = np.array(
    [c0_nn_12, c1_nn_12, c2_nn_12], dtype=np.float64
).transpose()
c_nn_13 = np.array(
    [c0_nn_13, c1_nn_13, c2_nn_13], dtype=np.float64
).transpose()
c_nn_14 = np.array(
    [c0_nn_14, c1_nn_14, c2_nn_14], dtype=np.float64
).transpose()
c_nn_15 = np.array(
    [c0_nn_15, c1_nn_15, c2_nn_15], dtype=np.float64
).transpose()
c_nn_22 = np.array(
    [c0_nn_22, c1_nn_22, c2_nn_22], dtype=np.float64
).transpose()
c_nn_23 = np.array(
    [c0_nn_23, c1_nn_23, c2_nn_23], dtype=np.float64
).transpose()
c_nn_24 = np.array(
    [c0_nn_24, c1_nn_24, c2_nn_24], dtype=np.float64
).transpose()
c_nn_33 = np.array(
    [c0_nn_33, c1_nn_33, c2_nn_33], dtype=np.float64
).transpose()
c_nn_44 = np.array(
    [c0_nn_44, c1_nn_44, c2_nn_44], dtype=np.float64
).transpose()
fillnan = np.full(c_nn_11.shape, np.nan, dtype=np.float64)
c_nn = np.array(
    [
        [c_nn_11, c_nn_12, c_nn_13, c_nn_14, c_nn_15],
        [fillnan, c_nn_22, c_nn_23, c_nn_24, fillnan],
        [fillnan, fillnan, c_nn_33, fillnan, fillnan],
        [fillnan, fillnan, fillnan, c_nn_44, fillnan],
    ],
    dtype=np.float64,
)

#############################################################################
# Table 1 of [Laricchiuta2007]_
# "Fitting parameters, entering in Eq. (16), of classical transport collision
# integrals \Omega^{(l,s), \star} for neutral–ion interactions (m = 4)."
c0_in_11 = [
    9.851755e-1,
    -4.737800e-1,
    7.080799e-1,
    -1.239439,
    -4.638467,
    3.841835,
    2.317342,
]
c1_in_11 = [
    -2.870704e-2,
    -1.370344e-3,
    4.575312e-3,
    -3.683605e-2,
    4.418904e-1,
    3.277658e-1,
    3.912768e-1,
]
c2_in_11 = [0, 0, 0, 0, -1.220292e-2, -2.660275e-2, -3.136223e-2]
c0_in_12 = [
    8.361751e-1,
    -4.707355e-1,
    1.771157e-1,
    -1.094937,
    -4.976384,
    3.645873,
    2.428864,
]
c1_in_12 = [
    -3.201292e-2,
    -1.783284e-3,
    1.172773e-2,
    -3.115598e-2,
    4.708074e-1,
    3.699452e-1,
    4.267351e-1,
]
c2_in_12 = [0, 0, 0, 0, -1.283818e-2, -2.988684e-2, -3.278874e-2]
c0_in_13 = [
    7.440562e-1,
    -4.656306e-1,
    -1.465752e-1,
    -1.080410,
    -5.233907,
    3.489814,
    2.529678,
]
c1_in_13 = [
    -3.453851e-2,
    -2.097901e-3,
    1.446209e-2,
    -2.712029e-2,
    4.846691e-1,
    4.140270e-1,
    4.515088e-1,
]
c2_in_13 = [0, 0, 0, 0, -1.280346e-2, -3.250138e-2, -3.339293e-2]
c0_in_14 = [
    6.684360e-1,
    -4.622014e-1,
    -3.464990e-1,
    -1.054374,
    -5.465789,
    3.374614,
    2.648622,
]
c1_in_14 = [
    -3.515695e-2,
    -2.135808e-3,
    1.336362e-2,
    -3.149321e-2,
    4.888443e-1,
    4.602468e-1,
    4.677409e-1,
]
c2_in_14 = [0, 0, 0, 0, -1.228090e-2, -3.463073e-2, -3.339297e-2]
c0_in_15 = [
    6.299083e-1,
    -4.560921e-1,
    -5.228598e-1,
    -1.124725,
    -5.687354,
    3.267709,
    2.784725,
]
c1_in_15 = [
    -3.720000e-2,
    -2.395779e-3,
    1.594610e-2,
    -2.862354e-2,
    4.714789e-1,
    5.281419e-1,
    4.840700e-1,
]
c2_in_15 = [0, 0, 0, 0, -1.056602e-2, -3.678869e-2, -3.265127e-2]
c0_in_22 = [
    9.124518e-1,
    -4.697184e-1,
    1.031053,
    -1.090782,
    -4.127243,
    4.059078,
    2.086906,
]
c1_in_22 = [
    -2.398461e-2,
    -7.809681e-4,
    4.069668e-3,
    -2.413508e-2,
    4.302667e-1,
    2.597379e-1,
    2.920310e-1,
]
c2_in_22 = [0, 0, 0, 0, -1.352874e-2, -2.169951e-2, -2.560437e-2]
c0_in_23 = [
    8.073459e-1,
    -4.663682e-1,
    6.256342e-1,
    -1.063437,
    -4.365989,
    3.854346,
    2.146207,
]
c1_in_23 = [
    -2.581232e-2,
    -1.030271e-3,
    4.086881e-3,
    -1.235489e-2,
    4.391454e-1,
    3.219224e-1,
    3.325620e-1,
]
c2_in_23 = [0, 0, 0, 0, -1.314615e-2, -2.587493e-2, -2.686959e-2]
c0_in_24 = [
    7.324117e-1,
    -4.625614e-1,
    3.315871e-1,
    -1.055706,
    -4.571022,
    3.686006,
    2.217893,
]
c1_in_24 = [
    -2.727580e-2,
    -1.224292e-3,
    7.216776e-3,
    -8.585500e-3,
    4.373660e-1,
    3.854493e-1,
    3.641196e-1,
]
c2_in_24 = [0, 0, 0, 0, -1.221457e-2, -2.937568e-2, -2.763824e-2]
c0_in_33 = [
    8.402943e-1,
    -4.727437e-1,
    4.724228e-1,
    -1.213660,
    -4.655574,
    3.817178,
    2.313186,
]
c1_in_33 = [
    -2.851694e-2,
    -1.328784e-3,
    7.706027e-3,
    -3.456656e-2,
    4.467685e-1,
    3.503180e-1,
    3.889828e-1,
]
c2_in_33 = [0, 0, 0, 0, -1.237864e-2, -2.806506e-2, -3.120619e-2]
c0_in_44 = [
    8.088842e-1,
    -4.659483e-1,
    6.092981e-1,
    -1.113323,
    -4.349145,
    3.828467,
    2.138075,
]
c1_in_44 = [
    -2.592379e-2,
    -1.041599e-3,
    1.428402e-3,
    -1.031574e-2,
    4.236246e-1,
    3.573461e-1,
    3.388072e-1,
]
c2_in_44 = [0, 0, 0, 0, -1.210668e-2, -2.759622e-2, -2.669344e-2]
c_in_11 = np.array(
    [c0_in_11, c1_in_11, c2_in_11], dtype=np.float64
).transpose()
c_in_12 = np.array(
    [c0_in_12, c1_in_12, c2_in_12], dtype=np.float64
).transpose()
c_in_13 = np.array(
    [c0_in_13, c1_in_13, c2_in_13], dtype=np.float64
).transpose()
c_in_14 = np.array(
    [c0_in_14, c1_in_14, c2_in_14], dtype=np.float64
).transpose()
c_in_15 = np.array(
    [c0_in_15, c1_in_15, c2_in_15], dtype=np.float64
).transpose()
c_in_22 = np.array(
    [c0_in_22, c1_in_22, c2_in_22], dtype=np.float64
).transpose()
c_in_23 = np.array(
    [c0_in_23, c1_in_23, c2_in_23], dtype=np.float64
).transpose()
c_in_24 = np.array(
    [c0_in_24, c1_in_24, c2_in_24], dtype=np.float64
).transpose()
c_in_33 = np.array(
    [c0_in_33, c1_in_33, c2_in_33], dtype=np.float64
).transpose()
c_in_44 = np.array(
    [c0_in_44, c1_in_44, c2_in_44], dtype=np.float64
).transpose()
fillnan = np.full(c_in_11.shape, np.nan, dtype=np.float64)
c_in = np.array(
    [
        [c_in_11, c_in_12, c_in_13, c_in_14, c_in_15],
        [fillnan, c_in_22, c_in_23, c_in_24, fillnan],
        [fillnan, fillnan, c_in_33, fillnan, fillnan],
        [fillnan, fillnan, fillnan, c_in_44, fillnan],
    ],
    dtype=np.float64,
)


#############################################################################
# Potentials functions.


def n_effective_electrons(nint, nout):
    return nout * (1 + (1 - nout / nint) * (nint / (nout + nint)) ** 2)


def pot_parameters_neut_neut(
    species_i: "Species", species_j: "Species"
) -> tuple[float, float]:
    r"""Calculate potential parameters for a neutral-neutral pair.

    Calculate the equilibrium distance and binding energy for a
    neutral-neutral pair.

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
    The equilibrium distance is given by eq. 6 of [Laricchiuta2007]_ by the
    formula:

    .. math::

        r_e = 1.767 \frac{\alpha_1^{1 / 3}+\alpha_2^{1 / 3}}
                         {\left(\alpha_1 \alpha_2\right)^{0.095}}

    where :math:`\alpha_i` is the polarisability of species :math:`i` in m^3.


    The binding energy is given by eq. 7 of [Laricchiuta2007]_ by the formula:

    .. math::

        \epsilon_0 = 0.72 \frac{C_d}{r_e^6}

    where :math:`C_d` is the effective long-range London coefficient, defined
    in eq. 8 of [Laricchiuta2007]_ by the formula:

    .. math::

        C_d = 15.7 \frac{\alpha_1 \alpha_2}
                    {\sqrt{\frac{\alpha_1}{n_1}} + \sqrt{\frac{\alpha_2}{n_2}}}

    with :math:`n_i` the effective number of electrons of species :math:`i`.
    """
    # Polarisabilities of the species, in m^3.
    alpha_i = species_i.polarisability * 1e30
    alpha_j = species_j.polarisability * 1e30
    # Effective long-range London coefficient, as defined in eq. 8 of
    # [Laricchiuta2007]_.
    if (
        species_i.effective_electrons is None
        or species_j.effective_electrons is None
    ):
        raise ValueError(
            "Effective number of electrons must be provided for neutral"
            " species."
        )
    n_eff_i = species_i.effective_electrons
    n_eff_j = species_j.effective_electrons

    # Effective long-range London coefficient, as defined in eq. 8 of
    # [Laricchiuta2007]_.
    C_d = (
        15.7
        * alpha_i
        * alpha_j
        / (np.sqrt(alpha_i / n_eff_i) + np.sqrt(alpha_j / n_eff_j))
    )
    # Equilibrium distance r_e, as defined in eq. 6 of [Laricchiuta2007]_.
    r_e = (
        1.767
        * (alpha_i ** (1 / 3) + alpha_j ** (1 / 3))
        / (alpha_i * alpha_j) ** 0.095
    )
    # Binding energy epsilon_0, as defined in eq. 7 of [Laricchiuta2007]_.
    epsilon_0 = 0.72 * C_d / r_e**6

    # Return the equilibrium distance and the binding energy.
    return r_e, epsilon_0


def pot_parameters_ion_neut(
    species_ion: "Species",
    species_neutral: "Species",
) -> tuple[float, float]:
    r"""Calculate potential parameters for an ion-neutral pair.

    Calculate the equilibrium distance and binding energy for an
    ion-neutral pair.

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
    The equilibrium distance is given by eq. 9 of [Laricchiuta2007]_ by the
    formula:

    .. math::

        r_e = 1.767 \frac{\alpha_i^{1 / 3}+\alpha_n^{1 / 3}}
                         {\left(\alpha_i \alpha_n
                         \left[ 1 + \frac{1}{\rho}\right] \right)^{0.095}}

    where :math:`\alpha_i` is the polarisability of the ion species and
    :math:`\alpha_n` is the polarisability of the neutral species, both in m^3.

    The binding energy is given by eq. 10 of [Laricchiuta2007]_ by the formula:

    .. math::

        \epsilon_0 = 5.2 \frac{z^2 \alpha_n}{r_e^4} \left(1 + \rho\right)

    where :math:`z` is the charge number of the ion species, :math:`\alpha_n`
    is the polarisability of the neutral species in m^3

    :math:`\rho` is representative of the relative role of dispersion and
    induction attraction components in proximity to the equilibrium distance,
    defined in eq. 11 of [Laricchiuta2007]_ by the formula:

    .. math::

        \rho = \frac{\alpha_i}
                    {z^2 \sqrt{\alpha_n} \left(1 + \left(2 \alpha_i /
                    \alpha_n\right)^{2 / 3}\right)}

    """
    # Polarisabilities of the species, in m^3.
    alpha_i = species_ion.polarisability * 1e30
    alpha_n = species_neutral.polarisability * 1e30
    # Charge number of the ion species.
    Z_ion = species_ion.charge_number

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
    :math:`\beta` is a parameter to estimate the hardness of interacting
    electronic distribution densities, and it is estimated in eq. 5 of
    [Laricchiuta2007]_:

    .. math::

        \beta = 6 + \frac{5}{s_1 + s_2}

    where :math:`s_i` is the softness.

    The softness s defined as the cubic root of the polarizability.
    For open-shell atoms and ions a multiplicative factor, which is the ground
    state spin multiplicity, should be also considered:

    .. math::

        s = \frac{\alpha^{1 / 3}}{m}
    """
    # Polarisabilities of the species, in m^3.
    alpha_i = species_i.polarisability * 1e30
    alpha_j = species_j.polarisability * 1e30

    # Compute the softness of the species.
    s_i = alpha_i ** (1 / 3) * species_i.multiplicity
    s_j = alpha_j ** (1 / 3) * species_j.multiplicity
    # Return the beta parameter.
    return 6 + 5 / (s_i + s_j)


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
    :math:`x_0` is defined in eq. 13 of [Laricchiuta2007]_ as a solution to a
    transcendal equation. It can be approximated by eq. 17,
    with the following formula:

    .. math::

        x_0 = \xi_1 \beta^{\xi_2}

    where :math:`\xi_1 = 0.8002` and :math:`\xi_2 = 0.049256`,
    as given in Table 3.
    """
    return 0.8002 * beta_value**0.049256


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
    :math:`x_0` is defined in eq. 13 of [Laricchiuta2007]_ as a solution to a
    transcendal equation. It can be approximated by eq. 17,
    with the following formula:

    .. math::

        x_0 = \xi_1 \beta^{\xi_2}

    where :math:`\xi_1 = 0.7564` and :math:`\xi_2 = 0.064605`,
    as given in Table 3.
    """
    return 0.7564 * beta_value**0.064605


def cl_charged(
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
            \left(\frac{n_i Z_i^2}{T} + \frac{n_i' Z_i'^2}{T} \right)^{1 / 2}
            \right)
    """
    T_eV = T * u.K_to_eV  # Convert temperature to eV.
    if species_i.name == "e" and species_j.name == "e":
        # Electron-electron collisions.
        ne_cgs = n_i * 1e-6  # m^-3 to cm^-3
        return (
            23.5
            - np.log(ne_cgs ** (1 / 2) * T_eV ** (-5 / 4))
            - (1e-5 + (np.log(T_eV) - 2) ** 2 / 16) ** (1 / 2)
        )
    elif species_i.name == "e":
        # Electron-ion collisions.
        ne_cgs = n_i * 1e-6  # m^-3 to cm^-3
        z_ion = species_j.charge_number
        return 23 - np.log(ne_cgs ** (1 / 2) * abs(z_ion) * T_eV ** (-3 / 2))
    elif species_j.name == "e":
        # Ion-electron collisions, same as electron-ion collisions.
        ne_cgs = n_j * 1e-6  # m^-3 to cm^-3
        z_ion = species_i.charge_number
        return 23 - np.log(ne_cgs ** (1 / 2) * abs(z_ion) * T_eV ** (-3 / 2))
    else:
        # Ion-ion collisions.
        ni_cgs, nj_cgs = n_i * 1e-6, n_j * 1e-6  # m^-3 to cm^-3
        z_ion_i = species_i.charge_number
        z_ion_j = species_j.charge_number
        return 23 - np.log(
            abs(z_ion_i * z_ion_j)
            / T_eV
            * (
                ni_cgs * abs(z_ion_i) ** 2 / T_eV
                + nj_cgs * abs(z_ion_j) ** 2 / T_eV
            )
            ** (1 / 2)
        )


def psiconst(s: int) -> float:
    r"""Calculate the constant psi(s).

    It is used in the fit function of the total resonant transfert
    cross section.

    Parameters
    ----------
    s : int
        Number of terms to sum.

    Returns
    -------
    float
        Constant psi(s).

    Notes
    -----
    :math:`\psi(s)` is defined as:

    .. math::

        \psi(s) = \sum_{n=1}^{s} \frac{1}{n}
    """
    if s == 1:
        return 0
    else:
        return np.sum(1 / np.arange(1, s))


def A(ionisation_energy: float) -> float:
    r"""Calculate the constant A.

    Constant A for the fit function of the total resonant transfert
    cross section.

    Parameters
    ----------
    ionisation_energy : float
        First ionisation energy of the species, in J.

    Returns
    -------
    float
        Constant A.

    Notes
    -----
    :math:`A` and :math:`B` are given by a simple empirical fit as a function
    of the ionisation energy to analytical expressions from [Rapp1962]_
    and [Smirnov1970]_.

    The fit function is given by eq. 11 of [Devoto1967]_:

    .. math::

        Q_{ T }^{T O T}=(1 / 2)[A-B \ln (g)]^2

    From equation 14 of [Rapp1962]_, we have:

    .. math::

        \begin{aligned}
            \sigma^{\frac{1}{2}}
                =\left(\frac{1}{2} \pi\right)^{\frac{1}{2}} b_1
                =-\left(\frac{1}{2} \pi\right)^{\frac{1}{2}} & \left(a_0 /
                \gamma\right) \ln v
                 +\left(\frac{1}{2} \pi\right)^{\frac{1}{2}}\left(a_0 / 2
                 \gamma\right) \\
                 & \times \ln \left[\frac{72 \bar{b}_1^3}{\pi \gamma a_0}
                 \left(\frac{I^2}{\hbar^2}\right)
                    \left(1+\frac{a_0}{\gamma \bar{b}_1}\right)^2\right]
        \end{aligned}

    where:

    - :math:`\gamma = \sqrt{\frac{I[eV]}{13.6}}`
      (cf. equation 10 of [Rapp1962]_),
    - :math:`a_0` is the Bohr radius,
    - :math:`I` is the ionisation energy in eV,
    - :math:`\hbar` is the reduced Planck constant, and
    - :math:`\bar{b}_1` can be found using equation 13 of [Rapp1962]_.

    Therefore, :math:`A` is given by:

    .. math::

        A = \sqrt{\pi} \frac{a_0}{2 \gamma}
            \ln \left[\frac{72 \bar{b}_1^3}{\pi \gamma a_0}\left(\frac{I^2}
                {\hbar^2}\right)
            \left(1+\frac{a_0}{\gamma \bar{b}_1}\right)^2\right]

    TODO: Check the formula for A.
    """
    ie_eV = ionisation_energy * u.J_to_eV  # Convert ionisation energy to eV.
    return np.sqrt(np.pi) * 9.81867945e-09 / ie_eV**0.729218856


def B(ionisation_energy: float) -> float:
    r"""Calculate the constant B.

    Constant B for the fit function of the total resonant transfert
    cross section.

    Parameters
    ----------
    ionisation_energy : float
        First ionisation energy of the species, in J.

    Returns
    -------
    float
        Constant B.

    Notes
    -----
    :math:`A` and :math:`B` are given by a simple empirical fit as a function
    of the ionisation energy to analytical expressions from [Rapp1962]_ and
    [Smirnov1970]_.

    The fit function is given by eq. 11 of [Devoto1967]_:

    .. math::

        Q_{ T }^{T O T}=(1 / 2)[A-B \ln (g)]^2

    From equation 14 of [Rapp1962]_, we have:

    .. math::

        \begin{aligned}
            \sigma^{\frac{1}{2}}
                =\left(\frac{1}{2} \pi\right)^{\frac{1}{2}} b_1
                =-\left(\frac{1}{2} \pi\right)^{\frac{1}{2}} & \left(a_0 /
                \gamma\right) \ln v
                 +\left(\frac{1}{2} \pi\right)^{\frac{1}{2}}\left(a_0 / 2
                 \gamma\right) \\
                 & \times \ln \left[\frac{72 \bar{b}_1^3}{\pi \gamma a_0}
                 \left(\frac{I^2}{\hbar^2}\right)
                    \left(1+\frac{a_0}{\gamma \bar{b}_1}\right)^2\right]
        \end{aligned}

    where:

    - :math:`\gamma = \sqrt{\frac{I[eV]}{13.6}}` (cf. equation 10 of
      [Rapp1962]_),
    - :math:`a_0` is the Bohr radius,
    - :math:`I` is the ionisation energy in eV,
    - :math:`\hbar` is the reduced Planck constant, and
    - :math:`\bar{b}_1` can be found using equation 13 of [Rapp1962]_.

    Therefore, :math:`B` is given by:

    .. math::

        B = \pi^{\frac{1}{2}} \frac{a_0}{\gamma}


    TODO: Check the following:

    .. math::

        B = \sqrt{\pi} \frac{a_0 \sqrt{13.6} }{\sqrt{I[eV]}}
          = \sqrt{\pi} \times \frac{1,95.10^{-10}}{ \sqrt{I[eV]}}

    However, the function returns
    :math:`\sqrt{\pi} \times \frac{4.78.10^{-10}}{ I[eV]^{0.657}}`.
    """
    ie_eV = ionisation_energy * u.J_to_eV  # Convert ionisation energy to eV.
    return np.sqrt(np.pi) * 4.78257679e-10 / ie_eV**0.657012657


def sum1(s: int) -> float:
    r"""Sum of the first s+1 terms of the harmonic series, minus gamma.

    Gamma is the Euler-Mascheroni constant.

    Parameters
    ----------
    s : int
        Number of terms to sum.

    Returns
    -------
    float
        Sum of the first s+1 terms of the harmonic series, minus gamma.

    Notes
    -----
    :math:`\zeta_1` is defined as:

    .. math::

        \zeta_1(s) = \sum_{n=1}^{s+1} \frac{1}{n} - \gamma
    """
    return np.sum(1 / np.arange(1, s + 2)) - egamma


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


### Collision cross section calculations ######################################


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
    Calculation of the electron-neutral collision integral :math:`\theta_e`
    from first principles is an extremely complex process and requires
    detailed knowledge of quantum mechanical properties of the target species.
    The complexity also increases rapidly as the atomic mass of the target
    increases and multiple excited states become relevant.
    In light of this, minplascalc opts for a simple empirical formulation which
    can be fitted to experimental or theoretical data to obtain an estimate of
    the collision integral for the neutral species of interest:

    .. math::

        \Omega_{ej}^{(l)} \approx
            D_1
            + D_2 \left( \frac{m_r g}{\hbar} \right) ^{D_3}
            \exp \left( -D_4 \left( \frac{m_r g}{\hbar} \right)^2 \right)

    In cases where insufficient data is available, a very crude hard sphere
    cross section approximation can be implemented by specifying only
    :math:`D_1` and setting the remaining :math:`D_i` to zero. In all other
    cases, the :math:`D_i` are fitted to momentum cross section curves obtained
    from literature. Performing the second collision integral integration step
    then yields:

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
    if isinstance(species_i.electron_cross_section, (tuple, list)):
        D1, D2, D3, D4 = species_i.electron_cross_section
    elif isinstance(species_i.electron_cross_section, float):
        D1, D2, D3, D4 = species_i.electron_cross_section, 0, 0, 0
    else:
        raise ValueError("Invalid electron cross section data.")
    barg = D3 / 2 + s + 2
    tau = np.sqrt(2 * u.m_e * u.k_b * T) / u.hbar
    return D1 + D2 * tau**D3 * gamma(barg) / (
        gamma(s + 2) * (D4 * tau**2 + 1) ** barg
    )


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
    The reduced collision integral :math:`\Omega^{(\ell, s) \star}` is
    computed, using eq. 15 of [Laricchiuta2007]_, as:

    .. math::

        \begin{aligned}
            \ln \Omega^{(\ell, s) \star} =
            & {\left[a_1(\beta)+a_2(\beta) x\right]
                \frac{e^{\left(x-a_3(\beta)\right) / a_4(\beta)}}
                {e^{\left(x-a_3(\beta)\right) / a_4(\beta)}+e^{
                    \left(a_3(\beta)-x\right) / a_4(\beta)}} } \\
            & +a_5(\beta)
                \frac{e^{\left(x-a_6(\beta)\right) / a_7(\beta)}}
                {e^{\left(x-a_6(\beta)\right) / a_7(\beta)}+e^{
                    \left(a_6(\beta)-x\right) / a_7(\beta)}}
        \end{aligned}

    where :math:`x=\ln T^{\star}`.

    The fitting parameters are :math:`c_j`.
    They are used in in eq. (16) of [Laricchiuta2007]_ to compute the
    polynomial functions :math:`a_i(\beta)`.

    .. math::

        a_i(\beta)=\sum_{j=0}^2 c_j \beta^j

    where :math:`\beta` is a parameter to estimate the hardness of interacting
    electronic distribution densities, and it is estimated in
    eq. 5 of [Laricchiuta2007]_.

    The reduced temperature is defined as
    :math:`T^{\star}=\frac{k_b T}{\epsilon}` in eq. 12 of [Laricchiuta2007]_,
    where :math:`\epsilon` is the binding energy, defined in eq. 7 of
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
    beta_array = np.array([1, beta_value, beta_value**2], dtype=np.float64)
    a = np.dot(
        c_nn[l - 1, s - 1],
        beta_array,
        out=np.zeros((7,), dtype=np.float64),
    )
    # Get the parameter sigma (Paragraph above eq. 13 of [Laricchiuta2007]).
    sigma = r_e * x0
    # Compute T* (eq. 12 of [Laricchiuta2007]).
    T_star = T * u.K_to_eV / epsilon_0
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
    The reduced collision integral :math:`\Omega^{(\ell, s) \star}` is
    computed, using eq. 15 of [Laricchiuta2007]_, as:

    .. math::

        \begin{aligned}
            \ln \Omega^{(\ell, s) \star} =
            & {\left[a_1(\beta)+a_2(\beta) x\right]
                \frac{e^{\left(x-a_3(\beta)\right) / a_4(\beta)}}
                {e^{\left(x-a_3(\beta)\right) / a_4(\beta)}+e^{
                    \left(a_3(\beta)-x\right) / a_4(\beta)}} } \\
            & +a_5(\beta)
                \frac{e^{\left(x-a_6(\beta)\right) / a_7(\beta)}}
                {e^{\left(x-a_6(\beta)\right) / a_7(\beta)}+e^{
                    \left(a_6(\beta)-x\right) / a_7(\beta)}}
        \end{aligned}

    where :math:`x=\ln T^{\star}`.

    The fitting parameters are :math:`c_j`.
    They are used in in eq. (16) of [Laricchiuta2007]_ to compute the
    polynomial functions :math:`a_i(\beta)`.

    .. math::

        a_i(\beta)=\sum_{j=0}^2 c_j \beta^j

    where :math:`\beta` is a parameter to estimate the hardness of interacting
    electronic distribution densities, and it is estimated in eq. 5 of
    [Laricchiuta2007]_.

    The reduced temperature is defined as
    :math:`T^{\star}=\frac{k_b T}{\epsilon}` in eq. 12 of [Laricchiuta2007]_,
    where :math:`\epsilon` is the binding energy, defined in eq. 10 of
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
    beta_array = np.array([1, beta_value, beta_value**2], dtype=np.float64)
    a = np.dot(
        c_in[l - 1, s - 1],
        beta_array,
        out=np.zeros((7,), dtype=np.float64),
    )
    # Get the parameter sigma (Paragraph above eq. 13 of [Laricchiuta2007]).
    sigma = r_e * x0
    # Compute T* (eq. 12 of [Laricchiuta2007]).
    T_star = T * u.K_to_eV / epsilon_0
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
    The resonant charge transfer collision integral is given by eq. 12 of
    [Devoto1967]_:

    .. math::

        \begin{aligned}
            \bar{Q}^{(1, s)}=A^2- & A B x+\left(\frac{B x}{2}\right)^2+\frac{
                B \zeta}{2}(B x-2 A) \\
            & +\frac{B^2}{4}\left(\frac{\pi^2}{6}-\sum_{n=1}^{s+1}
                \frac{1}{n^2}+\zeta^2\right) \\
            & +\frac{B}{2}[B(x+\zeta)-2 A] \ln \frac{T}{M} \\
            & +\left(\frac{B}{2} \ln \frac{T}{M}\right)^2
        \end{aligned}

    where:

    - :math:`A` and :math:`B` are given by a simple empirical fit as a
      function of the ionisation energy to analytical expressions
      from [Rapp1962]_ and [Smirnov1970]_.
    - :math:`x=\ln (4 R)`, with :math:`R` the gas constant,
    - :math:`\zeta=\sum_{n=1}^{s+1} \frac{1}{n} - \gamma`,
    - :math:`\gamma` is Euler's constant,
    - :math:`M` is the molar mass of the species,
      in :math:`\text{kg.mol}^{-1}`.

    See Also
    --------
    - https://www.wellesu.com/10.1007/978-1-4419-8172-1_4
    """
    if species_i.charge_number < species_j.charge_number:
        a, b = A(species_i.ionisation_energy), B(species_i.ionisation_energy)
        M = species_i.molar_mass
    else:
        a, b = A(species_j.ionisation_energy), B(species_j.ionisation_energy)
        M = species_j.molar_mass
    ln_term = np.log(4 * u.R * T / M)
    zeta_1, zeta_2 = sum1(s), sum2(s)
    cterm = np.pi**2 / 6 - zeta_2 + zeta_1**2

    # Same as eq. 12 of [Devoto1967], with rearranged terms.
    return (
        a**2
        - zeta_1 * a * b
        + (b / 2) ** 2 * cterm
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
    The Coulomb collision integrals are given by equation 5 of [Devoto1967]_:

    .. math::

        \begin{aligned}
        \bar{Q}^{(1,s)}=\frac{4\pi}{s(s+1)} b_0^2\left[\ln \Lambda-\frac{1}{2}
          -2\bar{\gamma}+\psi(s)\right]\\
        \bar{Q}^{(2,s)}=\frac{12\pi}{s(s+1)} b_0^2\left[\ln \Lambda-1
          -2 \bar{\gamma}+\psi(s)\right]\\
        \bar{Q}^{(3,s)}=\frac{12\pi}{s(s+1)} b_0^2\left[\ln \Lambda-\frac{7}{6}
          -2\bar{\gamma}+\psi(s)\right]\\
        \bar{Q}^{(4,s)}=\frac{16\pi}{s(s+1)} b_0^2\left[\ln \Lambda-\frac{4}{3}
          -2\bar{\gamma}+\psi(s)\right]
        \end{aligned}

    where:

    - :math:`b_0=\frac{Z_i Z_j e^2}{2k_B T}`,
    - :math:`\ln \Lambda` is the Coulomb logarithm,
    - :math:`\bar{\gamma=0.5772...}` is Euler's constant,
    - :math:`\psi(s) = \sum_{n=1}^{s+1} \frac{1}{n}`.
    """
    C1 = [4, 12, 12, 16]
    C2 = [1 / 2, 1, 7 / 6, 4 / 3]
    term1 = C1[l - 1] * np.pi / (s * (s + 1))
    term2 = (
        ke  # TODO: with is there a facotr ke=1/(4*pi*eps0)?
        # Error in documentation or code?
        * species_i.charge_number
        * species_j.charge_number
        * u.e**2
        / (2 * u.k_b * T)
    ) ** 2
    term3 = (
        cl_charged(species_i, species_j, n_i, n_j, T)
        + np.log(2)
        - C2[l - 1]
        - 2 * egamma
        + psiconst(s)
    )
    return term1 * term2 * term3


### Unified cross section calculations ########################################


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
    if species_i.charge_number != 0 and species_j.charge_number != 0:
        # For charged species, like ion-ion collisions,
        # use the Coulomb collision integral.
        return Qc(species_i, ni, species_j, nj, l, s, T)
    elif species_j.name == "e":
        # For neutral-electron collisions,
        # use the electron-neutral collision integral.
        return Qe(species_i, l, s, T)
    elif species_i.name == "e":
        # For electron-neutral collisions,
        # use the electron-neutral collision integral.
        return Qe(species_j, l, s, T)
    elif species_i.charge_number == 0 and species_j.charge_number == 0:
        # For neutral-neutral collisions,
        # use the neutral-neutral collision integral.
        return Qnn(species_i, species_j, l, s, T)
    elif (
        species_i.stoichiometry == species_j.stoichiometry
        and abs(species_i.charge_number - species_j.charge_number) == 1
        and l % 2 == 1
    ):
        # For neutral-ion (with ion charge difference of 1),
        # use the resonant charge transfer collisions.
        return Qtr(species_i, species_j, s, T)
    elif species_i.charge_number == 0:
        # For neutral-ion collisions,
        # use the ion-neutral collision integral.
        return Qin(species_j, species_i, l, s, T)
    elif species_j.charge_number == 0:
        # For ion-neutral collisions,
        # use the ion-neutral collision integral.
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

    # For all pairs of species in the mixture, calculate the corresponding
    # collision integral.
    for i, (ndi, species_i) in enumerate(
        zip(number_densities, mixture.species)
    ):
        for j, (ndj, species_j) in enumerate(
            zip(number_densities, mixture.species)
        ):
            Q_values[i, j] = Qij(
                species_i, ndi, species_j, ndj, l, s, mixture.T
            )

    # Return the collision integral matrix.
    return Q_values


### q-matrix calculations #####################################################


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
    The various elements of the q-matrix are calculated from the appendix of
    [Devoto1966]_.
    """
    nb_species = len(mixture.species)
    number_densities = mixture.calculate_composition()  # m^-3
    masses = np.array(
        [species.molar_mass / u.N_a for species in mixture.species]
    )  # kg

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
            sumval = 0
            for l in range(nb_species):
                term1 = (
                    number_densities[l]
                    * masses[i] ** (1 / 2)
                    / (masses[i] + masses[l]) ** (1 / 2)
                )
                term2 = number_densities[i] * (masses[l] / masses[j]) ** (
                    1 / 2
                ) * (delta(i, j) - delta(j, l)) - number_densities[j] * (
                    masses[l] * masses[j]
                ) ** (1 / 2) / masses[i] * (1 - delta(i, l))
                sumval += term1 * Q11[i, l] * term2
            q00[i, j] = 8 * sumval

    # Equation A4 of [Devoto1966]_.
    q01 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            sumval = 0
            for l in range(nb_species):
                term1 = (
                    number_densities[l]
                    * masses[l] ** (3 / 2)
                    / (masses[i] + masses[l]) ** (3 / 2)
                )
                term2 = (delta(i, j) - delta(j, l)) * (
                    5 / 2 * Q11[i, l] - 3 * Q12[i, l]
                )
                sumval += term1 * term2
            q01[i, j] = (
                8
                * number_densities[i]
                * (masses[i] / masses[j]) ** (3 / 2)
                * sumval
            )

    # Equation A6 of [Devoto1966]_.
    q11 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            sumval = 0
            for l in range(nb_species):
                term1 = (
                    number_densities[l]
                    * masses[l] ** (1 / 2)
                    / (masses[i] + masses[l]) ** (5 / 2)
                )
                term2 = (delta(i, j) - delta(j, l)) * (
                    5
                    / 4
                    * (6 * masses[j] ** 2 + 5 * masses[l] ** 2)
                    * Q11[i, l]
                    - 15 * masses[l] ** 2 * Q12[i, l]
                    + 12 * masses[l] ** 2 * Q13[i, l]
                ) + (delta(i, j) + delta(j, l)) * 4 * masses[j] * masses[
                    l
                ] * Q22[i, l]
                sumval += term1 * term2
            q11[i, j] = (
                8
                * number_densities[i]
                * (masses[i] / masses[j]) ** (3 / 2)
                * sumval
            )

    # Equation A7 of [Devoto1966]_.
    q02 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            sumval = 0
            for l in range(nb_species):
                term1 = (
                    number_densities[l]
                    * masses[l] ** (5 / 2)
                    / (masses[i] + masses[l]) ** (5 / 2)
                )
                term2 = (delta(i, j) - delta(j, l)) * (
                    35 / 8 * Q11[i, l] - 21 / 2 * Q12[i, l] + 6 * Q13[i, l]
                )
                sumval += term1 * term2
            q02[i, j] = (
                8
                * number_densities[i]
                * (masses[i] / masses[j]) ** (5 / 2)
                * sumval
            )

    # Equation A9 of [Devoto1966]_.
    q12 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            sumval = 0
            for l in range(nb_species):
                term1 = (
                    number_densities[l]
                    * masses[l] ** (3 / 2)
                    / (masses[i] + masses[l]) ** (7 / 2)
                )
                term2 = (delta(i, j) - delta(j, l)) * (
                    35
                    / 16
                    * (12 * masses[j] ** 2 + 5 * masses[l] ** 2)
                    * Q11[i, l]
                    - 63
                    / 2
                    * (masses[j] ** 2 + 5 / 4 * masses[l] ** 2)
                    * Q12[i, l]
                    + 57 * masses[l] ** 2 * Q13[i, l]
                    - 30 * masses[l] ** 2 * Q14[i, l]
                ) + (delta(i, j) + delta(j, l)) * (
                    14 * masses[j] * masses[l] * Q22[i, l]
                    - 16 * masses[j] * masses[l] * Q23[i, l]
                )
                sumval += term1 * term2
            q12[i, j] = (
                8
                * number_densities[i]
                * (masses[i] / masses[j]) ** (5 / 2)
                * sumval
            )

    # Equation A11 of [Devoto1966]_.
    q22 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            sumval = 0
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
                    + 24 * (masses[j] * masses[l]) ** 2 * Q33[i, l]
                ) + (delta(i, j) + delta(j, l)) * (
                    7
                    * masses[j]
                    * masses[l]
                    * (4 * (masses[j] ** 2 + 7 * masses[l] ** 2))
                    * Q22[i, l]
                    - 112 * masses[j] * masses[l] ** 3 * Q23[i, l]
                    + 80 * masses[j] * masses[l] ** 3 * Q24[i, l]
                )
                sumval += term1 * term2
            q22[i, j] = (
                8
                * number_densities[i]
                * (masses[i] / masses[j]) ** (5 / 2)
                * sumval
            )

    # Equation A12 of [Devoto1966]_.
    q03 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            sumval = 0
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
                sumval += term1 * term2
            q03[i, j] = (
                8
                * number_densities[i]
                * (masses[i] / masses[j]) ** (7 / 2)
                * sumval
            )

    # Equation A14 of [Devoto1966]_.
    q13 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            sumval = 0
            for l in range(nb_species):
                term1 = (
                    number_densities[l]
                    * masses[l] ** (5 / 2)
                    / (masses[i] + masses[l]) ** (9 / 2)
                )
                term2 = (delta(i, j) - delta(j, l)) * (
                    105
                    / 32
                    * (18 * masses[j] ** 2 + 5 * masses[l] ** 2)
                    * Q11[i, l]
                    - 63
                    / 4
                    * (9 * masses[j] ** 2 + 5 * masses[l] ** 2)
                    * Q12[i, l]
                    + 81 * (masses[j] ** 2 + 2 * masses[l] ** 2) * Q13[i, l]
                    - 160 * masses[l] ** 2 * Q14[i, l]
                    + 60 * masses[l] ** 2 * Q15[i, l]
                ) + (delta(i, j) + delta(j, l)) * masses[j] * masses[l] * (
                    63 / 2 * Q22[i, l] - 72 * Q23[i, l] + 40 * Q24[i, l]
                )
                sumval += term1 * term2
            q13[i, j] = (
                8
                * number_densities[i]
                * (masses[i] / masses[j]) ** (7 / 2)
                * sumval
            )

    # Equation A16 of [Devoto1966]_.
    q23 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            sumval = 0
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
                    - 5
                    / 2
                    * masses[l] ** 2
                    * (198 * masses[j] ** 2 + 301 * masses[l] ** 2)
                    * Q14[i, l]
                    + 615 * masses[l] ** 4 * Q15[i, l]
                    - 210 * masses[l] ** 4 * Q16[i, l]
                    + 108 * (masses[j] * masses[l]) ** 2 * Q33[i, l]
                    - 120 * (masses[j] * masses[l]) ** 2 * Q34[i, l]
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
                sumval += term1 * term2
            q23[i, j] = (
                8
                * number_densities[i]
                * (masses[i] / masses[j]) ** (7 / 2)
                * sumval
            )

    # Equation A18 of [Devoto1966]_.
    q33 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            sumval = 0
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
                    - 15
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
                    + 560 * masses[l] ** 6 * Q17[i, l]
                    + 18
                    * (masses[j] * masses[l]) ** 2
                    * (10 * masses[j] ** 2 + 27 * masses[l] ** 2)
                    * Q33[i, l]
                    - 1080 * masses[j] ** 2 * masses[l] ** 4 * Q34[i, l]
                    + 720 * masses[j] ** 2 * masses[l] ** 4 * Q35[i, l]
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
                sumval += term1 * term2
            q33[i, j] = (
                8
                * number_densities[i]
                * (masses[i] / masses[j]) ** (7 / 2)
                * sumval
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
    The various elements of the qhat-matrix are calculated from the appendix of
    [Devoto1966]_, from equation A19 to A22.
    """
    nb_species = len(mixture.species)
    number_densities = mixture.calculate_composition()
    masses = np.array([sp.molar_mass / u.N_a for sp in mixture.species])

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
            sumval = 0
            for l in range(nb_species):
                term1 = (
                    number_densities[l]
                    * masses[l] ** (1 / 2)
                    / (masses[i] + masses[l]) ** (3 / 2)
                )
                term2 = (delta(i, j) - delta(j, l)) * 10 / 3 * masses[j] * Q11[
                    i, l
                ] + (delta(i, j) + delta(j, l)) * 2 * masses[l] * Q22[i, l]
                sumval += term1 * term2
            qhat00[i, j] = (
                8 * number_densities[i] * (masses[i] / masses[j]) * sumval
            )

    # Equation A20 of [Devoto1966]_.
    qhat01 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            sumval = 0
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
                sumval += term1 * term2
            qhat01[i, j] = (
                8 * number_densities[i] * (masses[i] / masses[j]) ** 2 * sumval
            )

    # Equation A22 of [Devoto1966]_.
    qhat11 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            sumval = 0
            for l in range(nb_species):
                term1 = (
                    number_densities[l]
                    * masses[l] ** (1 / 2)
                    / (masses[i] + masses[l]) ** (7 / 2)
                )
                term2 = (delta(i, j) - delta(j, l)) * masses[j] * (
                    1
                    / 6
                    * (140 * masses[j] ** 2 + 245 * masses[l] ** 2)
                    * Q11[i, l]
                    - masses[l] ** 2
                    * (98 * Q12[i, l] - 64 * Q13[i, l] - 24 * Q33[i, l])
                ) + (delta(i, j) + delta(j, l)) * masses[l] * (
                    1
                    / 6
                    * (154 * masses[j] ** 2 + 147 * masses[l] ** 2)
                    * Q22[i, l]
                    - masses[l] ** 2 * (56 * Q23[i, l] - 40 * Q24[i, l])
                )
                sumval += term1 * term2
            qhat11[i, j] = (
                8 * number_densities[i] * (masses[i] / masses[j]) ** 2 * sumval
            )

    # Equation A21 of [Devoto1966]_.
    qhat10 = np.zeros((nb_species, nb_species))
    for i in range(nb_species):
        for j in range(nb_species):
            qhat10[i, j] = masses[j] / masses[i] * qhat01[i, j]

    qq = np.zeros((2 * nb_species, 2 * nb_species))

    qq[0 * nb_species : 1 * nb_species, 0 * nb_species : 1 * nb_species] = (
        qhat00
    )
    qq[0 * nb_species : 1 * nb_species, 1 * nb_species : 2 * nb_species] = (
        qhat01
    )

    qq[1 * nb_species : 2 * nb_species, 0 * nb_species : 1 * nb_species] = (
        qhat10
    )
    qq[1 * nb_species : 2 * nb_species, 1 * nb_species : 2 * nb_species] = (
        qhat11
    )

    return qq


### Transport property calculations ###########################################


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

        D_{ij} = \frac{\rho n_i}{2 n_{\text{tot}} m_j}
            \sqrt{\frac{2 k_B T}{m_i}} c_{i 0}^{j i}

    where:

    - :math:`D_{ij}` is the diffusion coefficient between species :math:`i`
      and :math:`j`,
    - :math:`\rho` is the density of the mixture,
    - :math:`n_i` is the number density of species :math:`i`,
    - :math:`n_{\text{tot}}` is the total number density of the mixture,
    - :math:`m_j` is the molar mass of species :math:`j`,
    - :math:`k_B` is the Boltzmann constant,
    - :math:`T` is the temperature of the mixture,


    The elements of :math:`c_{i 0}^{j i}` are given by equation 6 of
    [Devoto1966]_:

    .. math::

        \begin{aligned}
            & \sum_{j=1}^\nu \sum_{p=0}^M q_{i j}^{m p} c_{j p}^{h k}
                = 3 \pi^{\frac{1}{2}}\left(\delta_{i k}-\delta_{i h}\right)
                    \delta_{m 0} \\
            & \quad(i=1,2, \cdots \nu ; m=0, \cdots M)
        \end{aligned}

    where:

    - :math:`\nu` is the number of species in the mixture,
    - :math:`M` is the order of the approximation (:math:`M=3` in this case,
      and goes from 0 to 3 so that a fourth-order approximation is used),
    - :math:`q_{i j}^{m p}` are the elements of the :math:`q`-matrix.


    TODO: Add how the code works.
    TODO: Why not use equation 8?
    """
    nb_species = len(mixture.species)
    number_densities = mixture.calculate_composition()  # m^-3
    masses = np.array([sp.molar_mass / u.N_a for sp in mixture.species])  # kg
    rho = mixture.calculate_density()  # kg/m^3

    n_tot = np.sum(number_densities)  # m^-3

    diffusion_matrix = np.zeros((nb_species, nb_species))
    qq = q(mixture)  # Size (4*nb_species, 4*nb_species)

    inverse_q = np.linalg.inv(qq)
    b_vec = np.zeros(4 * nb_species)  # 4 for 4th order approximation

    for i in range(nb_species):
        for j in range(nb_species):
            # TODO: Check if this is correct
            # Equation 6 of [Devoto1966]_.
            dij = np.array(
                [delta(h, i) - delta(h, j) for h in range(0, nb_species)]
            )
            b_vec[:nb_species] = 3 * np.sqrt(np.pi) * dij
            cflat = inverse_q.dot(b_vec)
            cip = cflat.reshape(4, nb_species)

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

    Thermal diffusion coefficents, calculation per [Devoto1966]_
    (eq. 4 and eq. 5). Fourth-order approximation.

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
    The thermal diffusion coefficients are given by equation 4 of
    [Devoto1966]_:

    .. math::

        D_i^T = \frac{1}{2} n_i m_i \sqrt{\frac{2 k_B T}{m_i}} a_{i 0}

    where:

    - :math:`D_i^T` is the thermal diffusion coefficient of species :math:`i`,
    - :math:`n_i` is the number density of species :math:`i`,
    - :math:`m_i` is the molar mass of species :math:`i`,
    - :math:`k_B` is the Boltzmann constant,
    - :math:`T` is the temperature of the mixture.


    The elements of :math:`a_{i 0}` are given by equation 5 of [Devoto1966]_:

    .. math::

        \begin{aligned}
            & \sum_{j=1}^\nu \sum_{p=0}^M q_{i j}^{m p} a_{j p}
                =-\frac{15 \pi^{\frac{1}{2}} n_i}{2} \delta_{m 1} \\
            & \quad(i=1,2, \cdots \nu ; m=0, \cdots M)
        \end{aligned}

    where:

    - :math:`\nu` is the number of species in the mixture,
    - :math:`M` is the order of the approximation (:math:`M=3` in this case,
      and goes from 0 to 3 so that a fourth-order approximation is used),
    - :math:`q_{i j}^{m p}` are the elements of the :math:`q`-matrix.


    TODO: Add how the code works.
    TODO: Why not use equation 9?
    """
    nb_species = len(mixture.species)
    number_densities = mixture.calculate_composition()
    masses = np.array([sp.molar_mass / u.N_a for sp in mixture.species])

    qq = q(mixture)

    inverse_q = np.linalg.inv(qq)
    b_vec = np.zeros(4 * nb_species)  # 4 for 4th order approximation
    # Only the first element is non-zero
    b_vec[nb_species : 2 * nb_species] = (
        -15 / 2 * np.sqrt(np.pi) * number_densities
    )
    aflat = inverse_q.dot(b_vec)
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
                =5 n_i\left(\frac{2 \pi m_i}{k_b T}\right)^{\frac{1}{2}}
                  \delta_{m 0} \\
            & \quad(i=1,2, \cdots \nu ; m=0, 1)
        \end{aligned}

    where:

    - :math:`\nu` is the number of species in the mixture,
    - :math:`\hat{q}_{i j}^{m p}` are the elements of the
      :math:`\hat{q}`-matrix.


    TODO: Add how the code works.
    TODO: Why not use equation 21?
    """
    nb_species = len(mixture.species)
    number_densities = mixture.calculate_composition()
    masses = np.array([sp.molar_mass / u.N_a for sp in mixture.species])

    qqhat = qhat(mixture)

    inverse_qhat = np.linalg.inv(qqhat)
    b_vec = np.zeros(2 * nb_species)  # 2 for 2nd order approximation
    b_vec[:nb_species] = (
        5
        * number_densities
        * np.sqrt(2 * np.pi * masses / (u.k_b * mixture.T))
    )
    bflat = inverse_qhat.dot(b_vec)
    bip = bflat.reshape(2, nb_species)

    return 0.5 * u.k_b * mixture.T * np.sum(number_densities * bip[0])


def electrical_conductivity(mixture: "LTE") -> float:
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

        \sigma=\frac{e^{2} n_{\text{tot}}}{\rho k_{B} T} \sum_{j=2}^{\zeta}
                n_{j} m_{j} z_{j} D_{1 j}

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
    charge_numbers = np.array([sp.charge_number for sp in mixture.species])
    number_densities = mixture.calculate_composition()
    masses = np.array([sp.molar_mass / u.N_a for sp in mixture.species])
    rho = mixture.calculate_density()

    D1 = Dij(mixture)[-1, :]
    n_tot = np.sum(number_densities)

    sum_val = 0.0
    for charge_number, D1j, mj, nj in zip(
        charge_numbers, D1, masses, number_densities
    ):
        # TODO: Check if this is correct. Electrons should be discarded.
        # TODO: Check if this is correct. Neutral species should be discarded
        # (but ok, since they have 0 charge).
        sum_val += nj * mj * charge_number * D1j

    pre_mult = u.e**2 * n_tot / (rho * u.k_b * mixture.T)

    return pre_mult * sum_val


def thermal_conductivity(
    mixture: "LTE",
    rel_delta_T: float,
    DTterms_yn: bool,
    ni_limit: float,
) -> float:
    r"""Thermal conductivity.

    Thermal conductivity, calculation per [Devoto1966]_ (eq. 2, 13 and 18).
    Fourth-order approximation.
    Numerical derivative performed to obtain :math:`\frac{dx_i}{dT}` for
    :math:`\vec{\nabla} x` in the :math:`\vec{d_i}` expression.

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
    The total heat flux :math:`\vec{q}` is given by equation 18 of
    [Devoto1966]_:

    .. math::

        \begin{array}{r}
            \vec{q} =\sum_{j=1}^\nu\left(\frac{n^2 m_j}{\rho} \sum_{i=1}^\nu
                m_i h_i D_{i j}
                    -\frac{n k_b T D_j^T}{n_j m_j}\right) \vec{d_i} \\
            -\left(\lambda^{\prime}+\sum_{i=1}^\nu \frac{n^2 h_i D_i^T}{\rho T}
                \right) \vec{\nabla} T
        \end{array}

    where:

    - :math:`\vec{q}` is the total heat flux,
    - :math:`\nu` is the number of species in the mixture,
    - :math:`n` is the total number density of the mixture,
    - :math:`m_j` is the mass of species :math:`j`,
    - :math:`\rho` is the density of the mixture,
    - :math:`m_i` is the mass of species :math:`i`,
    - :math:`h_i` is the enthalpy of species :math:`i`,
    - :math:`D_{i j}` is the diffusion coefficient between species :math:`i`
      and :math:`j`,
    - :math:`D_j^T` is the thermal diffusion coefficient of species :math:`j`,
    - :math:`D_i^T` is the thermal diffusion coefficient of species :math:`i`,


    :math:`\vec{d_i}` contains diffusion forces due to concentration
    :math:`x_i` and pressure gradients, and from external forces
    :math:`\vec{X_i}`, and is given by equation 2 of [Devoto1966]_:

    .. math::

        \vec{d_i} = \vec{\nabla} x_i
                  + \left(x_i-\frac{\rho_i}{\rho}\right) \vec{\nabla} \ln p
                  - \frac{\rho_i}{p\rho}\left(\frac{\rho}{m_i} \vec{X_i}-
                   \sum_{i=1}^{\nu} n_l \vec{X_l}\right)

    Assuming there is no pressure gradient and no external forces,
    the equation simplifies to:

    .. math::

        \vec{d_i} = \vec{\nabla} x_i

    It can be rewritten as: (TODO: check if correct)

    .. math::

        \vec{d_i} = \frac{d x_i}{d T} \vec{\nabla} T

    Injecting back into the total heat flux equation, we get:

    .. math::

        \vec{q} = \left[\sum_{j=1}^\nu\left(\frac{n^2 m_j}{\rho} \sum_{i=1}^\nu
                 m_i h_i D_{i j}
                -\frac{n k_b T D_j^T}{n_j m_j}\right) \frac{d x_i}{d T}
            -\left(\lambda^{\prime}+\sum_{i=1}^\nu \frac{n^2 h_i D_i^T}{\rho T}
                \right) \right]\vec{\nabla} T

    The thermal conductivity (:math:`\vec{q} = - \lambda \vec{\nabla} T`)
    is then given by:

    .. math::

        \lambda = \sum_{j=1}^\nu\left(\frac{n k_b T D_j^T}{n_j m_j}
                    -\frac{n^2 m_j}{\rho} \sum_{i=1}^\nu m_i h_i D_{i j}\right)
                    \frac{d x_i}{d T}
            +\left(\lambda^{\prime}+\sum_{i=1}^\nu \frac{n^2 h_i D_i^T}{\rho T}
            \right)


    In this equation, :math:`\lambda^{\prime}` is given by equation 13 of
    [Devoto1966]_:

    .. math::

        \lambda^{\prime} = -\frac{5 k_b}{4}
            \sum_{j=1}^\nu n_j\left(\frac{2 k_b T}{m_j}\right)^{\frac{1}{2}}
              a_{j 1}
    """
    nb_species = len(mixture.species)
    number_densities = mixture.calculate_composition()
    masses = np.array([sp.molar_mass / u.N_a for sp in mixture.species])
    n_tot = np.sum(number_densities)
    rho = mixture.calculate_density()
    hv = mixture.calculate_species_enthalpies()

    # Rescale species enthalpies relative to average molar mass.
    # TODO: why?
    average_molar_mass = rho / n_tot
    hv = hv * masses / average_molar_mass

    ### translational tk components ###
    # Solve equation 5 of [Devoto1966]_ to get the `a` matrix.
    qq = q(mixture)

    inverse_q = np.linalg.inv(qq)
    b_vec = np.zeros(4 * nb_species)
    b_vec[nb_species : 2 * nb_species] = (
        -15 / 2 * np.sqrt(np.pi) * number_densities
    )
    aflat = inverse_q.dot(b_vec)
    aip = aflat.reshape(4, nb_species)
    # Equation 13 of [Devoto1966]_.
    k_dash = (
        -5
        / 4
        * u.k_b
        * np.sum(
            number_densities * np.sqrt(2 * u.k_b * mixture.T / masses) * aip[1]
        )
    )
    # TODO: Why not use equation 14?

    ### thermal diffusion tk components ###
    if DTterms_yn:
        # TODO: This looks like the second term in the second parenthesis of 18
        # of [Devoto1966]_.
        # TODO: Check if it is correct. Where are the term n² and rho?
        locDTi = DTi(mixture)
        kdt = np.sum(hv * locDTi / mixture.T)
    else:
        kdt = 0

    ### reactional tk components - normal diffusion term ###

    # Compute the derivative of the number densities with respect to
    # temperature. x is the concentration of species i, x = ni / ntot
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
            # TODO: This looks like the first term in the first parenthesis of
            # 18 of [Devoto1966]_.
            # TODO: Check if it is correct.
            krxn_enth += masses[j] * masses[i] * hv[i] * locDij[i, j] * dxdT[j]
    krxn_enth *= -(n_tot**2) / rho

    ### reactional tk components - thermal diffusion term ###
    if DTterms_yn:
        # TODO: This looks like the second term in the first parenthesis of 18
        # of [Devoto1966]_. TODO: Check if it is correct.
        dxdTfilt = np.where(
            number_densities < ni_limit, np.zeros(nb_species), dxdT
        )
        krxn_therm = (
            n_tot
            * u.k_b
            * Tval
            * np.sum(locDTi * dxdTfilt / (number_densities * masses))
        )
    else:
        krxn_therm = 0.0

    return k_dash + kdt + krxn_enth + krxn_therm
