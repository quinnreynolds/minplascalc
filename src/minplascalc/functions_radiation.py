"""Radiation functions module."""

from typing import TYPE_CHECKING

import numpy

from minplascalc import units as u

if TYPE_CHECKING:
    from minplascalc.mixture import LTE


def total_emission_coefficient(mix: "LTE") -> float:
    r"""
    Compute the LTE total radiation emission coefficient of the plasma.

    The total radiation emission coefficient is returned in W/m^3/sr.

    Parameters
    ----------
    mix : LTE
        Plasma mixture object containing species and temperature information.

    Returns
    -------
    float
        The total radiation emission coefficient in W/m^3/sr.

    Notes
    -----
    The explicit expression for the emission coefficient of a spectral line
    emitted by an atom or an ion as a function of temperature is given by
    equation 5 of chapter 20 of [Boulos2023]_,
    (and also at eq. 75 of chapter 7 of [Boulos2023]_):

    .. math::

        \varepsilon_L=\frac{1}{4 \pi} A_{u \ell}^r n_r \frac{g_{r u}}{Q_r}
                \exp \left(- \frac{ E_{r u} }{k_b T} \right) h \nu_{u \ell}

    where:

    * :math:`\varepsilon_L` is the emission coefficient of the spectral line
      in :math:`\text{W.m}^{-3}.sr^{-1}`,
    * :math:`A_{u \ell}^r` is the Einstein coefficient (or transition
      probability) for spontaneous emission in :math:`\text{s}^{-1}`,
    * :math:`n_r` is the number density of the species,
      in :math:`\text{m}^{-3}`,
    * :math:`g_{r u}` is the statistical weight of the upper state,
    * :math:`Q_r` is the internal partition function of the species,
    * :math:`E_{r u}` is the energy difference between the upper and lower
      states, in :math:`\text{J}`,
    * :math:`k_b` is Boltzmann's constant, in :math:`\text{J.K}^{-1}`,
    * :math:`T` is the temperature, in :math:`\text{K}`,
    * :math:`h` is Planck's constant, in :math:`\text{J.s}`,
    * :math:`\nu_{u \ell}` is the frequency of the transition,
      in :math:`\text{Hz}`,
    * :math:`\frac{1}{4 \pi}` is the solid angle, in :math:`\text{sr}`.


    Then, the total emission coefficient is calculated by summing the
    contributions of each emission line of each species in the mixture,
    using equation 6 of chapter 20 of [Boulos2023]_.

    The number densities of the species are calculated by the mixture object,
    and the internal partition functions are calculated using the species
    object.

    The formula used is derived from the Einstein coefficients for spontaneous
    emission and the Boltzmann distribution for the population of excited
    states.

    The total emission coefficient is finally given by the following
    expression, expressing :math:`\nu_{u \ell}=\frac{c}{\lambda_{u \ell}}`:

    .. math::

        \varepsilon = \frac{h c}{4 \pi} \cdot \sum_i \left(
            \frac{n_i  A_{ij} }{Q_i \lambda_{ij}}
                \exp\left(-\frac{E_i}{k_B T}\right)
            \right)

    where:

    * :math:`\varepsilon` is the total emission coefficient,
      in :math:`\text{W.m}^{-3}.sr^{-1}`,
    * :math:`h` is Planck's constant, in :math:`\text{J.s}`,
    * :math:`c` is the speed of light, in :math:`\text{m.s}^{-1}`,
    * :math:`\pi` is Pi,
    * :math:`n_i` is the number density of species :math:`i`,
      in :math:`\text{m}^{-3}`,
    * :math:`A_{ij}` is the Einstein coefficient for spontaneous emission,
      in :math:`\text{s}^{-1}`,
    * :math:`E_i` is the energy of the lower state, in :math:`\text{J}`,
    * :math:`k_B` is Boltzmann's constant, in :math:`\text{J.K}^{-1}`,
    * :math:`T` is the temperature, in :math:`\text{K}`,
    * :math:`Q_i` is the internal partition function of species :math:`i`,
    * :math:`\lambda_{ij}` is the wavelength of the transition,
      in :math:`\text{m}`.

    See Also
    --------
    TODO: Equation 131 of chapter 7 of [Boulos2023]_ for net emission
    coefficient (NEC).
    """
    # Calculate the number densities of species in the mixture.
    nd = mix.calculate_composition()

    # Initialize the total emission coefficient.
    total_emission_coefficient = 0.0

    # Calculate the pre-constant of the emission line.
    line_pre_constant = u.h * u.c / (4 * u.pi)

    # Iterate over species and their number densities
    for nv, species in zip(nd[:-1], mix.species[:-1]):
        # Calculate the internal partition function of the species at the given
        # temperature.
        Qi = species.internal_partition_function(mix.T, 0)

        # Iterate over emission lines of the species.
        for emission_line in species.emission_lines:
            # Extract wavelength, Einstein coefficient, and lower state energy
            # of the emission line.
            wavele, gA, Ek = emission_line
            # Calculate the contribution of this emission line to the total
            # emission coefficient.
            total_emission_coefficient += (
                line_pre_constant
                * nv
                * gA
                * numpy.exp(-Ek / (u.k_b * mix.T))
                / (Qi * wavele)
            )

    # Return the total emission coefficient.
    return total_emission_coefficient
