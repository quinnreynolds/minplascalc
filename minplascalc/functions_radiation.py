from typing import TYPE_CHECKING

import numpy

from minplascalc.units import Units

if TYPE_CHECKING:
    from minplascalc.mixture import LTE


u = Units()


line_preconst = u.h * u.c / (4 * u.pi)


def total_emission_coefficient(mix: "LTE") -> float:
    r"""
    Compute the LTE total radiation emission coefficient of the plasma in W/m3.sr.

    Parameters
    ----------
    mix : LTE
        The plasma mixture object containing species and temperature information.

    Returns
    -------
    float
        The total radiation emission coefficient in W/m^3/sr.

    Notes
    -----
    The total emission coefficient is calculated by summing the contributions of each
    emission line of each species in the mixture.

    The number densities of the species are calculated by the mixture object,
    and the internal partition functions are calculated using the species object.

    The formula used is derived from the Einstein coefficients for spontaneous emission
    and the Boltzmann distribution for the population of excited states.

    The emission coefficient is given by:

    .. math::

        \varepsilon = \frac{h c}{4 \pi} \cdot \sum_i \left(
            \frac{n_i  A_{ij} }{Q_i \lambda_{ij}} \exp\left(-\frac{E_i}{k_B T}\right)
            \right)

    where:

    * :math:`\varepsilon` is the total emission coefficient, in W/m^3/sr,
    * :math:`h` is Planck's constant, in J.s,
    * :math:`c` is the speed of light, in m/s,
    * :math:`\pi` is Pi,
    * :math:`n_i` is the number density of species :math:`i`, in m^-3,
    * :math:`A_{ij}` is the Einstein coefficient for spontaneous emission, in s^-1,
    * :math:`E_i` is the energy of the lower state, in J,
    * :math:`k_B` is Boltzmann's constant, in J/K,
    * :math:`T` is the temperature, in K,
    * :math:`Q_i` is the internal partition function of species :math:`i`,
    * :math:`\lambda_{ij}` is the wavelength of the transition, in m.

    References
    ----------
    TODO: Add references.
    """
    # Calculate the number densities of species in the mixture.
    nd = mix.calculate_composition()

    # Initialize the total emission coefficient.
    total_emission_coefficient = 0.0

    # Iterate over species and their number densities
    for nv, species in zip(nd[:-1], mix.species[:-1]):
        # Calculate the internal partition function of the species at the given temperature.
        Qi = species.partitionfunction_internal(mix.T, 0)

        # Iterate over emission lines of the species.
        for emission_line in species.emissionlines:
            # Extract wavelength, Einstein coefficient, and lower state energy of the emission line.
            wavele, gA, Ek = emission_line
            # Calculate the contribution of this emission line to the total emission coefficient.
            total_emission_coefficient += (
                line_preconst
                * nv
                * gA
                * numpy.exp(-Ek / (u.k_b * mix.T))
                / (Qi * wavele)
            )

    # Return the total emission coefficient.
    return total_emission_coefficient
