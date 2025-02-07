from typing import TYPE_CHECKING

import numpy

from minplascalc.units import Units

if TYPE_CHECKING:
    from minplascalc.mixture import LTE


u = Units()


line_preconst = u.h * u.c / (4 * u.pi)


def total_emission_coefficient(mix: "LTE") -> float:
    """Compute the LTE total radiation emission coefficient of the plasma in W/m3.sr."""
    nd = mix.calculate_composition()
    rt = 0
    for nv, sp in zip(nd[:-1], mix.species[:-1]):
        Qi = sp.partitionfunction_internal(mix.T, 0)
        for eml in sp.emissionlines:
            wl, gA, Ek = eml
            rt += line_preconst * nv * gA * numpy.exp(-Ek / (u.k_b * mix.T)) / (Qi * wl)
    return rt
