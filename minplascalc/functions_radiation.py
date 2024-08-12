import numpy
from scipy import constants

kb = constants.Boltzmann
line_preconst = constants.Planck * constants.c / (4*constants.pi)

def total_emission_coefficient(mix):
    """Calculate the LTE total radiation emission coefficient of the plasma 
    in W/m3.sr.
    """
    nd = mix.calculate_composition()
    rt = 0
    for nv, sp in zip(nd[:-1], mix.species[:-1]):
        Qi = sp.partitionfunction_internal(mix.T, 0)
        for eml in sp.emissionlines:
            wl, gA, Ek = eml
            rt += line_preconst * nv * gA * numpy.exp(-Ek/(kb*mix.T)) / (Qi*wl)
    return rt
