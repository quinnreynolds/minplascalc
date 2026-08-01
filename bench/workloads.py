"""Shared benchmark workloads for minplascalc performance investigation.

These mirror the heaviest paths exercised by the test suite and the
``examples/plot_tutorial_10_*`` script, but with knobs for the number of
temperature points so profiling runs stay tractable.
"""

import numpy as np

import minplascalc as mpc

SIMPLE_SPECIES = ["O2", "O2+", "O", "O-", "O+", "O++"]
SIMPLE_X0 = [1, 0, 0, 0, 0, 0]

SICO_SPECIES = [
    "O2",
    "O2+",
    "O",
    "O+",
    "O++",
    "CO",
    "CO+",
    "C",
    "C+",
    "C++",
    "SiO",
    "SiO+",
    "Si",
    "Si+",
    "Si++",
]


def sico_x0(sio: float) -> list[float]:
    return [0, 0, 0, 0, 0, 1 - sio, 0, 0, 0, 0, sio, 0, 0, 0, 0]


def make_sico(sio: float = 0.5, T: float = 1000.0, P: float = 101325.0):
    return mpc.mixture.lte_from_names(SICO_SPECIES, sico_x0(sio), T, P)


def make_simple(T: float = 1000.0, P: float = 101325.0):
    return mpc.mixture.lte_from_names(SIMPLE_SPECIES, SIMPLE_X0, T, P)


def sico_transport_sweep(n_T: int = 20, n_mixtures: int = 3):
    """Tutorial-10 style workload: all four properties over a T sweep."""
    mixtures = [make_sico(sio) for sio in [0.1, 0.5, 0.9][:n_mixtures]]
    temperatures = np.linspace(1000, 25000, n_T)
    out = []
    for mixture in mixtures:
        for T in temperatures:
            mixture.T = T
            out.append(
                (
                    mixture.calculate_viscosity(),
                    mixture.calculate_electrical_conductivity(),
                    mixture.calculate_total_emission_coefficient(),
                    mixture.calculate_thermal_conductivity(),
                )
            )
    return out


def sico_composition_sweep(n_T: int = 20, n_mixtures: int = 3):
    """Composition (Gibbs minimisation) only, no transport."""
    mixtures = [make_sico(sio) for sio in [0.1, 0.5, 0.9][:n_mixtures]]
    temperatures = np.linspace(1000, 25000, n_T)
    out = []
    for mixture in mixtures:
        for T in temperatures:
            mixture.T = T
            out.append(mixture.calculate_composition())
    return out


def sico_viscosity_sweep(n_T: int = 20, n_mixtures: int = 3):
    mixtures = [make_sico(sio) for sio in [0.1, 0.5, 0.9][:n_mixtures]]
    temperatures = np.linspace(1000, 25000, n_T)
    out = []
    for mixture in mixtures:
        for T in temperatures:
            mixture.T = T
            out.append(mixture.calculate_viscosity())
    return out


WORKLOADS = {
    "transport": sico_transport_sweep,
    "composition": sico_composition_sweep,
    "viscosity": sico_viscosity_sweep,
}
