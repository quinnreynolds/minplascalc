import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from minplascalc.units import Units

u = Units()


def fit_from_lxcat_to_minplascalc(energy, cross_section):
    """Fit the electron cross section from LXCat to the formula used in minplascalc.

    Parameters
    ----------
    energy : np.ndarray
        Energy values in eV.
    cross_section : np.ndarray
        Cross section values in m^2.

    Returns
    -------
    np.ndarray
        The values of D1, D2, D3, D4.
    """
    # Fit the cross section from LXCat to the formula used in minplascalc.
    # The formula used in minplascalc is:
    # Omega = D1 + D2 * tau**D3 * np.exp(-D4 * tau**2)
    # where tau = np.sqrt(2 * u.m_e * u.k_b * temperatures) / u.hbar
    # and D1, D2, D3, D4 are the parameters to fit.
    # The values of D1, D2, D3, D4 are taken from the `./data/species/O.json` file.
    # These are the default values for the electron cross section of the oxygen atom.
    # The values of D1, D2, D3, D4 are fitted to the cross section from LXCat.

    temperatures = 2 / 3 * energy * u.eV_to_K  # eV to K
    tau = np.sqrt(2 * u.m_e * u.k_b * temperatures) / u.hbar

    # Fit the cross section from LXCat to the formula used in minplascalc.
    def fit_function(tau, D1, D2, D3, D4):
        return D1 + D2 * tau**D3 * np.exp(-D4 * tau**2)

    curve_fit_parameters, _ = curve_fit(
        fit_function, tau, cross_section, p0=[1e-20, 1e-34, 1, 1e-20], maxfev=5000
    )
    return curve_fit_parameters


if __name__ == "__main__":
    # LXCAT
    # Elastic E + O → E + O (m/M = 0.000034286) | Integral elastic cross section
    # (integrated from differential) | Source: integrated from [Williams&Allen J.
    # Phys. B: At. Mol. Opt. Phys. 22, 3529 (1989)] , Above 10 eV: ELSEPA. Updated: 3 January 2024.
    energy_cross_section = np.array(
        [
            [0.000000e0, 1.400000e-20],
            [1.000000e-3, 1.610000e-20],
            [1.000000e-2, 2.060000e-20],
            [5.400000e-1, 2.864490e-20],
            [2.180000e0, 5.774350e-20],
            [3.400000e0, 7.063500e-20],
            [4.900000e0, 7.498520e-20],
            [8.700000e0, 8.400360e-20],
            [1.000000e1, 7.350530e-20],
            [1.100000e1, 7.295590e-20],
            [1.200000e1, 7.217360e-20],
            [1.300000e1, 7.123100e-20],
            [1.400000e1, 7.017720e-20],
            [1.500000e1, 6.905010e-20],
            [1.600000e1, 6.787900e-20],
            [1.700000e1, 6.668690e-20],
            [1.800000e1, 6.548620e-20],
            [1.900000e1, 6.429090e-20],
            [2.000000e1, 6.310860e-20],
            [2.500000e1, 5.758450e-20],
            [3.000000e1, 5.285330e-20],
            [3.500000e1, 4.887070e-20],
            [4.000000e1, 4.548370e-20],
            [4.500000e1, 4.256410e-20],
            [5.000000e1, 4.002110e-20],
            [5.500000e1, 3.779270e-20],
            [6.000000e1, 3.583240e-20],
            [6.500000e1, 3.409700e-20],
            [7.000000e1, 3.255030e-20],
            [7.500000e1, 3.116150e-20],
            [8.000000e1, 2.990720e-20],
            [8.500000e1, 2.876730e-20],
            [9.000000e1, 2.772530e-20],
            [9.500000e1, 2.677080e-20],
            [1.000000e2, 2.589340e-20],
            [1.100000e2, 2.433430e-20],
            [1.200000e2, 2.299300e-20],
            [1.300000e2, 2.182630e-20],
            [1.400000e2, 2.080030e-20],
            [1.500000e2, 1.988950e-20],
            [1.600000e2, 1.907450e-20],
            [1.700000e2, 1.834010e-20],
            [1.800000e2, 1.767460e-20],
            [1.900000e2, 1.706760e-20],
            [2.000000e2, 1.651170e-20],
            [2.500000e2, 1.429610e-20],
            [3.000000e2, 1.270460e-20],
            [3.500000e2, 1.149850e-20],
            [4.000000e2, 1.054770e-20],
            [4.500000e2, 9.768040e-21],
            [5.000000e2, 9.110740e-21],
            [5.500000e2, 8.546710e-21],
            [6.000000e2, 8.055050e-21],
            [6.500000e2, 7.621540e-21],
            [7.000000e2, 7.235750e-21],
            [7.500000e2, 6.889220e-21],
            [8.000000e2, 6.576020e-21],
            [8.500000e2, 6.291600e-21],
            [9.000000e2, 6.031970e-21],
            [9.500000e2, 5.793590e-21],
            [1.000000e3, 5.573770e-21],
            [1.250000e3, 4.416190e-21],
            [1.500000e3, 3.812160e-21],
            [1.750000e3, 3.349080e-21],
            [2.000000e3, 2.991300e-21],
            [2.500000e3, 2.468910e-21],
            [3.000000e3, 2.103330e-21],
            [3.500000e3, 1.833100e-21],
            [4.000000e3, 1.625020e-21],
            [4.500000e3, 1.460000e-21],
            [5.000000e3, 1.325610e-21],
            [5.500000e3, 1.214340e-21],
            [6.000000e3, 1.120540e-21],
            [6.500000e3, 1.040410e-21],
            [7.000000e3, 9.712030e-22],
            [7.500000e3, 9.108240e-22],
            [8.000000e3, 8.614902e-22],
            [9.000000e3, 7.681121e-22],
            [1.000000e4, 6.933725e-22],
            [1.500000e4, 4.871660e-22],
            [2.000000e4, 3.706550e-22],
            [2.500000e4, 3.007340e-22],
            [3.000000e4, 2.541200e-22],
            [4.000000e4, 1.958630e-22],
            [5.000000e4, 1.609260e-22],
            [6.000000e4, 1.376520e-22],
            [7.000000e4, 1.210420e-22],
            [8.000000e4, 1.085980e-22],
            [9.000000e4, 9.893100e-23],
            [1.000000e5, 9.120730e-23],
            [2.000000e5, 5.670680e-23],
            [5.000000e5, 3.682820e-23],
            [1.000000e6, 3.096330e-23],
            [2.000000e6, 2.860810e-23],
            [5.000000e6, 2.766180e-23],
            [1.000000e7, 2.748900e-23],
            [2.000000e7, 2.744110e-23],
            [5.000000e7, 2.742690e-23],
            [1.000000e8, 2.742480e-23],
            [2.000000e8, 2.742430e-23],
            [5.000000e8, 2.742420e-23],
            [1.000000e9, 2.742410e-23],
        ]
    )

    # LXCAT
    # Morgan (Kinema Research & Software)
    # Data Group [H]
    # e / H
    # lastic E + H → E + H (m/M = 0.000548193, complete set) | Updated: 16 July 2010.
    energy_cross_section = np.array(
        [
            [0.000000e0, 3.940000e-19],
            [1.000000e-1, 3.910000e-19],
            [5.000000e-1, 2.940000e-19],
            [1.000000e0, 2.300000e-19],
            [1.500000e0, 1.941000e-19],
            [2.000000e0, 1.707000e-19],
            [3.000000e0, 1.408000e-19],
            [4.000000e0, 1.214000e-19],
            [5.000000e0, 1.066000e-19],
            [7.000000e0, 8.500000e-20],
            [9.000000e0, 7.030000e-20],
            [1.400000e1, 4.440000e-20],
            [1.500000e1, 4.120000e-20],
            [1.700000e1, 3.570000e-20],
            [2.000000e1, 2.940000e-20],
            [2.500000e1, 2.230000e-20],
            [3.000000e1, 1.770000e-20],
            [3.500000e1, 1.451000e-20],
            [4.000000e1, 1.218000e-20],
            [5.000000e1, 9.040000e-21],
            [7.000000e1, 6.340000e-21],
            [1.000000e2, 4.220000e-21],
            [2.000000e2, 1.733000e-21],
            [5.000000e2, 6.130000e-22],
            [1.000000e3, 2.930000e-22],
        ]
    )

    energy = energy_cross_section[:, 0]
    cross_section = energy_cross_section[:, 1]

    max_energy = 1e2  # eV
    mask = energy <= max_energy
    energy = energy[mask]
    cross_section = cross_section[mask]

    curve_fit_parameters = fit_from_lxcat_to_minplascalc(energy, cross_section)

    print(curve_fit_parameters)

    # Plot the cross section from LXCat and the fitted curve.
    temperatures = 2 / 3 * energy * u.eV_to_K  # eV to K
    tau = np.sqrt(2 * u.m_e * u.k_b * temperatures) / u.hbar

    def fit_function(tau, D1, D2, D3, D4):
        return D1 + D2 * tau**D3 * np.exp(-D4 * tau**2)

    plt.plot(energy, cross_section, label="LXCat")
    plt.plot(energy, fit_function(tau, *curve_fit_parameters), label="Minplascalc")
    plt.xlabel("Energy [eV]")
    plt.ylabel("Cross section [m^2]")
    plt.xscale("log")
    plt.legend()
    plt.show()
