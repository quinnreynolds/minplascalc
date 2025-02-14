"""Fit the electron cross section from LXCat to the formula used in minplascalc.

WORK IN PROGRESS.
"""

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
    taus = np.sqrt(2 * u.m_e * u.k_b * temperatures) / u.hbar

    D1 = cross_section[0]
    cross_section_part_to_fit = (cross_section - D1) / D1

    # Fit the cross section from LXCat to the formula used in minplascalc.
    def fit_function(tau, D2, D3, D4):
        return D2 / D1 * tau**D3 * np.exp(-D4 * tau**2)
        # return np.log(D2) + D3 * np.log(tau) - D4 * tau**2

    p0 = [1e-20, 1, 1e-30]
    curve_fit_parameters, _ = curve_fit(
        fit_function,
        xdata=taus[1:],  # Skip the first value to avoid division by zero.
        ydata=cross_section_part_to_fit[
            1:
        ],  # Skip the first value to avoid division by zero.
        # ydata=np.log(
        #     cross_section_part_to_fit[1:]
        # ),  # Skip the first value to avoid division by zero.
        p0=p0,
        maxfev=10_000,
        bounds=(0, [1e-10, 10, 1e-15]),
    )

    return [D1] + list(curve_fit_parameters)


if __name__ == "__main__":
    # LXCAT
    # DATABASE:         Morgan (Kinema Research  Software)
    # PERMLINK:         www.lxcat.net/Morgan
    # DESCRIPTION:      Assembled over the course of 30 years WL Morgan and suitable for use with 2-term Boltzmann solvers.
    # CONTACT:          W. Lowell Morgan, Kinema Research  Software
    # e / C
    # Effective E + C → E + C (m/M = 0.000045683, complete set) | (p. 1664, Vol. IV). Updated: 6 June 2011.
    energy_cross_section = np.array(
        [
            [0.000000e0, 1.050000e-19],
            [5.000000e-2, 1.070000e-19],
            [1.340000e-1, 1.250000e-19],
            [2.710000e-1, 1.620000e-19],
            [3.890000e-1, 2.230000e-19],
            [4.400000e-1, 2.930000e-19],
            [4.720000e-1, 3.960000e-19],
            [5.590000e-1, 4.060000e-19],
            [6.280000e-1, 3.970000e-19],
            [7.640000e-1, 3.420000e-19],
            [1.080000e0, 2.790000e-19],
            [1.410000e0, 2.370000e-19],
            [2.180000e0, 2.030000e-19],
            [3.020000e0, 1.970000e-19],
            [3.880000e0, 1.900000e-19],
            [4.820000e0, 1.880000e-19],
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
    taus = np.sqrt(2 * u.m_e * u.k_b * temperatures) / u.hbar

    def fit_function(tau, D1, D2, D3, D4):
        return D1 + D2 * tau**D3 * np.exp(-D4 * tau**2)

    plt.plot(energy, cross_section, label="LXCat")
    plt.plot(energy, fit_function(taus, *curve_fit_parameters), label="Minplascalc fit")
    electron_cross_section = [
        2.7716909516406733e-20,
        1.2091239912854124e-85,
        7.039441779968125,
        2.3051925774489085e-19,
    ]
    plt.plot(
        energy, fit_function(taus, *electron_cross_section), label="Minplascalc default"
    )
    plt.title("Electron cross section of atomic carbon (C)")
    plt.xlabel("Energy [eV]")
    plt.ylabel("Cross section [m^2]")
    plt.xscale("log")
    plt.legend()
    plt.show()
