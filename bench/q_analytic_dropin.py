"""Drop-in analytic ``Qnn`` / ``Qin`` using the sympy-derived derivative.

Replaces the recursive central-difference evaluation of eq. 18 with the
closed-form derivative from ``q_analytic_derivative``.  Same inputs, same
potential parameters, same fit coefficients -- only the derivative changes,
from a unit-step finite difference to the exact one.
"""

import numpy as np
from q_analytic_compact import BASE_S, omega_star

import minplascalc.functions_transport as ft
from minplascalc import units as u


def _fit_coeffs(table, l, s, beta_value):
    """Fit coefficients a0..a6 for (l, s), eq. 16 of [Laricchiuta2007]_."""
    s_eff = min(s, BASE_S[l])
    beta_array = np.array([1.0, beta_value, beta_value**2])
    return table[l - 1, s_eff - 1] @ beta_array


def Qnn_analytic(species_i, species_j, l, s, T):
    r_e, epsilon_0 = ft.pot_parameters_neut_neut(species_i, species_j)
    beta_value = ft.beta(species_i, species_j)
    x0 = ft.x0_neut_neut(beta_value)
    a = _fit_coeffs(ft.c_nn, l, s, beta_value)
    sigma = r_e * x0
    x = np.log(T * u.K_to_eV / epsilon_0)
    s0 = BASE_S[l]
    omega_reduced = omega_star(x, a, max(0, s - s0), s0)
    return omega_reduced * np.pi * sigma**2 * 1e-20


def Qin_analytic(species_i, species_j, l, s, T):
    r_e, epsilon_0 = ft.pot_parameters_ion_neut(species_i, species_j)
    beta_value = ft.beta(species_i, species_j)
    x0 = ft.x0_ion_neut(beta_value)
    a = _fit_coeffs(ft.c_in, l, s, beta_value)
    sigma = r_e * x0
    x = np.log(T * u.K_to_eV / epsilon_0)
    s0 = BASE_S[l]
    omega_reduced = omega_star(x, a, max(0, s - s0), s0)
    return omega_reduced * np.pi * sigma**2 * 1e-20


def _omega_fd(kind, species_i, species_j, l, s, T, h):
    """Evaluate the current recursion with a tunable difference step.

    With ``h = 0.5`` this reproduces ``functions_transport`` exactly: there
    the divisor ``2h == 1`` is dropped rather than written.
    """
    if s > BASE_S[l]:
        lo = _omega_fd(kind, species_i, species_j, l, s - 1, T - h, h)
        hi = _omega_fd(kind, species_i, species_j, l, s - 1, T + h, h)
        mid = _omega_fd(kind, species_i, species_j, l, s - 1, T, h)
        return mid + T / (s + 1) * (hi - lo) / (2 * h)

    if kind == "nn":
        r_e, epsilon_0 = ft.pot_parameters_neut_neut(species_i, species_j)
        beta_value = ft.beta(species_i, species_j)
        x0 = ft.x0_neut_neut(beta_value)
        table = ft.c_nn
    else:
        r_e, epsilon_0 = ft.pot_parameters_ion_neut(species_i, species_j)
        beta_value = ft.beta(species_i, species_j)
        x0 = ft.x0_ion_neut(beta_value)
        table = ft.c_in

    a = table[l - 1, s - 1] @ np.array([1.0, beta_value, beta_value**2])
    sigma = r_e * x0
    x = np.log(T * u.K_to_eV / epsilon_0)
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
    return np.exp(lnS1 + lnS2) * np.pi * sigma**2 * 1e-20


def omega_fd(kind, species_i, species_j, l, s, T, h=0.5):
    return _omega_fd(kind, species_i, species_j, l, s, T, h)


def patch():
    """Install the analytic versions; returns an undo callable."""
    real_nn, real_in = ft.Qnn, ft.Qin
    ft.Qnn = Qnn_analytic
    ft.Qin = Qin_analytic

    def undo():
        ft.Qnn = real_nn
        ft.Qin = real_in

    return undo
