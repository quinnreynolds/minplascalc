"""Physical constants and unit conversions.

This module provides a class with physical constants and unit conversions.
All units are by default in the International System of Units (SI).

Values are taken from the `scipy.constants` module, which itself uses the 2022
CODATA recommended values.
"""

import numpy as np
from scipy import constants  # type: ignore

pi = np.pi
"""Pi."""

m_e = constants.electron_mass
r"""Electron mass [:math:`\text{kg}`]."""

e = constants.elementary_charge
r"""Elementary charge [:math:`\text{C}`]."""

k_b = constants.Boltzmann
r"""Boltzmann constant [:math:`\text{J.K}^{-1}`]."""

N_a = constants.Avogadro
r"""Avogadro's number [:math:`\text{mol}^{-1}`]."""

M_e = m_e * N_a
r"""Electron molar mass [:math:`\text{kg.mol}^{-1}`]."""

eV_to_K = e / k_b
r"""Conversion factor from :math:`\text{eV}` to :math:`\text{K}`."""

K_to_eV = k_b / e
r"""Conversion factor from :math:`\text{K}` to :math:`\text{eV}`."""

eV_to_J = e
r"""Conversion factor from :math:`\text{eV}` to :math:`\text{J}`."""

J_to_eV = 1 / e
r"""Conversion factor from :math:`\text{J}` to :math:`\text{eV}`."""

K_to_J = k_b
r"""Conversion factor from :math:`\text{K}` to :math:`\text{J}`."""

J_to_K = 1 / k_b
r"""Conversion factor from :math:`\text{J}` to :math:`\text{K}`."""

Da = constants.atomic_mass
r"""Dalton or atomic mass constant [:math:`\text{kg}`]."""

R = constants.gas_constant
r"""Ideal gas constant [:math:`\text{J.mol}^{-1}.\text{K}^{-1}`]."""

R_kmol = R * 1e-3
r"""Ideal gas constant [:math:`\text{J.kmol}^{-1}.\text{K}^{-1}`]."""

P_1_bar = 1e5
r"""1 bar [:math:`\text{Pa}`]."""

h = constants.Planck
r"""Planck constant [:math:`\text{J.s}`]."""

hbar = constants.hbar
r"""Reduced Planck constant [:math:`\text{J.s}`]."""

c = constants.speed_of_light
r"""Speed of light [:math:`\text{m.s}^{-1}`]."""

epsilon_0 = constants.epsilon_0
r"""Vacuum permittivity [:math:`\text{F.m}^{-1}`]."""

stefan_boltzmann = 2 * np.pi**5 * k_b**4 / (15 * h**3 * c**2)
r"""Stefan-Boltzmann constant [:math:`\text{W.m}^{-2}.\text{K}^{-4}`]."""

spitzer_constant = (
    (4 * np.pi * epsilon_0 / e) ** 2
    * (k_b) ** (3 / 2)
    / (4 / 3 * (2 * np.pi * m_e) ** (1 / 2))
) * 1.96
r"""Spitzer conductivity constant [:math:`\Omega^{-1}.\text{m}^{-1}.\text{K}^{-3/2}`].

The conductivity of a fully-ionized plasma is given by the Spitzer conductivity:

.. math::

    \sigma = \text{Spitzer constant} \times \frac{T_e^{3/2}}{\log(\lambda)}

with :

* :math:`T_e` the electron temperature, in :math:`\text{K}`, and
* :math:`\log(\lambda)` the Coulomb logarithm.

The constant is given by:

.. math::

    \begin{align}
        \text{Spitzer constant}
            &= \left(\frac{4 \pi \varepsilon_0}{e}\right)^2
                \times \left(\frac{k_b^{3/2}}{\frac{4}{3} \sqrt{2 \pi m_e} }
                \right)
                \times 1.96
            &= 1.53 \times 10^{-2} \, \Omega^{-1} \, \text{m}^{-1} \,
            \text{K}^{-3/2}
    \end{align}

with:

* :math:`\varepsilon_0` the vacuum permittivity,
* :math:`e` the elementary charge,
* :math:`k_b` the Boltzmann constant,
* :math:`m_e` the electron mass.
"""  # noqa: E501
