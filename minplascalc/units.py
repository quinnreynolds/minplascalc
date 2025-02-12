import numpy as np
from scipy import constants


class Units:
    """Physical constants and unit conversions.

    All units are by default in the International System of Units (SI).
    """

    def __init__(self):
        self.pi = np.pi
        """Pi."""

        self.m_e = constants.electron_mass
        """Electron mass [kg]."""

        self.e = constants.elementary_charge
        """Elementary charge [C]."""

        self.k_b = constants.Boltzmann
        """Boltzmann constant [J/K]."""

        self.N_a = constants.Avogadro
        """Avogadro's number [mol^-1]."""

        self.M_e = self.m_e * self.N_a
        """Electron molar mass [kg/mol]."""

        self.eV_to_K = self.e / self.k_b
        """Conversion factor from eV to K."""

        self.K_to_eV = self.k_b / self.e
        """Conversion factor from K to eV."""

        self.eV_to_J = self.e
        """Conversion factor from eV to J."""

        self.J_to_eV = 1 / self.e
        """Conversion factor from J to eV."""

        self.K_to_J = self.k_b
        """Conversion factor from K to J."""

        self.J_to_K = 1 / self.k_b
        """Conversion factor from J to K."""

        self.Da = constants.atomic_mass
        """Dalton or atomic mass constant [kg]."""

        self.m_N2 = 28.013_4 * self.Da
        """N2 mass [kg]."""

        self.R = constants.gas_constant
        """Ideal gas constant [J/(mol K)]."""

        self.R_kmol = self.R * 1e-3
        """Ideal gas constant [J/(kmol K)]."""

        self.P_1_bar = 1e5
        """1 bar [Pa]."""

        self.h = constants.Planck
        """Planck constant [J s]."""

        self.hbar = constants.hbar
        """Reduced Planck constant [J s]."""

        self.c = constants.speed_of_light
        """Speed of light [m/s]."""

        self.epsilon_0 = constants.epsilon_0
        """Vacuum permittivity [F/m]."""

        self.stefan_boltzmann = (
            2 * np.pi**5 * self.k_b**4 / (15 * self.h**3 * self.c**2)
        )
        """Stefan-Boltzmann constant [W/(m^2 K^4)]."""

        self.spitzer_constant = (
            (4 * np.pi * self.epsilon_0 / self.e) ** 2
            * (self.k_b) ** (3 / 2)
            / (4 / 3 * (2 * np.pi * self.m_e) ** (1 / 2))
        ) * 1.96
        r"""Spitzer conductivity constant in Ohm^(-1) . m^(-1) . K^(-3/2).

        The conductivity of a fully-ionized plasma is given by the Spitzer conductivity:

        .. math::

            \sigma = \text{Spitzer constant} \times \frac{T_e^{3/2}}{\log(\lambda)}

        with :

        * :math:`T_e` the electron temperature in K,
        * :math:`\log(\lambda)` the Coulomb logarithm.

        The constant is given by:

        .. math::

            \begin{align}
                \text{Spitzer constant}
                    &= \left(\frac{4 \pi \varepsilon_0}{e}\right)^2
                       \times \left(\frac{k_b^{3/2}}{\frac{4}{3} \sqrt{2 \pi m_e} }\right)
                       \times 1.96
                    &= 1.53 \times 10^{-2} \, \Omega^{-1} \, \text{m}^{-1} \, \text{K}^{-3/2}
            \end{align}

        with:

        * :math:`\varepsilon_0` the vacuum permittivity,
        * :math:`e` the elementary charge,
        * :math:`k_b` the Boltzmann constant,
        * :math:`m_e` the electron mass.
        """
