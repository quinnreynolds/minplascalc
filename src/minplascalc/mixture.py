"""Module for handling LTE plasma mixtures.

This includes species composition and thermodynamic properties.
"""

import logging
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from minplascalc import functions_radiation, functions_transport
from minplascalc import species as _sp
from minplascalc import units as u

__all__ = ["lte_from_names", "LTE", "LTEWithoutElectrons"]


@dataclass(frozen=True)
class _EquilibriumState:
    """Derived quantities for one converged mixture state."""

    T: float
    P: float
    particle_numbers: np.ndarray
    number_densities: np.ndarray
    mole_fractions: np.ndarray
    masses: np.ndarray
    charge_numbers: np.ndarray
    reference_energies: np.ndarray
    ionization_lowering: np.ndarray
    kbt: float
    volume: float
    n_tot: float
    rho: float
    mean_particle_mass: float


@dataclass(frozen=True)
class _EquilibriumTemperatureTangent:
    """Piecewise derivative of the constrained equilibrium state."""

    particle_derivative: np.ndarray
    mole_fraction_derivative: np.ndarray
    reference_energy_derivative: np.ndarray


class LTE:
    def __init__(
        self,
        species: list[_sp.Monatomic | _sp.Diatomic | _sp.Polyatomic],
        x0: list[float],
        T: float,
        P: float,
        gfe_initial_particles: float,
        gfe_rtol: float,
        gfe_max_iter: int,
    ):
        r"""Local Thermodynamic Equilibrium (LTE) plasma mixture object.

        Class representing a thermal plasma specification with multiple
        species, and methods for calculating equilibrium species concentrations
        at different temperatures and pressures using the principle of Gibbs
        free energy minimisation.

        Parameters
        ----------
        species : list[_sp.Monatomic | _sp.Diatomic | _sp.Polyatomic]
            All species participating in the mixture (excluding electrons which
            are added automatically), as minplascalc Species objects.
        x0 : list[float]
            Constraint mole fractions for each species, typically the
            room-temperature composition of the plasma-generating gas.
            It should be the same length as species.
        T : float
            LTE plasma temperature, in :math:`\text{K}`.
        P : float
            LTE plasma pressure, in :math:`\text{Pa}`.
        gfe_initial_particles : float
            Gibbs Free Energy minimiser solution control: Starting estimate for
            number of particles of each species. Typically O(1e20).
        gfe_rtol : float
            Gibbs Free Energy minimiser solution control: Relative tolerance at
            which solution for particle numbers is considered converged.
            Typically O(1e-10).
        gfe_max_iter : int
            Gibbs Free Energy minimiser solution control: Bailout loop count
            value for iterative solver. Typically O(1e3).

        Raises
        ------
        ValueError
            If the species list includes an electron species.
        ValueError
            If the species list and constraint mole fractions list are not the
            same length.
        """
        # Check for electron species in the species list.
        if "e" in [sp.name for sp in species]:
            raise ValueError(
                "Electrons are added automatically, please "
                "don't include them in your species list."
            )
        # Check equal length of species and constraint mole fractions lists.
        if len(species) != len(x0):
            raise ValueError("Lists species and x0 must be the same length.")

        # Add electron species to the species list.
        self.__species = self._setup_species_list(species)

        # Per-species constants. The species tuple is immutable (its setter
        # raises), so these are derived once here rather than rebuilt by
        # every property calculation that needs them.
        self.__masses = np.array(
            [sp.molar_mass / u.N_a for sp in self.__species]
        )
        self.__charge_numbers = np.array(
            [sp.charge_number for sp in self.__species]
        )

        self.__state: _EquilibriumState | None = None
        self.__transport_workspace = None
        self.__collision_model = None

        self.x0 = x0
        self.T = T
        self.P = P
        self.gfe_initial_particles = gfe_initial_particles
        self.gfe_rtol = gfe_rtol
        self.gfe_max_iter = gfe_max_iter

        self.__isLTE = (
            False  # Flag to indicate if LTE composition has been calculated.
        )

        # Number of particles of each species.
        self.__Ni: np.ndarray = np.zeros(len(self.species))

    @property
    def species(self):
        return self.__species

    @species.setter
    def species(self, species):
        raise TypeError(
            "Attribute species is read-only. Please create a new "
            "Mixture object if you wish to change the plasma "
            "species."
        )

    @property
    def masses(self) -> np.ndarray:
        r"""Mass of each species, in :math:`\text{kg.particle}^{-1}`."""
        return self.__masses

    @property
    def charge_numbers(self) -> np.ndarray:
        """Charge number of each species, in units of the electron charge."""
        return self.__charge_numbers

    @property
    def x0(self):
        return self.__x0

    @x0.setter
    def x0(self, x0):
        self._validate_x0_length(x0)
        self.__x0 = self._format_x0(x0)
        self._invalidate_equilibrium()

    @property
    def T(self):
        return self.__T

    @T.setter
    def T(self, T):
        self.__T = T
        self._invalidate_equilibrium()

    @property
    def P(self):
        return self.__P

    @P.setter
    def P(self, P):
        self.__P = P
        self._invalidate_equilibrium()

    @property
    def gfe_initial_particles(self):
        return self.__gfe_initial_particles

    @gfe_initial_particles.setter
    def gfe_initial_particles(self, value):
        self.__gfe_initial_particles = value
        self._invalidate_equilibrium()

    @property
    def gfe_rtol(self):
        return self.__gfe_rtol

    @gfe_rtol.setter
    def gfe_rtol(self, value):
        self.__gfe_rtol = value
        self._invalidate_equilibrium()

    @property
    def gfe_max_iter(self):
        return self.__gfe_max_iter

    @gfe_max_iter.setter
    def gfe_max_iter(self, value):
        self.__gfe_max_iter = value
        self._invalidate_equilibrium()

    def _invalidate_equilibrium(self) -> None:
        """Discard quantities derived from mutable mixture inputs."""
        self.__isLTE = False
        self.__state = None
        self.__transport_workspace = None

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(species={self.species},"
            f"x0={self.x0},T={self.T},P={self.P},"
            f"gfe_initial_particles={self.gfe_initial_particles},"
            f"gfe_rtol={self.gfe_rtol},gfe_max_iter={self.gfe_max_iter})"
        )

    def __str__(self):
        species_str, x0_str = self._format_species_string()
        return (
            f"LTE mixture species: {species_str}\n"
            f"Initial composition: {x0_str}\n"
            f"Temperature: {self.T} K\nPressure: {self.P} Pa"
        )

    def _setup_species_list(
        self,
        species: list[_sp.Monatomic | _sp.Diatomic | _sp.Polyatomic],
    ) -> tuple:
        """Set up the species list, adding electrons by default."""
        return tuple(list(species) + [_sp.Electron()])

    def _validate_x0_length(self, x0: list[float]) -> None:
        """Validate constraint mole fractions length (electrons case)."""
        if len(x0) == len(self.species) - 1:
            return  # Valid: x0 for all species except electrons
        else:
            raise ValueError(
                "Please specify constraint mole fractions for all "
                "species (except electrons)."
            )

    def _format_x0(self, x0: list[float]) -> tuple[float, ...]:
        """Format x0 for electrons case."""
        # Add electron mole fraction, set to zero
        return tuple(list(x0) + [0.0])

    def _format_species_string(self) -> tuple[tuple, tuple]:
        """Format species and x0 strings for display (electrons case)."""
        species_tuple = tuple([sp.name for sp in self.species[:-1]])
        x0_tuple = self.x0[:-1]
        return species_tuple, x0_tuple

    def _get_constraint_dimensions(
        self, elements_count: int
    ) -> tuple[int, int]:
        """Calculate matrix dimensions for solver (electrons case)."""
        nb_species = len(self.species)
        minimiser_dof = (
            nb_species + elements_count + 1
        )  # +1 for charge constraint
        constraints_dof = elements_count + 1
        return minimiser_dof, constraints_dof

    def _setup_charge_constraints(
        self, A_matrix: np.ndarray, A_matrix_transpose: np.ndarray
    ) -> None:
        """Set up charge neutrality constraints (electrons case)."""
        for j, qc in enumerate(self.charge_numbers):
            A_matrix[j, -1] = qc
            A_matrix_transpose[-1, j] = qc

    def _constraint_matrix(self) -> np.ndarray:
        """Return species-by-conservation-law coefficients."""
        element_names = sorted(
            {element for sp in self.species for element in sp.stoichiometry}
        )
        _, constraint_count = self._get_constraint_dimensions(
            len(element_names)
        )
        matrix = np.zeros((len(self.species), constraint_count))
        for column, element in enumerate(element_names):
            matrix[:, column] = [
                sp.stoichiometry.get(element, 0) for sp in self.species
            ]
        self._setup_charge_constraints(matrix, matrix.T)
        return matrix

    def _calculate_ionization_lowering(
        self, number_densities: np.ndarray
    ) -> np.ndarray:
        """Calculate ionization energy lowering (electrons case)."""
        nb_species = len(self.species)
        kbt = u.k_b * self.T
        dE = np.zeros(nb_species)

        # Calculate the effective charge number z*.
        charge_numbers = self.charge_numbers
        weighted_charge_sum_squared, weighted_charge_sum = 0.0, 0.0
        for z_i, nd in zip(charge_numbers, number_densities):
            if z_i > 0:  # Only consider positively charged species.
                weighted_charge_sum += nd * z_i
                weighted_charge_sum_squared += nd * z_i**2
        z_star = weighted_charge_sum_squared / weighted_charge_sum

        # Get the electron number density.
        n_e = number_densities[-1]  # m^-3

        # Calculate the Debye sphere radius, to the power 3.
        debye_pow3 = (
            u.epsilon_0 * kbt / (4 * np.pi * (z_star + 1) * n_e * u.e**2)
        ) ** (3 / 2)

        # Calculate ionisation energy lowering for each positive ion.
        for i, charge_number in enumerate(charge_numbers):
            if charge_number > 0:
                # Calculate the ion-sphere radius, to the power 3.
                ai_pow3 = 3 * charge_number / (4 * np.pi * n_e)
                # Calculate the ionisation energy lowering.
                dE[i] = (
                    kbt
                    * ((ai_pow3 / debye_pow3 + 1) ** (2 / 3) - 1)
                    / (2 * (z_star + 1))
                )

        return dE

    def _ionization_lowering_derivatives(
        self, particle_numbers: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ionisation-lowering derivatives for the active species."""
        count = len(self.species)
        dE_dN = np.zeros((count, count))
        dE_dT = np.zeros(count)
        charges = self.charge_numbers
        positive = charges > 0

        charge_sum = particle_numbers[positive] @ charges[positive]
        charge_square_sum = particle_numbers[positive] @ charges[positive] ** 2
        z_star = charge_square_sum / charge_sum
        denominator = z_star + 1

        dzstar_dN = np.zeros(count)
        dzstar_dN[positive] = (
            charges[positive] ** 2 * charge_sum
            - charge_square_sum * charges[positive]
        ) / charge_sum**2

        total_particles = particle_numbers.sum()
        electron_index = count - 1
        electron_particles = particle_numbers[electron_index]
        volume = total_particles * u.k_b * self.T / self.P
        electron_density = electron_particles / volume
        debye_pow3 = (
            u.epsilon_0
            * u.k_b
            * self.T
            / (4 * np.pi * denominator * electron_density * u.e**2)
        ) ** (3 / 2)

        dlog_ratio_dN = 1.5 * dzstar_dN / denominator - 0.5 / total_particles
        dlog_ratio_dN[electron_index] += 0.5 / electron_particles

        for i, charge in enumerate(charges):
            if charge <= 0:
                continue
            ion_sphere_pow3 = 3 * charge / (4 * np.pi * electron_density)
            ratio = ion_sphere_pow3 / debye_pow3
            shape = (ratio + 1) ** (2 / 3) - 1
            shape_derivative = 2 / 3 * (ratio + 1) ** (-1 / 3)
            prefactor = u.k_b * self.T / 2

            ratio_dN = ratio * dlog_ratio_dN
            dE_dN[i] = prefactor * (
                shape_derivative * ratio_dN / denominator
                - shape * dzstar_dN / denominator**2
            )
            dE_dT[i] = (
                u.k_b
                / (2 * denominator)
                * (shape - 2 * ratio * shape_derivative)
            )

        return dE_dN, dE_dT

    def _get_species_for_iteration(
        self, number_densities: np.ndarray
    ) -> tuple[np.ndarray, list]:
        """Get species and densities for iteration (electrons case)."""
        return number_densities[:-1], self.species[:-1]

    def __get_reference_energies(self) -> tuple[np.ndarray, np.ndarray]:
        r"""Calculate the reference energy values for all species.

        Calculate the reference energy values for all species, including
        ionisation energy lowering from limitation theory of [Stewart1966]_
        Note that lowering only applied to positive ions.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Reference energy and ionisation energy lowering of each species
            in the mixture, in :math:`\text{J}`.

        Notes
        -----
        The reference energy :math:`E_i^0` of each species is calculated as:

        * For uncharged monatomic species and electrons, :math:`E_i^0 = 0`,
        * For uncharged polyatomic species, :math:`E_i^0` is the negative of
          the dissociation energy,
        * For charged species, :math:`E_i^0` is :math:`E_i^0` of the species
          with one fewer charge number plus the lowered ionisation energy of
          that species.

        The lowered ionisation energy :math:`\Delta E_i` of each species is
        using the equation 5 of [Stewart1966]_ (:math:`J` in the article is
        the lowered ionisation energy):

        .. math::

            \frac{\delta E_i}{k_B T} = \frac{
                \left [ \left (\frac{a_i}{l_D} \right )^3 + 1
                    \right ]^\frac{2}{3} -1
                }{2 \left( z^*+1 \right)}

        where:

        .. math::

            z^* = \left ( \frac{\sum z_j^2 n_j}{\sum z_j n_j}
                  \right )_{j \neq e}, \quad
            a_i = \left ( \frac{3 z_i}{4 \pi n_e} \right )^\frac{1}{3}, \quad
            l_D = \left ( \frac{\epsilon_0 k_B T}{4 \pi e^2 \left ( z^* + 1
                  \right ) n_e} \right )^\frac{1}{2}

        Here,

        * :math:`\delta E_i` is the amount the ionisation energy of species i
          is lowered by, in :math:`\text{J}`,
        * :math:`a_i` is the ion-sphere radius of species i,
        * :math:`l_D` is the Debye sphere radius,
        * :math:`z^*` is the effective charge number in a plasma consisting of
          a mixture of species of different charges,
        * :math:`z_j` is the charge number of species j,
        * :math:`n_j` is the number density (particles per cubic meter) of
           species j,
        * :math:`e` is the electron charge.
        """
        nb_species = len(self.species)
        kbt = u.k_b * self.T

        # Array of number densities of each species in the plasma.
        N_i: np.ndarray = self.__Ni  # Number of particles of each species.
        N_tot = N_i.sum()  # Total number of particles in the plasma.
        V = N_tot * kbt / self.P  # Volume of the plasma, in m3.
        # Number density of each species, in particles/m3.
        number_densities = N_i / V

        # Initialise arrays for reference energy and ionisation energy
        # lowering.
        E0 = np.zeros(nb_species)
        dE = np.zeros(nb_species)

        # For (uncharged) polyatomic species, the reference energy is the
        # negative of the dissociation energy.
        for i, sp in enumerate(self.species):
            if sum(sp.stoichiometry.values()) >= 2:
                E0[i] = -sp.dissociation_energy

        # Calculate ionization energy lowering (electron-containing only)
        dE = self._calculate_ionization_lowering(number_densities)

        # Get the neutral species.
        neutral_species = [sp for sp in self.species if sp.charge_number == 0]

        # Calculate the reference energy for each species.
        for neutral_sp in neutral_species:
            # Get the negatively charged species with the same stoichiometry.
            negatively_charged_sp = [
                (i, sp)
                for i, sp in enumerate(self.species)
                if (
                    sp.stoichiometry == neutral_sp.stoichiometry
                    and sp.charge_number <= 0
                )
            ]
            # Sort the negatively charged species by charge number in
            # descending order.
            # Example: -2, -1, 0.
            negatively_charged_sp.sort(
                key=lambda sp: sp[1].charge_number, reverse=True
            )

            # Get the positively charged species with the same stoichiometry.
            positively_charged_sp = [
                (i, sp)
                for i, sp in enumerate(self.species)
                if (
                    sp.stoichiometry == neutral_sp.stoichiometry
                    and sp.charge_number >= 0
                )
            ]
            # Sort the positively charged species by charge number in
            # ascending order.
            # Example: 0, 1, 2.
            positively_charged_sp.sort(
                key=lambda sp: sp[1].charge_number, reverse=False
            )

            # Calculate the reference energy for non-neutral species.
            # .. Positive ions.
            for (ifrom, spfrom), (ito, spto) in zip(
                positively_charged_sp[:-1], positively_charged_sp[1:]
            ):
                # The reference energy is the reference energy of the species
                # with one fewer charge number,
                # plus the lowered ionisation energy of that species.
                E0[ito] = E0[ifrom] + spfrom.ionisation_energy - dE[ifrom]

                # Code example:
                # positively_charged_sp = [(index_H, H), (index_H+, H+), (index_H2+, H2+)]  # noqa: E501
                # positively_charged_sp[:-1] = [(index_H, H), (index_H+, H+)]
                # positively_charged_sp[1:] = [(index_H+, H+), (index_H2+, H2+)]  # noqa: E501
                #
                # 1st iteration:
                #   (ifrom, spfrom) = (index_H, H)
                #   (ito, spto) = (index_H+, H+)
                #   E0[index_H+] = E0[index_H] + H.ionisation_energy - dE[index_H]  # noqa: E501
                #                = 0 + H.ionisation_energy - 0
                # 2nd iteration:
                #   (ifrom, spfrom) = (index_H+, H+)
                #   (ito, spto) = (index_H2+, H2+)
                #   E0[index_H2+] = E0[index_H+] + H+.ionisation_energy - dE[index_H+]  # noqa: E501
            # .. Negative ions.
            for (ifrom, spfrom), (ito, spto) in zip(
                negatively_charged_sp[:-1], negatively_charged_sp[1:]
            ):
                E0[ito] = E0[ifrom] - spto.ionisation_energy + dE[ito]
                # NOTE: For negative ions, dE is equal to zero.

        # Return the reference energy and ionisation energy lowering.
        return E0, dE

    def __get_reference_energy_derivatives(
        self,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return derivatives of reference energies with respect to N and T."""
        dE_dN, dE_dT = self._ionization_lowering_derivatives(self.__Ni)
        E0_dN = np.zeros_like(dE_dN)
        E0_dT = np.zeros_like(dE_dT)

        neutral_species = [sp for sp in self.species if sp.charge_number == 0]
        for neutral_sp in neutral_species:
            negative = sorted(
                (
                    (i, sp)
                    for i, sp in enumerate(self.species)
                    if sp.stoichiometry == neutral_sp.stoichiometry
                    and sp.charge_number <= 0
                ),
                key=lambda item: item[1].charge_number,
                reverse=True,
            )
            positive = sorted(
                (
                    (i, sp)
                    for i, sp in enumerate(self.species)
                    if sp.stoichiometry == neutral_sp.stoichiometry
                    and sp.charge_number >= 0
                ),
                key=lambda item: item[1].charge_number,
            )
            for (source, _), (target, _) in zip(positive[:-1], positive[1:]):
                E0_dN[target] = E0_dN[source] - dE_dN[source]
                E0_dT[target] = E0_dT[source] - dE_dT[source]
            for (source, _), (target, _) in zip(negative[:-1], negative[1:]):
                E0_dN[target] = E0_dN[source] + dE_dN[target]
                E0_dT[target] = E0_dT[source] + dE_dT[target]

        return E0_dN, E0_dT

    def calculate_composition(self) -> np.ndarray:
        r"""Calculate the LTE composition of the plasma in m^-3.

        An iterative Lagrange multiplier approach is used to minimise the Gibbs
        free energy of the plasma, subject to the constraints of constant
        temperature, pressure, species mole fractions and charge neutrality.

        Returns
        -------
        np.ndarray
            Number density of each species in the plasma as listed in
            :meth:`~minplascalc.mixture.Mixture.species`,
            in :math:`\text{particles.m}^{-3}`.

        Notes
        -----
        The Gibbs free energy minimisation problem is solved iteratively by
        solving a linear system of equations. The system is defined by the
        following equations:

        .. math::

            \left( dG \right)_{P, T} = 0 = \sum_j \mu_j dN_j

        where :math:`dG` is the change in Gibbs free energy, :math:`P` is the
        pressure, :math:`T` is the temperature, :math:`\mu_j` is the chemical
        potential of species :math:`j`, and :math:`N_j` is the number of
        particles of species :math:`j`. The chemical potential of each species
        is given by:

        .. math::

            \mu_i = \frac{\partial G}{\partial N_i}, \quad i = 1, 2, \ldots, n

        where :math:`G` is the Gibbs free energy of the plasma, :math:`N_i` is
        the number of particles of species :math:`i`, and :math:`\mu_i` is the
        chemical potential of species :math:`i`.

        The Gibbs free energy of the plasma is given by:

        .. math::

            G = G^0 + \sum_i \mu_i N_i
              = \sum_i \left ( E_i^0 - k_B T \log \left
                ( \frac{Z_{\text{tot},i}}{N_i} \right ) \right ) N_i

        where:

        * :math:`G^0` is the Gibbs free energy at zero temperature,
        * :math:`E_i^0` is the reference energy of species :math:`i`,
        * :math:`Z_{\text{tot}, i}` is the total partition function of species
          :math:`i`,
        * :math:`k_B` is the Boltzmann constant,
        * :math:`T` is the temperature,
        * :math:`N_i` is the number of particles of species :math:`i`.

        The total partition function of each species is given by:

        .. math::

            Z_{\text{tot}, i} = Z_{tr, i} Z_{rot, i} Z_{vib, i} Z_{el, i}

        TODO: write how the minimisation is done.
        """
        nb_species = len(self.species)
        kbt = u.k_b * self.T

        # If the composition has already been calculated, return it.
        # Otherwise, calculate it.
        if self.__isLTE:
            N_i = self.__Ni  # Number of particles of each species.
            N_tot = N_i.sum()  # Total number of particles in the plasma.
            V = N_tot * kbt / self.P  # Volume of the plasma, in m3.
            return N_i / V  # Number density of each species, in particles/m3.

        # Get the set of unique elements in the species.
        # Electrons are discarded.
        # Example: if species = {N2, O2, NO}, then unique_elements = {N, O}.
        unique_elements = set(
            s for sp in self.species for s in sp.stoichiometry
        )
        # For each unique element, create a dictionary with the element name,
        # stoichiometric coefficient in each species, and total number of that
        # element in the plasma.
        elements = [
            {"name": name, "stoich_coeff": None, "N_tot": 0}
            for name in sorted(unique_elements)
        ]
        # Fill in the stoichiometric coefficients.
        # Example: if species = {N2, O2, NO},
        #          then elements = [{N, [2, 0, 1], 0}, {O, [0, 2, 1], 0}].
        for element in elements:
            element["stoich_coeff"] = [
                sp.stoichiometry.get(element["name"], 0) for sp in self.species
            ]
        # Calculate the total number density of each element in the plasma.
        # Example: if species = {N2, O2, NO}, and x0 = [0.7, 0.2, 0.1], then
        # elements = [{N, [2, 0, 1], 1.5e24}, {O, [0, 2, 1], 0.5e24}].
        # TODO: Check if the factor 1e24 is arbitrary.
        for element in elements:
            element["N_tot"] = sum(
                1e24 * c * x0
                for c, x0 in zip(element["stoich_coeff"], self.x0)
            )

        # Create the Gibbs free energy minimisation matrix and vector.
        # Example:
        #   if species = {N2, O2, NO, N2+, e-}, and x0 = [0.7, 0.2, 0.1, 0],
        #   then:
        #   -nb_species = 5,
        #   -elements = [{N, [2, 0, 1, 2], 1.5e24}, {O, [0, 2, 1, 0], 0.5e24}],
        #   -and minimiser_dof = 5 + 2 + 1 = 8.
        #
        #  gfe_matrix = [     N2  O2  NO  N2+  e-  N  O  charge
        #           ┌ N2    [  0,  0,  0,  0,  0,  2,  0,  0],
        #           │ O2    [  0,  0,  0,  0,  0,  0,  2,  0],
        #   species ┥ NO    [  0,  0,  0,  0,  0,  1,  1,  0],
        #           │ N2+   [  0,  0,  0,  0,  0,  2,  0,  1],
        #           └ e-    [  0,  0,  0,  0,  0,  0,  0, -1],
        #   element ┌  N    [  2,  0,  1,  2,  0,  0,  0,  0],
        #           └  O    [  0,  2,  1,  0,  0,  0,  0,  0],
        #   charge          [  0,  0,  0,  1, -1,  0,  0,  0],
        # ]
        # gfe_vector = [
        #           ┌ N2     0,
        #           │ O2     0,
        #   species ┥ NO     0,
        #           │ N2+    0,
        #           └ e-     0,
        #   element ┌  N     1.5e24,
        #           └  O     0.5e24,
        #   charge           0,
        # ]
        #
        minimiser_dof, constraints_dof = self._get_constraint_dimensions(
            len(elements)
        )
        gfe_matrix = np.zeros((minimiser_dof, minimiser_dof))
        gfe_vector = np.zeros(minimiser_dof)
        # The first nb_species rows and columns are for the species.
        # The next len(self._elements) rows and columns are for the elements.
        # The last row and column are for the charge neutrality.
        A_matrix_constraints = self._constraint_matrix()
        A_matrix_constraints_transpose = A_matrix_constraints.T
        b_vector_constraints = np.zeros(constraints_dof)

        for i, element in enumerate(elements):
            b_vector_constraints[i] = element["N_tot"]

        gfe_matrix[:nb_species, nb_species:] = A_matrix_constraints
        gfe_matrix[nb_species:, :nb_species] = A_matrix_constraints_transpose
        gfe_vector[nb_species:] = b_vector_constraints

        # Initialise the number of particles of each species.
        # The estimate is the same for all species, and is given by the user.
        # It is typically O(1e20).
        self.__Ni = np.full(nb_species, self.gfe_initial_particles)

        # Minimise the Gibbs free energy.
        # The minimisation is done iteratively, with a relaxation factor to
        # prevent large changes in the number of particles of each species.

        minimiser_success = (
            False  # Flag to indicate if the minimiser has converged.
        )
        # Factors to control the relaxation.
        # The relaxation factor is decreased at each failed iteration.
        governor_factors = np.linspace(0.9, 0.1, 9)
        governor_iters = 0  # Iteration counter for the relaxation factor.

        while not minimiser_success and governor_iters < len(governor_factors):
            minimiser_success = True  # Assume the minimiser will converge.
            governor_factor = governor_factors[
                governor_iters
            ]  # Relaxation factor.
            # Initial relative tolerance.
            relative_tolerance = self.gfe_rtol * 10
            minimiser_iters = 0  # Iteration counter for the minimiser.

            while relative_tolerance > self.gfe_rtol:
                # Calculate reference energy and ionisation energy lowering.
                self.__E0, self.__dE = self.__get_reference_energies()
                # Total number of particles in the plasma.
                N_tot = self.__Ni.sum()
                V = N_tot * kbt / self.P  # Volume of the plasma, in m3.

                #  gfe_matrix[:nb_species, :nb_species] = [
                #               N2                         O2                  NO               N2+  e-  # noqa: E501
                #   N2    [  -kbt/N_tot + kbt/N_N2, -kbt/N_tot           , -kbt/N_tot           , ...],  # noqa: E501
                #   O2    [  -kbt/N_tot           , -kbt/N_tot + kbt/N_O2, -kbt/N_tot           , ...],  # noqa: E501
                #   NO    [  -kbt/N_tot           , -kbt/N_tot           , -kbt/N_tot + kbt/N_NO, ...],  # noqa: E501
                #   N2+   [  ...                  ,                                                  ],  # noqa: E501
                #   e-    [  ...                  ,                                                  ],  # noqa: E501
                # ]
                off_diag = -kbt / N_tot * np.ones(nb_species)
                on_diag = np.diag(kbt / self.__Ni)
                gfe_matrix[:nb_species, :nb_species] = off_diag + on_diag

                # Calculate the total partition function of each species.
                total = [
                    species.total_partition_function(V, self.T, dE)
                    for species, dE in zip(self.species, self.__dE)
                ]

                # Calculate the chemical potential of each species.
                mu = -kbt * np.log(total / self.__Ni) + self.__E0

                # gfe_vector[:nb_species] = [
                #     -( E_0_N2 - kbt * log(Z_tot / N_N2) ),
                #     -( E_0_O2 - kbt * log(Z_tot / N_O2) ),
                #     ...,
                #     ...,
                #     ...,
                # ]
                gfe_vector[:nb_species] = -mu

                # Solve the linear system of equations.
                # The solution is the estimated number of particles of
                # each species.
                solution = np.linalg.solve(gfe_matrix, gfe_vector)

                # New number of particles of each species.
                new_Ni = solution[0:nb_species]
                # Absolute change in the number of particles.
                delta_Ni = abs(new_Ni - self.__Ni)
                max_Ni_index = new_Ni.argmax()
                relative_tolerance = (
                    delta_Ni[max_Ni_index] / solution[max_Ni_index]
                )
                # TODO: Why not take the maximume relative tolerance of all
                # species, instead of the relative tolerance of the species
                # with the maximum number of particles?

                # .. Apply relaxation factor to the new number of particles.
                # Maximum allowed change.
                max_allowed_delta_Ni = governor_factor * self.__Ni
                # Clip the change to the maximum allowed change.
                delta_Ni = delta_Ni.clip(min=max_allowed_delta_Ni)
                # Calculate the relaxation factor.
                new_relaxation_factors = max_allowed_delta_Ni / delta_Ni
                relaxation_factor = new_relaxation_factors.min()
                # Apply the relaxation factor to the new number of particles.
                self.__Ni = (
                    1 - relaxation_factor
                ) * self.__Ni + relaxation_factor * new_Ni

                minimiser_iters += 1
                if minimiser_iters > self.gfe_max_iter:
                    minimiser_success = False
                    break
            governor_iters += 1
        if not minimiser_success:
            warnings.warn(
                "Minimiser could not find a converged solution, "
                "results may be inaccurate."
            )
        logging.debug(governor_iters, relaxation_factor, relative_tolerance)
        logging.debug(self.__Ni)

        self.__isLTE = True
        self.__state = None

        N_i = self.__Ni  # Number of particles of each species.
        N_tot = N_i.sum()  # Total number of particles in the plasma.
        V = N_tot * kbt / self.P  # Volume of the plasma, in m3.
        return N_i / V  # Number density of each species, in particles/m3.

    def _equilibrium_temperature_tangent(
        self,
    ) -> _EquilibriumTemperatureTangent:
        """Differentiate the full constrained equilibrium state."""
        self.calculate_composition()
        particle_numbers = self.__Ni
        particle_total = particle_numbers.sum()
        count = len(self.species)
        kbt = u.k_b * self.T
        volume = particle_total * kbt / self.P

        _, lowering = self.__get_reference_energies()
        reference_dN, reference_dT = self.__get_reference_energy_derivatives()
        partitions = np.array(
            [
                species.total_partition_function(volume, self.T, dE)
                for species, dE in zip(self.species, lowering)
            ]
        )
        log_partition_ratio = np.log(partitions / particle_numbers)
        dlog_partition_dT = np.array(
            [
                1 / self.T + species.dlog_total_partition_dT(self.T, dE)
                for species, dE in zip(self.species, lowering)
            ]
        )
        chemical_potential_dT = (
            reference_dT
            - u.k_b * log_partition_ratio
            - kbt * dlog_partition_dT
        )

        constraints = self._constraint_matrix()
        constraint_count = constraints.shape[1]
        system = np.zeros((count + constraint_count,) * 2)
        system[:count, :count] = (
            -kbt / particle_total
            + np.diag(kbt / particle_numbers)
            + reference_dN
        )
        system[:count, count:] = constraints
        system[count:, :count] = constraints.T

        rhs = np.zeros(count + constraint_count)
        rhs[:count] = -chemical_potential_dT
        particle_derivative = np.linalg.solve(system, rhs)[:count]
        total_derivative = particle_derivative.sum()
        mole_fraction_derivative = (
            particle_derivative / particle_total
            - particle_numbers * total_derivative / particle_total**2
        )
        reference_energy_derivative = (
            reference_dT + reference_dN @ particle_derivative
        )
        return _EquilibriumTemperatureTangent(
            particle_derivative=particle_derivative,
            mole_fraction_derivative=mole_fraction_derivative,
            reference_energy_derivative=reference_energy_derivative,
        )

    def calculate_composition_temperature_derivative(self) -> np.ndarray:
        r"""Calculate the piecewise analytical derivative of mole fractions.

        The equilibrium conditions are differentiated implicitly at the
        converged state. Electronic levels below the ionisation-lowered cutoff
        retain their current active/inactive status, so the result is the
        one-sided analytical branch between discrete level crossings.

        Returns
        -------
        np.ndarray
            :math:`dx_i/dT`, in :math:`\text{K}^{-1}`.
        """
        return self._equilibrium_temperature_tangent().mole_fraction_derivative

    def _equilibrium_state(self) -> _EquilibriumState:
        """Return all commonly used quantities for the current LTE state."""
        if self.__state is not None:
            return self.__state

        number_densities = self.calculate_composition()
        kbt = u.k_b * self.T
        particle_numbers = self.__Ni
        particle_total = particle_numbers.sum()
        volume = particle_total * kbt / self.P
        n_tot = number_densities.sum()
        rho = number_densities @ self.masses

        self.__state = _EquilibriumState(
            T=self.T,
            P=self.P,
            particle_numbers=particle_numbers,
            number_densities=number_densities,
            mole_fractions=number_densities / n_tot,
            masses=self.masses,
            charge_numbers=self.charge_numbers,
            reference_energies=self.__E0,
            ionization_lowering=self.__dE,
            kbt=kbt,
            volume=volume,
            n_tot=n_tot,
            rho=rho,
            mean_particle_mass=rho / n_tot,
        )
        return self.__state

    def _transport_workspace(self):
        """Return lazily evaluated transport intermediates for this state."""
        if self.__transport_workspace is None:
            self.__transport_workspace = (
                functions_transport._TransportWorkspace(self)
            )
        return self.__transport_workspace

    def _collision_model(self):
        """Return temperature-independent numeric collision descriptors."""
        if self.__collision_model is None:
            self.__collision_model = functions_transport._CollisionModel(
                self.species
            )
        return self.__collision_model

    @contextmanager
    def _at_temperature(self, T: float) -> Iterator[None]:
        """Temporarily change temperature, restoring the complete LTE state."""
        self._equilibrium_state()
        original = (
            self.__T,
            self.__isLTE,
            self.__Ni,
            self.__E0,
            self.__dE,
            self.__state,
            self.__transport_workspace,
        )
        self.T = T
        try:
            yield
        finally:
            (
                self.__T,
                self.__isLTE,
                self.__Ni,
                self.__E0,
                self.__dE,
                self.__state,
                self.__transport_workspace,
            ) = original

    def calculate_density(self) -> float:
        r"""Calculate the LTE density of the plasma.

        Returns
        -------
        float
            Plasma density, in :math:`\text{kg.m}^{-3}`.

        Notes
        -----
        The plasma density is calculated as:

        .. math::

            \rho = \frac{1}{N_A} \sum_i n_i M_i

        where:

        * :math:`\rho` is the plasma density, in :math:`\text{kg.m}^{-3}`,
        * :math:`N_A` is Avogadro's number, in :math:`\text{mol}^{-1}`,
        * :math:`n_i` is the number density of species :math:`i`,
          in :math:`\text{particles.m}^{-3}`,
        * :math:`M_i` is the molar mass of species :math:`i`,
          in :math:`\text{kg.mol}^{-1}`.
        """
        return self._equilibrium_state().rho

    def calculate_species_enthalpies(self) -> np.ndarray:
        r"""Calculate the LTE enthalpy for each component in the plasma.

        These are needed for calculation of the effective thermal conductivity.

        Returns
        -------
        np.ndarray
            Enthalpies of each species, in :math:`\text{J.kg}^{-1}`.

        Notes
        -----
        The enthalpy of each species is calculated as:

        .. math::

            H_i = U_i + E_i^0 + k_B T

        where:

        * :math:`H_i` is the enthalpy of species :math:`i`,
          in :math:`\text{J.particle}^{-1}`,
        * :math:`U_i` is the internal energy of species :math:`i`,
          in :math:`\text{J.particle}^{-1}`,
        * :math:`E_i^0` is the reference energy of species :math:`i`,
          in :math:`\text{J}`,
        * :math:`k_B` is the Boltzmann constant, in :math:`\text{J.K}^{-1}`,
        * :math:`T` is the temperature, in :math:`\text{K}`.

        The enthalpy is then divided by the mass of the species to obtain
        the enthalpy per unit mass.

        .. math::

            h_i = \frac{H_i}{m_i} = \frac{H_i}{M_i / N_A}

        where:

        * :math:`h_i` is the enthalpy of species :math:`i`,
          in :math:`\text{J.kg}^{-1}`,
        * :math:`m_i` is the mass of species :math:`i`,
          in :math:`\text{kg.particle}^{-1}`,
        * :math:`M_i` is the molar mass of species :math:`i`,
          in :math:`\text{kg.mol}^{-1}`.
        """
        state = self._equilibrium_state()
        internal_energies = [
            sp.internal_energy(state.T, dE)
            for sp, dE in zip(self.species, state.ionization_lowering)
        ]  # J/particle

        enthalpies = [
            (u_i + E0_i + state.kbt)
            for u_i, E0_i in zip(internal_energies, state.reference_energies)
        ]  # J/particle

        # (kg/mol) / (particle/mol) = kg/particle
        return np.array(enthalpies) / self.masses  # J/kg

    def calculate_enthalpy(self) -> float:
        r"""Calculate the LTE enthalpy of the plasma.

        Referenced to zero at zero Kelvin.

        Returns
        -------
        float
            Enthalpy, in :math:`\text{J.kg}^{-1}`.

        Notes
        -----
        The enthalpy of the plasma is calculated as:

        .. math::

            H = \frac{1}{\rho} \sum_i n_i \left ( H_i -
                    \frac{E_{i=min}^0 M_{i=min}}{M_i} \right)

        where:

        * :math:`H` is the enthalpy of the plasma, in :math:`\text{J.kg}^{-1}`,
        * :math:`\rho` is the plasma density, in :math:`\text{kg.m}^{-3}`,
        * :math:`n_i` is the number density of species :math:`i`,
          in :math:`\text{particles.m}^{-3}`,
        * :math:`H_i` is the enthalpy of species :math:`i`,
          in :math:`\text{J.kg}^{-1}`,
        * :math:`E_{i=min}^0` is the reference energy of the species
          with the lowest reference energy, in :math:`\text{J}`,
        * :math:`M_{i=min}` is the molar mass of the species with the lowest
          reference energy, in :math:`\text{kg.mol}^{-1}`,
        * :math:`M_i` is the molar mass of species :math:`i`,
          in :math:`\text{kg.mol}^{-1}`.
        """
        state = self._equilibrium_state()
        number_densities = state.number_densities
        molar_masses = [sp.molar_mass for sp in self.species]  # kg/mol

        density = state.rho

        mass_enthalpies = self.calculate_species_enthalpies()  # J/kg
        masses = self.masses  # kg/particle
        enthalpies = mass_enthalpies * masses  # J/particle

        # Get the species with the lowest reference energy.
        # Index of the species with the lowest reference energy.
        i_min = np.argmin(state.reference_energies)
        # J/(kg/mol)
        h_mol_0 = (
            state.reference_energies[i_min] / self.species[i_min].molar_mass
        )

        weighted_enthalpy = sum(
            n_i * (h_i - h_mol_0 * M_i)
            for n_i, h_i, M_i in zip(
                number_densities, enthalpies, molar_masses
            )
        )

        return weighted_enthalpy / density

    def calculate_heat_capacity(self, rel_delta_T: float = 0.001) -> float:
        r"""Calculate the LTE heat capacity at constant pressure of the plasma.

        The equilibrium enthalpy is differentiated analytically along the
        constrained composition tangent. The electronic-level active set and
        the species with the lowest reference energy are held fixed, making
        this a piecewise derivative between discrete model transitions.

        Parameters
        ----------
        rel_delta_T : float, optional
            Retained for API compatibility; the analytical derivative does not
            use a temperature step.

        Returns
        -------
        float
            Heat capacity, in :math:`\text{J.kg}^{-1}.\text{K}^{-1}`.

        Notes
        -----
        The heat capacity at constant pressure of the plasma is calculated as:

        .. math::

            C_p = \frac{dH}{dT}

        where:

        * :math:`C_p` is the heat capacity at constant pressure,
          in :math:`\text{J.kg}^{-1}.\text{K}^{-1}`,
        * :math:`H` is the enthalpy of the plasma, in :math:`\text{J.kg}^{-1}`,
        * :math:`T` is the temperature, in :math:`\text{K}`.
        """
        del rel_delta_T
        state = self._equilibrium_state()
        tangent = self._equilibrium_temperature_tangent()
        internal_energies = np.array(
            [
                species.internal_energy(state.T, lowering)
                for species, lowering in zip(
                    self.species, state.ionization_lowering
                )
            ]
        )
        internal_energy_derivatives = np.array(
            [
                species.dinternal_energy_dT(state.T, lowering)
                for species, lowering in zip(
                    self.species, state.ionization_lowering
                )
            ]
        )
        particle_enthalpies = (
            internal_energies + state.reference_energies + state.kbt
        )
        particle_enthalpy_derivatives = (
            internal_energy_derivatives
            + tangent.reference_energy_derivative
            + u.k_b
        )

        minimum = int(np.argmin(state.reference_energies))
        mass_ratios = state.masses / state.masses[minimum]
        relative_enthalpies = (
            particle_enthalpies
            - state.reference_energies[minimum] * mass_ratios
        )
        relative_enthalpy_derivatives = (
            particle_enthalpy_derivatives
            - tangent.reference_energy_derivative[minimum] * mass_ratios
        )

        mole_fractions = state.mole_fractions
        mole_fraction_derivative = tangent.mole_fraction_derivative
        mean_mass = mole_fractions @ state.masses
        mean_mass_derivative = mole_fraction_derivative @ state.masses
        enthalpy_per_particle = mole_fractions @ relative_enthalpies
        enthalpy_derivative = (
            mole_fraction_derivative @ relative_enthalpies
            + mole_fractions @ relative_enthalpy_derivatives
        )
        return (
            enthalpy_derivative * mean_mass
            - enthalpy_per_particle * mean_mass_derivative
        ) / mean_mass**2

    def calculate_viscosity(self) -> float:
        r"""Calculate the LTE viscosity of the plasma in :math:`\text{Pa.s}`.

        Calculate the LTE viscosity of the plasma in Pa.s based on current
        conditions and species composition.

        Returns
        -------
        float
            Viscosity, in :math:`\text{Pa.s}`.
        """
        return functions_transport.viscosity(self)

    def calculate_thermal_conductivity(
        self,
        rel_delta_T: float = 0.001,
        DTterms_yn: bool = True,
        ni_limit: float = 1e8,
    ) -> float:
        r"""Calculate the LTE thermal conductivity of the plasma.

        The thermal conductivity is returned in
        :math:`\text{W.m}^{-1}.\text{K}^{-1}`.

        Parameters
        ----------
        rel_delta_T : float, optional
            Retained for API compatibility; the piecewise analytical
            composition derivative does not use a temperature increment.
        DTterms_yn : bool, optional
            TODO:Flag to include the temperature-dependent terms in the
            calculation, by default True.
        ni_limit : float, optional
            TODO:Number density limit for the calculation of the thermal
            conductivity, by default 1e8.

        Returns
        -------
        float
            Thermal conductivity, in :math:`\text{W.m}^{-1}.\text{K}^{-1}`.
        """
        return functions_transport.thermal_conductivity(
            self, rel_delta_T, DTterms_yn, ni_limit
        )

    def calculate_electrical_conductivity(self) -> float:
        r"""Calculate the LTE electrical conductivity of the plasma.

        The electrical conductivity is returned in :math:`\text{S.m}^{-1}`.

        Returns
        -------
        float
            Electrical conductivity, in :math:`\text{S.m}^{-1}`.
        """
        return self._calculate_electrical_conductivity()

    def _calculate_electrical_conductivity(self) -> float:
        """Template method for electrical conductivity calculation."""
        return functions_transport.electrical_conductivity(self)

    def calculate_total_emission_coefficient(self) -> float:
        r"""Calculate the LTE total emission coefficient of the plasma.

        The total radiation emission coefficient of the plasma is returned in
        :math:`\text{W.m}^{-3}`.

        Returns
        -------
        float
            Total radiation emission coefficient, in :math:`\text{W.m}^{-3}`.
        """
        return functions_radiation.total_emission_coefficient(self)


class LTEWithoutElectrons(LTE):
    """LTE mixture class specifically for cases without electrons."""

    def _setup_species_list(
        self,
        species: list[_sp.Monatomic | _sp.Diatomic | _sp.Polyatomic],
    ) -> tuple:
        """Set up the species list without adding electrons."""
        return tuple(list(species))

    def _validate_x0_length(self, x0: list[float]) -> None:
        """Validate constraint mole fractions length (no electrons case)."""
        if len(x0) == len(self.species):
            return  # Valid: x0 for all species (no electrons)
        else:
            raise ValueError(
                "Please specify constraint mole fractions for all species."
            )

    def _format_x0(self, x0: list[float]) -> tuple[float, ...]:
        """Format x0 for no electrons case."""
        return tuple(list(x0))

    def _format_species_string(self) -> tuple[tuple, tuple]:
        """Format species and x0 strings for display (no electrons case)."""
        species_tuple = tuple([sp.name for sp in self.species])
        x0_tuple = self.x0
        return species_tuple, x0_tuple

    def _get_constraint_dimensions(
        self, elements_count: int
    ) -> tuple[int, int]:
        """Calculate matrix dimensions for solver (no electrons case)."""
        nb_species = len(self.species)
        minimiser_dof = nb_species + elements_count
        constraints_dof = elements_count
        return minimiser_dof, constraints_dof

    def _setup_charge_constraints(
        self, A_matrix: np.ndarray, A_matrix_transpose: np.ndarray
    ) -> None:
        """Set up charge neutrality constraints (no electrons case)."""
        # No charge constraints needed without electrons
        pass

    def _calculate_ionization_lowering(
        self, number_densities: np.ndarray
    ) -> np.ndarray:
        """Calculate ionization energy lowering (no electrons case)."""
        nb_species = len(self.species)
        return np.zeros(nb_species)  # No ionization lowering without electrons

    def _ionization_lowering_derivatives(
        self, particle_numbers: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return zero lowering derivatives for electron-free mixtures."""
        count = len(self.species)
        return np.zeros((count, count)), np.zeros(count)

    def _get_species_for_iteration(
        self, number_densities: np.ndarray
    ) -> tuple[np.ndarray, list]:
        """Get species and densities for iteration (no electrons case)."""
        return number_densities, self.species

    def _calculate_electrical_conductivity(self) -> float:
        """Electrical conductivity is zero without electrons."""
        return 0.0


def lte_from_names(
    names: list[str],
    x0: list[float],
    T: float,
    P: float,
    electrons_yn: bool = True,
) -> LTE:
    r"""Create a LTE mixture from a list of species names.

    The species database, in ./data/species is used to create the species
    objects from the names. The electron species is added automatically, and
    should not be included in the list of species names.

    Parameters
    ----------
    names : list[str]
        Names of the species.
    x0 : list[float]
        Initial value of mole fractions for each species, typically the
        room-temperature composition of the plasma-generating gas.
    T : float
        LTE plasma temperature, in :math:`\text{K}`.
    P : float
        LTE plasma pressure, in :math:`\text{Pa}`.
    electrons_yn: bool
        Whether or not to include electrons in the calculation (default True).

    Returns
    -------
    An LTE object instance.
    """
    if "e" in names:
        raise ValueError(
            "Electrons are added automatically, please don't "
            "include them in your species list."
        )
    species = [_sp.from_name(name) for name in names]

    if electrons_yn:
        return LTE(species, x0, T, P, 1e20, 1e-10, 1000)
    else:
        return LTEWithoutElectrons(species, x0, T, P, 1e20, 1e-10, 1000)
