"""Module for handling LTE plasma mixtures.

This includes species composition and thermodynamic properties.
"""

import logging
import warnings

import numpy as np

from minplascalc import functions_radiation, functions_transport
from minplascalc import species as _sp
from minplascalc import units as u

__all__ = ["lte_from_names", "LTE"]


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
                "Electrons are added automatically, please don't "
                "include them in your species list."
            )
        # Check equal length of species and constraint mole fractions lists.
        if len(species) != len(x0):
            raise ValueError("Lists species and x0 must be the same length.")

        # Add electron species to the species list.
        self.__species = tuple(list(species) + [_sp.Electron()])

        self.x0 = x0
        self.T = T
        self.P = P
        self.gfe_initial_particles = gfe_initial_particles
        self.gfe_rtol = gfe_rtol
        self.gfe_max_iter = gfe_max_iter

        self.__isLTE = (
            False  # Flag to indicate if LTE composition has been calculated.
        )

        self.__Ni: np.ndarray = np.zeros(
            len(self.species)
        )  # Number of particles of each species.

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
    def x0(self):
        return self.__x0

    @x0.setter
    def x0(self, x0):
        if len(x0) == len(self.species) - 1:
            # Reset LTE composition flag.
            self.__isLTE = False
            # Add electron mole fraction, set to zero.
            self.__x0 = tuple(list(x0) + [0.0])
        else:
            raise ValueError(
                "Please specify constraint mole fractions for all "
                "species except electrons."
            )

    @property
    def T(self):
        return self.__T

    @T.setter
    def T(self, T):
        self.__isLTE = False  # Reset LTE composition flag.
        self.__T = T

    @property
    def P(self):
        return self.__P

    @P.setter
    def P(self, P):
        self.__isLTE = False  # Reset LTE composition flag.
        self.__P = P

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(species={self.species},"
            f"x0={self.x0},T={self.T},P={self.P},gfe_initial_particles={self.gfe_initial_particles},"
            f"gfe_rtol={self.gfe_rtol},gfe_max_iter={self.gfe_max_iter})"
        )

    def __str__(self):
        return (
            f"LTE mixture species: "
            f"{tuple([sp.name for sp in self.species[:-1]])}\n"
            f"Initial composition: {self.x0[:-1]}\n"
            f"Temperature: {self.T} K\nPressure: {self.P} Pa"
        )

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
        number_densities = (
            N_i / V
        )  # Number density of each species, in particles/m3.

        # Initialise arrays for reference energy and ionisation energy
        # lowering.
        E0 = np.zeros(nb_species)
        dE = np.zeros(nb_species)

        # For (uncharged) polyatomic species, the reference energy is the
        # negative of the dissociation energy.
        for i, sp in enumerate(self.species):
            if sum(sp.stoichiometry.values()) >= 2:
                E0[i] = -sp.dissociation_energy

        # Calculate the effective charge number z*.
        # The effective charge number is the sum of the square of the charge
        # number of each species multiplied by the number density of that
        # species.
        charge_numbers = np.array([sp.charge_number for sp in self.species])
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

        # Calculate the ionisation energy lowering for each (positively)
        # charged species.
        for i, charge_number in enumerate(charge_numbers):
            if charge_number > 0:
                # Electron are discarded, but so are every negatively charged
                # species.
                # TODO: Check if negative ions should be considered.

                # Calculate the ion-sphere radius, to the power 3.
                ai_pow3 = 3 * charge_number / (4 * np.pi * n_e)
                # Calculate the ionisation energy lowering.
                dE[i] = (
                    kbt
                    * ((ai_pow3 / debye_pow3 + 1) ** (2 / 3) - 1)
                    / (2 * (z_star + 1))
                )

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
        nb_species = len(self.species)  # It includes the electron species.
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
        minimiser_dof = nb_species + len(elements) + 1
        gfe_matrix = np.zeros((minimiser_dof, minimiser_dof))
        gfe_vector = np.zeros(minimiser_dof)
        # The first nb_species rows and columns are for the species.
        # The next len(self._elements) rows and columns are for the elements.
        # The last row and column are for the charge neutrality.
        A_matrix_constraints = np.zeros((len(self.species), len(elements) + 1))
        A_matrix_constraints_transpose = np.zeros(
            (len(elements) + 1, len(self.species))
        )
        b_vector_constraints = np.zeros(len(elements) + 1)

        for i, element in enumerate(elements):
            stoichiometric_coefficients = element["stoich_coeff"]
            assert isinstance(stoichiometric_coefficients, list)
            for j, sc in enumerate(stoichiometric_coefficients):
                A_matrix_constraints[j, i] = sc
                A_matrix_constraints_transpose[i, j] = sc
            b_vector_constraints[i] = element["N_tot"]

        for j, qc in enumerate(sp.charge_number for sp in self.species):
            A_matrix_constraints[j, -1] = qc
            A_matrix_constraints_transpose[-1, j] = qc

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
            relative_tolerance = (
                self.gfe_rtol * 10
            )  # Initial relative tolerance.
            minimiser_iters = 0  # Iteration counter for the minimiser.

            while relative_tolerance > self.gfe_rtol:
                # Calculate reference energy and ionisation energy lowering.
                self.__E0, self.__dE = self.__get_reference_energies()
                N_tot = (
                    self.__Ni.sum()
                )  # Total number of particles in the plasma.
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

        N_i = self.__Ni  # Number of particles of each species.
        N_tot = N_i.sum()  # Total number of particles in the plasma.
        V = N_tot * kbt / self.P  # Volume of the plasma, in m3.
        return N_i / V  # Number density of each species, in particles/m3.

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
        number_densities = self.calculate_composition()  # particules/m^3
        molar_masses = [sp.molar_mass for sp in self.species]  # kg/mol
        return (
            sum(n_i * M_i for n_i, M_i in zip(number_densities, molar_masses))
            / u.N_a
        )  # kg/m3 = (particules/m^3 * kg/mol) / (particules/mol)

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
        internal_energies = [
            sp.internal_energy(self.T, dE)
            for sp, dE in zip(self.species, self.__dE)
        ]  # J/particle

        enthalpies = [
            (u_i + E0_i + u.k_b * self.T)
            for u_i, E0_i in zip(internal_energies, self.__E0)
        ]  # J/particle

        masses = [
            sp.molar_mass / u.N_a for sp in self.species
        ]  # (kg/mol) / (particle/mol) = kg/particle

        return np.array(enthalpies) / np.array(masses)  # J/kg

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
        number_densities = self.calculate_composition()  # m^-3
        molar_masses = [sp.molar_mass for sp in self.species]  # kg/mol

        density = self.calculate_density()  # kg/m3

        mass_enthalpies = self.calculate_species_enthalpies()  # J/kg
        masses = np.array(
            [sp.molar_mass / u.N_a for sp in self.species]
        )  # kg/particle
        enthalpies = mass_enthalpies * masses  # J/particle

        # Get the species with the lowest reference energy.
        i_min = np.argmin(
            self.__E0
        )  # Index of the species with the lowest reference energy.
        h_mol_0 = (
            self.__E0[i_min] / self.species[i_min].molar_mass
        )  # J/(kg/mol)

        weighted_enthalpy = sum(
            n_i * (h_i - h_mol_0 * M_i)
            for n_i, h_i, M_i in zip(
                number_densities, enthalpies, molar_masses
            )
        )

        return weighted_enthalpy / density

    def calculate_heat_capacity(self, rel_delta_T: float = 0.001) -> float:
        r"""Calculate the LTE heat capacity at constant pressure of the plasma.

        Calculate the LTE heat capacity at constant pressure of the plasma
        based on current conditions and species composition.

        This is done by performing multiple LTE composition recalculations and
        can be time-consuming to execute - when performing large quantities of
        Cp calculations at different temperatures, it is more efficient to
        calculate enthalpies and perform a numerical derivative external to
        minplascalc.

        Parameters
        ----------
        rel_delta_T : float, optional
            Relative change in temperature to calculate the numerical
            derivative, by default 0.001.

        Returns
        -------
        float
            Heat capacity, in :math:`\text{J.kg}^{-1}.\text{K}^{-1}`.

        Notes
        -----
        The heat capacity at constant pressure of the plasma is calculated as:

        .. math::

            C_p = \frac{dH}{dT}
                \approx \frac{H(T + \Delta T) - H(T - \Delta T)}{2 \Delta T}

        where:

        * :math:`C_p` is the heat capacity at constant pressure,
          in :math:`\text{J.kg}^{-1}.\text{K}^{-1}`,
        * :math:`H` is the enthalpy of the plasma, in :math:`\text{J.kg}^{-1}`,
        * :math:`T` is the temperature, in :math:`\text{K}`, and
        * :math:`\Delta T` is the relative change in temperature.
        """
        start_temperature = self.T
        self.T = start_temperature * (1 - rel_delta_T)
        enthalpy_low = self.calculate_enthalpy()
        self.T = start_temperature * (1 + rel_delta_T)
        enthalpy_high = self.calculate_enthalpy()
        self.T = start_temperature
        return (enthalpy_high - enthalpy_low) / (2 * rel_delta_T * self.T)

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
            TODO:Relative change in temperature to calculate the numerical
            derivative, by default 0.001.
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


def lte_from_names(
    names: list[str], x0: list[float], T: float, P: float
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
    return LTE(species, x0, T, P, 1e20, 1e-10, 1000)
