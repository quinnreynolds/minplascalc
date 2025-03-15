"""Module for defining species in the plasma."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from minplascalc import units as u
from minplascalc.utils import get_path_to_data

__all__ = [
    "SPECIES_PATH",
    "from_file",
    "from_name",
    "Monatomic",
    "Diatomic",
    "Polyatomic",
    "Electron",
]

DATA_PATH = get_path_to_data()
SPECIES_PATH = DATA_PATH / "species"


class BaseSpecies:
    def __init__(self):
        self.molar_mass: float
        r"""Molar mass of the species in :math:`\text{kg.mol}^{-1}`."""

    def total_partition_function(self, V: float, T: float, dE: float) -> float:
        r"""Calculate the total partition function for the species.

        Parameters
        ----------
        V : float
            Volume, in :math:`\text{m}^3`.
        T : float
            Temperature, in :math:`\text{K}`.
        dE : float
            Ionisation energy lowering, in :math:`\text{J}`.

        Returns
        -------
        float
            The total partition function.

        Notes
        -----
        The total partition function is given by:

        .. math::

            Z = V Z_{tr} Z_{int}

        where :math:`Z_{tr}` is the translational partition function and
        :math:`Z_{int}` is the internal partition function.

        NOTE: Check if this assumes that energy can be decomposed into
        translational and internal components. If not, the method should be
        overridden in the subclass.
        """
        return (
            V
            * self.translational_partition_function(T)
            * self.internal_partition_function(T, dE)
        )

    def translational_partition_function(self, T: float) -> float:
        r"""Calculate the volumic translational partition function.

        Parameters
        ----------
        T : float
            Temperature, in :math:`\text{K}`.

        Returns
        -------
        float
            The volumic translational partition function,
            in :math:`\text{m}^{-3}`.

        Notes
        -----
        The volumic translational partition function is given by:

        .. math::

            Z_{tr} = \left(\frac{2 \pi m k_B T}{h^2}\right)^{1.5}
        """
        return (
            (2 * u.pi * self.molar_mass * u.k_b * T) / (u.N_a * u.h**2)
        ) ** 1.5

    def internal_partition_function(self, T, dE):
        raise NotImplementedError

    def internal_energy(self, T, dE):
        raise NotImplementedError


class Species(BaseSpecies):
    def __init__(
        self,
        name: str,
        stoichiometry: dict[str, int],
        molar_mass: float,
        charge_number: int,
        polarisability: float,
        multiplicity: float,
        effective_electrons: float | None,
        electron_cross_section: float
        | tuple[float, float, float, float]
        | None,
        emission_lines: list[tuple[float, float, float]],
    ):
        r"""Heavy particle base class.

        Monatomic, diatomic, or polyatomic chemical species in the plasma,
        e.g. O2 or Si+. Not for electrons.

        Parameters
        ----------
        name : str
            A unique identifier for the species.
        stoichiometry : dictionary
            Dictionary describing the elemental stoichiometry of the species.
        molar_mass : float
            Molar mass of the species, in :math:`\text{kg.mol}^{-1}`.
        charge_number : int
            Charge on the species (in integer units of the fundamental charge).
        polarisability : float
            Polarisability of the species, in :math:`\text{m}^3`.
        multiplicity : float
            Spin multiplicity (2S + 1) of the ground state.
        effective_electrons : float | None
            Effective number of electrons in valence shell,
            per eq.6 of [Cambi1991]_ (only required for neutral species).
        electron_cross_section : float | tuple[float, ...] | None
            Cross section for elastic electron collisions, in math:`\text{m}^2`
            (only required for neutral species).
            Either a single constant value, or a 4-tuple of empirical
            fitting parameters.
            Could be None if not available.
        emission_lines : list[tuple[float, float, float]]
            Radiation emission line data - each entry in the list contains
            three values giving
            the line's wavelength :math:`\lambda` in :math:`\text{m}`,
            its :math:`g \times A` constant in :math:`\text{s}^{-1}`,
            and its emission strength in :math:`\text{J}`.
        """
        self.name = name
        self.stoichiometry = deepcopy(stoichiometry)
        self.molar_mass = molar_mass
        self.charge_number = charge_number
        self.polarisability = polarisability
        self.multiplicity = multiplicity
        self.effective_electrons = effective_electrons
        self.electron_cross_section = electron_cross_section
        self.emission_lines = emission_lines

        self.ionisation_energy: float
        r"""Ionisation energy of the species in :math:`\text{J}`."""

    def to_file(self, datafile=None):
        """Save a Species object to a file for easy re-use.

        Parameters
        ----------
        datafile : str or Path, optional
            The file to which the output should be saved (full path). The
            default is None in which case the Species' name attribute will be
            used for the file name, and it will be saved to the cwd.
        """
        if datafile:
            with open(datafile, "w") as f:
                json.dump(self.__dict__, f, indent=4)
        else:
            with open(self.name + ".json", "w") as f:
                json.dump(self.__dict__, f, indent=4)


class Monatomic(Species):
    def __init__(
        self,
        name: str,
        stoichiometry: dict[str, int],
        molar_mass: float,
        charge_number: int,
        ionisation_energy: float,
        energylevels: list[tuple[float, float]],
        polarisability: float,
        multiplicity: float,
        effective_electrons: float | None,
        electron_cross_section: float
        | tuple[float, float, float, float]
        | None,
        emission_lines: list[tuple[float, float, float]],
        sources: list[str],
    ):
        r"""Class for monatomic plasma species (single atoms and ions).

        Parameters
        ----------
        name : string
            A unique identifier for the species.
        stoichiometry : dictionary
            Dictionary describing the elemental stoichiometry of the species
            (e.g. {'O': 1} for O or O+).
        molar_mass : float
            Molar mass of the species, in :math:`\text{kg.mol}^{-1}`.
        charge_number : int
            Charge on the species (in integer units of the fundamental charge).
        ionisation_energy : float
            Ionisation energy of the species, in :math:`\text{J}`.
        energylevels : list[tuple[float, float]]
            Atomic energy level data - each entry in the list contains a pair
            of values giving the level's quantum number and its energy
            respectively, with energy in :math:`\text{J}`.
        polarisability : float
            Polarisability of the species, in :math:`\text{m}^3`.
        multiplicity : float
            Spin multiplicity (2S + 1) of the ground state.
        effective_electrons : float | None
            Effective number of electrons in valence shell, per eq.6 of
            [Cambi1991]_ (only required for neutral species).
            Could be None if not available.
        electron_cross_section : float | tuple[float, ...] | None
            Cross section for elastic electron collisions, in math:`\text{m}^2`
            (only required for neutral species).
            Either a single constant value, or a 4-tuple of empirical fitting
            parameters.
            Could be None if not available.
        emission_lines : list[tuple[float, float, float]]
            Radiation emission line data - each entry in the list contains
            three values giving
            the line's wavelength :math:`\lambda` in :math:`\text{m}`,
            its :math:`g \times A` constant in :math:`\text{s}^{-1}`,
            and its emission strength in :math:`\text{J}`.
        sources : list of str
            Each entry represents a reference from which the data was obtained.
        """
        super().__init__(
            name,
            stoichiometry,
            molar_mass,
            charge_number,
            polarisability,
            multiplicity,
            effective_electrons,
            electron_cross_section,
            emission_lines,
        )

        self.ionisation_energy = ionisation_energy
        self.energylevels = deepcopy(energylevels)
        self.sources = deepcopy(sources)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(name={self.name},"
            f"stoichiometry={self.stoichiometry},"
            f"molar_mass={self.molar_mass},charge_number={self.charge_number},"
            f"ionisation_energy={self.ionisation_energy},"
            f"energylevels={self.energylevels},"
            f"polarisability={self.polarisability},"
            f"multiplicity={self.multiplicity},"
            f"effective_electrons={self.effective_electrons},"
            f"electron_cross_section={self.electron_cross_section},"
            f"emission_lines={self.emission_lines},sources={self.sources})"
        )

    def __str__(self):
        if np.isclose(0, self.charge_number):
            sptype = "Monatomic atom"
        else:
            sptype = "Monatomic ion"
        return (
            f"Species: {self.name}\nType: {sptype}\n"
            f"Stoichiometry: {self.stoichiometry}\n"
            f"Molar mass: {self.molar_mass} kg/mol\n"
            f"Charge number: {self.charge_number}\n"
            f"Ionisation energy: {self.ionisation_energy} J\n"
            f"Energy levels: {len(self.energylevels)}\n"
            f"Polarisability: {self.polarisability} m^3\n"
            f"Multiplicity: {self.multiplicity}\n"
            f"Effective valence electrons: {self.effective_electrons}\n"
            f"Electron cross section data: {self.electron_cross_section}\n"
            f"Emission lines: {len(self.emission_lines)}"
        )

    def internal_partition_function(self, T: float, dE: float) -> float:
        r"""Calculate the internal partition function for an atomic species.

        Parameters
        ----------
        T : float
            Temperature, in :math:`\text{K}`.
        dE : float
            Ionisation energy lowering, in :math:`\text{J}`.

        Returns
        -------
        float
            The internal partition function.

        Notes
        -----
        The internal partition function of an atomic species is equal to its
        electronic partition function, which is given by:

        .. math::

            Z_{int} = Z_{el} = \sum_{i} g_i e^{-E_i / k_B T}

        where:

        - :math:`g_i` is the degeneracy of the energy level :math:`i`,
        - :math:`E_i` is the energy of the energy level :math:`i`,
          in :math:`\text{J}`,
        - :math:`k_B` is the Boltzmann constant, in :math:`\text{J.K}^{-1}`,
        - :math:`T` is the temperature, in :math:`\text{K}`.

        The sum is taken over all energy levels of the species, up to the
        ionisation energy lowered by :math:`dE` (:math:`dE` is the amont the
        ionisation energy is lowered by, in :math:`\text{J}`).

        The degeneracy of the electronic energy levels is given by:

        .. math::

            g_i = 2J_i + 1
        """
        beta = 1 / (u.k_b * T)  # Inverse temperature, in J^-1.

        # Calculate the electronic partition function.
        electron_partition_function = 0.0

        for J_i, E_i in self.energylevels:
            if E_i < (self.ionisation_energy - dE):
                # Only include energy levels below the ionisation energy.
                g_i = 2 * J_i + 1  # Degeneracy of the energy level.
                electron_partition_function += g_i * np.exp(-beta * E_i)
            else:
                # Stop summing when the ionisation energy is reached.
                break

        return electron_partition_function

    def internal_energy(self, T: float, dE: float) -> float:
        r"""Calculate the internal energy of an atomic species.

        Parameters
        ----------
        T : float
            Temperature, in :math:`\text{K}`.
        dE : float
            Ionisation energy lowering, in :math:`\text{J}`.

        Returns
        -------
        float
            The internal energy of the species, in :math:`\text{J}`.

        Notes
        -----
        The internal energy of an atomic species is given by:

        .. math::

            U_{int} = U_{el} = \frac{1}{Z_{el}}
                    \sum_{i} g_i E_i e^{-E_i / k_B T}

        where:

        - :math:`g_i` is the degeneracy of the energy level :math:`i`,
        - :math:`E_i` is the energy of the energy level :math:`i`,
          in :math:`\text{J}`,
        - :math:`k_B` is the Boltzmann constant, in :math:`\text{J.K}^{-1}`,
        - :math:`T` is the temperature, in :math:`\text{K}`.

        The sum is taken over all energy levels of the species, up to the
        ionisation energy lowered by :math:`dE` (:math:`dE` is the amont
        the ionisation energy is lowered by, in :math:`\text{J}`).

        The degeneracy of the electronic energy levels is given by:

        .. math::

            g_i = 2J_i + 1

        TODO: Check this --> The internal energy is defined as the sum of the
            translational energy and the electronic energy.
        """
        beta = 1 / (u.k_b * T)  # Inverse temperature, in J^-1.

        # Calculate the translational energy.
        translational_energy = 3 / 2 * u.k_b * T

        # Calculate the electronic energy.
        electronic_energy = 0.0
        for J_i, E_i in self.energylevels:
            if E_i < (self.ionisation_energy - dE):
                # Only include energy levels below the ionisation energy.
                g_i = 2 * J_i + 1  # Degeneracy of the energy level.
                electronic_energy += g_i * E_i * np.exp(-beta * E_i)
            else:
                # Stop summing when the ionisation energy is reached.
                break

        electronic_energy /= self.internal_partition_function(T, dE)
        return translational_energy + electronic_energy


class Diatomic(Species):
    def __init__(
        self,
        name: str,
        stoichiometry: dict[str, int],
        molar_mass: float,
        charge_number: int,
        ionisation_energy: float,
        dissociation_energy: float,
        sigma_s: int,
        g0: float,
        w_e: float,
        b_e: float,
        polarisability: float,
        multiplicity: float,
        effective_electrons: float | None,
        electron_cross_section: float
        | tuple[float, float, float, float]
        | None,
        emission_lines: list[tuple[float, float, float]],
        sources: list[str],
    ):
        r"""Class for diatomic plasma species.

        Diatomic species is defined as bonded pairs of atoms,
        like neutral particles or ions.

        Parameters
        ----------
        name : str
            A unique identifier for the species.
        stoichiometry : dict[str, int]
            Dictionary describing the elemental stoichiometry of the species
            (e.g. {'Si': 1, 'O': 1} for SiO or SiO+).
        molar_mass : float
            Molar mass of the species, in :math:`\text{kg.mol}^{-1}`.
        charge_number : int
            Charge on the species (in integer units of the fundamental charge).
        ionisation_energy : float
            Ionisation energy of the species, in :math:`\text{J}`.
        dissociation_energy : float
            Dissociation energy of the species, in :math:`\text{J}`.
        sigma_s : int
            Symmetry constant (=2 for homonuclear molecules, =1 for
            heteronuclear).
        g0 : float
            Ground state electronic energy level degeneracy.
        w_e : float
            Vibrational energy level constant, in :math:`\text{J}`.
        b_e : float
            Rotational energy level constant, in :math:`\text{J}`.
        polarisability : float
            Polarisability of the species, in :math:`\text{m}^3`.
        multiplicity : float
            Spin multiplicity (2S + 1) of the ground state.
        effective_electrons : float | None
            Effective number of electrons, in valence shell, per eq.6 of
            [Cambi1991]_ (only required for neutral species)
        electron_cross_section : float | tuple[float, ...] | None
            Cross section for elastic electron collisions, in math:`\text{m}^2`
            (only required for neutral species).
            Either a single constant value, or a 4-tuple of empirical fitting
            parameters.
            Could be None if not available.
        emission_lines : list[tuple[float, float, float]]
            Radiation emission line data - each entry in the list contains
            three values giving
            the line's wavelength :math:`\lambda` in :math:`\text{m}`,
            its :math:`g \times A` constant in :math:`\text{s}^{-1}`,
            and its emission strength in :math:`\text{J}`.
        sources : list[str]
            Each dictionary represents a reference source from which the data
            was obtained.
        """
        super().__init__(
            name,
            stoichiometry,
            molar_mass,
            charge_number,
            polarisability,
            multiplicity,
            effective_electrons,
            electron_cross_section,
            emission_lines,
        )

        self.dissociation_energy = dissociation_energy
        self.ionisation_energy = ionisation_energy
        self.sigma_s = sigma_s
        self.g0 = g0
        self.w_e = w_e
        self.b_e = b_e
        self.sources = deepcopy(sources)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(name={self.name},"
            f"stoichiometry={self.stoichiometry},"
            f"molar_mass={self.molar_mass},"
            f"charge_number={self.charge_number},"
            f"dissociation_energy={self.dissociation_energy},"
            f"ionisation_energy={self.ionisation_energy},"
            f"sigma_s={self.sigma_s},g0={self.g0},w_e={self.w_e},"
            f"b_e={self.b_e},"
            f"polarisability={self.polarisability},"
            f"multiplicity={self.multiplicity},"
            f"effective_electrons={self.effective_electrons},"
            f"electron_cross_section={self.electron_cross_section},"
            f"emission_lines={self.emission_lines},sources={self.sources})"
        )

    def __str__(self):
        if np.isclose(0, self.charge_number):
            sptype = "Diatomic molecule"
        else:
            sptype = "Diatomic ion"
        return (
            f"Species: {self.name}\nType: {sptype}\n"
            f"Stoichiometry: {self.stoichiometry}\n"
            f"Molar mass: {self.molar_mass} kg/mol\n"
            f"Charge number: {self.charge_number}\n"
            f"Dissociation energy: {self.dissociation_energy} J\n"
            f"Ionisation energy: {self.ionisation_energy} J\n"
            f"sigma_s: {self.sigma_s}\ng0: {self.g0}\nw_e: {self.w_e} J\n"
            f"B_e: {self.b_e} J\n"
            f"Polarisability: {self.polarisability} m^3\n"
            f"Multiplicity: {self.multiplicity}\n"
            f"Effective valence electrons: {self.effective_electrons}\n"
            f"Electron cross section data: {self.electron_cross_section}\n"
            f"Emission lines: {len(self.emission_lines)}"
        )

    def internal_partition_function(self, T: float, dE: float) -> float:
        r"""Calculate the internal partition function for a diatomic species.

        Parameters
        ----------
        T : float
            Temperature, in :math:`\text{K}`.
        dE : float
            Ionisation energy lowering, in :math:`\text{J}`.

        Returns
        -------
        float
            The internal partition function.

        Notes
        -----
        The internal partition function of a diatomic species is equal to the
        product of its electronic partition function, its vibrational partition
        function, and its rotational partition function, which are given by:

        .. math::

            Z_{int} = Z_{el} Z_{vib} Z_{rot}

        where:

        .. math::

            Z_{el} = \sum_{i} g_i e^{-E_i / k_B T} \approx g_0

        is the electronic partition function, with :math:`g_i` the degeneracy
        of the energy level :math:`i` and :math:`E_i` the energy of the energy
        level :math:`i`, :math:`k_B` the Boltzmann constant, and :math:`T` the
        temperature. The sum is taken over all energy levels of the species, up
        to the ionisation energy lowered by :math:`dE` (:math:`dE` is the amont
        the ionisation energy is lowered by, in :math:`\text{J}`).

        For diatomic species, the current implementation of electronic
        partition function is equal to the ground state electronic energy level
        degeneracy. Since these species are generally present only at low
        temperatures where electronic excitation is limited compared to
        vibrational and rotational states, this approximation is reasonable.

        .. math::

            Z_{vib} = \frac{e^{-w_e / (2 k_B T)}}{1 - e^{-w_e / k_B T}}

        is the vibrational partition function, with :math:`w_e` the vibrational
        energy level constant for vibration mode, in :math:`\text{J}`.

        .. math::

            Z_{rot} = \frac{k_B T}{\sigma_s B_e}

        is the rotational partition function, with :math:`\sigma_s` the
        rotational symmetry constant and :math:`B_e` the rotational energy
        level constant, in :math:`\text{J}`.
        """
        kbt = u.k_b * T

        # Calculate the electronic partition function.
        # The electronic partition function is approximated by the ground state
        # electronic energy level degeneracy.
        electronic_partition_function = self.g0

        # Calculate the vibrational partition function.
        vibrational_partition_function = np.exp(-self.w_e / (2 * kbt)) / (
            1 - np.exp(-self.w_e / kbt)
        )

        # Calculate the rotational partition function.
        rotational_partition_function = kbt / (self.sigma_s * self.b_e)

        # Return the total internal partition function.
        return (
            electronic_partition_function
            * vibrational_partition_function
            * rotational_partition_function
        )

    def internal_energy(self, T: float, dE: float) -> float:
        r"""Calculate the internal energy of a diatomic species.

        Parameters
        ----------
        T : float
            Temperature, in :math:`\text{K}`.
        dE : float
            Ionisation energy lowering, in :math:`\text{J}`.

        Returns
        -------
        float
            The internal energy of the species, in :math:`\text{J}`.

        Notes
        -----
        The internal energy of a diatomic species is equal to the sum of its
        translational energy, electronic energy, rotational energy, and
        vibrational energy, which are given by:

        .. math::

            U_{int} = U_{tr} + U_{el} + U_{rot} + U_{vib}

        where:

        .. math::

            U_{tr} = \frac{3}{2} k_B T

        is the translational energy, with :math:`k_B` the Boltzmann constant
        and :math:`T` the temperature.

        .. math::

            U_{el} = 0

        is the electronic energy (since the electronic partition function is
        approximated by the ground state electronic energy level degeneracy).

        .. math::

            U_{rot} = k_B T

        is the rotational energy.

        .. math::

            U_{vib} = \frac{w_e}{2 \tanh(w_e / (2 k_B T))}

        is the vibrational energy, with :math:`w_e` the vibrational energy
        level constant for vibration mode, in :math:`\text{J}`.


        TODO: Check this --> The internal energy is defined as the sum of the
            translational energy and internal energy?
        """
        kbt = u.k_b * T

        # Calculate the translational energy.
        translational_energy = 3 / 2 * kbt

        # Calculate the electronic energy.
        # The electronic energy is zero since the electronic partition function
        # is approximated by the ground state electronic energy level
        # degeneracy.
        electronic_energy = 0

        # Calculate the rotational energy.
        rotational_energy = kbt

        # Calculate the vibrational energy.
        vibrational_energy = self.w_e / (2 * np.tanh(self.w_e / (2 * kbt)))

        # Return the total internal energy.
        return (
            translational_energy
            + electronic_energy
            + rotational_energy
            + vibrational_energy
        )


class Polyatomic(Species):
    def __init__(
        self,
        name: str,
        stoichiometry: dict[str, int],
        molar_mass: float,
        charge_number: int,
        ionisation_energy: float,
        dissociation_energy: float,
        linear_yn: bool,
        sigma_s: int,
        g0: float,
        wi_e: list[float],
        abc_e: list[float],
        polarisability: float,
        multiplicity: float,
        effective_electrons: float | None,
        electron_cross_section: float
        | tuple[float, float, float, float]
        | None,
        emission_lines: list[tuple[float, float, float]],
        sources: list[str],
    ):
        r"""Class for polyatomic plasma species.

        Polyatomic species is defined as molecules (bonded sets of atoms)
        or ions.

        Parameters
        ----------
        name : str
            A unique identifier for the species.
        stoichiometry : dict[str, int]
            Dictionary describing the elemental stoichiometry of the species
            (e.g. {'H': 2, 'O': 1} for H2O or H2O+).
        molar_mass : float
            Molar mass of the species in :math:`\text{kg.mol}^{-1}`.
        charge_number : int
            Charge on the species (in integer units of the fundamental charge).
        ionisation_energy : float
            Ionisation energy of the species in :math:`\text{J}`.
        dissociation_energy : float
            Dissociation energy of the species in :math:`\text{J}`.
        linear_yn : bool
            For linear molecules, only the B rotation constant is used in
            calculation of the rotational partition function. For non-linear
            molecules, all three are used.
        sigma_s : int
            Rotational symmetry constant.
        g0 : float
            Ground state electronic energy level degeneracy.
        wi_e : list[float]
            Vibrational energy level constants for each vibration mode,
            in :math:`\text{J}`.
        abc_e : list[float]
            A, B, and C rotational energy level constants in :math:`\text{J}`.
        polarisability : float
            Polarisability of the species, in :math:`\text{m}^3`.
        multiplicity : float
            Spin multiplicity (2S + 1) of the ground state.
        effective_electrons : float | None
            Effective number of electrons in valence shell, per eq.6 of
            [Cambi1991]_ (only required for neutral species)
        electron_cross_section : float | tuple[float, ...] | None
            Cross section for elastic electron collisions, in math:`\text{m}^2`
            (only required for neutral species).
            Either a single constant value, or a 4-tuple of empirical fitting
            parameters.
            Could be None if not available.
        emission_lines : list[tuple[float, float, float]]
            Radiation emission line data - each entry in the list contains
            three values giving
            the line's wavelength :math:`\lambda` in :math:`\text{m}`,
            its :math:`g \times A` constant in :math:`\text{s}^{-1}`,
            and its emission strength in :math:`\text{J}`.
        sources : list[str]
            Each dictionary represents a reference source from which the data
            was obtained.
        """
        super().__init__(
            name,
            stoichiometry,
            molar_mass,
            charge_number,
            polarisability,
            multiplicity,
            effective_electrons,
            electron_cross_section,
            emission_lines,
        )

        self.dissociation_energy = dissociation_energy
        self.ionisation_energy = ionisation_energy
        self.linear_yn = linear_yn
        self.sigma_s = sigma_s
        self.g0 = g0
        self.wi_e = wi_e
        self.abc_e = abc_e
        self.sources = deepcopy(sources)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(name={self.name},"
            f"stoichiometry={self.stoichiometry},"
            f"molar_mass={self.molar_mass},"
            f"charge_number={self.charge_number},"
            f"dissociation_energy={self.dissociation_energy},"
            f"ionisation_energy={self.ionisation_energy},"
            f"linear_yn={self.linear_yn},sigma_s={self.sigma_s},"
            f"g0={self.g0},wi_e={self.wi_e},abc_e={self.abc_e},"
            f"polarisability={self.polarisability},"
            f"multiplicity={self.multiplicity},"
            f"effective_electrons={self.effective_electrons},"
            f"electron_cross_section={self.electron_cross_section},"
            f"emission_lines={self.emission_lines},sources={self.sources})"
        )

    def __str__(self):
        if np.isclose(0, self.charge_number):
            sptype = "Polyatomic molecule"
        else:
            sptype = "Polyatomic ion"
        return (
            f"Species: {self.name}\nType: {sptype}\n"
            f"Stoichiometry: {self.stoichiometry}\n"
            f"Molar mass: {self.molar_mass} kg/mol\n"
            f"Charge number: {self.charge_number}\n"
            f"Dissociation energy: {self.dissociation_energy} J\n"
            f"Ionisation energy: {self.ionisation_energy} J\n"
            f"linear_yn: {self.linear_yn}\nsigma_s: {self.sigma_s}\n"
            f"g0: {self.g0}\nwi_e: {self.wi_e} J\nABC_e: {self.abc_e} J\n"
            f"Polarisability: {self.polarisability} m^3\n"
            f"Multiplicity: {self.multiplicity}\n"
            f"Effective valence electrons: {self.effective_electrons}\n"
            f"Electron cross section data: {self.electron_cross_section}\n"
            f"Emission lines: {len(self.emission_lines)}"
        )

    def internal_partition_function(self, T: float, dE: float) -> float:
        r"""Calculate the internal partition function for a polyatomic species.

        Parameters
        ----------
        T : float
            Temperature, in :math:`\text{K}`.
        dE : float
            Ionisation energy lowering, in :math:`\text{J}`.

        Returns
        -------
        float
            The internal partition function.

        Notes
        -----
        The internal partition function of a polyatomic species is equal to the
        product of its electronic partition function, its vibrational partition
        function, and its rotational partition function, which are given by:

        .. math::

            Z_{int} = Z_{el} Z_{vib} Z_{rot}

        where:

        .. math::

            Z_{el} = \sum_{i} g_i e^{-E_i / k_B T} \approx g_0

        is the electronic partition function, with :math:`g_i` the degeneracy
        of the energy level :math:`i` and :math:`E_i` the energy of the energy
        level :math:`i`, :math:`k_B` the Boltzmann constant, and :math:`T` the
        temperature. The sum is taken over all energy levels of the species, up
        to the ionisation energy lowered by :math:`dE` (:math:`dE` is the amont
        the ionisation energy is lowered by, in :math:`\text{J}`).

        For polyatomic species, the current implementation of electronic
        partition function is equal to the ground state electronic energy level
        degeneracy. Since these species are generally present only at low
        temperatures where electronic excitation is limited compared to
        vibrational and rotational states, this approximation is reasonable.

        .. math::

            Z_{vib} = \prod_{i} \frac{e^{-w_{e, i} / (2 k_B T)}}
                                {1 - e^{-w_{e, i} / k_B T}}

        is the vibrational partition function, with :math:`w_{e, i}` the
        vibrational energy level constant for vibration mode :math:`i`,
        in :math:`\text{J}`.

        .. math::

            Z_{rot} = \begin{cases}
                \frac{k_B T}{\sigma_s B_e} & \text{if linear} \\
                \sqrt{\pi} \frac{\sqrt{k_B^3 T^3}}{\sqrt{A_e B_e C_e}} &
                    \text{if non-linear}
            \end{cases}

        is the rotational partition function, with :math:`\sigma_s` the
        rotational symmetry constant, :math:`B_e` the rotational energy level
        constant, and :math:`A_e`, :math:`B_e`, and :math:`C_e` the rotational
        energy level constants, in :math:`\text{J}`.
        """
        kbt = u.k_b * T

        # Calculate the electronic partition function.
        # The electronic partition function is approximated by the ground state
        # electronic energy level degeneracy.
        electronic_partition_function = self.g0

        # Calculate the vibrational partition function.
        vibrationalpartition = np.prod(
            [
                np.exp(-wi / (2 * kbt)) / (1 - np.exp(-wi / kbt))
                for wi in self.wi_e
            ]
        )

        # Calculate the rotational partition function.
        if self.linear_yn:
            rotationalpartition = kbt / (self.sigma_s * self.abc_e[1])
        else:
            ABC = np.prod(self.abc_e)
            rotationalpartition = (
                np.sqrt(np.pi) / self.sigma_s * np.sqrt(kbt**3 / ABC)
            )

        # Return the total internal partition function.
        return (
            electronic_partition_function
            * vibrationalpartition
            * rotationalpartition
        )

    def internal_energy(self, T: float, dE: float) -> float:
        r"""Calculate the internal energy of a polyatomic species.

        Parameters
        ----------
        T : float
            Temperature, in :math:`\text{K}`.
        dE : float
            Ionisation energy lowering, in :math:`\text{J}`.

        Returns
        -------
        float
            The internal energy of the species, in :math:`\text{J}`.

        Notes
        -----
        The internal energy of a polyatomic species is equal to the sum of its
        translational energy, electronic energy, rotational energy, and
        vibrational energy, which are given by:

        .. math::

            U_{int} = U_{tr} + U_{el} + U_{rot} + U_{vib}

        where:

        .. math::

            U_{tr} = \frac{3}{2} k_B T

        is the translational energy, with :math:`k_B` the Boltzmann constant
        and :math:`T` the temperature (in :math:`\text{K}`).

        .. math::

            U_{el} = 0

        is the electronic energy (since the electronic partition function is
        approximated by the ground state electronic energy level degeneracy).

        .. math::

            U_{rot} = \begin{cases}
                k_B T & \text{if linear} \\
                \frac{3}{2} k_B T & \text{if non-linear}
            \end{cases}

        is the rotational energy.

        .. math::

            U_{vib} = \sum_{i} \frac{w_{e, i}}{2 \tanh(w_{e, i} / (2 k_B T))}

        is the vibrational energy, with :math:`w_{e, i}` the vibrational energy
        level constant for vibration mode :math:`i`.


        TODO: Check this --> The internal energy is defined as the sum of the
            translational energy and internal energy?
        """
        kbt = u.k_b * T

        # Calculate the translational energy.
        translational_energy = 1.5 * kbt

        # Calculate the electronic energy.
        # The electronic energy is zero since the electronic partition function
        # is approximated by the ground state electronic energy level
        # degeneracy.
        electronic_energy = 0

        # Calculate the rotational energy.
        if self.linear_yn:
            rotational_energy = kbt
        else:
            rotational_energy = 1.5 * kbt

        # Calculate the vibrational energy.
        vibrational_energy = np.sum(
            [wi / (2 * np.tanh(wi / (2 * kbt))) for wi in self.wi_e]
        )

        # Return the total internal energy.
        return (
            translational_energy
            + electronic_energy
            + rotational_energy
            + vibrational_energy
        )


class Electron(BaseSpecies):
    def __init__(self):
        """Class for electrons as a plasma species."""
        self.name = "e"
        self.stoichiometry = {}
        self.molar_mass = u.m_e * u.N_a
        self.charge_number = -1

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(name={self.name},"
            f"molar_mass={self.molar_mass},charge_number={self.charge_number})"
        )

    def __str__(self):
        return (
            f"Species: e\nType: Electron\n"
            f"Molar mass: {self.molar_mass} kg/mol\n"
            f"Charge number: {self.charge_number}"
        )

    # noinspection PyUnusedLocal
    def internal_partition_function(self, T: float, dE: float) -> float:
        r"""Calculate the internal partition function for an electron.

        Parameters
        ----------
        T : float
            Temperature, in :math:`\text{K}`.
        dE : float
            Ionisation energy lowering, in :math:`\text{J}`.

        Returns
        -------
        float
            The internal partition function.

        Notes
        -----
        The internal partition function of an electron is equal to its
        electronic partition function, which is equal to the degeneracy of the
        electron, which is 2.

        .. math::

            Z_{int} = Z_{el} = g_{el} = 2
        """
        return 2.0

    def internal_energy(self, T: float, dE: float) -> float:
        r"""Calculate the internal energy of an electron.

        Parameters
        ----------
        T : float
            Temperature, in :math:`\text{K}`.
        dE : float
            Ionisation energy lowering, in :math:`\text{J}`.

        Returns
        -------
        float
            The internal energy of the species, in :math:`\text{J}`.

        Notes
        -----
        The internal energy of an electron is equal to the sum of its
        translational energy and electronic energy, which are given by:

        .. math::

            U_{int} = U_{tr} + U_{el}

        where:

        .. math::

            U_{tr} = \frac{3}{2} k_B T

        is the translational energy, with :math:`k_B` the Boltzmann constant
        and :math:`T` the temperature.

        .. math::

            U_{el} = 0

        is the electronic energy (since the electronic partition function is
        equal to the degeneracy of the electron, which is 2).
        """
        # Calculate the translational energy.
        translational_energy = 1.5 * u.k_b * T
        # Calculate the electronic energy.
        electronic_energy = 0.0
        # Return the total internal energy.
        return translational_energy + electronic_energy


def from_file(datafile: str | Path) -> Monatomic | Diatomic | Polyatomic:
    """Create a species from a data file in JSON format.

    Parameters
    ----------
    datafile : str | Path
        Path to a JSON data file describing the electronic and molecular
        properties of the species.

    Returns
    -------
    species : Monatomic or Diatomic or Polyatomic
        A species object with the properties described in the data file.
    """
    # Load the data from the file.
    with open(datafile) as f:
        species_data = json.load(f)

    # Determine the number of atoms in the species.
    number_atoms = sum(species_data["stoichiometry"].values())

    # Create the appropriate species object.
    if number_atoms == 1:
        return Monatomic(
            species_data["name"],
            species_data["stoichiometry"],
            species_data["molar_mass"],
            species_data["charge_number"],
            species_data["ionisation_energy"],
            species_data["energylevels"],
            species_data["polarisability"],
            species_data["multiplicity"],
            species_data["effective_electrons"],
            species_data["electron_cross_section"],
            species_data["emission_lines"],
            species_data["sources"],
        )
    elif number_atoms == 2:
        return Diatomic(
            species_data["name"],
            species_data["stoichiometry"],
            species_data["molar_mass"],
            species_data["charge_number"],
            species_data["ionisation_energy"],
            species_data["dissociation_energy"],
            species_data["sigma_s"],
            species_data["g0"],
            species_data["w_e"],
            species_data["b_e"],
            species_data["polarisability"],
            species_data["multiplicity"],
            species_data["effective_electrons"],
            species_data["electron_cross_section"],
            species_data["emission_lines"],
            species_data["sources"],
        )
    else:
        return Polyatomic(
            species_data["name"],
            species_data["stoichiometry"],
            species_data["molar_mass"],
            species_data["charge_number"],
            species_data["ionisation_energy"],
            species_data["dissociation_energy"],
            species_data["linear_yn"],
            species_data["sigma_s"],
            species_data["g0"],
            species_data["wi_e"],
            species_data["abc_e"],
            species_data["polarisability"],
            species_data["multiplicity"],
            species_data["effective_electrons"],
            species_data["electron_cross_section"],
            species_data["emission_lines"],
            species_data["sources"],
        )


def from_name(name: str) -> Monatomic | Diatomic | Polyatomic:
    """Create a species from the species database.

    Parameters
    ----------
    name : str
        Name of the species.

    Returns
    -------
    species : Monatomic or Diatomic or Polyatomic
        A species object with the properties described in the data file.
    """
    filename = SPECIES_PATH / (name + ".json")

    # Check if the file exists.
    if not filename.exists():
        raise FileNotFoundError(f"Species data file {filename} not found.")

    return from_file(str(filename))
