#!/usr/bin/env python3
#
# Classes for gas/plasma data and LTE calculations with simple species
#
# Q Reynolds 2017

import json
import numpy as np
import collections
import logging
import warnings
import pathlib

DATAPATH = pathlib.Path(__file__).parent / 'data'
SPECIESPATH = DATAPATH / 'species'
MIXTUREPATH = DATAPATH / 'mixtures'

# utility functions ############################################################

def molar_mass_calculator(protons, neutrons, electrons):
    """Estimate the molar mass in kg/mol of a species based on its nuclear and
    electronic structure, if you can't get it anywhere else for some reason.
    """

    return constants.avogadro * (protons * constants.protonmass
                                 + electrons * constants.electronmass
                                 + neutrons * (constants.protonmass
                                               + constants.electronmass))


def parse_values(nist_line):
    """Helper function to tidy up a string of data copied from NIST online
    databases.
    """

    table = str.maketrans('', '', '+x?[]')
    line = "".join(nist_line.split()).translate(table)
    records = line.split("|")[:-1]

    values = []
    for record in records:
        if "/" in record:
            num, den = record.split("/")
            value = float(num) / float(den)
        else:
            value = float(record)
        values.append(value)
    return values


def read_energylevels(data):
    """ Read a NIST energy level file

    Parameters
    ----------
    data : file-like
        NIST energy level file data


    Return
    ------
    energylevels : list of dict
         Energy levels. Each dict contains the energy of the level Ei and the 
         associated quantum number J.
    """
    energylevels = []

    try:
        name = data.name
    except AttributeError:
        name = 'input'

    for i, line in enumerate(data):
        try:
            j, ei = parse_values(line)
            energylevels.append({"J": j, "Ei": ei})
        except ValueError as exception:
            logging.debug("Ignoring line %i in %s", i, name)
            logging.debug(exception)

    return energylevels


def build_monatomic_species_json(name, stoichiometry, molarmass, chargenumber,
                                 ionisationenergy, energylevels, sources=None):
    """Function to take text data retrieved from NIST websites or other sources

    and build a data dictionary for a monatomic plasma species, with specified
    electron energy levels and degeneracies.

    Parameters
    ----------
    name : string
        A unique identifier for the species (also the name of the JSON output
        file)
    stoichiometry : dictionary
        Dictionary describing the elemental stoichiometry of the species (e.g.
        {"O": 1} for O or O+)
    molarmass : float
        Molar mass of the species in kg/mol
    chargenumber : int
        Charge on the species (in integer units of the fundamental charge)
    ionisationenergy : float
        Ionisation energy of the species in 1/cm
    energylevels : list of dict
        Path to text file containing raw energy level data (in NIST Atomic
        Spectra Database format)
    sources : list of strings
        Each entry represents a reference source from which the data was
        obtained (defaults to NIST Atomic Spectra Database)
    """

    if sources is None:
        sources = ["""NIST Atomic Spectra Database (ver. 5.3), [Online]. 
                   A Kramida, Yu Ralchenko, J Reader, and NIST ASD Team, 
                   National Institute of Standards and Technology, Gaithersburg 
                   MD., http://physics.nist.gov/asd"""]

    speciesdict = collections.OrderedDict([
        ("name", name),
        ("stoichiometry", stoichiometry),
        ("molarMass", molarmass),
        ("chargeNumber", chargenumber),
        ("monatomicData", collections.OrderedDict([
            ("ionisationEnergy", ionisationenergy),
            ("energyLevels", energylevels),
        ])),
        ("energyUnit", "1/cm"),
        ("molarMassUnit", "kg/mol"),
        ("sources", sources),
    ])

    return speciesdict


def build_diatomic_species_json(name, stoichiometry, molarmass, chargenumber,
                                ionisationenergy, dissociationenergy, sigma_s,
                                g0, w_e, b_e, sources=None):
    """Function to take text data retrieved from NIST websites or other sources
    and build a data dictionary file for a diatomic plasma species, with specified
    ground state degeneracy and rotational & vibrational parameters.

    Parameters
    ----------
    name : string
        A unique identifier for the species (also the name of the JSON output
        file)

    stoichiometry : dictionary
        Dictionary describing the elemental stoichiometry of the species (e.g.
        {"Si": 1, "O": 1} for SiO or SiO+)
    molarmass : float
        Molar mass of the species in kg/mol
    chargenumber : int
        Charge on the species (in integer units of the fundamental charge)
    ionisationenergy : float
        Ionisation energy of the species in 1/cm
    dissociationenergy : float
        Dissociation energy of the species in 1/cm
    sigma_s : int
        Symmetry constant (=2 for homonuclear molecules, =1 for heteronuclear)
    g0 : float
        Ground state electronic energy level degeneracy
    w_e : float
        Vibrational energy level constant in 1/cm
    b_e : float
        Rotational energy level constant in 1/cm
    sources : list of dictionaries
        Each dictionary represents a reference source from which the data was
        obtained (defaults to NIST Chemistry Webbook)
    """
    if sources is None:
        sources = ["""NIST Chemistry WebBook, NIST Standard Reference Database 
                   Number 69. PJ Linstrom and WG Mallard (Editors), National 
                   Institute of Standards and Technology, Gaithersburg MD., 
                   http://webbook.nist.gov/chemistry/, doi:10.18434/T4D303"""]

    speciesdict = collections.OrderedDict([
        ("name", name),
        ("stoichiometry", stoichiometry),
        ("molarMass", molarmass),
        ("chargeNumber", chargenumber),
        ("diatomicData", collections.OrderedDict([
            ("ionisationEnergy", ionisationenergy),
            ("dissociationEnergy", dissociationenergy),
            ("sigmaS", sigma_s),
            ("g0", g0),
            ("we", w_e),
            ("Be", b_e),
        ])),
        ("energyunit", "1/cm"),
        ("molarmassunit", "kg/mol"),
        ("sources", sources),
    ])

    return speciesdict

# classes ######################################################################


class constants:
    """A collection of physical and unit-conversion constants useful in plasma
    calculations.
    """

    protonmass = 1.6726219e-27
    electronmass = 9.10938356e-31
    fundamentalcharge = 1.60217662e-19
    avogadro = 6.0221409e23
    boltzmann = 1.38064852e-23
    planck = 6.62607004e-34
    c = 2.99792458e8
    electronvolt_to_kelvin = 11604.505
    electronvolt_to_joule = 1.60217653e-19
    invcm_to_joule = 1.9864456e-23
    joulepermol_to_invcm = 0.0835934811
    electronvolt_to_invcm = 8065.54446


def species_from_file(datafile, numberofparticles=0, x0=0):
    """Create a species from a data file.

    Parameters
    ----------
    datafile : string
        Path to a JSON data file describing the electronic and molecular
        properties of the species
    numberofparticles : float
        Initial particle count (default 0)
    x0 : float
        initial x
    """
    # Construct a data object from JSON data file
    with open(datafile) as df:
        jsondata = json.load(df)

    if 'monatomicData' in jsondata:
        return MonatomicSpecies(jsondata, numberofparticles, x0)
    else:
        return DiatomicSpecies(jsondata, numberofparticles, x0)


def species_from_name(name, numberofparticles=0, x0=0):
    """ Create a species from the species database

    Parameters
    ----------
    name : str
        Name of the species
    numberofparticles : float
        Initial particle count (default 0)
    x0 : float
        initial x
    """

    filename = SPECIESPATH / (name + '.json')
    return species_from_file(str(filename), numberofparticles, x0)


class BaseSpecies:
    def partitionfunction_total(self, V, T):
        return (V * self.partitionfunction_translational(T)
                * self.partitionfunction_internal(T))

    def partitionfunction_translational(self, T):
        return ((2 * np.pi * self.molarmass * constants.boltzmann * T)
                / (constants.avogadro * constants.planck ** 2)) ** 1.5

    def partitionfunction_internal(self, T):
        raise NotImplementedError

    def internal_energy(self, T):
        raise NotImplementedError


# Diatomic molecules, single atoms, and ions
class Species(BaseSpecies):
    def __init__(self, jsondata, numberofparticles=0, x0=0):
        """Base class for species. Either single monatomic or diatomic chemical
        species in the plasma, eg O2 or Si+

        Parameters
        ----------
        jsondata : dict
            JSON data describing the electronic and molecular
            properties of the species
        numberofparticles : float
            Initial particle count (default 0)
        x0 : float
        """

        self.numberofparticles = numberofparticles
        self.numberdensity = 0.
        self.x0 = x0

        # General species data
        self.name = jsondata["name"]
        self.stoichiometry = jsondata["stoichiometry"]
        self.molarmass = jsondata["molarMass"]
        self.chargenumber = jsondata["chargeNumber"]

        if self.chargenumber < 0:
            # TODO is this the right exception to raise?
            raise ValueError("Error! Negatively charged ions not implemented yet.")


class MonatomicSpecies(Species):
    def __init__(self, jsondata, numberofparticles=0, x0=0):
        super().__init__(jsondata, numberofparticles, x0)

        self.ionisationenergy = constants.invcm_to_joule * jsondata["monatomicData"]["ionisationEnergy"]
        self.deltaionisationenergy = 0.
        self.energylevels = []
        for energylevel in jsondata["monatomicData"]["energyLevels"]:
            self.energylevels.append([2. * energylevel["J"] + 1.,
                                      constants.invcm_to_joule * energylevel["Ei"]])
        self.e0 = 0

    def partitionfunction_internal(self, temperature):
        partitionval = 0.
        for twojplusone, eij in self.energylevels:
            if eij < (self.ionisationenergy - self.deltaionisationenergy):
                partitionval += twojplusone * np.exp(-eij / (constants.boltzmann * temperature))
        return partitionval

    def internal_energy(self, temperature):
        translational_energy = 1.5 * constants.boltzmann * temperature
        electronic_energy = 0.
        for twojplusone, eij in self.energylevels:
            if eij < (self.ionisationenergy - self.deltaionisationenergy):
                electronic_energy += twojplusone * eij * np.exp(-eij / (constants.boltzmann * temperature))
        electronic_energy /= self.partitionfunction_internal(temperature)
        return translational_energy + electronic_energy


class DiatomicSpecies(Species):
    def __init__(self, jsondata, numberofparticles=0, x0=0):
        super().__init__(jsondata, numberofparticles, x0)

        self.dissociationenergy = constants.invcm_to_joule * jsondata["diatomicData"][
            "dissociationEnergy"]
        self.ionisationenergy = constants.invcm_to_joule * jsondata["diatomicData"][
            "ionisationEnergy"]
        self.deltaionisationenergy = 0.
        self.sigma_s = jsondata["diatomicData"]["sigmaS"]
        self.g0 = jsondata["diatomicData"]["g0"]
        self.w_e = constants.invcm_to_joule * jsondata["diatomicData"]["we"]
        self.b_e = constants.invcm_to_joule * jsondata["diatomicData"]["Be"]
        self.e0 = -self.dissociationenergy

    def partitionfunction_internal(self, temperature):
        electronicpartition = self.g0
        vibrationalpartition = 1. / (1. - np.exp(-self.w_e / (constants.boltzmann * temperature)))
        rotationalpartition = constants.boltzmann * temperature / (self.sigma_s * self.b_e)
        return electronicpartition * vibrationalpartition * rotationalpartition

    def internal_energy(self, temperature):
        translational_energy = 1.5 * constants.boltzmann * temperature
        electronic_energy = 0.
        rotational_energy = constants.boltzmann * temperature
        vibrational_energy = self.w_e * np.exp(-self.w_e / (constants.boltzmann * temperature)) / (1. - np.exp(-self.w_e / (constants.boltzmann * temperature)))
        return translational_energy + electronic_energy + rotational_energy + vibrational_energy


class ElectronSpecies(BaseSpecies):
    def __init__(self, numberofparticles=0):
        """Class describing electrons as a species in the plasma.

        Parameters
        ----------
        numberofparticles : float
            Initial particle count (default 0)
        """

        self.name = "e"
        self.stoichiometry = {}
        self.molarmass = constants.electronmass * constants.avogadro
        self.chargenumber = -1
        self.numberofparticles = numberofparticles
        self.numberdensity = 0.
        self.e0 = 0
        self.x0 = 0

    # noinspection PyUnusedLocal
    def partitionfunction_internal(self, temperature):
        return 2.

    def internal_energy(self, temperature):
        translational_energy = 1.5 * constants.boltzmann * temperature
        electronic_energy = 0.
        return translational_energy + electronic_energy


class Element:
    def __init__(self, name="", stoichiometriccoeffts=None, totalnumber=0.):
        """Class acting as struct to hold some information about different
        elements in the plasma.

        Parameters
        ----------
        name : string
            Name of element, eg "O" (default empty string)
        stoichiometriccoeffts : array_like
            List of number of atoms of this element present in each species, in
            same order as Mixture.species (default empty list)
        totalnumber : float
            Total number of atoms of this element present in the simulation
            (conserved), calculated from initial conditions during instantiation
            of Mixture (default 0)
        """

        self.name = name
        self.stoichiometriccoeffts = [] if stoichiometriccoeffts is None else stoichiometriccoeffts
        self.totalnumber = totalnumber


class Mixture:
    def __init__(self, mixture_file, temperature=10000., pressure=101325):
        """Class representing a thermal plasma specification with multiple
        species, and methods for calculating equilibrium species concentrations
        at different temperatures and pressures using the principle of Gibbs
        free energy minimisation.

        Parameters
        ----------
        mixture_file : string
            Path to a JSON data file containing species and initial mole
            fractions
        temperature : float
            Temperature value in K, for initialisation (default 10000)
        pressure : float
            Pressure value in Pa, for initialisation (default 101325)
        """

        self.temperature = temperature
        self.pressure = pressure

        with open(mixture_file) as sf:
            jsondata = json.load(sf)

        # Random order upsets the nonlinearities in the minimiser resulting in
        # non-reproducibility between runs. Make sure this order is maintained
        self.species = [species_from_name(spdata["species"], x0=spdata["x0"])
                        for spdata in jsondata["speciesList"]]
        self.species.append(ElectronSpecies())

        # Random order upsets the nonlinearities in the minimiser resulting in
        # non-reproducibility between runs
        elementset = sorted(set(s for sp in self.species
                                for s in sp.stoichiometry))
        self.elements = tuple(Element(name=element) for element in elementset)

        self.maxchargenumber = max(sp.chargenumber for sp in self.species)
        # Set species which each +ve charged ion originates from
        for sp in self.species:
            if sp.chargenumber > 0:
                for sp2 in self.species:
                    if sp2.stoichiometry == sp.stoichiometry and sp2.chargenumber == sp.chargenumber - 1:
                        sp.ionisedFrom = sp2

        # Set stoichiometry and charge coefficient arrays for mass action and
        # electroneutrality constraints
        for elm in self.elements:
            elm.stoichiometriccoeffts = [sp.stoichiometry.get(elm.name, 0.)
                                         for sp in self.species]

        self.chargeCoeffts = [sp.chargenumber for sp in self.species]

        # Set element totals for constraints from provided initial conditions
        nT0 = self.pressure / (constants.boltzmann * self.temperature)
        for elm in self.elements:
            elm.totalnumber = sum(nT0 * c * sp.x0
                                  for c, sp in zip(elm.stoichiometriccoeffts,
                                                   self.species))

        # Set up A matrix, b and ni vectors for GFE minimiser
        minimiserDOF = len(self.species) + len(self.elements) + 1
        self.gfeMatrix = np.zeros((minimiserDOF, minimiserDOF))
        self.gfeVector = np.zeros(minimiserDOF)
        self.ni = np.zeros(len(self.species))

        for i, elm in enumerate(self.elements):
            self.gfeVector[len(self.species) + i] = elm.totalnumber
            for j, sC in enumerate(elm.stoichiometriccoeffts):
                self.gfeMatrix[len(self.species) + i, j] = sC
                self.gfeMatrix[j, len(self.species) + i] = sC
        for j, qC in enumerate(self.chargeCoeffts):
            self.gfeMatrix[-1, j] = qC
            self.gfeMatrix[j, -1] = qC

    def initialiseNi(self, ni):
        for j, sp in enumerate(self.species):
            self.ni[j] = ni[j]
            sp.numberofparticles = ni[j]

    def readNi(self):
        for j, sp in enumerate(self.species):
            self.ni[j] = sp.numberofparticles

    def writeNi(self):
        for j, sp in enumerate(self.species):
            sp.numberofparticles = self.ni[j]

    def writeNumberDensity(self):
        V = self.ni.sum() * constants.boltzmann * self.temperature / self.pressure
        for j, sp in enumerate(self.species):
            sp.numberdensity = self.ni[j] / V

    def recalcE0i(self):
        # deltaionisationenergy recalculation, using limitation theory of
        # Stewart & Pyatt 1966
        ni = self.ni
        T = self.temperature
        P = self.pressure

        V = ni.sum() * constants.boltzmann * T / P
        weightedChargeSumSqd = 0
        weightedChargeSum = 0
        for j, sp in enumerate(self.species):
            if sp.chargenumber > 0:
                weightedChargeSum += (ni[j] / V) * sp.chargenumber
                weightedChargeSumSqd += (ni[j] / V) * sp.chargenumber ** 2
        zStar = weightedChargeSumSqd / weightedChargeSum
        debyeD = np.sqrt(constants.boltzmann * T
                         / (4. * np.pi * (zStar + 1.) * (ni[-1] / V)
                            * constants.fundamentalcharge ** 2))
        for j, sp in enumerate(self.species):
            if sp.name != "e":
                ai = (3. * (sp.chargenumber + 1.)
                      / (4. * np.pi * (ni[-1] / V))) ** (1 / 3)
                sp.deltaionisationenergy = (constants.boltzmann * T * 
                                            (((ai / debyeD) ** 3 + 1) ** (2 / 3)
                                             - 1) / (2. * (zStar + 1)))

        for cn in range(1, self.maxchargenumber + 1):
            for sp in self.species:
                if sp.chargenumber == cn:
                    sp.e0 = (sp.ionisedFrom.e0
                             + sp.ionisedFrom.ionisationenergy
                             - sp.ionisedFrom.deltaionisationenergy)

    def recalcGfeArrays(self):
        ni = self.ni
        T = self.temperature
        P = self.pressure

        niSum = ni.sum()
        V = niSum * constants.boltzmann * T / P
        offDiagonal = -constants.boltzmann * T / niSum
        nspecies = len(self.species)

        onDiagonal = constants.boltzmann * T / ni
        self.gfeMatrix[:nspecies, :nspecies] = offDiagonal + np.diag(onDiagonal)
        total = [sp.partitionfunction_total(V, T) for sp in self.species]
        E0 = [sp.e0 for sp in self.species]
        mu = -constants.boltzmann * T * np.log(total / ni) + E0
        self.gfeVector[:nspecies] = -mu


    def solveGfe(self, relativeTolerance=1e-10, maxIters=1000):
        self.readNi()

        governorFactors = np.linspace(0.9, 0.1, 9)
        successYN = False
        governorIters = 0
        while not successYN and governorIters < len(governorFactors):
            successYN = True
            governorFactor = governorFactors[governorIters]
            relTol = relativeTolerance * 10.
            minimiserIters = 0
            while relTol > relativeTolerance:
                self.recalcE0i()
                self.recalcGfeArrays()

                solution = np.linalg.solve(self.gfeMatrix, self.gfeVector)

                new_ni = solution[0:len(self.species)]
                deltaNi = abs(new_ni - self.ni)
                maxAllowedDeltaNi = governorFactor * self.ni

                maxNiIndex = new_ni.argmax()
                relTol = deltaNi[maxNiIndex] / solution[maxNiIndex]

                deltaNi = deltaNi.clip(min=maxAllowedDeltaNi)
                newRelaxFactors = maxAllowedDeltaNi / deltaNi
                relaxFactor = newRelaxFactors.min()

                self.ni = (1. - relaxFactor) * self.ni + relaxFactor * new_ni

                minimiserIters += 1
                if minimiserIters > maxIters:
                    successYN = False
                    break

            governorIters += 1

        if not successYN:
            warnings.warn("Minimiser could not find a converged solution, results may be inaccurate.")

        # noinspection PyUnboundLocalVariable
        logging.debug(governorIters, relaxFactor, relTol)
        logging.debug(self.ni)

        self.writeNi()
        self.writeNumberDensity()

    def calculateDensity(self):
        """Calculate the density of the plasma in kg/m3 based on current
        conditions and species composition.
        """

        return sum(sp.numberdensity * sp.molarmass / constants.avogadro
                   for sp in self.species)

    def calculate_heat_capacity(self, init_ni=1e20, rel_delta_t=0.001):
        """Calculate the heat capacity at constant pressure of the plasma in
        J/kg.K based on current conditions and species composition. Note that
        this is done by performing two full composition simulations when this
        function is called - can be time-consuming.
        """

        T = self.temperature

        self.initialiseNi([init_ni for i in range(len(self.species))])
        self.temperature = (1 - rel_delta_t) * T
        self.solveGfe()
        enthalpy_low = self.calculate_enthalpy()

        self.initialiseNi([init_ni for i in range(len(self.species))])
        self.temperature = (1 + rel_delta_t) * T
        self.solveGfe()
        enthalpy_high = self.calculate_enthalpy()

        self.temperature = T

        return (enthalpy_high - enthalpy_low) / (2. * rel_delta_t * T)

    def calculate_enthalpy(self):
        """Calculate the enthalpy of the plasma in J/kg based on current
        conditions and species composition. Note that the value returned is not
        absolute, it is relative to an arbitrary reference which may be
        negative or positive depending on the reference energies of the diatomic
        species present.
        """

        T = self.temperature
        weighted_enthalpy = sum(constants.avogadro * sp.numberofparticles * (sp.internal_energy(T) + sp.e0 + constants.boltzmann * T)
                                for sp in self.species)
        weighted_molmass = sum(sp.numberofparticles * sp.molarmass
                               for sp in self.species)
        return weighted_enthalpy / weighted_molmass

    def calculateViscosity(self):
        """Calculate the viscosity of the plasma in Pa.s based on current
        conditions and species composition.
        """

        raise NotImplementedError

    def calculateThermalConductivity(self):
        """Calculate the thermal conductivity of the plasma in W/m.K based on
        current conditions and species composition.
        """

        raise NotImplementedError

    def calculateElectricalConductivity(self):
        """Calculate the electrical conductivity of the plasma in 1/ohm.m based
        on current conditions and species composition.
        """

        raise NotImplementedError

    def calculateTotalEmissionCoefficient(self):
        """Calculate the total radiation emission coefficient of the plasma in
        W/m3 based on current conditions and species composition.
        """

        raise NotImplementedError
