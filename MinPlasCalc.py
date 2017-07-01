#!/usr/bin/env python3
#
# Classes for gas/plasma data and LTE calculations with simple species
#
# Q Reynolds 2017

import json
import numpy as np
import collections


# utility functions ############################################################


def molarMassCalculator(nProtons, nNeutrons, nElectrons):
    """Estimate the molar mass in kg/mol of a species based on its nuclear and
    electronic structure, if you can't get it anywhere else for some reason.
    """

    return constants.avogadro * (nProtons * constants.protonMass
                                 + nElectrons * constants.electronMass
                                 + nNeutrons * (constants.protonMass
                                                + constants.electronMass))


def nistCleanAndSplit(nistString):
    """Helper function to tidy up a string of data copied from NIST online 
    databases.
    """

    table = str.maketrans('', '', '+x?[]')
    dataStr = "".join(nistString.split()).translate(table)
    dataSplit = dataStr.split("|")[:-1]

    returnData = []

    for dataVal in dataSplit:
        if "/" in dataVal:
            num, den = dataVal.split("/")
            dataVal = float(num) / float(den)
        returnData.append(float(dataVal))

    return returnData


def buildMonatomicSpeciesJSON(name, stoichiometry, molarMass, chargeNumber,
                              ionisationEnergy, nistDataFile, sources=None):
    """Function to take text data retrieved from NIST websites or other sources 
    and build a JSON object file for a monatomic plasma species, with specified
    electron energy levels and degeneracies.
    
    Parameters
    ----------
    name : string
        A unique identifier for the species (also the name of the JSON output
        file)
    stoichiometry : dictionary
        Dictionary describing the elemental stoichiometry of the species (e.g.
        {"O": 1} for O or O+)
    molarMass : float
        Molar mass of the species in kg/mol
    chargeNumber : int
        Charge on the species (in integer units of the fundamental charge)
    ionisationEnergy : float
        Ionisation energy of the species in 1/cm
    nistDataFile : string
        Path to text file containing raw energy level data (in NIST Atomic 
        Spectra Database format)
    sources : list of dictionaries
        Each dictionary represents a reference source from which the data was 
        obtained (defaults to NIST Atomic Spectra Database)
    """

    energylevels = []
    speciesDict = collections.OrderedDict([
        ("name", name),
        ("stoichiometry", stoichiometry),
        ("molarMass", molarMass),
        ("chargeNumber", chargeNumber),
        ("monatomicData", collections.OrderedDict([
            ("ionisationEnergy", ionisationEnergy),
            ("energyLevels", energylevels),
        ])),
        ("energyUnit", "1/cm"),
        ("molarMassUnit", "kg/mol"),
        ("sources", [] if sources is None else sources),
    ])

    with open(nistDataFile) as data:
        for i, dataLine in enumerate(data):
            try:
                J, Ei = nistCleanAndSplit(dataLine)
                energylevels.append({"J": J, "Ei": Ei})
            except ValueError as exception:
                print("Ignoring line", i, "in", nistDataFile)
                print(exception)

    if len(speciesDict["sources"]) == 0:
        speciesDict["sources"].append(collections.OrderedDict([
            ("title", "NIST Atomic Spectra Database (ver. 5.3), [Online]."),
            ("author", "A Kramida, Yu Ralchenko, J Reader, and NIST ASD Team"),
            ("publicationInfo", "National Institute of Standards and Technology, Gaithersburg, MD."),
            ("http", "http://physics.nist.gov/asd"),
            ("doi", "NA"),
        ]))

    with open(speciesDict["name"] + ".json", "w") as jf:
        json.dump(speciesDict, jf, indent=4)


def buildDiatomicSpeciesJSON(name, stoichiometry, molarMass, chargeNumber,
                             ionisationEnergy, dissociationEnergy, sigmaS,
                             g0, we, Be, sources=None):
    """Function to take text data retrieved from NIST websites or other sources 
    and build a JSON object file for a diatomic plasma species, with specified
    ground state degeneracy and rotational & vibrational parameters.
    
    Parameters
    ----------
    name : string
        A unique identifier for the species (also the name of the JSON output
        file)
    stoichiometry : dictionary
        Dictionary describing the elemental stoichiometry of the species (e.g.
        {"Si": 1, "O": 1} for SiO or SiO+)
    molarMass : float
        Molar mass of the species in kg/mol
    chargeNumber : int
        Charge on the species (in integer units of the fundamental charge)
    ionisationEnergy : float
        Ionisation energy of the species in 1/cm
    dissociationEnergy : float
        Dissociation energy of the species in 1/cm
    sigmaS : int
        Symmetry constant (=2 for homonuclear molecules, =1 for heteronuclear)
    g0 : float
        Ground state electronic energy level degeneracy
    we : float
        Vibrational energy level constant in 1/cm
    Be : float
        Rotational energy level constant in 1/cm
    sources : list of dictionaries
        Each dictionary represents a reference source from which the data was 
        obtained (defaults to NIST Chemistry Webbook)
    """

    speciesDict = collections.OrderedDict([
        ("name", name),
        ("stoichiometry", stoichiometry),
        ("molarMass", molarMass),
        ("chargeNumber", chargeNumber),
        ("diatomicData", collections.OrderedDict([
            ("ionisationEnergy", ionisationEnergy),
            ("dissociationEnergy", dissociationEnergy),
            ("sigmaS", sigmaS),
            ("g0", g0),
            ("we", we),
            ("Be", Be),
        ])),
        ("energyUnit", "1/cm"),
        ("molarMassUnit", "kg/mol"),
        ("sources", [] if sources is None else sources),
    ])

    if len(speciesDict["sources"]) == 0:
        speciesDict["sources"].append(collections.OrderedDict([
            ("title", "NIST Chemistry WebBook, NIST Standard Reference Database Number 69"),
            ("author", "PJ Linstrom and WG Mallard (Editors)"),
            ("publicationInfo", "National Institute of Standards and Technology, Gaithersburg MD., 20899"),
            ("http", "http://webbook.nist.gov/chemistry/"),
            ("doi", "10.18434/T4D303"),
        ]))

    with open(speciesDict["name"] + ".json", "w") as jf:
        json.dump(speciesDict, jf, indent=4)


# classes ######################################################################


class constants:
    """A collection of physical and unit-conversion constants useful in plasma 
    calculations.
    """

    protonMass = 1.6726219e-27
    electronMass = 9.10938356e-31
    fundamentalCharge = 1.60217662e-19
    avogadro = 6.0221409e23
    boltzmann = 1.38064852e-23
    planck = 6.62607004e-34
    c = 2.99792458e8
    eVtoK = 11604.505
    eVtoJ = 1.60217653e-19
    invCmToJ = 1.9864456e-23
    JperMolToInvCm = 0.0835934811
    eVtoInvCm = 8065.54446


# Diatomic molecules, single atoms, and ions
class Species:
    def __init__(self, dataFile, numberOfParticles=0):
        """Class describing a single monatomic or diatomic chemical species in the plasma, eg O2 or Si+
        
        Parameters
        ----------
        dataFile : string
            Path to a JSON data file describing the electronic and molecular properties of the species
        numberOfParticles : float
            Initial particle count (default 0)
        """

        self.numberOfParticles = numberOfParticles
        self.numberDensity = 0.

        # Construct a data object from JSON data file
        with open(dataFile) as df:
            jsonData = json.load(df)

        # General species data
        self.name = jsonData["name"]
        self.stoichiometry = jsonData["stoichiometry"]
        self.molarMass = jsonData["molarMass"]
        self.chargeNumber = jsonData["chargeNumber"]

        if "monatomicData" in jsonData:
            # Monatomic-specific species data
            self.monatomicYN = True
            self.ionisationEnergy = constants.invCmToJ * jsonData["monatomicData"]["ionisationEnergy"]
            self.deltaIonisationEnergy = 0.
            self.energyLevels = []
            for energyLevelLine in jsonData["monatomicData"]["energyLevels"]:
                self.energyLevels.append([2. * energyLevelLine["J"] + 1., constants.invCmToJ * energyLevelLine["Ei"]])
        else:
            # Diatomic-specific species data
            self.monatomicYN = False
            self.dissociationEnergy = constants.invCmToJ * jsonData["diatomicData"]["dissociationEnergy"]
            self.ionisationEnergy = constants.invCmToJ * jsonData["diatomicData"]["ionisationEnergy"]
            self.deltaIonisationEnergy = 0.
            self.sigmaS = jsonData["diatomicData"]["sigmaS"]
            self.g0 = jsonData["diatomicData"]["g0"]
            self.we = constants.invCmToJ * jsonData["diatomicData"]["we"]
            self.Be = constants.invCmToJ * jsonData["diatomicData"]["Be"]

        if self.chargeNumber < 0:
            # TODO is this the right exception to raise?
            raise ValueError("Error! Negatively charged ions not implemented yet.")

    def translationalPartitionFunction(self, T):
        return ((2 * np.pi * self.molarMass * constants.boltzmann * T)
                / (constants.avogadro * constants.planck ** 2)) ** 1.5

    def internalPartitionFunction(self, T):
        if self.monatomicYN:
            partitionVal = 0.
            for eLevel in self.energyLevels:
                if eLevel[1] < (self.ionisationEnergy - self.deltaIonisationEnergy):
                    partitionVal += eLevel[0] * np.exp(-eLevel[1] / (constants.boltzmann * T))
            return partitionVal
        else:
            electronicPartition = self.g0
            vibrationalPartition = 1. / (1. - np.exp(-self.we / (constants.boltzmann * T)))
            rotationalPartition = constants.boltzmann * T / (self.sigmaS * self.Be)
            return electronicPartition * vibrationalPartition * rotationalPartition


class ElectronSpecies:
    def __init__(self, numberOfParticles=0):
        """Class describing electrons as a species in the plasma.
        
        Parameters
        ----------
        numberOfParticles : float
            Initial particle count (default 0)
        """

        self.name = "e"
        self.stoichiometry = {}
        self.molarMass = constants.electronMass * constants.avogadro
        self.chargeNumber = -1
        self.numberOfParticles = numberOfParticles
        self.numberDensity = 0.

    def translationalPartitionFunction(self, T):
        return ((2 * np.pi * self.molarMass * constants.boltzmann * T)
                / (constants.avogadro * constants.planck ** 2)) ** 1.5

    def internalPartitionFunction(self, T):
        return 2.


class Element:
    def __init__(self, name="", stoichiometricCoeffts=None, totalNumber=0.):
        """Class acting as struct to hold some information about different 
        elements in the plasma.
        
        Parameters
        ----------
        name : string
            Name of element, eg "O" (default empty string)
        stoichiometricCoeffts : array_like
            List of number of atoms of this element present in each species, in
            same order as compositionGFE.species (default empty list)
        totalNumber : float
            Total number of atoms of this element present in the simulation 
            (conserved), calculated from initial conditions during instantiation
            of compositionGFE (default 0)
        """

        self.name = name
        self.stoichiometricCoeffts = [] if stoichiometricCoeffts is None else stoichiometricCoeffts
        self.totalNumber = totalNumber


class compositionGFE:
    def __init__(self, compositionFile, T=10000., P=101325):
        """Class representing a thermal plasma specification with multiple 
        species, and methods for calculating equilibrium species concentrations 
        at different temperatures and pressures using the principle of Gibbs 
        free energy minimisation.
        
        Parameters
        ----------
        compositionFile : string
            Path to a JSON data file containing species and initial mole
            fractions
        T : float
            Temperature value in K, for initialisation (default 10000)
        P : float
            Pressure value in Pa, for initialisation (default 101325)
        """

        self.T = T
        self.P = P

        with open(compositionFile) as sf:
            jsonData = json.load(sf)

        # Random order upsets the nonlinearities in the minimiser resulting in
        # non-reproducibility between runs - hence OrderedDict
        self.species = collections.OrderedDict()
        for spData in jsonData["speciesList"]:
            sp = Species(dataFile=spData["species"])
            self.species[sp.name] = sp
            self.species[sp.name].x0 = spData["x0"]
        self.species["e"] = ElectronSpecies()
        self.species["e"].x0 = 0.

        # Random order upsets the nonlinearities in the minimiser resulting in
        # non-reproducibility between runs - hence OrderedDict
        self.elements = collections.OrderedDict()
        elmList = []
        for spKey, sp in self.species.items():
            for stKey in sp.stoichiometry:
                elmList.append(stKey)
        elmList = list(set(elmList))
        for elmKey in elmList:
            self.elements[elmKey] = Element(name=elmKey)

        # Set species which each +ve charged ion originates from
        self.maxChargeNumber = 0
        for spKey, sp in self.species.items():
            if sp.chargeNumber > self.maxChargeNumber:
                self.maxChargeNumber = sp.chargeNumber
            if sp.chargeNumber > 0:
                for spKey2, sp2 in self.species.items():
                    if sp2.stoichiometry == sp.stoichiometry and sp2.chargeNumber == sp.chargeNumber - 1:
                        sp.ionisedFrom = spKey2

        # Set species reference energies
        for spKey, sp in self.species.items():
            if sp.chargeNumber == 0:
                if sp.monatomicYN:
                    sp.E0 = 0.
                else:
                    sp.E0 = -sp.dissociationEnergy
        self.species["e"].E0 = 0.

        # Set stoichiometry and charge coefficient arrays for mass action and 
        # electroneutrality constraints
        for elmKey, elm in self.elements.items():
            elm.stoichiometricCoeffts = []
            for spKey, sp in self.species.items():
                elm.stoichiometricCoeffts.append(
                    sp.stoichiometry.get(elmKey, 0.))

        self.chargeCoeffts = []
        for spKey, sp in self.species.items():
            self.chargeCoeffts.append(sp.chargeNumber)

        # Set element totals for constraints from provided initial conditions
        nT0 = self.P / (constants.boltzmann * self.T)
        for elmKey, elm in self.elements.items():
            elm.totalNumber = 0
            for j, spKey in enumerate(self.species):
                elm.totalNumber += nT0 * elm.stoichiometricCoeffts[j] * self.species[spKey].x0

        # Set up A matrix, b and ni vectors for GFE minimiser
        minimiserDOF = len(self.species) + len(self.elements) + 1
        self.gfeMatrix = np.zeros((minimiserDOF, minimiserDOF))
        self.gfeVector = np.zeros(minimiserDOF)
        self.ni = np.zeros(len(self.species))

        for i, elm in enumerate(self.elements):
            self.gfeVector[len(self.species) + i] = self.elements[elm].totalNumber
            for j, sC in enumerate(self.elements[elm].stoichiometricCoeffts):
                self.gfeMatrix[len(self.species) + i, j] = sC
                self.gfeMatrix[j, len(self.species) + i] = sC
        for j, qC in enumerate(self.chargeCoeffts):
            self.gfeMatrix[-1, j] = qC
            self.gfeMatrix[j, -1] = qC

    def initialiseNi(self, ni):
        for j, spKey in enumerate(self.species):
            self.ni[j] = ni[j]
            self.species[spKey].numberOfParticles = ni[j]

    def readNi(self):
        for j, spKey in enumerate(self.species):
            self.ni[j] = self.species[spKey].numberOfParticles

    def writeNi(self):
        for j, spKey in enumerate(self.species):
            self.species[spKey].numberOfParticles = self.ni[j]

    def writeNumberDensity(self):
        V = self.ni.sum() * constants.boltzmann * self.T / self.P
        for j, spKey in enumerate(self.species):
            self.species[spKey].numberDensity = self.ni[j] / V

    def recalcE0i(self):
        # deltaIonisationEnergy recalculation, using limitation theory of 
        # Stewart & Pyatt 1966
        weightedChargeSumSqd = 0
        weightedChargeSum = 0
        for j, spKey in enumerate(self.species):
            if self.species[spKey].chargeNumber > 0:
                weightedChargeSum += self.ni[j] * self.species[spKey].chargeNumber
                weightedChargeSumSqd += self.ni[j] * self.species[spKey].chargeNumber ** 2
        zStar = weightedChargeSumSqd / weightedChargeSum
        debyeD = np.sqrt(constants.boltzmann * self.T
                         / (4. * np.pi * (zStar + 1.) * self.ni[-1]
                            * constants.fundamentalCharge ** 2))
        for j, spKey in enumerate(self.species):
            if spKey != "e":
                ai = (3. * (self.species[spKey].chargeNumber + 1.)
                      / (4. * np.pi * self.ni[-1])) ** (1/3)
                self.species[spKey].deltaIonisationEnergy = constants.boltzmann * self.T * (((ai / debyeD) ** (2 / 3) + 1) - 1) / (2. * (zStar + 1))

        for cn in range(1, self.maxChargeNumber + 1):
            for key, sp in self.species.items():
                if sp.chargeNumber == cn:
                    self.species[key].E0 = (self.species[sp.ionisedFrom].E0
                                            + self.species[sp.ionisedFrom].ionisationEnergy
                                            - self.species[sp.ionisedFrom].deltaIonisationEnergy)

    def recalcGfeArrays(self):
        niSum = self.ni.sum()
        V = niSum * constants.boltzmann * self.T / self.P
        offDiagonal = -constants.boltzmann * self.T / niSum

        for j, spKey in enumerate(self.species):
            onDiagonal = constants.boltzmann * self.T / self.ni[j]

            for j2, spKey2 in enumerate(self.species):
                self.gfeMatrix[j, j2] = offDiagonal
            self.gfeMatrix[j, j] += onDiagonal

            totalPartitionFunction = (V * self.species[spKey].translationalPartitionFunction(self.T)
                                        * self.species[spKey].internalPartitionFunction(self.T))
            mu = -constants.boltzmann * self.T * np.log(totalPartitionFunction / self.ni[j]) + self.species[spKey].E0

            self.gfeVector[j] = -mu + onDiagonal * self.ni[j]
            for j2, spKey2 in enumerate(self.species):
                self.gfeVector[j] += offDiagonal * self.ni[j2]

    def solveGfe(self, relativeTolerance=1e-10, maxIters=1000):
        self.readNi()

        governorFactors = np.linspace(0.9, 0.1, 9)
        successYN = False
        governorIters = 0
        while not successYN:
            successYN = True
            governorFactor = governorFactors[governorIters]
            relTol = relativeTolerance * 10.
            minimiserIters = 0
            while relTol > relativeTolerance:
                self.recalcE0i()
                self.recalcGfeArrays()

                newNi = np.linalg.solve(self.gfeMatrix, self.gfeVector)

                deltaNi = abs(newNi[0:len(self.species)] - self.ni)
                maxAllowedDeltaNi = governorFactor * self.ni

                maxNiIndex = newNi.argmax()
                relTol = deltaNi[maxNiIndex] / newNi[maxNiIndex]

                lowDeltaNiYN = deltaNi < maxAllowedDeltaNi
                deltaNi[lowDeltaNiYN] = maxAllowedDeltaNi[lowDeltaNiYN]
                newRelaxFactors = maxAllowedDeltaNi / deltaNi
                relaxFactor = newRelaxFactors.min()

                self.ni = (1. - relaxFactor) * self.ni + relaxFactor * newNi[0:len(self.species)]

                minimiserIters += 1
                if minimiserIters > maxIters:
                    successYN = False
                    break

            governorIters += 1

        if not successYN:
            # TODO need to raise a proper warning or even exception here
            print("Warning! Minimiser could not find a converged solution, results may be inaccurate.")

        print(governorIters, relaxFactor, relTol)
        print(self.ni)

        self.writeNi()
        self.writeNumberDensity()

    def calculateDensity(self):
        """Calculate the density of the plasma in kg/m3 based on current 
        conditions and species composition.
        """

        density = 0
        for spKey, sp in self.species.items():
            density += sp.numberDensity * sp.molarMass / constants.avogadro
        return density

    def calculateHeatCapacity(self):
        """Calculate the heat capacity of the plasma in J/kg.K based on current 
        conditions and species composition.
        """

        raise NotImplementedError

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

################################################################################
