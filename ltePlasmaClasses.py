#!/usr/bin/env python3
#
# Classes for gas/plasma data and LTE calculations with simple species
#
# Q Reynolds 2016-2017

import math
import json

################################################################################

class constants:
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


# Single atoms and ions
class monatomicSpecie:
    def __init__(self, **kwargs):
        self.numberDensity = kwargs.get("numberDensity", 0.)

        # Construct a data object from JSON data file
        with open(kwargs.get("dataFile")) as df:
            jsonData = json.load(df)

        # General specie data
        self.name = jsonData["name"]
        self.stoichiometry = jsonData["stoichiometry"]
        self.molarMass = jsonData["molarMass"]

        # Monatomic-specific specie data
        self.ionisationEnergy = constants.invCmToJ * jsonData["monatomicData"]["ionisationEnergy"]
        self.deltaIonisationEnergy = 0.
        self.energyLevels = []
        for energyLevelLine in jsonData["monatomicData"]["energyLevels"]:
            self.energyLevels.append([2. * energyLevelLine["J"] + 1., constants.invCmToJ * energyLevelLine["Ei"]])
            
    def internalPartitionFunction(self, T):
        partitionVal = 0.
        for eLevel in self.energyLevels:
            if eLevel[1] < (self.ionisationEnergy - self.deltaIonisationEnergy):
                partitionVal += eLevel[0] * math.exp(-eLevel[1] / (constants.boltzmann * T))
                
        return partitionVal


# A bonded atom pair
class diatomicSpecie:
    def __init__(self, **kwargs):
        self.numberDensity = kwargs.get("numberDensity", 0.)

        # Construct a data object from JSON data file
        with open(kwargs.get("dataFile")) as df:
            jsonData = json.load(df)

        # General specie data
        self.name = jsonData["name"]
        self.stoichiometry = jsonData["stoichiometry"]
        self.molarMass = jsonData["molarMass"]

        # Diatomic-specific specie data
        self.dissociationEnergy = constants.invCmToJ * jsonData["diatomicData"]["dissociationEnergy"]
        self.sigmaS = jsonData["diatomicData"]["sigmaS"]
        self.g0 = jsonData["diatomicData"]["g0"]
        self.we = constants.invCmToJ * jsonData["diatomicData"]["we"]
        self.Be = constants.invCmToJ * jsonData["diatomicData"]["Be"]
        
    def internalPartitionFunction(self, T):
        electronicPartition = self.g0
        vibrationalPartition = 1. / (1. - math.exp(-self.we / (constants.boltzmann * T)))
        rotationalPartition = constants.boltzmann * T / (self.sigmaS * self.Be)
        
        return electronicPartition * vibrationalPartition * rotationalPartition


# Group representing an element in the presence of oxygen. First specie is the 
# diatomic single bond with O, second is the neutral atom, third is the singly
# charged ion, etc. Eg indices for Silicon: 
# 0: SiO
# 1: Si
# 2: Si+
# 3: Si++
# ...etc
class slagElementGroup:
    def __init__(self, **kwargs):
        self.element = kwargs.get("element")
        self.maximumIonCharge = kwargs.get("maximumIonCharge")
        self.dataFilePath = kwargs.get("dataFilePath")

        self.nSpecies = self.maximumIonCharge + 2
        self.equilibriumTerms = [0.] * (self.nSpecies - 1)
        
        self.oxygenAtomSpecie = monatomicSpecie(
            name = "OI",
            chargeNumber = 0,
            dataFile = self.dataFilePath + "OI.csv")
        
        self.species = []
        self.species.append(diatomicSpecie(
            name = self.element + "O",
            chargeNumber = 0,
            dataFile = self.dataFilePath + self.element + "O.csv"))
        
        for n in range(self.maximumIonCharge + 1):
            chargeString = "I"
            for i in range(n):
                chargeString += "I"
            
            self.species.append(monatomicSpecie(
                name = self.element + chargeString, 
                chargeNumber = n,
                dataFile = self.dataFilePath + self.element + chargeString + ".csv"))
        
        self.diatomicReducedMass = self.oxygenAtomSpecie.molecularMass * self.species[1].molecularMass / (self.oxygenAtomSpecie.molecularMass + self.species[1].molecularMass)
        
    def _equilibriumRHS(self, n, T):
        if n < (self.nSpecies - 1):
            if n == 0:
                partitionTerm = self.species[n+1].internalPartitionFunction(T) * self.oxygenAtomSpecie.internalPartitionFunction(T) / self.species[n].internalPartitionFunction(T)
                translationTerm = (2. * math.pi * self.diatomicReducedMass * constants.boltzmann * T / constants.planck ** 2) ** 1.5
                expTerm = math.exp(-(self.species[n].dissociationEnergy) / (constants.boltzmann * T))
            else:                
                partitionTerm = 2. * self.species[n+1].internalPartitionFunction(T) / self.species[n].internalPartitionFunction(T)
                translationTerm = (2. * math.pi * constants.electronMass * constants.boltzmann * T / constants.planck ** 2) ** 1.5
                expTerm = math.exp(-(self.species[n].ionisationEnergy - self.species[n].deltaIonisationEnergy) / (constants.boltzmann * T))
            return partitionTerm * translationTerm * expTerm
        else:
            # TODO: Handle with proper exception
            return "Error - requested Saha ratio for n too high"

    def recalcEquilibriumTerms(self, T):
        for n in range(len(self.equilibriumTerms)):
            self.equilibriumTerms[n] = self._equilibriumRHS(n, T)


class slagComposition:
    def __init__(self, **kwargs):
        speciesDict = kwargs.get("speciesDict")
        self.dataFilePath = kwargs.get("dataFilePath")
        self.ne = 0.
        
        self.elementGroups = []
        self.elementStartingMoleFractions = []
        for key, val in speciesDict.items():
            self.elementGroups.append(slagElementGroup(element = key, maximumIonCharge = val[0], energyLevelFilePath = self.energyLevelFilePath))
            self.elementMoleFractions.append(val[1])
        
        
################################################################################

