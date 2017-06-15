#!/usr/bin/env python3
#
# Simple classes for monatomic gas/plasma data and LTE calculations
#
# Q Reynolds 2016-2017

import math

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
        self.name = kwargs.get("name")
        self.chargeNumber = kwargs.get("chargeNumber")
        self.dataFile = kwargs.get("dataFile")
        
        self.numberDensity = 0.

        with open(self.dataFile) as f:
            energyFileData = f.readlines()  
        self.molecularMass = float(energyFileData.pop(0)) / constants.avogadro
        self.ionisationEnergy = constants.invCmToJ * float(energyFileData.pop(0))
        self.deltaIonisationEnergy = 0.

        self.energyLevels = []
        for energyLevelLine in energyFileData:
            energyLevelLineData = energyLevelLine.split(",")
            energyLevelLineData[0] = float(energyLevelLineData[0])
            energyLevelLineData[1] = constants.invCmToJ * float(energyLevelLineData[1])
            self.energyLevels.append(energyLevelLineData)
            
    def internalPartitionFunction(self, T):
        partitionVal = 0.
        for eLevel in self.energyLevels:
            if eLevel[1] < (self.ionisationEnergy - self.deltaIonisationEnergy):
                partitionVal += (2. * eLevel[0] + 1.) * math.exp(-eLevel[1] / (constants.boltzmann * T))
        return partitionVal


# A bonded atom pair
class diatomicSpecie:
    def __init__(self, **kwargs):
        self.name = kwargs.get("name")
        self.chargeNumber = kwargs.get("chargeNumber")
        self.dataFile = kwargs.get("dataFile")
        
        self.numberDensity = 0.

        with open(self.dataFile) as f:
            energyFileData = f.readlines()  
        self.molecularMass = float(energyFileData.pop(0)) / constants.avogadro
        self.dissociationEnergy = constants.invCmToJ * float(energyFileData.pop(0))
        self.sigmaS = float(energyFileData.pop(0))
        
        moleculeData = energyFileData[0].split(",")
        self.g0 = float(moleculeData[0])
        self.we = constants.invCmToJ * float(moleculeData[1])
        self.Be = constants.invCmToJ * float(moleculeData[2])
        
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
        
    def _equilibriumRHS(self, n, T):
        if n < (self.nSpecies - 1):
            if n == 0:
                partitionTerm = (self.species[n+1].internalPartitionFunction(T)) ** 2. / self.species[n].internalPartitionFunction(T)
                translationTerm = (2. * math.pi * self.species[n+1].molecularMass * constants.boltzmann * T / constants.planck ** 2) ** 1.5
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
        self.elementMoleFractions = []
        for key, val in speciesDict.items():
            self.elementGroups.append(slagElementGroup(element = key, chargeNumbers = val[0], energyLevelFilePath = self.energyLevelFilePath))
            
        
        
################################################################################

