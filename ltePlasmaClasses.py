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
    eVtoK = 11604.505
    eVtoJ = 1.60217653e-19
    invCmToJ = 1.9864456e-23

    
class specie:
    def __init__(self, **kwargs):
        self.name = kwargs.get("name")
        self.chargeNumber = kwargs.get("chargeNumber")
        self.energyLevelFile = kwargs.get("energyLevelFile")
        
        self.numberDensity = 0.

        with open(self.energyLevelFile) as f:
            energyLevelFileData = f.readlines()  
        self.molWeight = float(energyLevelFileData.pop(0))
        self.ionisationEnergy = constants.invCmToJ * float(energyLevelFileData.pop(0))
        self.deltaIonisationEnergy = 0.

        self.energyLevels = []
        for energyLevelLine in energyLevelFileData:
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


class elementGroup:
    def __init__(self, **kwargs):
        self.element = kwargs.get("element")
        self.chargeNumbers = kwargs.get("chargeNumbers")
        self.energyLevelFilePath = kwargs.get("energyLevelFilePath")

        self.species = []
        for n in range(len(self.chargeNumbers)):
            chargeString = "I"
            for i in range(self.chargeNumbers[n]):
                chargeString += "I"
            
            self.species.append(specie(
                name = self.element + chargeString, 
                chargeNumber = self.chargeNumbers[n],
                energyLevelFile = self.energyLevelFilePath + self.element + chargeString + ".csv"))
        
    def sahaRHS(self, n, T):
        partitionTerm = 2. * self.species[n+1].internalPartitionFunction(T) / self.species[n].internalPartitionFunction(T)
        translationTerm = (2. * math.pi * constants.electronMass * constants.boltzmann * T / constants.planck ** 2) ** 1.5
        expTerm = math.exp(-(self.species[n].ionisationEnergy - self.species[n].deltaIonisationEnergy) / (constants.boltzmann * T))
        return partitionTerm * translationTerm * expTerm
    
################################################################################

