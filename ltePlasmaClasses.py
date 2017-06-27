#!/usr/bin/env python3
#
# Classes for gas/plasma data and LTE calculations with simple species
#
# Q Reynolds 2016-2017

import json
import numpy as np

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


# Diatomic molecules, single atoms, and ions
class specie:
    def __init__(self, **kwargs):
        self.numberDensity = kwargs.get("numberDensity", 0.)

        # Construct a data object from JSON data file
        with open(kwargs.get("dataFile")) as df:
            jsonData = json.load(df)

        # General specie data
        self.name = jsonData["name"]
        self.stoichiometry = jsonData["stoichiometry"]
        self.molarMass = jsonData["molarMass"]
        self.chargeNumber = jsonData["chargeNumber"]
            
        if "monatomicData" in jsonData:
            # Monatomic-specific specie data
            self.monatomicYN = True
            self.ionisationEnergy = constants.invCmToJ * jsonData["monatomicData"]["ionisationEnergy"]
            self.deltaIonisationEnergy = 0.
            self.energyLevels = []
            for energyLevelLine in jsonData["monatomicData"]["energyLevels"]:
                self.energyLevels.append([2. * energyLevelLine["J"] + 1., constants.invCmToJ * energyLevelLine["Ei"]])
        
        else:
            # Diatomic-specific specie data
            self.monatomicYN = False
            self.dissociationEnergy = constants.invCmToJ * jsonData["diatomicData"]["dissociationEnergy"]
            self.ionisationEnergy = constants.invCmToJ * jsonData["diatomicData"]["ionisationEnergy"]
            self.deltaIonisationEnergy = 0.
            self.sigmaS = jsonData["diatomicData"]["sigmaS"]
            self.g0 = jsonData["diatomicData"]["g0"]
            self.we = constants.invCmToJ * jsonData["diatomicData"]["we"]
            self.Be = constants.invCmToJ * jsonData["diatomicData"]["Be"]

        if self.chargeNumber < 0:
            raise ValueError("Error! Negatively charged ions not implemented yet.")
            
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

        
class electronSpecie:
    def __init__(self, **kwargs):
        self.name = "e"
        self.stoichiometry = {}
        self.molarMass = constants.electronMass * constants.avogadro
        self.chargeNumber = -1
        self.numberDensity = kwargs.get("numberDensity", 0.)
        
    def internalPartitionFunction(self, T):
        return 2.


# Composition class for Gibbs Free Energy minimisation calculation
class compositionGFE:
    def __init__(self, **kwargs):
        with open(kwargs.get("compositionFile")) as sf:
            jsonData = json.load(sf)
                
        self.species = {}
        for spData in jsonData["speciesList"]:
            sp = specie(dataFile = spData["specie"])
            self.species[sp.name] = sp
            self.species[sp.name].x0 = spData["x0"]        
        self.species["e"] = electronSpecie()
        self.species["e"].x0 = 0.
        
        self.elements = []
        for key, sp in self.species.items():
            for skey in sp.stoichiometry:
                self.elements.append(skey)
        self.elements = list(set(self.elements))
        
        # set ion source specie
        self.maxChargeNumber = 0
        for key, sp in self.species.items():
            if sp.chargeNumber > self.maxChargeNumber:
                self.maxChargeNumber = sp.chargeNumber
            if sp.chargeNumber > 0:
                for key2, sp2 in self.species.items():
                    if sp2.stoichiometry == sp.stoichiometry and sp2.chargeNumber == sp.chargeNumber - 1:
                        self.species[key].ionisedFrom = key2
        
        # set specie reference energies
        for key, sp in self.species.items():
            if sp.chargeNumber == 0:
                if sp.monatomicYN:
                    self.species[key].E0 = 0.
                else:
                    self.species[key].E0 = -self.species[key].dissociationEnergy
        self.species["e"].E0 = 0.
        
        # set up A matrix and b vector for GFE minimiser
        self.gfeMatrix = np.zeros((
            len(self.species) + len(self.elements) + 1, 
            len(self.species) + len(self.elements) + 1))
        self.gfeVector = np.zeros(len(self.species) + len(self.elements) + 1)

        self.speciesMultiplierDeltaV = []
        for key, sp in self.species.items():
            atomSum = 0
            for key2, st in sp.stoichiometry.items():
                atomSum += st
            self.speciesMultiplierDeltaV.append(atomSum - 1)
        
        self.elementFractions = []
        atomsNT = 0.
        for j, key in enumerate(self.species):
            atomsNT += self.species[key].x0 * (self.speciesMultiplierDeltaV[j] + 1)
        for elm in self.elements:
            eMF = 0.
            for key, sp in self.species.items():
                eMF += self.species[key].x0 * self.species[key].stoichiometry.get(elm, 0.)
            self.elementFractions.append(eMF / atomsNT)
        
        for i, elm in enumerate(self.elements):
            for j, key in enumerate(self.species):
                self.gfeMatrix[len(self.species) + i][j] = self.species[key].stoichiometry.get(elm, 0.) - self.elementFractions[i] * self.speciesMultiplierDeltaV[j]
                self.gfeMatrix[j][len(self.species) + i] = self.gfeMatrix[len(self.species) + i][j]

        for j, key in enumerate(self.species):
            self.gfeMatrix[-1][j] = self.species[key].chargeNumber
            self.gfeMatrix[j][-1] = self.gfeMatrix[-1][j]
        
        self.setTP(kwargs.get("T", 10000.), kwargs.get("P", 101325.))
        
        self.setInitialGuess()
        
    def setTP(self, newT, newP):
        self.T = newT
        self.P = newP
        self.totalNumberDensity = self.P / (constants.boltzmann * self.T)
        for i, elm in enumerate(self.elements):
            self.gfeVector[len(self.species) + i] = self.elementFractions[i] * self.totalNumberDensity
                
    def setInitialGuess(self):
        neMultiplier = 0
        for key, sp in self.species.items():
            if key != "e":            
                neMultiplier += sp.chargeNumber
        nsp = len(self.species) - 1
        niInitial = self.totalNumberDensity / (neMultiplier + nsp)
        for key, sp in self.species.items():
            sp.numberDensity = niInitial
        self.species["e"].numberDensity = neMultiplier * niInitial
        
    def recalcE0i(self):
        #recalculation of deltaIonisationEnergy's here
        for cn in range(1, self.maxChargeNumber + 1):
            for key, sp in self.species.items():
                if sp.chargeNumber == cn:
                    self.species[key].E0 = self.species[sp.ionisedFrom].E0 + self.species[sp.ionisedFrom].ionisationEnergy - self.species[sp.ionisedFrom].deltaIonisationEnergy

    def recalcCoeffts(self):
        for j, key in enumerate(self.species):
            self.gfeMatrix[j][j] = constants.boltzmann * self.T / self.species[key].numberDensity
            self.gfeVector[j] = (np.log(self.species[key].totalPartitionFunction(self.T) / self.species[key].numberDensity) + 1.) * constants.boltzmann * self.T - self.species[key].E0
    
    def solveGFE(self):
        self.recalcE0i()
        self.recalcCoeffts()
        ni = np.linalg.solve(self.gfeMatrix, self.gfeVector)
        for j, key in enumerate(self.species):
            if ni[j] < 0:
                ni[j] = 1e8
            self.species[key].numberDensity = ni[j]
        
################################################################################

