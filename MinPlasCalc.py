#!/usr/bin/env python3
#
# Classes for gas/plasma data and LTE calculations with simple species
#
# Q Reynolds 2016-2017

import json
import numpy as np
from scipy.optimize import minimize

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
        self.numberOfParticles = kwargs.get("numberOfParticles", 0.)
        self.numberDensity = 0.

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

    def translationalPartitionFunction(self, T):
        return ((2 * np.pi * self.molarMass * constants.boltzmann * T) / (constants.avogadro * constants.planck ** 2)) ** 1.5
            
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
        self.numberOfParticles = kwargs.get("numberOfParticles", 0.)
        self.numberDensity = 0.
        
    def translationalPartitionFunction(self, T):
        return ((2 * np.pi * self.molarMass * constants.boltzmann * T) / (constants.avogadro * constants.planck ** 2)) ** 1.5

    def internalPartitionFunction(self, T):
        return 2.


class element:
    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "")
        self.stoichiometricCoeffts = kwargs.get("stoichiometricCoeffts", [])
        self.totalNumber = kwargs.get("totalNumber", 0.)
    
    
# Composition class for Gibbs Free Energy minimisation calculation
class compositionGFE:
    def __init__(self, **kwargs):
        self.T = kwargs.get("T", 10000.)
        self.P = kwargs.get("P", 101325.)

        with open(kwargs.get("compositionFile")) as sf:
            jsonData = json.load(sf)
                
        self.species = {}
        for spData in jsonData["speciesList"]:
            sp = specie(dataFile = spData["specie"])
            self.species[sp.name] = sp
            self.species[sp.name].x0 = spData["x0"]        
        self.species["e"] = electronSpecie()
        self.species["e"].x0 = 0.
        
        self.elements = {}
        elmList = []
        for spKey, sp in self.species.items():
            for stKey in sp.stoichiometry:
                elmList.append(stKey)
        elmList = list(set(elmList))
        for elmKey in elmList:
            self.elements[elmKey] = element(name = elmKey)
        
        # set specie which each +ve charged ion derives from
        self.maxChargeNumber = 0
        for spKey, sp in self.species.items():
            if sp.chargeNumber > self.maxChargeNumber:
                self.maxChargeNumber = sp.chargeNumber
            if sp.chargeNumber > 0:
                for spKey2, sp2 in self.species.items():
                    if sp2.stoichiometry == sp.stoichiometry and sp2.chargeNumber == sp.chargeNumber - 1:
                        sp.ionisedFrom = spKey2
        
        # set specie reference energies
        for spKey, sp in self.species.items():
            if sp.chargeNumber == 0:
                if sp.monatomicYN:
                    sp.E0 = 0.
                else:
                    sp.E0 = -sp.dissociationEnergy
        self.species["e"].E0 = 0.
        self.recalcE0i()
        
        # set stoichiometry and charge coefficient arrays for mass action and 
        # electroneutrality constraints
        for elmKey, elm in self.elements.items():
            elm.stoichiometricCoeffts = []
            for spKey, sp in self.species.items():
                elm.stoichiometricCoeffts.append(sp.stoichiometry.get(elmKey, 0.))
        
        self.chargeCoeffts = []
        for spKey, sp in self.species.items():
            self.chargeCoeffts.append(sp.chargeNumber)
        
        #set element totals for constraints from provided initial conditions
        nT0 = self.P / (constants.boltzmann * self.T)
        for elmKey, elm in self.elements.items():
            elm.totalNumber = 0
            for j, spKey in enumerate(self.species):
                elm.totalNumber += nT0 * elm.stoichiometricCoeffts[j] * self.species[spKey].x0

        # set up A matrix,  b and ni vectors for GFE minimiser
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
        # deltaIonisationEnergy recalculation here...
        for cn in range(1, self.maxChargeNumber + 1):
            for key, sp in self.species.items():
                if sp.chargeNumber == cn:
                    self.species[key].E0 = self.species[sp.ionisedFrom].E0 + self.species[sp.ionisedFrom].ionisationEnergy - self.species[sp.ionisedFrom].deltaIonisationEnergy

    def recalcGfeArrays(self):
        niSum = self.ni.sum()
        V = niSum * constants.boltzmann * self.T / self.P
        offDiagonal = -constants.boltzmann * self.T / niSum

        for j, spKey in enumerate(self.species):
            onDiagonal = constants.boltzmann * self.T / self.ni[j]
            
            for j2, spKey2 in enumerate(self.species):
                self.gfeMatrix[j, j2] = offDiagonal
            self.gfeMatrix[j, j] += onDiagonal
            
            totalPartitionFunction = V * self.species[spKey].translationalPartitionFunction(self.T) * self.species[spKey].internalPartitionFunction(self.T)
            mu = -constants.boltzmann * self.T * np.log(totalPartitionFunction / self.ni[j]) + self.species[spKey].E0
            
            self.gfeVector[j] = -mu + onDiagonal * self.ni[j]
            for j2, spKey2 in enumerate(self.species):
                self.gfeVector[j] += offDiagonal * self.ni[j2]
            
    def solveGfe(self, **kwargs):
        governorFactor = kwargs.get("governorFactor", 0.5)
        relativeTolerance = kwargs.get("relativeTolerance", 1e-10)
        maxIters = kwargs.get("maxIters", 1000)
        
        self.readNi()

        relTol = relativeTolerance * 10.
        iters = 0
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
            
            iters += 1
            if iters > maxIters:
                # TODO need to raise proper warning here
                print("Warning! Max iters reached")
                break
            

        print(iters, relaxFactor, relTol)
        print(self.ni)

        self.writeNi()
        self.writeNumberDensity()        


################################################################################

