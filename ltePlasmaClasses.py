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
        self.numberDensity = kwargs.get("numberDensity", 0.)
        
    def translationalPartitionFunction(self, T):
        return ((2 * np.pi * self.molarMass * constants.boltzmann * T) / (constants.avogadro * constants.planck ** 2)) ** 1.5

    def internalPartitionFunction(self, T):
        return 2.


class element:
    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "")
        self.stoichiometricCoeffts = np.array(kwargs.get("stoichiometricCoeffts", []))
        self.totalNumber = kwargs.get("totalNumber", 0.)
    
    
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
        
        self.elements = {}
        elmList = []
        for key, sp in self.species.items():
            for skey in sp.stoichiometry:
                elmList.append(skey)
        elmList = list(set(elmList))
        for elm in elmList:
            self.elements[elm] = element(name = elm)
        
        # set ion source specie
        self.maxChargeNumber = 0
        for key, sp in self.species.items():
            if sp.chargeNumber > self.maxChargeNumber:
                self.maxChargeNumber = sp.chargeNumber
            if sp.chargeNumber > 0:
                for key2, sp2 in self.species.items():
                    if sp2.stoichiometry == sp.stoichiometry and sp2.chargeNumber == sp.chargeNumber - 1:
                        self.species[key].ionisedFrom = key2
        
        # set reference specie energies
        for key, sp in self.species.items():
            if sp.chargeNumber == 0:
                if sp.monatomicYN:
                    self.species[key].E0 = 0.
                else:
                    self.species[key].E0 = -self.species[key].dissociationEnergy
        self.species["e"].E0 = 0.
        self.recalcE0i()
        
        # set stoichiometry and charge coefficient arrays for constraints
        for i, elm in enumerate(self.elements):
            self.elements[elm].stoichiometricCoeffts = np.zeros(len(self.species))
            for j, key in enumerate(self.species):
                self.elements[elm].stoichiometricCoeffts[j] = self.species[key].stoichiometry.get(elm, 0.)
        
        self.chargeCoeffts = np.zeros(len(self.species))
        for j, key in enumerate(self.species):
            self.chargeCoeffts[j] = self.species[key].chargeNumber
        
        #set element totals from initial conditions for constraints
        self.T = kwargs.get("T", 10000.)
        self.P = kwargs.get("P", 101325.)
        nT0 = self.P / (constants.boltzmann * self.T)
        for keyElm, elm in self.elements.items():
            elm.totalNumber = 0
            for j, key in enumerate(self.species):
                elm.totalNumber += nT0 * elm.stoichiometricCoeffts[j] * self.species[key].x0

        # set up constraints and bounds for optimiser
        self.constraints = []
        for i, elm in enumerate(self.elements):
            self.constraints.append({})
            self.constraints[-1]["type"] = "eq"
            self.constraints[-1]["fun"] = lambda x: np.dot(x, self.elements[elm].stoichiometricCoeffts) - self.elements[elm].totalNumber
        self.constraints.append({})
        self.constraints[-1]["type"] = "eq"
        self.constraints[-1]["fun"] = lambda x: np.dot(x, self.chargeCoeffts)
        self.constraints = tuple(self.constraints)
        
        self.bounds = []
        for j, key in enumerate(self.species):
            self.bounds.append((0., 1e24))
        self.bounds = tuple(self.bounds)
                
    def recalcE0i(self):
        # TODO recalculation of deltaIonisationEnergies here...
        for cn in range(1, self.maxChargeNumber + 1):
            for key, sp in self.species.items():
                if sp.chargeNumber == cn:
                    self.species[key].E0 = self.species[sp.ionisedFrom].E0 + self.species[sp.ionisedFrom].ionisationEnergy - self.species[sp.ionisedFrom].deltaIonisationEnergy

    def calculateGFE(self):
        self.recalcE0i()
        x = np.full(len(self.species), 1e23)
            
        res = minimize(
            self.minFunctionGFE, 
            x, 
            method = "SLSQP", 
            jac = self.jacFunctionGFE, 
            bounds = self.bounds, 
            constraints = self.constraints)

        print(res)
################################################################################

