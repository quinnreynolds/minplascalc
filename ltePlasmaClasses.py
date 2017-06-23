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
            self.sigmaS = jsonData["diatomicData"]["sigmaS"]
            self.g0 = jsonData["diatomicData"]["g0"]
            self.we = constants.invCmToJ * jsonData["diatomicData"]["we"]
            self.Be = constants.invCmToJ * jsonData["diatomicData"]["Be"]

        if self.chargeNumber != 0 and self.monatomicYN == False:
            print("Warning! Charged diatomic molecules not implemented yet, results will probably be wrong")

        if self.chargeNumber < 0:
            print("Warning! Negatively charged ions not implemented yet, results will probably be wrong")
    
    def getRxnEnergy(self):
        if self.monatomicYN:
            return self.ionisationEnergy - self.deltaIonisationEnergy
        else:
            return self.dissociationEnergy
        
    def internalPartitionFunction(self, T):
        if self.monatomicYN:
            partitionVal = 0.
            for eLevel in self.energyLevels:
                if eLevel[1] < (self.ionisationEnergy - self.deltaIonisationEnergy):
                    partitionVal += eLevel[0] * math.exp(-eLevel[1] / (constants.boltzmann * T))                    
            return partitionVal
        else:
            electronicPartition = self.g0
            vibrationalPartition = 1. / (1. - math.exp(-self.we / (constants.boltzmann * T)))
            rotationalPartition = constants.boltzmann * T / (self.sigmaS * self.Be)            
            return electronicPartition * vibrationalPartition * rotationalPartition


# Composition class form Gibbs Free Energy minimisation calculation
class compositionGFE:
    def __init__(self, **kwargs):
        with open(kwargs.get("compositionFile")) as sf:
            jsonData = json.load(sf)
        
        self.species = {}
        for spData in jsonData["speciesList"]:
            sp = specie(dataFile = spData["specie"])
            self.species[sp.name] = sp
            self.species[sp.name].x0 = spData["x0"]
        
        self.elements = []
        for key, sp in self.species.items():
            for skey in sp.stoichiometry:
                self.elements.append(skey)
        self.elements = list(set(self.elements))
               
        print(self.elements)

        # set reference element energies
        # (free electron reference energy also assumed to be zero)
        
        for key, sp in self.species.items():
            if sp.monatomicYN and sp.chargeNumber > 0:
                for key2, sp2 in self.species.items():
                    if sp2.stoichiometry == sp.stoichiometry and sp2.chargeNumber == sp.chargeNumber - 1:
                        self.species[key].ionisedFrom = key2
                        print(key, key2)
                        
        for key, sp in self.species.items():
            if sp.chargeNumber == 0:
                if sp.monatomicYN and sp.chargeNumber == 0:
                    self.species[key].E0 = 0.
                if sp.monatomicYN == False:
                    self.species[key].E0 = -self.species[key].getRxnEnergy()
            
    def recalcE0i(self):
        for key, sp in self.species.items():
            if sp.monatomicYN and sp.chargeNumber == 1:
                
                
                
            
            

                
        
################################################################################

