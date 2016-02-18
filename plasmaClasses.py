#!/usr/bin/env python
#
# Simple classes for plasma data and calculations
#
# Q Reynolds 2016

import math
import numpy

################################################################################

class constants:
    protonMass = 1.6726219e-27
    electronMass = 9.10938356e-31
    fundamentalCharge = 1.60217662e-19
    avogadroNumber = 6.0221409e23
    boltzmann = 1.38064852e-23
    eVtoK = 11604.505
    ionClassicalRadiusFactor = 4.8e-10
    ionElectronRecombination = 1.1e-18
    graphiteRichardsonDushmannA = 60e4
    graphiteWorkFunction = 4.62


    
class specie:
    def __init__(self, name, molWeight, chargeNumber, radius):
        self.name = str(name)
        self.molWeight = float(molWeight)
        self.chargeNumber = int(chargeNumber)
        self.radius = float(radius)
            
    def mass(self):
        return self.molWeight / (1000. * constants.avogadroNumber)
   
    def charge(self):
        return float(self.chargeNumber) * constants.fundamentalCharge



class composition:
    def __init__(self, T, compArray):
        self.T = float(T)
        self.n = []
        for i in range(len(compArray)):
            self.n.append(float(compArray[i]))

    # len(n) must = len(spArr) in the following functions
    
    def totalEqDensity(self, spArr):
        sumN = 0.
        for j in range(len(spArr)):
            sumN += self.n[j]
        return sumN
        
    def heavyEqDensity(self, spArr):
        sumN = 0.
        for j in range(len(spArr)-1):
            sumN += self.n[j]
        return sumN
        
    def neutralEqDensity(self, spArr):
        sumN = 0.
        for j in range(len(spArr)):
            if (spArr[j].chargeNumber == 0):
                sumN += self.n[j]
        return sumN

    def crossSection(self, i, spArr):
        moleFractions = []
        for j in range(len(spArr)):
            moleFractions.append(self.n[j])
        invSumMF = 1. / math.fsum(moleFractions)
        for j in range(len(spArr)):
            moleFractions[j] *= invSumMF
        xSection = 0.
        for j in range(len(spArr)):
            xSection += moleFractions[j] * collisionCrossSection(spArr[i], spArr[j], self.T)
        return xSection

    def Ds(self, i, spArr):
        return 2. / math.sqrt(math.pi * 2.) * math.sqrt(constants.boltzmann * self.T / spArr[i].mass()) / (self.totalEqDensity(spArr) * self.crossSection(i, spArr))

    def averageDs(self, spArr):
        ionFractions = []
        for j in range(len(spArr)-1):   # -1 to omit electrons
            if (spArr[j].chargeNumber != 0):
                ionFractions.append(self.n[j])
            else:
                ionFractions.append(0.)
        ionFractions.append(0.)   # omit electrons (last specie)
        invSumMF = 1. / math.fsum(ionFractions)
        for j in range(len(spArr)):
            ionFractions[j] *= invSumMF
        avgDi = 0.
        for j in range(len(spArr)):
            avgDi += ionFractions[j] * self.Ds(j, spArr)
        return avgDi

    def DA(self, spArr):
        return 2. * self.averageDs(spArr)
   
    def cGamma(self, spArr):
        return constants.ionElectronRecombination / self.T ** 4.5
   
    def electricalConductivity(self, spArr):
        electronXS = self.crossSection(len(spArr)-1, self.T)
        electronN = self.n[len(spArr)-1]
        totalN = self.totalEqDensity(spArr)
        return electronN * constants.fundamentalCharge ** 2. / (math.sqrt(2. * math.pi * constants.boltzmann * self.T * constants.electronMass) * totalN * electronXS)



class plasmaData:
    def __init__(self, specieArray, compArray):
        self.species = specieArray
        self.compositions = compArray

    def nDensity(self, i, T):
        if (T < self.compositions[0].T):
            returnN = self.compositions[0].n[i]
        elif (T > self.compositions[len(self.compositions)-1].T):
            returnN = self.compositions[len(self.compositions)-1].n[i]
        else:
            for j in range(1,len(self.compositions)):
                if (self.compositions[j-1].T <= T < self.compositions[j].T):
                    ratioT = (T - self.compositions[j-1].T) / (self.compositions[j].T - self.compositions[j-1].T)
                    returnN = self.compositions[j-1].n[i] + ratioT * (self.compositions[j].n[i] - self.compositions[j-1].n[i])
                    break
        return returnN

    def totalEqDensity(self, T):
        sumN = 0.
        for j in range(len(self.species)):
            sumN += self.nDensity(j, T)
        return sumN
        
    def heavyEqDensity(self, T):
        sumN = 0.
        for j in range(len(self.species)-1):
            sumN += self.nDensity(j, T)
        return sumN
        
    def neutralEqDensity(self, T):
        sumN = 0.
        for j in range(len(self.species)):
            if (self.species[j].chargeNumber == 0):
                sumN += self.nDensity(j, T)
        return sumN

    def crossSection(self, i, T):
        moleFractions = []
        for j in range(len(self.species)):
            moleFractions.append(self.nDensity(j, T))
        invSumMF = 1. / math.fsum(moleFractions)
        for j in range(len(self.species)):
            moleFractions[j] *= invSumMF
        xSection = 0.
        for j in range(len(self.species)):
            xSection += moleFractions[j] * collisionCrossSection(self.species[i], self.species[j], T)
        return xSection

    def Ds(self, i, T):
        return 2. / math.sqrt(math.pi * 2.) * math.sqrt(constants.boltzmann * T / self.species[i].mass()) / (self.totalEqDensity(T) * self.crossSection(i, T))

    def averageDs(self, T):
        ionFractions = []
        for j in range(len(self.species)-1):   # -1 to omit electrons
            if (self.species[j].chargeNumber != 0):
                ionFractions.append(self.nDensity(j, T))
            else:
                ionFractions.append(0.)
        ionFractions.append(0.)   # omit electrons (last element)
        invSumMF = 1. / math.fsum(ionFractions)
        for j in range(len(self.species)):
            ionFractions[j] *= invSumMF
        avgDi = 0.
        for j in range(len(self.species)):
            avgDi += ionFractions[j] * self.Ds(j, T)
        return avgDi

    def DA(self, T):
        return 2. * self.averageDs(T)
   
    def cGamma(self, T):
        return constants.ionElectronRecombination / T ** 4.5
   
    def electricalConductivityEq(self, T):
        electronXS = self.crossSection(len(self.species)-1, T)
        electronN = self.nDensity(len(self.species)-1, T)
        totalN = self.totalEqDensity(T)
        return electronN * constants.fundamentalCharge ** 2. / (math.sqrt(2. * math.pi * constants.boltzmann * T * constants.electronMass) * totalN * electronXS)
        
class lowkeSheathModel:
    def __init__(self, Tc, Ta, xMax, nSteps, locPlasmaData):
        self.Tc = Tc
        self.Ta = Ta
        self.xMax = xMax
        
        self.jR = constants.graphiteRichardsonDushmannA * self.Tc**2. * math.exp(-constants.graphiteWorkFunction * constants.eVtoK / self.Tc)
        self.neC = self.jR / constants.fundamentalCharge * math.sqrt(math.pi * constants.electronMass / (8. * constants.boltzmann * self.Tc))
        self.neA = locPlasmaData.nDensity(len(locPlasmaData.species)-1, self.Ta)

        self.ne = [0.] * nSteps
        self.x = [0.] * nSteps
        self.T = [0.] * nSteps
        self.Da = [0.] * nSteps
        self.cGamma = [0.] * nSteps
        self.neq = [0.] * nSteps
        self.E = [0.] * nSteps
        self.tdmaAW = [0.] * nSteps
        self.tdmaAP = [0.] * nSteps
        self.tdmaAE = [0.] * nSteps
        self.tdmaD = [0.] * nSteps
        
        for i in range(nSteps):
            self.x[i] = self.xMax * float(i) / float(nSteps-1)
            self.T[i] = self.Tc + (self.Ta - self.Tc) * self.x[i] / self.xMax
            self.Da[i] = locPlasmaData.DA(self.T[i])
            self.cGamma[i] = locPlasmaData.cGamma(self.T[i])
            self.neq[i] = locPlasmaData.nDensity(len(locPlasmaData.species)-1, self.T[i])
            self.ne[i] = self.neq[i]
            
        for i in range(nSteps):
            if (0 < i < (nSteps-1)):
                self.tdmaAW[i] = hmean(self.Da[i-1], self.Da[i]) / (0.5 * (self.x[i+1] - self.x[i-1]) * (self.x[i] - self.x[i-1]))
                self.tdmaAE[i] = hmean(self.Da[i+1], self.Da[i]) / (0.5 * (self.x[i+1] - self.x[i-1]) * (self.x[i+1] - self.x[i]))
                #self.tdmaAP[i] = self.tdmaAE[i] + self.tdmaAW[i]
            else:
                self.tdmaAW[i] = 0.
                self.tdmaAE[i] = 0.
                self.tdmaAP[i] = 1.
            
            self.tdmaD[0] = self.neC
            self.tdmaD[nSteps-1] = self.neA

            
    def iterateNe(self):
        for i in range(1,len(self.ne)-1):
            dSdne = self.cGamma[i] * (self.neq[i] ** 2. - 3. * self.ne[i] ** 2.)
            self.tdmaD[i] = -2. * self.cGamma[i] * self.ne[i] ** 3.
            self.tdmaAP[i] = -(self.tdmaAE[i] + self.tdmaAW[i]) + dSdne
        self.ne = solveTDMA(self.tdmaAW, self.tdmaAP, self.tdmaAE, self.tdmaD)

    def solveNe(self, tolerance):
        converged = 10. * tolerance
        ne0 = [0.] * len(self.ne)
        while (converged > tolerance):
            for i in range(len(self.ne)):
                ne0[i] = self.ne[i]
            self.iterateNe()
            converged = 0.
            for i in range(len(self.ne)):
                testConv = abs(self.ne[i] - ne0[i]) / ne0[i]
                if (testConv > converged):
                    converged = testConv
            print str(converged) + ", " + str(self.ne[i])
    
    def electricalConductivityNonEq(self, T):
        electronXS = self.crossSection(len(self.species)-1, T)
        electronN = self.nDensity(len(self.species)-1, T)
        totalN = self.totalEqDensity(T)
        return electronN * constants.fundamentalCharge ** 2. / (math.sqrt(2. * math.pi * constants.boltzmann * T * constants.electronMass) * totalN * electronXS)
        
    def effectiveConductivity(self):
        jTest = 1e7
        
        
################################################################################

def hmean(x1, x2):
    return 2. * x1 * x2 / (x1 + x2)
    
def readSpeciesFile(fileName):
    componentArray = []
    f = open(fileName, "r")
    dataLine = f.readline()
    names = dataLine.split("\t")
    dataLine = f.readline()
    molwt = dataLine.split("\t")
    dataLine = f.readline()
    charges = dataLine.split("\t")
    dataLine = f.readline()
    radii = dataLine.split("\t")
    for i in range(len(names)):
        componentArray.append(specie(names[i], molwt[i], charges[i], radii[i]))        
    f.close()
    return componentArray

def readNumberDensityFile(fileName):
    compositionArray = []
    f = open(fileName, "r")
    while True:
        dataLine = f.readline()
        if ((dataLine == '') or (dataLine == '\n')):
            break
        else:
            data = dataLine.split("\t")
            temp = data.pop(0)
            compositionArray.append(composition(temp, data))        
    f.close()
    return compositionArray
    
def collisionCrossSection(sp1, sp2, T):
    if ((sp1.chargeNumber != 0) and (sp1.chargeNumber == sp2.chargeNumber)):
        rIon = float(sp1.chargeNumber * sp2.chargeNumber) * constants.ionClassicalRadiusFactor * constants.eVtoK / T
        return math.pi * rIon ** 2.
    else:
        return math.pi * (sp1.radius + sp2.radius) ** 2.

def solveTDMA(a, b, c, d):
    nf = len(a)
    ac, bc, cc, dc = map(numpy.array, (a, b, c, d))
    for it in xrange(1, nf):
        mc = ac[it] / bc[it-1]
        bc[it] = bc[it] - mc * cc[it-1] 
        dc[it] = dc[it] - mc * dc[it-1]

    xc = ac
    xc[-1] = dc[-1] / bc[-1]

    for il in xrange(nf-2, -1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il+1]) / bc[il]

    del bc, cc, dc

    return xc

################################################################################

