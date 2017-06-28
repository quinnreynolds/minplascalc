#!/usr/bin/env python3
#
# Q Reynolds 2017

import argparse
import numpy as np
from matplotlib import pyplot
import ltePlasmaClasses as lpc


parser = argparse.ArgumentParser(
    description = "Test driver for ltePlasmaClasses."
    )
parser.add_argument("-ts", help = "Temperature to start calculating at, K", type = float, default = 500.)
parser.add_argument("-te", help = "Temperature to stop calculating at, K", type = float, default = 25000.)
parserArgs = parser.parse_args()


myOxygenMolecule = lpc.specie(dataFile = "NistData/O2.json")
myOxygenAtom = lpc.specie(dataFile = "NistData/O.json")
temperatures = np.linspace(parserArgs.ts, parserArgs.te, 100)

pressure = 101325.
totalOxygens = 2. * pressure / (lpc.constants.boltzmann * temperatures[0])
print("totalOxygens = " + str(totalOxygens))

gfeMatrix = np.zeros((3, 3))
gfeVector = np.zeros(3)

gfeMatrix[0][2] = 2.
gfeMatrix[1][2] = 1. 
gfeMatrix[2][0] = 2.
gfeMatrix[2][1] = 1. 
gfeVector[2] = totalOxygens

def recalcCoeffts(newNi, T):
    V = newNi.sum() * lpc.constants.boltzmann * T / pressure
    totalPartitionO2 = V * myOxygenMolecule.translationalPartitionFunction(T) * myOxygenMolecule.internalPartitionFunction(T)
    totalPartitionO = V * myOxygenAtom.translationalPartitionFunction(T) * myOxygenAtom.internalPartitionFunction(T)

    muO2 = -lpc.constants.boltzmann * T * np.log(totalPartitionO2 / newNi[0]) - myOxygenMolecule.dissociationEnergy
    muO = -lpc.constants.boltzmann * T * np.log(totalPartitionO / newNi[1])
    dmuIdJ = -lpc.constants.boltzmann * T / newNi.sum()
    dmuO2dO2 = lpc.constants.boltzmann * T / newNi[0] + dmuIdJ
    dmuOdO = lpc.constants.boltzmann * T / newNi[1] + dmuIdJ
    
    gfeMatrix[0,0] = dmuO2dO2
    gfeMatrix[0,1] = dmuIdJ
    gfeMatrix[1,0] = dmuIdJ
    gfeMatrix[1,1] = dmuOdO
    
    gfeVector[0] = -muO2 + dmuO2dO2 * newNi[0] + dmuIdJ * newNi[1]
    gfeVector[1] = -muO + dmuIdJ * newNi[0] + dmuOdO * newNi[1]

print(gfeMatrix)
print(gfeVector)

ni = np.array([1e23, 1e23])

governorFactor = 0.75
nO2 = []
nO = []
ni = np.array([1e23, 1e23])
for temperature in temperatures:
    for n in range(100):
        recalcCoeffts(ni, temperature)

        x = np.linalg.solve(gfeMatrix, gfeVector)
        
        deltaNi = abs(x[0:2] - ni)
        maxDeltaNi = governorFactor * ni
        newRelaxFactors = maxDeltaNi / deltaNi
        relaxFactor = min(newRelaxFactors.min(), 1.)
        
        ni = (1 - relaxFactor) * ni + relaxFactor * x[0:2]
        
        #print(relaxFactor, ni)
        
    print(x)
    
    V = ni.sum() * lpc.constants.boltzmann * temperature / pressure
    nO2.append(ni[0] / V)
    nO.append(ni[1] / V)
    
fig, ax1 = pyplot.subplots(1, 1, sharex = True)

ax1.semilogy(temperatures, nO2)
ax1.semilogy(temperatures, nO)

pyplot.show()
