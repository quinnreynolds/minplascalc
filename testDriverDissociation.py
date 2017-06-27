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
parser.add_argument("-ts", help = "Temperature to start calculating at, K", type = float, default = 1000.)
parser.add_argument("-te", help = "Temperature to stop calculating at, K", type = float, default = 25000.)
parserArgs = parser.parse_args()


myOxygenMolecule = lpc.specie(dataFile = "NistData/O2.json")
myOxygenAtom = lpc.specie(dataFile = "NistData/O.json")



Temps = np.linspace(parserArgs.ts, parserArgs.te, 100)

nT = []
nO = []
nO2 = []
for Temp in Temps:    
    numberRatioTerm1 = myOxygenAtom.totalPartitionFunction(Temp) ** 2 / myOxygenMolecule.totalPartitionFunction(Temp)
    numberRatioTerm2 = np.exp(-myOxygenMolecule.dissociationEnergy / (lpc.constants.boltzmann * Temp))
    nR = numberRatioTerm1 * numberRatioTerm2

    nT.append(101325. / (lpc.constants.boltzmann * Temp))
    nO.append(0.5 * (np.sqrt(nR ** 2 + 4 * nT[-1] * nR) - nR))
    nO2.append(nT[-1] - nO[-1])
    
fig, ax2 = pyplot.subplots(1, 1)

ax2.semilogy(Temps, nO)
ax2.semilogy(Temps, nO2)
ax2.semilogy(Temps, nT)

pyplot.show()
