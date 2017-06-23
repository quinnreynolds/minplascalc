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
parser.add_argument("-te", help = "Temperature to stop calculating at, K", type = float, default = 35000.)
parserArgs = parser.parse_args()

myOxygenMolecule = lpc.specie(dataFile = "NistData/OO.json")
myOxygenAtom = lpc.specie(dataFile = "NistData/OI.json")
myOxygenPlus = lpc.specie(dataFile = "NistData/OII.json")
myOxygenPlusPlus = lpc.specie(dataFile = "NistData/OIII.json")

print(myOxygenPlus.stoichiometry.get("O", 0))
print(myOxygenPlus.stoichiometry.get("Si", 0))

Temps = np.linspace(parserArgs.ts, parserArgs.te, 1000)
pFuncOO = []
pFuncO = []
pFuncOp = []
pFuncOpp = []
for Temp in Temps:
    pFuncOO.append(myOxygenMolecule.internalPartitionFunction(Temp))
    pFuncO.append(myOxygenAtom.internalPartitionFunction(Temp))
    pFuncOp.append(myOxygenPlus.internalPartitionFunction(Temp))
    pFuncOpp.append(myOxygenPlusPlus.internalPartitionFunction(Temp))

fig, ax = pyplot.subplots()

ax.semilogy(Temps, pFuncOO)
ax.semilogy(Temps, pFuncO)
ax.semilogy(Temps, pFuncOp)
ax.semilogy(Temps, pFuncOpp)

pyplot.show()
