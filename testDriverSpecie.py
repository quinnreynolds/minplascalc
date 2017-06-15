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

myOxygenMolecule = lpc.diatomicSpecie(
    name = "OO", 
    chargeNumber = 0,
    dataFile = "NistData/OO.csv"
    )

myOxygenAtom = lpc.monatomicSpecie(
    name = "OI", 
    chargeNumber = 0,
    dataFile = "NistData/OI.csv"
    )

myOxygenPlus = lpc.monatomicSpecie(
    name = "OII", 
    chargeNumber = 1,
    dataFile = "NistData/OII.csv"
    )

myOxygenPlusPlus = lpc.monatomicSpecie(
    name = "OIII", 
    chargeNumber = 2,
    dataFile = "NistData/OIII.csv"
    )

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
