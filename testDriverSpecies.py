#!/usr/bin/env python3
#
# Q Reynolds 2017

import argparse
import numpy as np
from matplotlib import pyplot
import MinPlasCalc as mpc

parser = argparse.ArgumentParser(description="Test driver for MinPlasCalc - oxygen plasma species calculations.")
parser.add_argument("-ts", help="Temperature to start calculating at, K",
                    type=float, default=1000.)
parser.add_argument("-te", help="Temperature to stop calculating at, K",
                    type=float, default=25000.)
parserArgs = parser.parse_args()


# Load up some species
myOxygenMolecule = mpc.Species(dataFile ="NistData/O2.json")
myOxygenAtom = mpc.Species(dataFile ="NistData/O.json")
myOxygenPlus = mpc.Species(dataFile ="NistData/O+.json")
myOxygenPlusPlus = mpc.Species(dataFile ="NistData/O++.json")


# Calculate their internal partition functions
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


# Draw a nice graph
fig, ax = pyplot.subplots()

ax.semilogy(Temps, pFuncOO)
ax.semilogy(Temps, pFuncO)
ax.semilogy(Temps, pFuncOp)
ax.semilogy(Temps, pFuncOpp)

pyplot.show()