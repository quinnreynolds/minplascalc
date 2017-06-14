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
parser.add_argument("-ts", help = "Temperature to start calculating at, K", type = float, default = 5000.)
parser.add_argument("-te", help = "Temperature to stop calculating at, K", type = float, default = 25000.)
parserArgs = parser.parse_args()

myOxygenGroup = lpc.elementGroup(
    element = "O", 
    chargeNumbers = [0, 1, 2],
    energyLevelFilePath = "NistData/LevelDataParsed/"
    )

Temps = np.linspace(parserArgs.ts, parserArgs.te, 1000)
pFuncO = []
pFuncOp = []
pFuncOpp = []
ratio01 = []
ratio12 = []
for Temp in Temps:
    pFuncO.append(myOxygenGroup.species[0].internalPartitionFunction(Temp))
    pFuncOp.append(myOxygenGroup.species[1].internalPartitionFunction(Temp))
    pFuncOpp.append(myOxygenGroup.species[2].internalPartitionFunction(Temp))
    ratio01.append(myOxygenGroup.sahaRHS(0, Temp))
    ratio12.append(myOxygenGroup.sahaRHS(1, Temp))
    
fig, (ax1, ax2) = pyplot.subplots(nrows = 2)

ax1.plot(Temps, pFuncO, "b-")
ax1.plot(Temps, pFuncOp, "g--")
ax1.plot(Temps, pFuncOpp, "r--")

ax2.plot(Temps, ratio01, "b")
ax2.semilogy(Temps, ratio12, "g--")
