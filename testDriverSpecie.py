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

myOxygenAtom = lpc.specie(
    name = "OI", 
    chargeNumber = 0,
    energyLevelFile = "NistData/LevelDataParsed/OI.csv"
    )

myOxygenPlus = lpc.specie(
    name = "OII", 
    chargeNumber = 0,
    energyLevelFile = "NistData/LevelDataParsed/OII.csv"
    )

myOxygenPlusPlus = lpc.specie(
    name = "OIII", 
    chargeNumber = 0,
    energyLevelFile = "NistData/LevelDataParsed/OIII.csv"
    )

Temps = np.linspace(parserArgs.ts, parserArgs.te, 1000)
pFuncO = []
pFuncOp = []
pFuncOpp = []
for Temp in Temps:
    pFuncO.append(myOxygenAtom.internalPartitionFunction(Temp))
    pFuncOp.append(myOxygenPlus.internalPartitionFunction(Temp))
    pFuncOpp.append(myOxygenPlusPlus.internalPartitionFunction(Temp))

fig, ax = pyplot.subplots()

ax.plot(Temps, pFuncO)
ax.plot(Temps, pFuncOp)
ax.plot(Temps, pFuncOpp)

