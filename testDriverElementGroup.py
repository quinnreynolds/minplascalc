#!/usr/bin/env python3
#
# Q Reynolds 2017

import argparse
import numpy as np
import timeit
from matplotlib import pyplot
import ltePlasmaClasses as lpc


parser = argparse.ArgumentParser(
    description = "Test driver for ltePlasmaClasses."
    )
parser.add_argument("-ts", help = "Temperature to start calculating at, K", type = float, default = 5000.)
parser.add_argument("-te", help = "Temperature to stop calculating at, K", type = float, default = 25000.)
parserArgs = parser.parse_args()

myOxygenGroup = lpc.slagElementGroup(
    element = "O", 
    maximumIonCharge = 2,
    dataFilePath = "NistData/"
    )

time1 = timeit.default_timer()

Temps = np.linspace(parserArgs.ts, parserArgs.te, 1000)
pFuncOO = []
pFuncO = []
pFuncOp = []
pFuncOpp = []
ratio01 = []
ratio12 = []
ratio23 = []
for Temp in Temps:
    pFuncOO.append(myOxygenGroup.species[0].internalPartitionFunction(Temp))
    pFuncO.append(myOxygenGroup.species[1].internalPartitionFunction(Temp))
    pFuncOp.append(myOxygenGroup.species[2].internalPartitionFunction(Temp))
    pFuncOpp.append(myOxygenGroup.species[3].internalPartitionFunction(Temp))
    ratio01.append(myOxygenGroup._equilibriumRHS(0, Temp))
    ratio12.append(myOxygenGroup._equilibriumRHS(1, Temp))
    ratio23.append(myOxygenGroup._equilibriumRHS(2, Temp))
    
print(timeit.default_timer() - time1)

fig, (ax1, ax2) = pyplot.subplots(nrows = 2)

ax1.semilogy(Temps, pFuncOO)
ax1.semilogy(Temps, pFuncO)
ax1.semilogy(Temps, pFuncOp)
ax1.semilogy(Temps, pFuncOpp)

ax2.semilogy(Temps, ratio01)
ax2.semilogy(Temps, ratio12)
ax2.semilogy(Temps, ratio23)

pyplot.show()
