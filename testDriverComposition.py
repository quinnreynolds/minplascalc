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

myComposition = lpc.slagComposition(
    speciesDictionary = { "O": [2, 1.] },
    energyLevelFilePath = "NistData/LevelDataParsed/"
    )

Temps = np.linspace(parserArgs.ts, parserArgs.te, 1000)
ratio01 = []
ratio12 = []
for Temp in Temps:
    myComposition.elementGroups[0].recalcSahaTerms(Temp)
    ratio01.append(myComposition.elementGroups[0].sahaTerms[0])
    ratio12.append(myComposition.elementGroups[0].sahaTerms[1])
    
fig, ax1 = pyplot.subplots()

ax1.plot(Temps, ratio01, "b")
ax1.semilogy(Temps, ratio12, "g--")

pyplot.show()
