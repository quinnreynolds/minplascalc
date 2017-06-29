#!/usr/bin/env python3
#
# Q Reynolds 2017

import argparse
import numpy as np
from matplotlib import pyplot
import MinPlasCalc as mpc

parser = argparse.ArgumentParser(description = "Test driver for MinPlasCalc - simple oxygen plasma composition calculation.")
parser.add_argument("-ts", help = "Temperature to start calculating at, K", type = float, default = 1000.)
parser.add_argument("-te", help = "Temperature to stop calculating at, K", type = float, default = 25000.)
parser.add_argument("-p", help = "Pressure to calculate at, Pa", type = float, default = 101325.)
parserArgs = parser.parse_args()


# Set up the composition class using a JSON input file.
# T, P are just initial placeholder values, can be changed at any time.
myComposition = mpc.compositionGFE(
    compositionFile = "Compositions/OxygenPlasma5sp.json",
    T = parserArgs.ts,
    P = parserArgs.p)


# Run the GFE minimiser calculation at a range of temperatures
temperatures = np.linspace(parserArgs.ts, parserArgs.te, num = 100)
ndi = [ [] for j in range(len(myComposition.species)) ]
plotText = []
for T in temperatures:
    myComposition.initialiseNi([1e23 for i in range(len(myComposition.species))])
    myComposition.T = T
    myComposition.solveGfe(governorFactor = 0.7)

    for j, spKey in enumerate(myComposition.species):
        ndi[j].append(myComposition.species[spKey].numberDensity)
        plotText.append(spKey)


# Draw a graph of the results
fig, ax = pyplot.subplots()

ax.set_xlabel(r"$T, K$")
ax.set_ylabel(r"$n_i, m^{-3}$")
ax.set_ylim(1e15, 5e25)

positionIndexT = [5, 30, 25, 15, 80, 75]
positionFactorN = [2, 2, 2, 0.25, 0.25, 2]
plotColours = ["blue", "red", "green", "darkcyan", "darkred", "y"]

for j in range(len(ndi)):
    ax.semilogy(temperatures, ndi[j], lw = 2, color = plotColours[j])

for j in range(len(positionIndexT)):
    ax.text(
        temperatures[positionIndexT[j]], 
        positionFactorN[j] * ndi[j][positionIndexT[j]], 
        plotText[j], 
        fontdict = {"color": plotColours[j]} )

pyplot.show()

