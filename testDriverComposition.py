#!/usr/bin/env python3
#
# Q Reynolds 2017

import argparse
import numpy as np
from matplotlib import pyplot
import minplascalc as mpc

parser = argparse.ArgumentParser(description="Test driver for minplascalc - simple oxygen plasma composition calculation.")
parser.add_argument("-ts", help="Temperature to start calculating at, K",
                    type=float, default=1000.)
parser.add_argument("-te", help="Temperature to stop calculating at, K",
                    type=float, default=25000.)
parser.add_argument("-p", help="Pressure to calculate at, Pa",
                    type=float, default=101325.)
args = parser.parse_args()


# Instantiate a plasma mixture object using data from a JSON file.
# T, P are just initial placeholder values, can be changed at any time.
mixture = mpc.Mixture(mixture_file="mixtures/OxygenPlasma5sp.json",
                      T=args.ts,
                      P=args.p)

# Run the GFE minimiser calculation at a range of temperatures, and calculate 
# the plasma density
temperatures = np.linspace(args.ts, args.te, num=100)
ndi = [[] for j in range(len(mixture.species))]
plotText = []
density = []
for T in temperatures:
    mixture.initialiseNi([1e20] * len(mixture.species))
    mixture.T = T
    mixture.solveGfe()

    density.append(mixture.calculateDensity())
    for j, sp in enumerate(mixture.species):
        ndi[j].append(sp.numberDensity)
        plotText.append(sp.name)

# Draw a graph of the results
fig, (ax1, ax2) = pyplot.subplots(2, 1, figsize=[7.5, 10], sharex=True)

ax1.set_ylabel(r"$n_i, m^{-3}$")
ax1.set_ylim(1e15, 5e25)

# NB: These are specific to the 5-species test case at 100 kPa
positionIndexT = [5, 30, 25, 15, 80, 75]
positionFactorN = [2, 2, 2, 0.25, 0.25, 2]
plotColours = ["blue", "red", "green", "darkcyan", "darkred", "y"]

for j in range(len(ndi)):
    ax1.semilogy(temperatures, ndi[j], lw=2, color=plotColours[j])

for j in range(len(positionIndexT)):
    ax1.text(
        temperatures[positionIndexT[j]], 
        positionFactorN[j] * ndi[j][positionIndexT[j]], 
        plotText[j], 
        fontdict={"color": plotColours[j]})

ax2.set_xlabel(r"$T, K$")
ax2.set_ylabel(r"$\rho, kg/m^3$")
ax2.semilogy(temperatures, density, lw=2)

pyplot.show()

