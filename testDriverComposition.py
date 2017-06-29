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

myComposition = mpc.compositionGFE(
    compositionFile = "Compositions/OxygenPlasma5sp.json",
    T = parserArgs.ts,
    P = parserArgs.p)

temperatures = np.linspace(parserArgs.ts, parserArgs.te, num = 100)

nO2 = []
nO = []
nO2plus = []
nOplus = []
nOplusplus = []
ne = []

for T in temperatures:
    myComposition.T = T
    myComposition.initialiseNi([1e23 for i in range(len(myComposition.species))])
    myComposition.solveGfe()

    nO2.append(myComposition.species["O2"].numberDensity)
    nO.append(myComposition.species["O"].numberDensity)
    nO2plus.append(myComposition.species["O2+"].numberDensity)
    nOplus.append(myComposition.species["O+"].numberDensity)
    nOplusplus.append(myComposition.species["O++"].numberDensity)
    ne.append(myComposition.species["e"].numberDensity)

fig, ax = pyplot.subplots()

ax.set_ylim(1e15, 2e25)

ax.semilogy(temperatures, nO2)
ax.semilogy(temperatures, nO)
ax.semilogy(temperatures, nO2plus)
ax.semilogy(temperatures, nOplus)
ax.semilogy(temperatures, nOplusplus)
ax.semilogy(temperatures, ne)

pyplot.show()
