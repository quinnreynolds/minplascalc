#!/usr/bin/env python3
#
# Q Reynolds 2017

import argparse
import numpy as np
import ltePlasmaClasses as lpc

parser = argparse.ArgumentParser(
    description = "Test driver for ltePlasmaClasses."
    )
parser.add_argument("-ts", help = "Temperature to start calculating at, K", type = float, default = 5000.)
parser.add_argument("-te", help = "Temperature to stop calculating at, K", type = float, default = 25000.)
parser.add_argument("-p", help = "Pressure to calculate at, Pa", type = float, default = 101325.)
parserArgs = parser.parse_args()

myComposition = lpc.compositionGFE(
    compositionFile = "Compositions/OxygenPlasma5sp.json",
    T = parserArgs.ts,
    P = parserArgs.p)

myComposition.initialiseNi([1e23 for i in range(len(myComposition.species))])

myComposition.solveGfe()

for spKey, sp in myComposition.species.items():
    print(sp.name, sp.numberDensity)