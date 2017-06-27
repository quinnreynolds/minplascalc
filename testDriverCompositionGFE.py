#!/usr/bin/env python3
#
# Q Reynolds 2017

import argparse
import ltePlasmaClasses as lpc

parser = argparse.ArgumentParser(
    description = "Test driver for ltePlasmaClasses."
    )
parser.add_argument("-ts", help = "Temperature to start calculating at, K", type = float, default = 5000.)
parser.add_argument("-te", help = "Temperature to stop calculating at, K", type = float, default = 25000.)
parserArgs = parser.parse_args()

myComposition = lpc.compositionGFE(
    compositionFile = "Compositions/OxygenPlasma4sp.json",
    T = 5000.,
    P = 101325.)
    
myComposition.recalcE0i()

for niter in range(50):
    myComposition.solveGFE()

print(myComposition.gfeMatrix)
print(myComposition.gfeVector)

for key, sp in myComposition.species.items():
    print(key, sp.numberDensity)