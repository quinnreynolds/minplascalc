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
parserArgs = parser.parse_args()

myComposition = lpc.compositionGFE(
    compositionFile = "Compositions/OxygenPlasma3sp.json",
    T = 10000.,
    P = 101325.)
    
for key, sp in myComposition.species.items():
    print(key, sp.numberDensity)

for key, elm in myComposition.elements.items():
    print(elm.stoichiometricCoeffts, elm.totalNumber)
print(myComposition.chargeCoeffts, 0.)

myComposition.calculateGFE()
