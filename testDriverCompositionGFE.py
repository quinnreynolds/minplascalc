#!/usr/bin/env python3
#
# Q Reynolds 2017

import argparse
import ltePlasmaClasses as lpc
import numpy as np

parser = argparse.ArgumentParser(
    description = "Test driver for ltePlasmaClasses."
    )
parser.add_argument("-ts", help = "Temperature to start calculating at, K", type = float, default = 5000.)
parser.add_argument("-te", help = "Temperature to stop calculating at, K", type = float, default = 25000.)
parserArgs = parser.parse_args()

myComposition = lpc.compositionGFE(compositionFile = "Compositions/OxygenPlasma.json")
myComposition.recalcE0i()

#myComposition.recalcMatrixCoeffts(10000.)
print(myComposition.gfeMatrix)
print(myComposition.gfeVector)
