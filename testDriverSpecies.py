#!/usr/bin/env python3
#
# Q Reynolds 2017

import argparse
import numpy as np
from matplotlib import pyplot
import MinPlasCalc as mpc

parser = argparse.ArgumentParser(
    description="Test driver for MinPlasCalc - "
                "oxygen plasma species calculations.")
parser.add_argument("-ts", help="Temperature to start calculating at, K",
                    type=float, default=1000.)
parser.add_argument("-te", help="Temperature to stop calculating at, K",
                    type=float, default=25000.)
args = parser.parse_args()


# Load up some species
species = [mpc.species_from_file(f) for f in ["NistData/O2.json",
                                              "NistData/O.json",
                                              "NistData/O+.json",
                                              "NistData/O++.json"]]

# Calculate their internal partition functions
temperatures = np.linspace(args.ts, args.te, 1000)

pfuncs = [[sp.internalPartitionFunction(temperature) for sp in species]
          for temperature in temperatures]

# Draw a nice graph
fig, ax = pyplot.subplots()

ax.semilogy(temperatures, pfuncs)

pyplot.show()
