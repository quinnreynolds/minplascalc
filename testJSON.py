#!/usr/bin/env python3
#
# Q Reynolds 2017

import argparse
import MinPlasCalc as mpc

parser = argparse.ArgumentParser(description = "Test driver JSON file conversion using NIST website data.")
parserArgs = parser.parse_args()

mpc.buildMonatomicSpeciesJSON(
    name = "C+", 
    stoichiometry = { "C": 1 },
    molarMass = 0.012,
    chargeNumber = 1,
    ionisationEnergy = 90820.45,
    nistDataFile = "NistData/RawData/nist_C+")

