#!/usr/bin/env python3
#
# Q Reynolds 2017

import minplascalc as mpc
import pathlib
import json
import collections

sourcefields = ('title', 'author', 'publicationInfo', 'http', 'doi')
DEFSOURCE = collections.OrderedDict([(field, 'test') for field in sourcefields])


def test_buildMonatomicSpeciesJSON():
    # Demo of how to build a JSON data file for a monatomic species
    mpc.buildMonatomicSpeciesJSON(
        name="C+",
        stoichiometry={"C": 1},
        molarMass=0.0120107,
        chargeNumber=1,
        ionisationEnergy=90820.45,
        nistDataFile=str(mpc.DATAPATH / "species_raw" / "nist_C+"))

    outputfile = pathlib.Path('C+.json')
    assert outputfile.exists()
    result = json.load(outputfile.open())
    assert len(result["monatomicData"]["energyLevels"]) == 85
    outputfile.unlink()


def test_buildMonatomicSpeciesJSON_sourced():
    # Demo of how to build a JSON data file for a monatomic species
    mpc.buildMonatomicSpeciesJSON(
        name="C+",
        stoichiometry={"C": 1},
        molarMass=0.0120107,
        chargeNumber=1,
        ionisationEnergy=90820.45,
        nistDataFile=str(mpc.DATAPATH / "species_raw" / "nist_C+"),
        sources=[DEFSOURCE])

    outputfile = pathlib.Path('C+.json')
    assert outputfile.exists()
    result = json.load(outputfile.open())
    assert len(result["monatomicData"]["energyLevels"]) == 85
    outputfile.unlink()


def test_buildDiatomicSpeciesJSON():
    # Demo of how to build a JSON data file for a diatomic species
    mpc.buildDiatomicSpeciesJSON(
        name="CO",
        stoichiometry={"C": 1, "O": 1},
        molarMass=0.0280101,
        chargeNumber=0,
        ionisationEnergy=113030.54,
        dissociationEnergy=89862.00,
        sigmaS=1,
        g0=1,
        we=2169.81358,
        Be=1.93128087)

    outputfile = pathlib.Path('CO.json')
    assert outputfile.exists()
    _ = json.load(outputfile.open())
    outputfile.unlink()


def test_buildDiatomicSpeciesJSON_sourced():
    # Demo of how to build a JSON data file for a diatomic species
    mpc.buildDiatomicSpeciesJSON(
        name="CO",
        stoichiometry={"C": 1, "O": 1},
        molarMass=0.0280101,
        chargeNumber=0,
        ionisationEnergy=113030.54,
        dissociationEnergy=89862.00,
        sigmaS=1,
        g0=1,
        we=2169.81358,
        Be=1.93128087,
        sources=[DEFSOURCE])

    outputfile = pathlib.Path('CO.json')
    assert outputfile.exists()
    _ = json.load(outputfile.open())
    outputfile.unlink()
