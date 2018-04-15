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
    speciesfile = mpc.DATAPATH / "species_raw" / "nist_C+"
    energylevels = mpc.read_energylevels(speciesfile.open())
    result = mpc.buildMonatomicSpeciesJSON(
        name="C+",
        stoichiometry={"C": 1},
        molarmass=0.0120107,
        chargenumber=1,
        ionisationenergy=90820.45,
        energylevels=energylevels)

    assert len(result["monatomicData"]["energyLevels"]) == 85


def test_buildMonatomicSpeciesJSON_sourced():
    # Demo of how to build a JSON data file for a monatomic species
    speciesfile = mpc.DATAPATH / "species_raw" / "nist_C+"
    energylevels = mpc.read_energylevels(speciesfile.open())

    result = mpc.buildMonatomicSpeciesJSON(
        name="C+",
        stoichiometry={"C": 1},
        molarmass=0.0120107,
        chargenumber=1,
        ionisationenergy=90820.45,
        energylevels=energylevels,
        sources=[DEFSOURCE])

    assert len(result["monatomicData"]["energyLevels"]) == 85
    assert result["sources"] is not None


def test_buildDiatomicSpeciesJSON():
    # Demo of how to build a JSON data file for a diatomic species
    result = mpc.buildDiatomicSpeciesJSON(
        name="CO",
        stoichiometry={"C": 1, "O": 1},
        molarmass=0.0280101,
        chargenumber=0,
        ionisationenergy=113030.54,
        dissociationenergy=89862.00,
        sigma_s=1,
        g0=1,
        w_e=2169.81358,
        b_e=1.93128087)


def test_buildDiatomicSpeciesJSON_sourced():
    # Demo of how to build a JSON data file for a diatomic species
    result = mpc.buildDiatomicSpeciesJSON(
        name="CO",
        stoichiometry={"C": 1, "O": 1},
        molarmass=0.0280101,
        chargenumber=0,
        ionisationenergy=113030.54,
        dissociationenergy=89862.00,
        sigma_s=1,
        g0=1,
        w_e=2169.81358,
        b_e=1.93128087,
        sources=[DEFSOURCE])
