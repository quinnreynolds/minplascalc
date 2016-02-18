#!/usr/bin/env python
#
# Lowke boundary condition model for plasma sheath at cathodes
#
# Q Reynolds 2016

import plasmaClasses as pc

nMax = 50

mp = pc.plasmaData(pc.readSpeciesFile("carbonPlasmaComponents"), pc.readNumberDensityFile("carbonPlasmaData"))

lsm = pc.lowkeSheathModel(4000., 4500., 0.01, nMax, mp)

lsm.solveNe(1e-10)

f = open("ne.csv", "w")
for i in range(nMax):
    f.write(str(lsm.x[i]) + "," + str(lsm.ne[i]) + "," + str(lsm.neq[i]) + "\n")
f.close()


