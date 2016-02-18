#!/usr/bin/env python
#
# Lowke boundary condition model for plasma sheath at cathodes
#
# Q Reynolds 2016

import plasmaClasses as pc

myComp1 = pc.component("C[+]", 12.011, 1, 29e-12)
myComp2 = pc.component("C", 12.011, 0, 67e-12)

print myComp1.mass()
print myComp1.charge()

print myComp1.name

print pc.collisionCrossSection(myComp1, myComp1, 4500.)
print pc.collisionCrossSection(myComp1, myComp2, 4500.)
