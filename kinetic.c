//A simple kinetic simulator for plasma materials, using classical mechanics
//Q Reynolds 2010

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//oxygen(0)
double atomM=15.999*1.660538782e-27; // kg
double atomR=1.4e-10; // m
double atomQ=0.; // C
//oxygen(I)
double ionM=15.999*1.660538782e-27; // kg
double ionR=2.2e-11; // m
double ionQ=1.602176487e-19; // C

static double coulombK=8.987551787e9; // Nm2/C

double T=10000; K
double 
