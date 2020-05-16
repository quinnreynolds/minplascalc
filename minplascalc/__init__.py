#!/usr/bin/env python3
#
# Classes for gas/plasma data and LTE calculations with simple species
#
# Q Reynolds 2017

import json
import numpy as np
import logging
import warnings
import pathlib
from copy import deepcopy
from . import constants

DATAPATH = pathlib.Path(__file__).parent / 'data'
SPECIESPATH = DATAPATH / 'species'

# utility functions ############################################################

def molar_mass_calculator(protons, neutrons, electrons):
    """Estimate the molar mass in kg/mol of a species based on its nuclear and
    electronic structure, if you can't get it anywhere else for some reason.
    """
    return constants.avogadro * (protons * constants.protonmass
                                 + electrons * constants.electronmass
                                 + neutrons * (constants.protonmass
                                               + constants.electronmass))


def parse_values(nist_line):
    """Helper function to tidy up a string of data copied from NIST online
    databases.
    """
    table = str.maketrans('', '', '+x?[]')
    line = ''.join(nist_line.split()).translate(table)
    records = line.split('|')[:-1]
    values = []
    for record in records:
        if '/' in record:
            num, den = record.split('/')
            value = float(num) / float(den)
        else:
            value = float(record)
        values.append(value)
    return values


def read_energylevels(data):
    """ Read a NIST energy level file

    Parameters
    ----------
    data : file-like
        NIST energy level file data

    Return
    ------
    energylevels : list of length-2 lists
         Energy levels. Each dict contains the energy of the level Ei and the 
         associated quantum number J.
    """
    energylevels = []
    try:
        name = data.name
    except AttributeError:
        name = 'input'
    for i, line in enumerate(data):
        try:
            j, ei = parse_values(line)
            energylevels.append([j, ei])
        except ValueError as exception:
            logging.debug('Ignoring line %i in %s', i, name)
            logging.debug(exception)

    return energylevels


# classes and class functions ##################################################

def species_to_file(sp, datafile=None):
    if datafile:
        with open(datafile, 'w') as f:
            json.dump(sp.__dict__, f, indent=4)
    else:
        with open(sp.name + '.json', 'w') as f:
            json.dump(sp.__dict__, f, indent=4)
        

def species_from_file(datafile):
    """Create a species from a data file in JSON format.

    Parameters
    ----------
    datafile : string
        Path to a JSON data file describing the electronic and molecular
        properties of the species
    """
    with open(datafile) as f:
        spdata = json.load(f)
    atomcount = sum(v for k, v in spdata['stoichiometry'].items())
    if atomcount == 1:
        return MonatomicSpecies(spdata['name'], spdata['stoichiometry'], 
                                spdata['molarmass'], spdata['chargenumber'], 
                                spdata['ionisationenergy'], 
                                spdata['energylevels'], spdata['sources'])
    else:
        return DiatomicSpecies(spdata['name'], spdata['stoichiometry'], 
                               spdata['molarmass'], spdata['chargenumber'], 
                               spdata['ionisationenergy'], 
                               spdata['dissociationenergy'], spdata['sigma_s'], 
                               spdata['g0'], spdata['w_e'], spdata['b_e'], 
                               spdata['sources'])


def species_from_name(name):
    """ Create a species from the species database

    Parameters
    ----------
    name : str
        Name of the species
    """
    filename = SPECIESPATH / (name + '.json')
    return species_from_file(str(filename))


def mixture_from_names(names, x0, T, P):
    """ Create a mixture from a list of species names using the species database

    Parameters
    ----------
    names : list of str
        Names of the species
    x0 : list of float
        Initial value of mole fractions for each species, typically the 
        room-temperature composition of the plasma-generating gas
    T : float
        LTE plasma temperature, in K
    P : float
        LTE plasma pressure, in Pa
        
    Returns
    -------
    A Mixture object instance.
    """
    if 'e' in names:
        raise ValueError('Electrons are added automatically, please don\'t '
                         'include them in your species list.')
    species = [species_from_name(nm) for nm in names]
    return Mixture(species, x0, T, P, 1e20, 1e-10, 1000)
    

class BaseSpecies:
    def partitionfunction_total(self, V, T, dE):
        return (V * self.partitionfunction_translational(T)
                * self.partitionfunction_internal(T, dE))

    def partitionfunction_translational(self, T):
        return ((2 * np.pi * self.molarmass * constants.boltzmann * T)
                / (constants.avogadro * constants.planck ** 2)) ** 1.5

    def partitionfunction_internal(self, T, dE):
        raise NotImplementedError

    def internal_energy(self, T, dE):
        raise NotImplementedError


class Species(BaseSpecies):
    def __init__(self, name, stoichiometry, molarmass, chargenumber):
        """Base class for heavy particles. Either single monatomic or diatomic 
        chemical species in the plasma, eg O2 or Si+

        Parameters
        ----------
        name : string
            A unique identifier for the species
        stoichiometry : dictionary
            Dictionary describing the elemental stoichiometry of the species
        molarmass : float
            Molar mass of the species in kg/mol
        chargenumber : int
            Charge on the species (in integer units of the fundamental charge)
        """
        self.name = name
        self.stoichiometry = deepcopy(stoichiometry)
        self.molarmass = molarmass
        self.chargenumber = chargenumber
        if self.chargenumber < 0:
            raise ValueError('Negatively charged ions not implemented.')


class MonatomicSpecies(Species):
    def __init__(self, name, stoichiometry, molarmass, chargenumber, 
                 ionisationenergy, energylevels, sources):
        """Class for monatomic plasma species (single atoms and ions).
    
        Parameters
        ----------
        name : string
            A unique identifier for the species
        stoichiometry : dictionary
            Dictionary describing the elemental stoichiometry of the species 
            (e.g. {'O': 1} for O or O+)
        molarmass : float
            Molar mass of the species in kg/mol
        chargenumber : int
            Charge on the species (in integer units of the fundamental charge)
        ionisationenergy : float
            Ionisation energy of the species in J
        energylevels : list of length-2 lists
            Atomic energy level data - each entry in the list contains a pair of
            values giving the level's quantum number and its energy 
            respectively, with energy in J
        sources : list of str
            Each entry represents a reference from which the data was
            obtained
        """
        super().__init__(name, stoichiometry, molarmass, chargenumber)
        
        self.ionisationenergy = ionisationenergy
        self.energylevels = deepcopy(energylevels)
        self.sources = deepcopy(sources)

    def partitionfunction_internal(self, T, dE):
        kbt = constants.boltzmann * T
        partitionval = 0       
        for j, eij in self.energylevels:
            if eij < (self.ionisationenergy - dE):
                partitionval += (2*j+1) * np.exp(-eij / kbt)
        return partitionval

    def internal_energy(self, T, dE):
        kbt = constants.boltzmann * T
        translationalenergy = 1.5 * kbt
        electronicenergy = 0
        for j, eij in self.energylevels:
            if eij < (self.ionisationenergy - dE):
                electronicenergy += (2*j+1) * eij * np.exp(-eij / kbt)
        electronicenergy /= self.partitionfunction_internal(T, dE)
        return translationalenergy + electronicenergy


class DiatomicSpecies(Species):
    def __init__(self, name, stoichiometry, molarmass, chargenumber,
                 ionisationenergy, dissociationenergy, sigma_s, g0, w_e, b_e, 
                 sources):
        """Class for diatomic plasma species (bonded pairs of atoms, as 
        neutral particles or ions).
    
        Parameters
        ----------
        name : string
            A unique identifier for the species
        stoichiometry : dictionary
            Dictionary describing the elemental stoichiometry of the species 
            (e.g. {'Si': 1, 'O': 1} for SiO or SiO+)
        molarmass : float
            Molar mass of the species in kg/mol
        chargenumber : int
            Charge on the species (in integer units of the fundamental charge)
        ionisationenergy : float
            Ionisation energy of the species in J
        dissociationenergy : float
            Dissociation energy of the species in J
        sigma_s : int
            Symmetry constant (=2 for homonuclear molecules, =1 for 
            heteronuclear)
        g0 : float
            Ground state electronic energy level degeneracy
        w_e : float
            Vibrational energy level constant in J
        b_e : float
            Rotational energy level constant in J
        sources : list of str
            Each dictionary represents a reference source from which the data 
            was obtained
        """
        super().__init__(name, stoichiometry, molarmass, chargenumber)

        self.dissociationenergy = dissociationenergy
        self.ionisationenergy = ionisationenergy
        self.sigma_s = sigma_s
        self.g0 = g0
        self.w_e = w_e
        self.b_e = b_e
        self.sources = deepcopy(sources)

    def partitionfunction_internal(self, T, dE):
        kbt = constants.boltzmann * T
        electronicpartition = self.g0
        vibrationalpartition = 1 / (1 - np.exp(-self.w_e / kbt))
        rotationalpartition = kbt / (self.sigma_s * self.b_e)
        return electronicpartition * vibrationalpartition * rotationalpartition

    def internal_energy(self, T, dE):
        kbt = constants.boltzmann * T
        translationalenergy = 1.5 * kbt
        electronicenergy = 0
        rotationalenergy = kbt
        vibrationalenergy = self.w_e * (np.exp(-self.w_e / kbt)
                                        / (1 - np.exp(-self.w_e / kbt)))
        return (translationalenergy + electronicenergy + rotationalenergy 
                + vibrationalenergy)


class ElectronSpecies(BaseSpecies):
    def __init__(self):
        """Class for electrons as a plasma species.
        """
        self.name = 'e'
        self.stoichiometry = {}
        self.molarmass = constants.electronmass * constants.avogadro
        self.chargenumber = -1

    # noinspection PyUnusedLocal
    def partitionfunction_internal(self, T, dE):
        return 2.

    def internal_energy(self, T, dE):
        translationalenergy = 1.5 * constants.boltzmann * T
        electronicenergy = 0
        return translationalenergy + electronicenergy


class Mixture:
    def __init__(self, species, x0, T, P, gfe_ni0, gfe_reltol, gfe_maxiter):
        """Class representing a thermal plasma specification with multiple
        species, and methods for calculating equilibrium species concentrations
        at different temperatures and pressures using the principle of Gibbs
        free energy minimisation.

        Parameters
        ----------
        species : list or tuple of obj
            All species participating in the mixture (excluding electrons which
            are added automatically), as minplascalc Species objects
        x0 : list or tuple of float
            Constraint mole fractions for each species, typically the 
            room-temperature composition of the plasma-generating gas
        T : float
            LTE plasma temperature, in K
        P : float
            LTE plasma pressure, in Pa            
        gfe_ni0 : float
            Gibbs Free Energy minimiser solution control: Starting estimate for 
            number of particles of each species. Typically O(1e20).         
        gfe_reltol : float
            Gibbs Free Energy minimiser solution control: Relative tolerance at
            which solution for particle numbers is considered converged. 
            Typically O(1e-10).
        gfe_maxiter : int
            Gibbs Free Energy minimiser solution control: Bailout loop count 
            value for iterative solver. Typically O(1e3).
        """
        if 'e' in [sp.name for sp in species]:
            raise ValueError('Electrons are added automatically, please don\'t '
                             'include them in your species list.')
        if len(species) != len(x0):
            raise ValueError('Lists species and x0 must be the same length.')
        self.__species = tuple(list(species) + [ElectronSpecies()])
        self.x0 = x0
        self.T = T
        self.P = P
        self.gfe_ni0 = gfe_ni0
        self.gfe_reltol = gfe_reltol
        self.gfe_maxiter = gfe_maxiter
        self.__isLTE = False
    
    @property
    def species(self):
        return self.__species
    
    @species.setter
    def species(self, species):
        raise TypeError('Attribute species is read-only. Please create a new '
                        'Mixture object if you wish to change the plasma '
                        'species.')

    @property
    def x0(self):
        return self.__x0
    
    @x0.setter
    def x0(self, x0):
        if len(x0) == len(self.species)-1:
            self.__isLTE = False
            self.__x0 = tuple(list(x0) + [0])
        else:
            raise ValueError('Please specify constraint mole fractions for all '
                             'species except electrons.')
        
    @property
    def T(self):
        return self.__T
    
    @T.setter
    def T(self, T):
        self.__isLTE = False
        self.__T = T
        
    @property
    def P(self):
        return self.__P
    
    @P.setter
    def P(self, P):
        self.__isLTE = False
        self.__P = P
    
    def __recalcE0i(self):
        """Calculate the ionisation energy lowering, using limitation theory of
        Stewart & Pyatt 1966
        """
        kbt = constants.boltzmann * self.T
        ndi = self.__ni * self.P / (self.__ni.sum() * kbt) 
        weightedchargesumsqd, weightedchargesum = 0, 0
        for sp, nd in zip(self.species, ndi):
            if sp.chargenumber > 0:
                weightedchargesum += nd * sp.chargenumber
                weightedchargesumsqd += nd * sp.chargenumber ** 2
        zstar = weightedchargesumsqd / weightedchargesum
        debyed3 = (kbt / (4 * np.pi * (zstar + 1) * ndi[-1] 
                          * constants.fundamentalcharge ** 2)) ** (3/2)
        for i, sp in enumerate(self.species):
            if sp.name != 'e':
                ai3 = 3 * (sp.chargenumber + 1) / (4 * np.pi * ndi[-1])
                de = kbt * ((ai3/debyed3 + 1) ** (2/3) - 1) / (2 * (zstar + 1))
                self.__dE[i] = de
        for cn in range(1, max(sp.chargenumber for sp in self.species) + 1):
            for i, (sp, ifrom) in enumerate(zip(self.species, 
                                                self.__ionisedfrom)):
                if sp.chargenumber == cn:
                    spfrom = self.species[ifrom]
                    self.__E0[i] = (self.__E0[ifrom] + spfrom.ionisationenergy 
                                    - self.__dE[ifrom])

    def calculate_composition(self):
        """Calculate the LTE composition of the plasma in particles/m3.
        
        Returns
        -------
        ndarray
            Number density of each species in the plasma as listed in 
            Mixture.species, in particles/m3
        """
        nspecies = len(self.species)
        kbt = constants.boltzmann*self.T

        self.__E0, self.__dE = np.zeros(nspecies), np.zeros(nspecies)
        for i, sp in enumerate(self.species):
            if sum(dv for kv, dv in sp.stoichiometry.items()) == 2:
                self.__E0[i] = -sp.dissociationenergy
        self.__ionisedfrom = [None] * nspecies
        for i, sp in enumerate(self.species):
            if sp.chargenumber > 0:
                for sp2 in self.species:
                    if (sp2.stoichiometry == sp.stoichiometry 
                        and sp2.chargenumber == sp.chargenumber-1):
                        for j, sp3 in enumerate(self.species):
                            if sp2.name == sp3.name:
                                self.__ionisedfrom[i] = j
        elements = [{'name': nm, 'stoichcoeff': None, 'ntot': 0}
                    for nm in sorted(set(s for sp in self.species
                                         for s in sp.stoichiometry))]
        for elm in elements:
            elm['stoichcoeff'] = [sp.stoichiometry.get(elm['name'], 0)
                                  for sp in self.species]
        for elm in elements:
            elm['ntot'] = sum(1e24 * c * x0loc
                              for c, x0loc in zip(elm['stoichcoeff'], self.x0))
        minimiser_dof = nspecies + len(elements) + 1
        gfematrix = np.zeros((minimiser_dof, minimiser_dof))
        gfevector = np.zeros(minimiser_dof)
        for i, elm in enumerate(elements):
            gfevector[nspecies + i] = elm['ntot']
            for j, sc in enumerate(elm['stoichcoeff']):
                gfematrix[nspecies + i, j] = sc
                gfematrix[j, nspecies + i] = sc
        for j, qc in enumerate(sp.chargenumber for sp in self.species):
            gfematrix[-1, j] = qc
            gfematrix[j, -1] = qc

        if not self.__isLTE:
            self.__ni = np.full(nspecies, self.gfe_ni0)
            governorfactors = np.linspace(0.9, 0.1, 9)
            successyn = False
            governoriters = 0
            while not successyn and governoriters < len(governorfactors):
                successyn = True
                governorfactor = governorfactors[governoriters]
                reltol = self.gfe_reltol * 10
                minimiseriters = 0
                while reltol > self.gfe_reltol:
                    self.__recalcE0i()
                    nisum = self.__ni.sum()
                    V = nisum * kbt / self.P
                    offdiag, ondiag = -kbt/nisum, np.diag(kbt/self.__ni)
                    gfematrix[:nspecies, :nspecies] = offdiag + ondiag
                    total = [sp.partitionfunction_total(V, self.T, dE) 
                             for sp, dE in zip(self.species, self.__dE)]
                    mu = -kbt * np.log(total/self.__ni) + self.__E0
                    gfevector[:nspecies] = -mu
                    solution = np.linalg.solve(gfematrix, gfevector)
                    newni = solution[0:nspecies]
                    deltani = abs(newni - self.__ni)
                    maxniindex = newni.argmax()
                    reltol = deltani[maxniindex] / solution[maxniindex]
    
                    maxalloweddeltani = governorfactor * self.__ni
                    deltani = deltani.clip(min=maxalloweddeltani)
                    newrelaxfactors = maxalloweddeltani / deltani
                    relaxfactor = newrelaxfactors.min()
                    self.__ni = (1-relaxfactor)*self.__ni + relaxfactor*newni
    
                    minimiseriters += 1
                    if minimiseriters > self.gfe_maxiter:
                        successyn = False
                        break 
                governoriters += 1
            if not successyn:
                warnings.warn('Minimiser could not find a converged solution, '
                              'results may be inaccurate.')
            # noinspection PyUnboundLocalVariable
            logging.debug(governoriters, relaxfactor, reltol)
            logging.debug(self.__ni)
        self.__isLTE = True
        return self.__ni*self.P / (self.__ni.sum()*kbt) 

    def calculate_density(self):
        """Calculate the LTE density of the plasma.

        Returns
        -------
        float
            Fluid density, in kg/m3
        """
        ndi = self.calculate_composition()
        return sum(nd * sp.molarmass / constants.avogadro
                   for sp, nd in zip(self.species, ndi))

    def calculate_enthalpy(self):
        """Calculate the LTE enthalpy of the plasma. 
        
        The value returned is relative to an arbitrary reference level which 
        may be negative, zero, or positive depending on the reference energies 
        of the plasma species present.

        Returns
        -------
        float
            Enthalpy, in J/kg
        """
        ndi = self.calculate_composition()
        weightedenthalpy = sum(constants.avogadro * nd 
                               * (sp.internal_energy(self.T, dE) + E0 
                                  + constants.boltzmann * self.T) 
                               for sp, nd, dE, E0 in zip(self.species, ndi, 
                                                         self.__dE, self.__E0))
        weightedmolmass = sum(nd * sp.molarmass
                              for sp, nd in zip(self.species, ndi))
        return weightedenthalpy / weightedmolmass

    def calculate_heat_capacity(self, rel_delta_T=0.001):
        """Calculate the LTE heat capacity at constant pressure of the plasma 
        based on current conditions and species composition. 
        
        This is done by performing multiple LTE composition recalculations and 
        can be time-consuming to execute - when performing large quantities of 
        Cp calculations at different temperatures, it is more efficient to 
        calculate enthalpies and perform a numerical derivative external to 
        minplascalc.

        Returns
        -------
        float
            Heat capacity, in J/kg.K
        """
        startT = self.T
        self.T = startT * (1-rel_delta_T)        
        enthalpylow = self.calculate_enthalpy()
        self.T = startT * (1+rel_delta_T)        
        enthalpyhigh = self.calculate_enthalpy()
        self.T = startT
        return (enthalpyhigh - enthalpylow) / (2 * rel_delta_T * self.T)

    def calculate_viscosity(self):
        """Calculate the LTE viscosity of the plasma in Pa.s based on current
        conditions and species composition.
        """

        raise NotImplementedError

    def calculate_thermal_conductivity(self):
        """Calculate the LTE thermal conductivity of the plasma in W/m.K.
        """

        raise NotImplementedError

    def calculate_electrical_conductivity(self):
        """Calculate the LTE electrical conductivity of the plasma in 1/ohm.m.
        """

        raise NotImplementedError

    def calculate_total_emission_coefficient(self):
        """Calculate the LTE total radiation emission coefficient of the plasma 
        in W/m3.
        """

        raise NotImplementedError
