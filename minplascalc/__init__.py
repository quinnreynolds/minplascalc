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
            raise ValueError('Error! Negatively charged ions not implemented'
                             ' yet.')


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
        vibrationalpartition = 1. / (1. - np.exp(-self.w_e / kbt))
        rotationalpartition = kbt / (self.sigma_s * self.b_e)
        return electronicpartition * vibrationalpartition * rotationalpartition

    def internal_energy(self, T, dE):
        kbt = constants.boltzmann * T
        translationalenergy = 1.5 * kbt
        electronicenergy = 0
        rotationalenergy = kbt
        vibrationalenergy = self.w_e * (np.exp(-self.w_e / kbt)
                                        / (1. - np.exp(-self.w_e / kbt)))
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
        species : list of obj
            All species participating in the mixture (excluding electrons which
            are added automatically), as minplascalc Species objects
        x0 : list of float
            Initial value of mole fractions for each species, typically the 
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
        self.species = deepcopy(species)
        self.species.append(ElectronSpecies())
        nspecies = len(self.species)
        self.x0 = np.zeros(nspecies)
        self.x0[:-1] = np.array(x0)
        self.T = T
        self.P = P
        
        self.gfe_ni0 = gfe_ni0
        self.gfe_reltol = gfe_reltol
        self.gfe_maxiter = gfe_maxiter
        
        self.ni = np.zeros(nspecies)
        self.numberdensity = np.zeros(nspecies)
        self.E0 = np.zeros(nspecies)
        for i, sp in enumerate(self.species):
            if sum(dv for kv, dv in sp.stoichiometry.items()) == 2:
                self.E0[i] = -sp.dissociationenergy
        self.dE = np.zeros(nspecies)

        self.maxchargenumber = max(sp.chargenumber for sp in self.species)
        self.ionisedfrom = [None] * nspecies
        for i, sp in enumerate(self.species):
            if sp.chargenumber > 0:
                for sp2 in self.species:
                    if (sp2.stoichiometry == sp.stoichiometry 
                        and sp2.chargenumber == sp.chargenumber-1):
                        for j, sp3 in enumerate(self.species):
                            if sp2.name == sp3.name:
                                self.ionisedfrom[i] = j
        
        elements = [{'name': nm, 'stoichometriccoeffts': None, 'totalnumber': 0}
                    for nm in sorted(set(s for sp in self.species
                                         for s in sp.stoichiometry))]
        for elm in elements:
            elm['stoichiometriccoeffts'] = [sp.stoichiometry.get(elm['name'], 0)
                                            for sp in self.species]

        for elm in elements:
            elm['totalnumber'] = sum(1e24 * c * x0loc
                                     for c, x0loc in zip(
                                             elm['stoichiometriccoeffts'],
                                             self.x0))

        minimiser_dof = nspecies + len(elements) + 1
        self.gfematrix = np.zeros((minimiser_dof, minimiser_dof))
        self.gfevector = np.zeros(minimiser_dof)

        for i, elm in enumerate(elements):
            self.gfevector[nspecies + i] = elm['totalnumber']
            for j, sc in enumerate(elm['stoichiometriccoeffts']):
                self.gfematrix[nspecies + i, j] = sc
                self.gfematrix[j, nspecies + i] = sc
        for j, qc in enumerate(sp.chargenumber for sp in self.species):
            self.gfematrix[-1, j] = qc
            self.gfematrix[j, -1] = qc

    def recalc_E0i(self):
        """Calculate the ionisation energy lowering, using limitation theory of
        Stewart & Pyatt 1966
        """
        T, P = self.T, self.P
        kbt = constants.boltzmann * T
        self.numberdensity = self.ni * P / (self.ni.sum() * kbt) 
        weightedchargesumsqd, weightedchargesum = 0, 0
        for sp, nd in zip(self.species, self.numberdensity):
            if sp.chargenumber > 0:
                weightedchargesum += nd * sp.chargenumber
                weightedchargesumsqd += nd * sp.chargenumber ** 2
        zstar = weightedchargesumsqd / weightedchargesum
        debyed3 = (kbt / (4 * np.pi * (zstar + 1) * self.numberdensity[-1] 
                          * constants.fundamentalcharge ** 2)) ** (3/2)
        for j, sp in enumerate(self.species):
            if sp.name != 'e':
                ai3 = 3 * (sp.chargenumber + 1) / (4 * np.pi 
                                                   * self.numberdensity[-1])
                de = kbt * ((ai3/debyed3 + 1) ** (2/3) - 1) / (2 * (zstar + 1))
                self.dE[j] = de
        
        for cn in range(1, self.maxchargenumber + 1):
            for i, (sp, ifrom) in enumerate(zip(self.species, self.ionisedfrom)):
                if sp.chargenumber == cn:
                    spfrom = self.species[ifrom]
                    self.E0[i] = (self.E0[ifrom] + spfrom.ionisationenergy 
                                  - self.dE[ifrom])

    def solve_gfe(self):
        T, P, nspecies = self.T, self.P, len(self.species)
        self.ni = np.full(nspecies, self.gfe_ni0)
        governorfactors = np.linspace(0.9, 0.1, 9)
        successyn = False
        governoriters = 0
        while not successyn and governoriters < len(governorfactors):
            successyn = True
            governorfactor = governorfactors[governoriters]
            reltol = self.gfe_reltol * 10
            minimiseriters = 0
            while reltol > self.gfe_reltol:
                self.recalc_E0i()

                nisum = self.ni.sum()
                V = nisum * constants.boltzmann*T / P
                offdiag = -constants.boltzmann*T / nisum
                ondiag = constants.boltzmann*T / self.ni
                self.gfematrix[:nspecies, :nspecies] = offdiag + np.diag(ondiag)
                total = [sp.partitionfunction_total(V, T, dE) 
                         for sp, dE in zip(self.species, self.dE)]
                mu = -constants.boltzmann*T * np.log(total / self.ni) + self.E0
                self.gfevector[:nspecies] = -mu

                solution = np.linalg.solve(self.gfematrix, self.gfevector)

                newni = solution[0:nspecies]
                deltani = abs(newni - self.ni)
                maxalloweddeltani = governorfactor * self.ni

                maxniindex = newni.argmax()
                reltol = deltani[maxniindex] / solution[maxniindex]

                deltani = deltani.clip(min=maxalloweddeltani)
                newrelaxfactors = maxalloweddeltani / deltani
                relaxfactor = newrelaxfactors.min()

                self.ni = (1 - relaxfactor) * self.ni + relaxfactor * newni

                minimiseriters += 1
                if minimiseriters > self.gfe_maxiter:
                    successyn = False
                    break

            governoriters += 1

        self.numberdensity = self.ni*P / (self.ni.sum()*constants.boltzmann*T) 

        if not successyn:
            warnings.warn('Minimiser could not find a converged solution, '
                          'results may be inaccurate.')
        
        # noinspection PyUnboundLocalVariable
        logging.debug(governoriters, relaxfactor, reltol)
        logging.debug(self.ni)

    def calculate_density(self):
        """Calculate the LTE density of the plasma in kg/m3.
        """
        self.solve_gfe()
        return sum(nd * sp.molarmass / constants.avogadro
                   for sp, nd in zip(self.species, self.numberdensity))

    def calculate_enthalpy(self):
        """Calculate the LTE enthalpy of the plasma in J/kg. Note that the 
        value returned is not absolute, it is relative to an arbitrary 
        reference which may be negative or positive depending on the reference 
        energies of the diatomic species present.
        """
        self.solve_gfe()
        weightedenthalpy = sum(constants.avogadro * ni * 
                               (sp.internal_energy(self.T, dE) + E0 
                                + constants.boltzmann * self.T) 
                               for sp, ni, dE, E0 in zip(self.species, self.ni, 
                                                         self.dE, self.E0))
        weightedmolmass = sum(ni * sp.molarmass
                              for sp, ni in zip(self.species, self.ni))
        return weightedenthalpy / weightedmolmass

    def calculate_heat_capacity(self, rel_delta_T=0.001):
        """Calculate the heat capacity at constant pressure of the plasma in
        J/kg.K based on current conditions and species composition. Note that
        this is done by performing two full composition simulations when this
        function is called - can be time-consuming.
        """
        T_start = self.T
        
        self.T = T_start * (1-rel_delta_T)        
        enthalpylow = self.calculate_enthalpy()

        self.T = T_start * (1+rel_delta_T)        
        enthalpyhigh = self.calculate_enthalpy()
        
        self.T = T_start

        return (enthalpyhigh - enthalpylow) / (2 * rel_delta_T * self.T)

    def calculate_viscosity(self):
        """Calculate the viscosity of the plasma in Pa.s based on current
        conditions and species composition.
        """

        raise NotImplementedError

    def calculate_thermal_conductivity(self):
        """Calculate the thermal conductivity of the plasma in W/m.K based on
        current conditions and species composition.
        """

        raise NotImplementedError

    def calculate_electrical_conductivity(self):
        """Calculate the electrical conductivity of the plasma in 1/ohm.m based
        on current conditions and species composition.
        """

        raise NotImplementedError

    def calculate_total_emission_coefficient(self):
        """Calculate the total radiation emission coefficient of the plasma in
        W/m3 based on current conditions and species composition.
        """

        raise NotImplementedError
