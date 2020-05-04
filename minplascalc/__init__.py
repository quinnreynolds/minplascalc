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
MIXTUREPATH = DATAPATH / 'mixtures'

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
        
        
def species_from_file(datafile, numberofparticles=0, x0=0):
    """Create a species from a data file in JSON format.

    Parameters
    ----------
    datafile : string
        Path to a JSON data file describing the electronic and molecular
        properties of the species
    numberofparticles : float
        Absolute particle count in plasma (default 0)
    x0 : float
        Initial mole fraction of species in plasma-gen gas (default 0)
    """
    with open(datafile) as f:
        spdata = json.load(f)

    atomcount = sum(v for k, v in spdata['stoichiometry'].items())
    if atomcount == 1:
        return MonatomicSpecies(spdata['name'], spdata['stoichiometry'], 
                                spdata['molarmass'], spdata['chargenumber'], 
                                spdata['ionisationenergy'], 
                                spdata['energylevels'], spdata['sources'], 
                                numberofparticles, x0)
    else:
        return DiatomicSpecies(spdata['name'], spdata['stoichiometry'], 
                               spdata['molarmass'], spdata['chargenumber'], 
                               spdata['ionisationenergy'], 
                               spdata['dissociationenergy'], spdata['sigma_s'], 
                               spdata['g0'], spdata['w_e'], spdata['b_e'], 
                               spdata['sources'], numberofparticles, x0)


def species_from_name(name, numberofparticles=0, x0=0):
    """ Create a species from the species database

    Parameters
    ----------
    name : str
        Name of the species
    numberofparticles : float
        Absolute particle count in plasma (default 0)
    x0 : float
        Initial mole fraction of species in plasma-gen gas (default 0)
    """

    filename = SPECIESPATH / (name + '.json')
    return species_from_file(str(filename), numberofparticles, x0)


class BaseSpecies:
    def partitionfunction_total(self, V, T):
        return (V * self.partitionfunction_translational(T)
                * self.partitionfunction_internal(T))

    def partitionfunction_translational(self, T):
        return ((2 * np.pi * self.molarmass * constants.boltzmann * T)
                / (constants.avogadro * constants.planck ** 2)) ** 1.5

    def partitionfunction_internal(self, T):
        raise NotImplementedError

    def internal_energy(self, T):
        raise NotImplementedError


class Species(BaseSpecies):
    def __init__(self, name, stoichiometry, molarmass, chargenumber,
                 numberofparticles, x0):
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
        numberofparticles : float
            Absolute particle count in plasma
        x0 : float
            Initial mole fraction of species in plasma-gen gas
        """
        self.name = name
        self.stoichiometry = deepcopy(stoichiometry)
        self.molarmass = molarmass
        self.chargenumber = chargenumber
        if self.chargenumber < 0:
            raise ValueError('Error! Negatively charged ions not implemented'
                             ' yet.')
        self.numberofparticles = numberofparticles
        self.x0 = x0

        self.numberdensity = 0


class MonatomicSpecies(Species):
    def __init__(self, name, stoichiometry, molarmass, chargenumber, 
                 ionisationenergy, energylevels, sources, numberofparticles, 
                 x0):
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
        numberofparticles : float
            Absolute particle count in plasma
        x0 : float
            Initial mole fraction of species in plasma-gen gas
        """
        super().__init__(name, stoichiometry, molarmass, chargenumber, 
                         numberofparticles, x0)
        
        self.ionisationenergy = ionisationenergy
        self.energylevels = deepcopy(energylevels)
        self.sources = deepcopy(sources)
        self.e0 = 0

        self.deltaionisationenergy = 0

    def partitionfunction_internal(self, T):
        kbt = constants.boltzmann * T
        partitionval = 0       
        for j, eij in self.energylevels:
            if eij < (self.ionisationenergy - self.deltaionisationenergy):
                partitionval += (2*j+1) * np.exp(-eij / kbt)
        return partitionval

    def internal_energy(self, T):
        kbt = constants.boltzmann * T
        translationalenergy = 1.5 * kbt
        electronicenergy = 0
        for j, eij in self.energylevels:
            if eij < (self.ionisationenergy - self.deltaionisationenergy):
                electronicenergy += (2*j+1) * eij * np.exp(-eij / kbt)
        electronicenergy /= self.partitionfunction_internal(T)
        return translationalenergy + electronicenergy


class DiatomicSpecies(Species):
    def __init__(self, name, stoichiometry, molarmass, chargenumber,
                 ionisationenergy, dissociationenergy, sigma_s, g0, w_e, b_e, 
                 sources, numberofparticles, x0):
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
        numberofparticles : float
            Absolute particle count in plasma
        x0 : float
            Initial mole fraction of species in plasma-gen gas
        """
        super().__init__(name, stoichiometry, molarmass, chargenumber, 
                         numberofparticles, x0)

        self.dissociationenergy = dissociationenergy
        self.ionisationenergy = ionisationenergy
        self.sigma_s = sigma_s
        self.g0 = g0
        self.w_e = w_e
        self.b_e = b_e
        self.sources = deepcopy(sources)
        self.e0 = -self.dissociationenergy

        self.deltaionisationenergy = 0

    def partitionfunction_internal(self, T):
        kbt = constants.boltzmann * T
        electronicpartition = self.g0
        vibrationalpartition = 1. / (1. - np.exp(-self.w_e / kbt))
        rotationalpartition = kbt / (self.sigma_s * self.b_e)
        return electronicpartition * vibrationalpartition * rotationalpartition

    def internal_energy(self, T):
        kbt = constants.boltzmann * T
        translationalenergy = 1.5 * kbt
        electronicenergy = 0
        rotationalenergy = kbt
        vibrationalenergy = self.w_e * (np.exp(-self.w_e / kbt)
                                        / (1. - np.exp(-self.w_e / kbt)))
        return (translationalenergy + electronicenergy + rotationalenergy 
                + vibrationalenergy)


class ElectronSpecies(BaseSpecies):
    def __init__(self, numberofparticles):
        """Class for electrons as a plasma species.

        Parameters
        ----------
        numberofparticles : float
            Absolute particle count in plasma
        """
        self.name = 'e'
        self.stoichiometry = {}
        self.molarmass = constants.electronmass * constants.avogadro
        self.chargenumber = -1
        self.numberdensity = 0
        self.e0 = 0

        self.numberofparticles = numberofparticles        
        self.x0 = 0

    # noinspection PyUnusedLocal
    def partitionfunction_internal(self, T):
        return 2.

    def internal_energy(self, T):
        translationalenergy = 1.5 * constants.boltzmann * T
        electronicenergy = 0
        return translationalenergy + electronicenergy


class Mixture:
    def __init__(self, mixture_file, T=10000, P=101325):
        """Class representing a thermal plasma specification with multiple
        species, and methods for calculating equilibrium species concentrations
        at different temperatures and pressures using the principle of Gibbs
        free energy minimisation.

        Parameters
        ----------
        mixture_file : string
            Path to a JSON data file containing species and initial mole
            fractions
        T : float
            Temperature value in K, for initialisation (default 10000)
        P : float
            Pressure value in Pa, for initialisation (default 101325)
        """

        self.T = T
        self.P = P

        with open(mixture_file) as sf:
            jsondata = json.load(sf)

        # Random order upsets the nonlinearities in the minimiser resulting in
        # non-reproducibility between runs. Make sure this order is maintained
        self.species = [species_from_name(spdata['species'], x0=spdata['x0'])
                        for spdata in jsondata['speciesList']]
        self.species.append(ElectronSpecies(0))

        # Random order upsets the nonlinearities in the minimiser resulting in
        # non-reproducibility between runs
        elements = [{'name': nm, 'stoichometriccoeffts': None, 'totalnumber': 0}
                    for nm in sorted(set(s for sp in self.species
                                         for s in sp.stoichiometry))]

        self.maxchargenumber = max(sp.chargenumber for sp in self.species)
        # Set species which each +ve charged ion originates from
        for sp in self.species:
            if sp.chargenumber > 0:
                for sp2 in self.species:
                    if (sp2.stoichiometry == sp.stoichiometry 
                        and sp2.chargenumber == sp.chargenumber-1):
                        sp.ionisedfrom = sp2

        # Set stoichiometry and charge coefficient arrays for mass action and
        # electroneutrality constraints
        for elm in elements:
            elm['stoichiometriccoeffts'] = [sp.stoichiometry.get(elm['name'], 0)
                                            for sp in self.species]

        # Set element totals for constraints from provided initial conditions
        nt0 = self.P / (constants.boltzmann * self.T)
        for elm in elements:
            elm['totalnumber'] = sum(nt0 * c * sp.x0
                                     for c, sp in zip(
                                             elm['stoichiometriccoeffts'],
                                             self.species))

        # Set up A matrix, b and ni vectors for GFE minimiser
        minimiser_dof = len(self.species) + len(elements) + 1
        self.gfematrix = np.zeros((minimiser_dof, minimiser_dof))
        self.gfevector = np.zeros(minimiser_dof)
        self.ni = np.zeros(len(self.species))

        for i, elm in enumerate(elements):
            self.gfevector[len(self.species) + i] = elm['totalnumber']
            for j, sc in enumerate(elm['stoichiometriccoeffts']):
                self.gfematrix[len(self.species) + i, j] = sc
                self.gfematrix[j, len(self.species) + i] = sc
        for j, qc in enumerate([sp.chargenumber for sp in self.species]):
            self.gfematrix[-1, j] = qc
            self.gfematrix[j, -1] = qc

    def initialise_ni(self, ni):
        for j, sp in enumerate(self.species):
            self.ni[j] = ni[j]
            sp.numberofparticles = ni[j]

    def read_ni(self):
        for j, sp in enumerate(self.species):
            self.ni[j] = sp.numberofparticles

    def write_ni(self):
        for j, sp in enumerate(self.species):
            sp.numberofparticles = self.ni[j]

    def write_numberdensity(self):
        V  = self.ni.sum() * constants.boltzmann * self.T / self.P
        for j, sp in enumerate(self.species):
            sp.numberdensity = self.ni[j] / V 

    def recalc_e0i(self):
        # deltaionisationenergy recalculation, using limitation theory of
        # Stewart & Pyatt 1966
        kbt = constants.boltzmann * self.T
        ndi = self.ni / (self.ni.sum() * kbt / self.P)
        weightedchargesumsqd = 0
        weightedchargesum = 0
        for j, sp in enumerate(self.species):
            if sp.chargenumber > 0:
                weightedchargesum += ndi[j] * sp.chargenumber
                weightedchargesumsqd += ndi[j] * sp.chargenumber ** 2
        zstar = weightedchargesumsqd / weightedchargesum
        debyed3 = (kbt / (4 * np.pi * (zstar + 1) * ndi[-1] 
                          * constants.fundamentalcharge ** 2)) ** (3/2)
        for j, sp in enumerate(self.species):
            if sp.name != 'e':
                ai3 = 3 * (sp.chargenumber + 1) / (4 * np.pi * ndi[-1])
                de = kbt * ((ai3/debyed3 + 1) ** (2/3) - 1) / (2 * (zstar + 1))
                self.deltaionisationenergy = de
        
        for cn in range(1, self.maxchargenumber + 1):
            for sp in self.species:
                if sp.chargenumber == cn:
                    sp.e0 = (sp.ionisedfrom.e0 + sp.ionisedfrom.ionisationenergy
                             - sp.ionisedfrom.deltaionisationenergy)

    def recalc_gfearrays(self):
        ni = self.ni
        T = self.T
        P = self.P

        nisum = ni.sum()
        V = nisum * constants.boltzmann * T / P
        offdiagonal = -constants.boltzmann * T / nisum
        nspecies = len(self.species)

        ondiagonal = constants.boltzmann * T / ni
        self.gfematrix[:nspecies, :nspecies] = offdiagonal + np.diag(ondiagonal)
        total = [sp.partitionfunction_total(V, T) for sp in self.species]
        e0 = [sp.e0 for sp in self.species]
        mu = -constants.boltzmann * T * np.log(total / ni) + e0
        self.gfevector[:nspecies] = -mu


    def solve_gfe(self, relativetolerance=1e-10, maxiters=1000):
        self.read_ni()

        governorfactors = np.linspace(0.9, 0.1, 9)
        successyn = False
        governoriters = 0
        while not successyn and governoriters < len(governorfactors):
            successyn = True
            governorfactor = governorfactors[governoriters]
            reltol = relativetolerance * 10
            minimiseriters = 0
            while reltol > relativetolerance:
                self.recalc_e0i()
                self.recalc_gfearrays()

                solution = np.linalg.solve(self.gfematrix, self.gfevector)

                newni = solution[0:len(self.species)]
                deltani = abs(newni - self.ni)
                maxalloweddeltani = governorfactor * self.ni

                maxniindex = newni.argmax()
                reltol = deltani[maxniindex] / solution[maxniindex]

                deltani = deltani.clip(min=maxalloweddeltani)
                newrelaxfactors = maxalloweddeltani / deltani
                relaxfactor = newrelaxfactors.min()

                self.ni = (1 - relaxfactor) * self.ni + relaxfactor * newni

                minimiseriters += 1
                if minimiseriters > maxiters:
                    successyn = False
                    break

            governoriters += 1

        if not successyn:
            warnings.warn('Minimiser could not find a converged solution, '
                          'results may be inaccurate.')

        # noinspection PyUnboundLocalVariable
        logging.debug(governoriters, relaxfactor, reltol)
        logging.debug(self.ni)

        self.write_ni()
        self.write_numberdensity()

    def calculate_density(self):
        """Calculate the density of the plasma in kg/m3 based on current
        conditions and species composition.
        """

        return sum(sp.numberdensity * sp.molarmass / constants.avogadro
                   for sp in self.species)

    def calculate_heat_capacity(self, init_ni=1e20, rel_delta_T=0.001):
        """Calculate the heat capacity at constant pressure of the plasma in
        J/kg.K based on current conditions and species composition. Note that
        this is done by performing two full composition simulations when this
        function is called - can be time-consuming.
        """

        T = self.T

        self.initialise_ni([init_ni for i in range(len(self.species))])
        self.T = (1 - rel_delta_T) * T
        self.solve_gfe()
        enthalpylow = self.calculate_enthalpy()

        self.initialise_ni([init_ni for i in range(len(self.species))])
        self.T = (1 + rel_delta_T) * T
        self.solve_gfe()
        enthalpyhigh = self.calculate_enthalpy()

        self.T = T

        return (enthalpyhigh - enthalpylow) / (2 * rel_delta_T * T)

    def calculate_enthalpy(self):
        """Calculate the enthalpy of the plasma in J/kg based on current
        conditions and species composition. Note that the value returned is not
        absolute, it is relative to an arbitrary reference which may be
        negative or positive depending on the reference energies of the diatomic
        species present.
        """

        T = self.T
        weightedenthalpy = sum(constants.avogadro * sp.numberofparticles 
                               * (sp.internal_energy(T) + sp.e0 
                                  + constants.boltzmann * T) 
                               for sp in self.species)
        weightedmolmass = sum(sp.numberofparticles * sp.molarmass
                              for sp in self.species)
        return weightedenthalpy / weightedmolmass

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
