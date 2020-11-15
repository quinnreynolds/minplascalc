import json
import numpy
import pathlib
from copy import deepcopy
from scipy import constants

__all__ = ['SPECIESPATH', 'to_file', 'from_file', 'from_name', 'Monatomic', 
           'Diatomic', 'Electron']

DATAPATH = pathlib.Path(__file__).parent / 'data'
SPECIESPATH = DATAPATH / 'species'
        

def from_file(datafile):
    """Create a species from a data file in JSON format.

    Parameters
    ----------
    datafile : string
        Path to a JSON data file describing the electronic and molecular
        properties of the species.
    """
    with open(datafile) as f:
        spdata = json.load(f)
    atomcount = sum(v for k, v in spdata['stoichiometry'].items())
    if atomcount == 1:
        return Monatomic(spdata['name'], spdata['stoichiometry'], 
                         spdata['molarmass'], spdata['chargenumber'], 
                         spdata['ionisationenergy'], spdata['energylevels'], 
                         spdata['sources'])
    else:
        return Diatomic(spdata['name'], spdata['stoichiometry'], 
                        spdata['molarmass'], spdata['chargenumber'], 
                        spdata['ionisationenergy'], 
                        spdata['dissociationenergy'], spdata['sigma_s'], 
                        spdata['g0'], spdata['w_e'], spdata['b_e'], 
                        spdata['sources'])


def from_name(name):
    """Create a species from the species database.

    Parameters
    ----------
    name : str
        Name of the species.
    """
    filename = SPECIESPATH / (name + '.json')
    return from_file(str(filename))


class BaseSpecies:
    def partitionfunction_total(self, V, T, dE):
        return (V * self.partitionfunction_translational(T)
                * self.partitionfunction_internal(T, dE))

    def partitionfunction_translational(self, T):
        return ((2 * numpy.pi * self.molarmass * constants.Boltzmann * T)
                / (constants.Avogadro * constants.Planck ** 2)) ** 1.5

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
            A unique identifier for the species.
        stoichiometry : dictionary
            Dictionary describing the elemental stoichiometry of the species.
        molarmass : float
            Molar mass of the species in kg/mol.
        chargenumber : int
            Charge on the species (in integer units of the fundamental charge).
        """
        self.name = name
        self.stoichiometry = deepcopy(stoichiometry)
        self.molarmass = molarmass
        self.chargenumber = chargenumber
        if self.chargenumber < 0:
            raise ValueError('Negatively charged ions not implemented.')

    def to_file(self, datafile=None):
        """Save a Species object to a file for easy re-use.
        
        Parameters
        ----------
        datafile : str or Path, optional
            The file to which the output should be saved (full path). The 
            default is None in which case the Species' name attribute will be 
            used for the file name, and it will be saved to the cwd.
        """
        if datafile:
            with open(datafile, 'w') as f:
                json.dump(self.__dict__, f, indent=4)
        else:
            with open(self.name + '.json', 'w') as f:
                json.dump(self.__dict__, f, indent=4)


class Monatomic(Species):
    def __init__(self, name, stoichiometry, molarmass, chargenumber, 
                 ionisationenergy, energylevels, sources):
        """Class for monatomic plasma species (single atoms and ions).
    
        Parameters
        ----------
        name : string
            A unique identifier for the species.
        stoichiometry : dictionary
            Dictionary describing the elemental stoichiometry of the species 
            (e.g. {'O': 1} for O or O+).
        molarmass : float
            Molar mass of the species in kg/mol.
        chargenumber : int
            Charge on the species (in integer units of the fundamental charge).
        ionisationenergy : float
            Ionisation energy of the species in J.
        energylevels : list of length-2 lists
            Atomic energy level data - each entry in the list contains a pair of
            values giving the level's quantum number and its energy
            respectively, with energy in J.
        sources : list of str
            Each entry represents a reference from which the data was
            obtained.
        """
        super().__init__(name, stoichiometry, molarmass, chargenumber)
        
        self.ionisationenergy = ionisationenergy
        self.energylevels = deepcopy(energylevels)
        self.sources = deepcopy(sources)

    def __repr__(self):
        return (f'{self.__class__.__name__ }(name={self.name},'
                f'stoichiometry={self.stoichiometry},'
                f'molarmass={self.molarmass},chargenumber={self.chargenumber},'
                f'ionisationenergy={self.ionisationenergy},'
                f'energylevels={self.energylevels},sources={self.sources})')

    def __str__(self):
        if numpy.isclose(0, self.chargenumber):
            sptype = 'Monatomic atom'
        else:
            sptype = 'Monatomic ion'
        return (f'Species: {self.name}\nType: {sptype}\n'
                f'Stoichiometry: {self.stoichiometry}\n'
                f'Molar mass: {self.molarmass} kg/mol\n'
                f'Charge number: {self.chargenumber}\n'
                f'Ionisation energy: {self.ionisationenergy} J\n'
                f'Energy levels: {len(self.energylevels)}')

    def partitionfunction_internal(self, T, dE):
        kbt = constants.Boltzmann * T
        partitionval = 0       
        for j, eij in self.energylevels:
            if eij < (self.ionisationenergy - dE):
                partitionval += (2*j+1) * numpy.exp(-eij / kbt)
        return partitionval

    def internal_energy(self, T, dE):
        kbt = constants.Boltzmann * T
        translationalenergy = 1.5 * kbt
        electronicenergy = 0
        for j, eij in self.energylevels:
            if eij < (self.ionisationenergy - dE):
                electronicenergy += (2*j+1) * eij * numpy.exp(-eij / kbt)
        electronicenergy /= self.partitionfunction_internal(T, dE)
        return translationalenergy + electronicenergy
    

class Diatomic(Species):
    def __init__(self, name, stoichiometry, molarmass, chargenumber,
                 ionisationenergy, dissociationenergy, sigma_s, g0, w_e, b_e, 
                 sources):
        """Class for diatomic plasma species (bonded pairs of atoms, as 
        neutral particles or ions).
    
        Parameters
        ----------
        name : string
            A unique identifier for the species.
        stoichiometry : dictionary
            Dictionary describing the elemental stoichiometry of the species 
            (e.g. {'Si': 1, 'O': 1} for SiO or SiO+).
        molarmass : float
            Molar mass of the species in kg/mol.
        chargenumber : int
            Charge on the species (in integer units of the fundamental charge).
        ionisationenergy : float
            Ionisation energy of the species in J.
        dissociationenergy : float
            Dissociation energy of the species in J.
        sigma_s : int
            Symmetry constant (=2 for homonuclear molecules, =1 for 
            heteronuclear).
        g0 : float
            Ground state electronic energy level degeneracy.
        w_e : float
            Vibrational energy level constant in J.
        b_e : float
            Rotational energy level constant in J.
        sources : list of str
            Each dictionary represents a reference source from which the data 
            was obtained.
        """
        super().__init__(name, stoichiometry, molarmass, chargenumber)

        self.dissociationenergy = dissociationenergy
        self.ionisationenergy = ionisationenergy
        self.sigma_s = sigma_s
        self.g0 = g0
        self.w_e = w_e
        self.b_e = b_e
        self.sources = deepcopy(sources)

    def __repr__(self):
        return (f'{self.__class__.__name__}(name={self.name},'
                f'stoichiometry={self.stoichiometry},'
                f'molarmass={self.molarmass},'
                f'chargenumber={self.chargenumber},'
                f'dissociationenergy={self.dissociationenergy},'
                f'ionisationenergy={self.ionisationenergy},'
                f'sigma_s={self.sigma_s},g0={self.g0},w_e={self.w_e},'
                f'b_e={self.b_e},sources={self.sources})')

    def __str__(self):
        if numpy.isclose(0, self.chargenumber):
            sptype = 'Diatomic molecule'
        else:
            sptype = 'Diatomic ion'
        return (f'Species: {self.name}\nType: {sptype}\n'
                f'Stoichiometry: {self.stoichiometry}\n'
                f'Molar mass: {self.molarmass} kg/mol\n'
                f'Charge number: {self.chargenumber}\n'
                f'Dissociation energy: {self.dissociationenergy} J\n'
                f'Ionisation energy: {self.ionisationenergy} J\n'
                f'sigma_s: {self.sigma_s}\ng0: {self.g0}\nw_e: {self.w_e} J\n'
                f'B_e: {self.b_e} J')

    def partitionfunction_internal(self, T, dE):
        kbt = constants.Boltzmann * T
        electronicpartition = self.g0
        vibrationalpartition = 1 / (1 - numpy.exp(-self.w_e / kbt))
        rotationalpartition = kbt / (self.sigma_s * self.b_e)
        return electronicpartition * vibrationalpartition * rotationalpartition

    def internal_energy(self, T, dE):
        kbt = constants.Boltzmann * T
        translationalenergy = 1.5 * kbt
        electronicenergy = 0
        rotationalenergy = kbt
        vibrationalenergy = self.w_e * (numpy.exp(-self.w_e / kbt)
                                        / (1 - numpy.exp(-self.w_e / kbt)))
        return (translationalenergy + electronicenergy + rotationalenergy 
                + vibrationalenergy)


class Electron(BaseSpecies):
    def __init__(self):
        """Class for electrons as a plasma species.
        """
        self.name = 'e'
        self.stoichiometry = {}
        self.molarmass = constants.electron_mass * constants.Avogadro
        self.chargenumber = -1

    def __repr__(self):
        return (f'{self.__class__.__name__}(name={self.name},'
                f'molarmass={self.molarmass},chargenumber={self.chargenumber})')

    def __str__(self):
        return (f'Species: e\nType: Electron\n'
                f'Molar mass: {self.molarmass} kg/mol\n'
                f'Charge number: {self.chargenumber}')

    # noinspection PyUnusedLocal
    def partitionfunction_internal(self, T, dE):
        return 2.

    def internal_energy(self, T, dE):
        translationalenergy = 1.5 * constants.Boltzmann * T
        electronicenergy = 0
        return translationalenergy + electronicenergy
