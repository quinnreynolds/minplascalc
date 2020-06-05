import logging
from scipy import constants


def molar_mass_calculator(protons, neutrons, electrons):
    """Estimate the molar mass in kg/mol of a species based on its nuclear and
    electronic structure, if you can't get it anywhere else for some reason.
    
    Returns
    -------
    float
        Molar mass in kg/mol.
    """
    return constants.Avogadro * (protons * constants.proton_mass
                                 + electrons * constants.electron_mass
                                 + neutrons * (constants.neutron_mass))


def parse_values(nist_line):
    """Helper function to tidy up a string of data copied from NIST online
    databases.
    
    Parameters
    ----------
    nist_line : str
        A string representing a line of information obtained from a NIST-style 
        data source.
    
    Returns
    -------
    list of float
        Each value in the data string.
    """
    table = str.maketrans('', '', '+x?[]()')
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
        NIST energy level file data.

    Return
    ------
    list of length-2 lists
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
