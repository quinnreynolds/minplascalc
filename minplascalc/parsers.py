import logging

__all__ = ["nist_string", "nist_energylevels"]


def nist_string(nist_line):
    """Parse a string of data in NIST online database format.

    Parameters
    ----------
    nist_line : str
        A string representing a line of information from a NIST-style data
        source.

    Returns
    -------
    list of float
        Each value in the data string.
    """
    table = str.maketrans("", "", "+x?[]()")
    line = "".join(nist_line.split()).translate(table)
    records = line.split("|")[:-1]
    values = []
    for record in records:
        if "/" in record:
            num, den = record.split("/")
            value = float(num) / float(den)
        else:
            value = float(record)
        values.append(value)
    return values


def nist_energylevels(data):
    """Parse a list of atomic energy level data with each entry in NIST online
    database format.

    Parameters
    ----------
    data : list of str
        NIST-style data line for each energy level.

    Return
    ------
    list of length-2 lists
         Energy levels. Each entry contains the energy of the level Ei and the
         associated quantum number J.
    """
    energylevels = []
    try:
        name = data.name
    except AttributeError:
        name = "input"
    for i, line in enumerate(data):
        try:
            j, ei = nist_string(line)
            energylevels.append([j, ei])
        except ValueError as exception:
            logging.debug("Ignoring line %i in %s", i, name)
            logging.debug(exception)

    return energylevels
