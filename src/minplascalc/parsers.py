__all__ = ["nist_string", "nist_energy_levels"]


def nist_string(nist_line: str) -> list[float]:
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
    # Create a translation table to remove unwanted characters.
    table = str.maketrans("", "", "+x?[]()")  # Remove all +x?[]() characters.

    # Remove all whitespace (including \n \r \t \f and spaces),
    # and apply the translation table.
    line: str = "".join(nist_line.split()).translate(table)

    # Split the line into records, at the pipe character (`|`).
    records: list[str] = line.split("|")[:-1]

    values: list[float] = []
    for record in records:
        if "/" in record:
            # If the record is a fraction, convert it to a float.
            # This can happen when the total angular momentum number J is a
            # fraction.
            num, den = record.split("/")
            value = float(num) / float(den)
        else:
            # For the rest of the records, convert them to floats.
            value = float(record)
        values.append(value)
    return values


def nist_energy_levels(data: list[str]) -> list[tuple[float, float]]:
    """Parse a list of atomic energy level data in NIST online database format.

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
    energy_levels: list[tuple[float, float]] = []

    for i, line in enumerate(data):
        try:
            # Assume that the line contains exatly two values:
            # - the quantum number J,
            # - the energy Ei.
            j, ei = nist_string(line)
            energy_levels.append((j, ei))
        except ValueError as exception:
            raise ValueError(
                f"Error parsing NIST energy level data at line {i}: {line}"
            ) from exception

    return energy_levels
