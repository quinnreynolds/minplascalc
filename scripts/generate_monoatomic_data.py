"""Generate data for a monatomic species.

This script generates data for a monatomic species, such as O+. The data is
stored in a JSON file in the `./data/species` folder.

The user must provide the atomic symbol and the charge number of the species.
Then, automatically:
- Energy levels and spectral lines are retrieved from the NIST Atomic Spectra Database.
- The stoichiometry is set to one atom of the given species.
- The molar mass is calculated from the atomic mass of the species.
What the user must provide:
- The ionisation energy of the species,
- The multiplicity of the species,
- The polarisability of the species,
- For neutral species, the number of effective electrons of the species,
- For neutral species, the electron cross section of the species.

"""

from minplascalc.atomic_data import ATOMIC_MASS, ATOMIC_SYMBOLS, ROMAN_NUMBER
from minplascalc.nist_parsers import get_nist_energy_levels, get_nist_spectral_lines
from minplascalc.species import Monatomic
from minplascalc.units import Units
from minplascalc.utils import get_path_to_data

u = Units()

species = "O+"

atomic_symbol = species.replace("+", "")
if atomic_symbol not in ATOMIC_SYMBOLS:
    raise ValueError(
        f"Atomic symbol '{atomic_symbol}' not found in the list of atomic symbols."
    )

charge_number = species.count("+")
roman_number = ROMAN_NUMBER[charge_number]
print(atomic_symbol, roman_number)

energy_levels = get_nist_energy_levels(atomic_symbol, roman_number)
spectral_lines = get_nist_spectral_lines(atomic_symbol, roman_number)

stochoiometry = {atomic_symbol: 1}

idx_symbol = ATOMIC_SYMBOLS.index(atomic_symbol)
molar_mass = ATOMIC_MASS[idx_symbol] * 1e-3  # kg/mol

# https://physics.nist.gov/cgi-bin/ASD/ie.pl?spectra=O&units=1&format=0&order=0&at_num_out=on&sp_name_out=on&ion_charge_out=on&el_name_out=on&seq_out=on&shells_out=on&level_out=on&ion_conf_out=on&e_out=0&unc_out=on&biblio=on&submit=Retrieve+Data
ionisation_energy = 35.12112 * u.eV_to_J  # J
multiplicity = 4

# For neutral, see https://ctcp.massey.ac.nz/2023Tablepol.pdf
# (and multiply the recommanded value by * 0.1481847113 * 1e-30)
# For ionized, try here:
# - https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/66ba88c901103d79c5c7418e/original/supporting-information-for-exploring-ion-polarizabilities-and-their-correlation-with-van-der-waals-radii-a-theoretical-investigation.pdf
# - https://www.osti.gov/servlets/purl/1851084
# - https://cccbdb.nist.gov/poolcalc2x.asp
# If no data available, use approximation https://physicspages.com/pdf/Electrodynamics/Polarizability%20of%20hydrogen.pdf
# alpha = 4 pi \epsilon_0 a^3 with a = 5.29e-11 m (Bohr radius) = 1.48 × 10−31 m3
# Or you could use the following approximation: divide by 2 the polarizability of the neutral species
# This is based on C/C+, O/O+ and Si/Si+ data
polarisability = 4.50711 * 0.1481847113 * 1e-30 / 2

# Only for neutral species
# See eq.6 of [Cambi1991]_
#
# E.g. for oxygen atom, electronic configuration is 1s2 2s2 2p4
# The 1s electrons in oxygen do not participate in bonding (i.e., chemistry)
# and are called core electrons (inner electrons).
# The valence electrons (i.e., the 2s2 2p4 part) are valence electrons (outer electrons),
# which do participate in the making and breaking of bonds.
n_int = 2  # Number of inner electrons
n_ext = 6  # Number of outer electrons
n_tot = n_int + n_ext
n_eff = n_ext * (1 + (1 - n_ext / n_int) * (n_int / n_tot) ** 2)  # Eq.6 of [Cambi1991]_

effective_electrons = None
if charge_number == 0:
    effective_electrons = n_eff
# effective_electrons = 1

# Only for neutral species
# See .\scripts\fit_electron_crosss_section.py
electron_cross_section = None
# electron_cross_section = [
#     5.5100053586152e-21,
#     5.9593311976618555e-34,
#     1.4298597572532863,
#     6.158720158223838e-21
# ],


oxygenplus = Monatomic(
    name=species,
    stoichiometry=stochoiometry,
    molarmass=molar_mass,
    chargenumber=charge_number,
    ionisationenergy=ionisation_energy,
    energylevels=energy_levels,
    polarisability=polarisability,
    multiplicity=multiplicity,
    effectiveelectrons=effective_electrons,
    electroncrosssection=electron_cross_section,
    emissionlines=spectral_lines,
    sources=[
        "Kramida, A., Ralchenko, Yu., Reader, J. and NIST ASD Team (2024)."
        "NIST Atomic Spectra Database (version 5.12), [Online]."
        "Available: https://physics.nist.gov/asd [Thu Feb 13 2025]."
        "National Institute of Standards and Technology, Gaithersburg, MD."
        "DOI: https://doi.org/10.18434/T4W30F"
    ],
)

species_folder = get_path_to_data("species")
oxygenplus.to_file(datafile=species_folder / f"{species}.json")

print(f"Data for {species} has been generated and saved in {species_folder}.")
