r"""
Tutorial 03: Adding a new polyatomic species entry to your local data store.
============================================================================

If you would like to do calculations using a plasma species that you haven't used before,
you will need to generate a minplascalc data entry for it first.
You only need to do this once< - minplascalc can store the species data in a plain-text file
formatted using JSON syntax, and it will then be available for use in any of your future calculations.

The procedure for generating a minplascalc data entry for a new polyatomic species is entirely
manual since the data required must be retrieved from various locations.
The majority of data for a wide range of species is available from the NIST Chemistry WebBook,
http://webbook.nist.gov/chemistry/form-ser/. Enter the formula for the polyatomic species
you're interested in in the search box at the top of the page, and click the Search button.
If you're looking for data for charged diatomic ions, make sure the "Exclude ions from the
search" box is unchecked.
"""  # noqa: D205

# %%
# Let's build a data entry for water, H2O:
#
# .. image:: ../../data/demo/img/webbook_landingpage_poly.png
#   :width: 300
#   :alt: NIST Chemistry WebBook
# .. image:: ../../data/demo/img/webbook_h2o.png
#   :width: 300
#   :alt: SiO WebBook entry

# %%
# The first pieces of information needed are the ionisation energy (to turn H2O into H2O+)
# and the bond dissociation energy (to turn H2O into monatomic H and O).
# The ionisation energy can be found on the "Gas phase ion energetics" link on the Chemistry WebBook
# species page and is usually in units of eV, so remember to convert:
#
# .. image:: ../../data/demo/img/webbook_ionenergetics_poly.png
#   :width: 400
#   :alt: H2O ionisation energy


# %%
# The dissociation energy can be found either on the NIST Computational Chemistry Comparison and
# Benchmark Database (http://cccbdb.nist.gov/introx.asp), or alternatively from chemistry textbooks
# or other sources.In the case of H2O, the value is 917.82 kJ/mol.
#
# We also need some information about the electronic ground state of the molecule, and its vibrational
# and rotational parameters. For polyatomics this can be obtained from the CCCBDB above,or other sources.
#
# .. image:: ../../data/demo/img/cccbdb_landing.png
#   :width: 250
#   :alt: CCCBDB data page for water
# .. image:: ../../data/demo/img/cccbdb_constants1.png
#   :width: 250
#   :alt: Ground state level entry


# %%
# The multiplicity and electronic degeneracy of the ground state, :math:`g_0`, is generally given by the
# superscripted number in the molecular term expression in the "State" column - in the case of H2O,
# it's 1.
#
# Additional pieces of data needed to create a polyatomic species object are:
#
# * The elemental stoichiometry of the species.
# * The molar mass of the species, in kg/mol.
# * The electric charge in units of the elementary charge.
# * The ionisation energy of the species, in J.
# * The dissociation (atomisation) energy of the species, in J, only used by neutral species.
# * The symmetry constant :math:`\sigma_s`, and a flag indicating whether the molecule is linear or not.
# * The vibrational constants :math:`\omega_{e,i}`, in J.
# * The rotational constants :math:`A`, :math:`B`, and :math:`C` in J.
# * The polarisability of the species, in :math:`O_2`.
# * The multiplicity of the ground state.
# * The number of effective valence electrons (see docstring for more information),
#   only used by neutral species.
# * Information about electron collision cross sections (see docstring for more information),
#   only used by neutral species.
# * A list of radiation emission line data, if available (see Tutorial 1 for more information).
# * A list of sources describing the provenance of the data.

# %%
# Now we have all the information to build a minplascalc data entry for the H2O species,
# which can be done by running the following code snippet:
import numpy
from scipy import constants

import minplascalc as mpc

invcm_to_joule = constants.Boltzmann / (
    0.01
    * constants.physical_constants["Boltzmann constant in inverse meters per kelvin"][0]
)
eV_to_joule = constants.elementary_charge

water = mpc.species.Polyatomic(
    name="H2O",
    stoichiometry={"H": 2, "O": 1},
    molar_mass=0.0180153,
    charge_number=0,
    ionisation_energy=eV_to_joule * 12.621,
    dissociation_energy=(498.7 + 428) * 1000 / constants.Avogadro,
    linear_yn=False,
    sigma_s=2,
    g0=1,
    wi_e=list(invcm_to_joule * numpy.array([3657, 1595, 3756])),
    abc_e=list(invcm_to_joule * numpy.array([27.877, 14.512, 9.285])),
    polarisability=1.501e-30,
    multiplicity=1,
    effective_electrons=7.04,
    electron_cross_section=(9.274e-36, 41.81, -2.090, 1.066e-20),
    emission_lines=[],
    sources=[
        "NIST Chemistry WebBook, NIST Standard Reference Database Number 69. "
        "PJ Linstrom and WG Mallard (Editors), National Institute of Standards "
        "and Technology, Gaithersburg MD., http://webbook.nist.gov/chemistry/, "
        "doi:10.18434/T4D303",
        "NIST Computational Chemistry Comparison and Benchmark Database. "
        "NIST Standard Reference Database Number 101, Release 21, August 2020, "
        "Editor: Russell D. Johnson III, http://cccbdb.nist.gov/, "
        "doi:10.18434/T47C7Z",
    ],
)

print(water)

water.to_file()

# %%
# What's happening here?
# First we import the minplascalc package, then we create a `Polyatomic` species object using our
# data, and finally we save the contents to a file.
#
# The `Polyatomic` class constructor takes as arguments the list of data described above.
# Once created, the object can be written out to disk using the Species object's `to_file`
# utility function, which saves data to a human-readable JSON-formatted file.
# It takes an optional file path argument - if omitted, the filename is created using the
# species' name and written to the current working directory.
#
# After this process it will be possible to create anH2O species object in any minplascalc
# calculation by importing it using either the explicit path to the JSON file, or (preferably)
# just the name of the species provided the JSON file is stored in any of the standard
# minplascalc data paths - see later demos for examples. Note that for this to work correctly
# in all cases, minplascalc's JSON files must be stored on a case-sensitive operating system
# or partition.

# %%
