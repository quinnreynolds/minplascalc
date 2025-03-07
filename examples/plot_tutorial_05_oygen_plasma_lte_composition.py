r"""
Tutorial 05: Calculating equilibrium compositions for an :math:`O_2` plasma.
============================================================================

Calculations for plasmas consisting of species of a single element are a useful starting point
because they are simple examples to illustrate the functionality of minplascalc, and their
experimental and theoretical/calculated properties are very well documented in literature;
they therefore make useful validation cases.

In order to define a plasma mixture in minplascalc, the user must specify all species present
in the plasma as well as a composition constraint - this is typically given as the composition
(in mole fractions) of the plasma-generating gas at low temperatures.
Note that the list of species does not include electrons, which are automatically assumed to
be present in any plasma.

Let's look at the case of an oxygen plasma, which we'll assume includes species
:math:`O_2`, :math:`O`, :math:`O^+`, :math:`O^{2+}`, :math:`O^-`, and :math:`O^{2-}`.
We specify the composition constraint as a mole fraction of 1 for :math:`O_2` and
zero for all the others, since the plasma originates from a pure oxygen gas at room temperature.
Note that in this case, any set of initial mole fractions of the various species will give the
same result provided they sum to unity - this is because oxygen is the only element present in
the plasma.

In order to calculate the composition of the plasma at various temperatures using these species,
execute the following code snippets in order.
The text in between indicates what each part of the code is doing.
"""  # noqa: D205

# %%
# Import the required libraries.
# ------------------------------
#
# We start by importing the modules we need:
#
# - matplotlib for drawing graphs,
# - numpy for array functions,
# - and of course minplascalc.

import matplotlib.pyplot as plt
import numpy as np

import minplascalc as mpc

# %%
# Create species objects for the species we're interested in.
# -----------------------------------------------------------
#
# Then, we create a list of Species objects for the species we want,
# as well as a list of the initial mole fractions.

species = [mpc.species.from_name(sp) for sp in ["O2", "O2+", "O", "O-", "O+", "O++"]]
x0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# %%
# Create a Mixture object for the species and mole fractions.
# -----------------------------------------------------------
#
# Next, we create a minplascalc LTE mixture object from the data above.
# The temperature in K and pressure in Pa are given to the constructor too,
# and attributes T and P will be initialised to those values.
#
# When using the raw Mixture constructor we must also provide some control information for
# the Gibbs free energy solver.

oxygen_mixture = mpc.mixture.LTE(
    species,
    x0,
    T=1000,
    P=101325,
    gfe_ni0=1e20,
    gfe_reltol=1e-10,
    gfe_maxiter=1000,
)

# %%
# Set a range of temperatures to calculate the equilibrium compositions at.
# -------------------------------------------------------------------------
#
# Next, set a range of temperatures to calculate the equilibrium compositions at - in this case
# we're going from 1000 to 25000 K in 100 K steps.
# Also initialise a list to store the composition result at each temperature.

temperatures = np.linspace(1000, 25000, 100)
species_names = [sp.name for sp in oxygen_mixture.species]
ni_list = []

# %%
# Perform the composition calculations.
# --------------------------------------
#
# Now we're ready to actually perform the composition calculations.
# We loop over all the temperatures, setting the LTE object's temperature attribute to the
# appropriate value and calculating the composition using the object's
# `calculate_composition()` function.
#
# Note that execution of this calculation is fairly compute intensive and the following code
# snippet may take a few seconds to complete.

for T in temperatures:
    oxygen_mixture.T = T
    ni_list.append(oxygen_mixture.calculate_composition())
ni = np.array(ni_list).transpose()

# %%
# Plot the results.
# -----------------
#
# Now we can visualise the results by plotting the plasma composition against temperature,
# to see how it varies.

fig, ax = plt.subplots(1, 1, figsize=(7, 4))

ax.set_title(r"$\mathregular{O_2}$ LTE plasma composition with temperature at 1 atm")
ax.set_xlabel("T [K]")
ax.set_ylabel(r"$\mathregular{n_i [m^{-3}]}$")
ax.set_ylim(1e15, 5e25)
for spn, sn in zip(ni, species_names):
    ax.semilogy(temperatures, spn, label=sn)
ax.legend(loc=(1.025, 0.25))


# %%
# Conclusion
# ----------
#
# The results obtained using minplascalc compare favourably with calculations by other authors,
# for example [Boulos2023]_. Some small deviations occur at very low concentration levels, but the
# impact of these low-concentration species on actual plasma behaviour is expected to be small since
# their concentrations are more than six orders of magnitude lower than the dominant components.

# %%
